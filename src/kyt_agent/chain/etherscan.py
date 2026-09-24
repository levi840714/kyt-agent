import time
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import httpx

from kyt_agent.chain.client import ContractInfo, TransactionDetail, Transfer

BASE_URL = "https://api.etherscan.io/v2/api"
WEI_PER_ETH = Decimal(10) ** 18


class EtherscanError(RuntimeError):
    """Etherscan 回傳錯誤，或重試後仍被限流。"""


class EtherscanClient:
    def __init__(
        self,
        api_key: str,
        *,
        page_size: int = 100,
        min_interval: float = 0.25,
        max_retries: int = 3,
        backoff: float = 1.0,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._api_key = api_key
        self._page_size = page_size
        self._min_interval = min_interval
        self._max_retries = max_retries
        self._backoff = backoff
        self._http = http or httpx.Client(timeout=30)
        self._sleep = sleep
        self._clock = clock
        self._last_call = float("-inf")

    def get_transfers(self, address: str) -> list[Transfer]:
        query = {
            "module": "account",
            "address": address,
            "page": 1,
            "offset": self._page_size,
            "sort": "desc",
        }
        transfers = [
            *(_from_native(tx) for tx in self._call(action="txlist", **query)),
            *(_from_native(tx) for tx in self._call(action="txlistinternal", **query)),
            *(_from_token(tx) for tx in self._call(action="tokentx", **query)),
        ]
        return [transfer for transfer in transfers if transfer is not None and transfer.amount > 0]

    def get_transaction(self, tx_hash: str) -> TransactionDetail | None:
        tx = self._call(module="proxy", action="eth_getTransactionByHash", txhash=tx_hash)
        if tx is None:
            return None
        return TransactionDetail(
            tx_hash=tx["hash"],
            sender=tx["from"],
            recipient=tx["to"],
            value_eth=Decimal(int(tx["value"], 16)) / WEI_PER_ETH,
            block_number=int(tx["blockNumber"] or "0x0", 16),
            method_id=tx["input"][:10],
        )

    def get_contract_info(self, address: str) -> ContractInfo:
        code = self._call(module="proxy", action="eth_getCode", address=address, tag="latest")
        if code in ("0x", "0x0"):
            return ContractInfo(address=address, is_contract=False)
        source = self._call(module="contract", action="getsourcecode", address=address)
        name = source[0].get("ContractName") if source else None
        return ContractInfo(address=address, is_contract=True, name=name or None)

    def _call(self, **params: Any) -> Any:
        params = {"chainid": 1, **params, "apikey": self._api_key}
        attempt = 0
        while True:
            self._throttle()
            response = self._http.get(BASE_URL, params=params)
            response.raise_for_status()
            body = response.json()
            if "jsonrpc" in body:
                if "error" in body:
                    raise EtherscanError(body["error"]["message"])
                return body["result"]
            if body["status"] == "1":
                return body["result"]
            if body["message"].startswith("No transactions found"):
                return []
            if "rate limit" in str(body["result"]).lower() and attempt < self._max_retries:
                self._sleep(self._backoff * 2**attempt)
                attempt += 1
                continue
            raise EtherscanError(f"{body['message']}: {body['result']}")

    def _throttle(self) -> None:
        wait = self._last_call + self._min_interval - self._clock()
        if wait > 0:
            self._sleep(wait)
        self._last_call = self._clock()


def _from_native(tx: dict[str, str]) -> Transfer | None:
    if tx["isError"] != "0":
        return None
    return Transfer(
        tx_hash=tx["hash"],
        timestamp=int(tx["timeStamp"]),
        sender=tx["from"],
        recipient=tx["to"] or tx["contractAddress"],
        amount=Decimal(tx["value"]) / WEI_PER_ETH,
        asset="ETH",
    )


def _from_token(tx: dict[str, str]) -> Transfer:
    decimals = int(tx["tokenDecimal"] or 0)
    return Transfer(
        tx_hash=tx["hash"],
        timestamp=int(tx["timeStamp"]),
        sender=tx["from"],
        recipient=tx["to"],
        amount=Decimal(tx["value"]) / Decimal(10) ** decimals,
        asset=tx["tokenSymbol"] or "UNKNOWN",
    )
