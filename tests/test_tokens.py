from decimal import Decimal

import pytest
from pydantic import ValidationError

from kyt_agent.tokens import KnownToken, TokenRegistry, classify_transfer
from tests.fakes import TARGET, UNKNOWN, transfer

USDT = "0x" + "d" * 40
FAKE = "0x" + "e" * 40
NATIVE_DUST = Decimal("0.0001")


def usdt(contract: str = USDT, symbol: str = "USDT") -> KnownToken:
    return KnownToken(
        contract=contract, symbol=symbol, decimals=6, dust_threshold=Decimal("0.01"), source="test"
    )


REGISTRY = TokenRegistry([usdt()])


def test_registry_looks_up_by_contract_and_symbol():
    assert REGISTRY.by_contract(USDT.upper().replace("0X", "0x")) == usdt()
    assert REGISTRY.by_symbol("usdt") == usdt()
    assert REGISTRY.by_contract(FAKE) is None
    assert REGISTRY.by_symbol("DAI") is None


def test_registry_rejects_duplicate_symbols():
    with pytest.raises(ValueError, match="USDT"):
        TokenRegistry([usdt(), usdt(contract=FAKE, symbol="usdt")])


def test_registry_rejects_duplicate_contracts():
    with pytest.raises(ValueError, match=USDT):
        TokenRegistry([usdt(), usdt(symbol="USDC")])


def test_from_csv_reads_rows_and_tolerates_missing_file(tmp_path):
    path = tmp_path / "tokens.csv"
    path.write_text(
        "contract,symbol,decimals,dust_threshold,source\n"
        f"{USDT.upper().replace('0X', '0x')},USDT,6,0.01,test\n",
        encoding="utf-8",
    )
    assert TokenRegistry.from_csv(path).by_contract(USDT) == usdt()
    assert TokenRegistry.from_csv(tmp_path / "missing.csv").by_contract(USDT) is None


def test_from_csv_rejects_unexpected_column(tmp_path):
    path = tmp_path / "tokens.csv"
    path.write_text(
        f"contract,symbol,decimals,dust_threshold,source,typo\n{USDT},USDT,6,0.01,test,oops\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="typo"):
        TokenRegistry.from_csv(path)


@pytest.mark.parametrize(
    ("amount", "asset", "contract", "expected"),
    [
        ("1", "ETH", None, set()),
        ("0.0001", "ETH", None, set()),
        ("0.00005", "ETH", None, {"dust"}),
        ("0.3", "ETH", FAKE, {"spoofed_token"}),
        ("100", "USDT", FAKE, {"spoofed_token"}),
        ("100", "usdt", FAKE, {"spoofed_token"}),
        ("5", "USDT", USDT, set()),
        ("0.001", "USDT", USDT, {"dust"}),
        ("0.0000001", "PEPE", FAKE, set()),
        ("0.01", "USDT", USDT, set()),
    ],
)
def test_classify_transfer(amount, asset, contract, expected):
    item = transfer(1, UNKNOWN, TARGET, amount, asset, token_contract=contract)
    assert classify_transfer(item, REGISTRY, NATIVE_DUST) == expected
