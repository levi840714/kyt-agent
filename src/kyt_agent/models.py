import re
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, model_validator

LowerStr = Annotated[str, AfterValidator(str.lower)]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "SEVERE"]
RISK_ORDER: tuple[RiskLevel, ...] = ("LOW", "MEDIUM", "HIGH", "SEVERE")
Category = Literal["sanctioned", "hack", "mixer", "bridge", "exchange", "defi"]
Decision = Literal["approve", "reject", "request_more"]
_TX_ALIAS = re.compile(r"T(\d+)", re.IGNORECASE)


def max_risk(*levels: RiskLevel) -> RiskLevel:
    return max(levels, key=RISK_ORDER.index)


class Label(BaseModel):
    address: LowerStr
    name: str
    category: Category
    source: str


class AddressNode(BaseModel):
    address: LowerStr
    depth: int
    parent: str | None = None
    label: Label | None = None
    expanded: bool = False


class Evidence(BaseModel):
    id: str
    kind: Literal["tx", "label"]
    summary: str
    ref: str = ""

    @model_validator(mode="after")
    def _default_ref(self) -> "Evidence":
        # v1.1 之前的 checkpoint 沒有 ref，當時 tx 證據直接以 hash 為 ID
        if not self.ref:
            self.ref = self.id.removeprefix("label:")
        return self


def evidence_key(cited: str) -> str:
    """將 LLM 引用的證據 ID 正規化：交易代號轉大寫，其餘（hash、label ID）轉小寫。"""
    cited = cited.strip()
    return cited.upper() if _TX_ALIAS.fullmatch(cited) else cited.lower()


def tx_alias_number(evidence_id: str) -> int | None:
    match = _TX_ALIAS.fullmatch(evidence_id)
    return int(match.group(1)) if match else None


class RuleHit(BaseModel):
    rule: str
    address: LowerStr
    level: RiskLevel
    detail: str


class Review(BaseModel):
    decision: Decision
    comment: str = ""
    reviewer: str
    round: int

    @model_validator(mode="after")
    def _comment_required(self) -> "Review":
        if self.decision != "approve" and not self.comment.strip():
            raise ValueError("駁回或要求補查時必須填寫意見")
        return self
