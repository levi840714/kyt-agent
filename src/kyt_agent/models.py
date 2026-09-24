from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, model_validator

LowerStr = Annotated[str, AfterValidator(str.lower)]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "SEVERE"]
RISK_ORDER: tuple[RiskLevel, ...] = ("LOW", "MEDIUM", "HIGH", "SEVERE")
Category = Literal["sanctioned", "hack", "mixer", "bridge", "exchange", "defi"]
Decision = Literal["approve", "reject", "request_more"]


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
