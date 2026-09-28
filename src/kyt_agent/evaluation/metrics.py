from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import BaseModel

from kyt_agent.models import RiskLevel

Expected = Literal["risky", "clean"]
FLAGGED: frozenset[RiskLevel] = frozenset({"HIGH", "SEVERE"})
# 每百萬 token 美元（input, output），2026-09-25 查詢
PRICING: dict[str, tuple[float, float]] = {
    "google_genai:gemini-3.8-flash": (0.75, 3.75),
    "google_genai:gemini-3.5-flash-lite": (0.30, 2.50),
    "openai:gpt-6-luna": (0.10, 0.50),
    "openai:gpt-6-sol": (2.00, 10.00),
    "anthropic:claude-sonnet-5": (2.00, 10.00),
    "anthropic:claude-haiku-4-5": (1.00, 5.00),
}


class CaseResult(BaseModel):
    address: str
    expected: Expected
    category: str
    predicted: RiskLevel | None
    baseline: RiskLevel
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    snapshot_misses: int = 0
    error: str | None = None


class CategoryMetrics(BaseModel):
    """flag_rate 對陽性類別即召回率，對陰性類別即誤報率。"""

    expected: Expected
    cases: int
    flag_rate: float
    baseline_flag_rate: float
    avg_tokens: float


class Summary(BaseModel):
    model: str
    cases: int
    errors: int
    recall: float | None
    false_positive_rate: float | None
    baseline_recall: float | None
    baseline_false_positive_rate: float | None
    avg_tool_calls: float
    avg_tokens: float
    avg_clean_tokens: float
    estimated_cost_usd: float | None
    snapshot_misses: int
    by_category: dict[str, CategoryMetrics]


def summarize(model: str, results: Sequence[CaseResult]) -> Summary:
    ok = [result for result in results if result.error is None]
    positives = [result for result in ok if result.expected == "risky"]
    negatives = [result for result in ok if result.expected == "clean"]
    return Summary(
        model=model,
        cases=len(results),
        errors=len(results) - len(ok),
        recall=_rate(positives, _flagged),
        false_positive_rate=_rate(negatives, _flagged),
        baseline_recall=_rate(positives, _baseline_flagged),
        baseline_false_positive_rate=_rate(negatives, _baseline_flagged),
        avg_tool_calls=_mean([result.tool_calls for result in ok]),
        avg_tokens=_mean([_tokens(result) for result in ok]),
        avg_clean_tokens=_mean([_tokens(result) for result in negatives]),
        estimated_cost_usd=estimate_cost(model, ok),
        snapshot_misses=sum(result.snapshot_misses for result in ok),
        by_category=_by_category(ok),
    )


def estimate_cost(model: str, results: Sequence[CaseResult]) -> float | None:
    price = PRICING.get(model)
    if price is None:
        return None
    tokens = sum(r.input_tokens * price[0] + r.output_tokens * price[1] for r in results)
    return tokens / 1_000_000


def _by_category(results: Sequence[CaseResult]) -> dict[str, CategoryMetrics]:
    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        groups[result.category].append(result)
    return {
        category: CategoryMetrics(
            expected=rows[0].expected,
            cases=len(rows),
            flag_rate=_mean([_flagged(row) for row in rows]),
            baseline_flag_rate=_mean([_baseline_flagged(row) for row in rows]),
            avg_tokens=_mean([_tokens(row) for row in rows]),
        )
        for category, rows in sorted(groups.items())
    }


def _flagged(result: CaseResult) -> bool:
    return result.predicted in FLAGGED


def _baseline_flagged(result: CaseResult) -> bool:
    return result.baseline in FLAGGED


def _tokens(result: CaseResult) -> int:
    return result.input_tokens + result.output_tokens


def _rate(rows: Sequence[CaseResult], predicate: Callable[[CaseResult], bool]) -> float | None:
    return sum(predicate(row) for row in rows) / len(rows) if rows else None


def _mean(values: Sequence[int]) -> float:
    return sum(values) / len(values) if values else 0.0
