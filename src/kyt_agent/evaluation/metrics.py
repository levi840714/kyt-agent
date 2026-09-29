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
    baseline: RiskLevel | None
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    snapshot_misses: int = 0
    error: str | None = None
    run: int = 1
    expanded: list[str] = []


class CategoryMetrics(BaseModel):
    """flag_rate 對陽性類別即召回率，對陰性類別即誤報率。"""

    expected: Expected
    cases: int
    flag_rate: float
    baseline_flag_rate: float
    avg_tokens: float


class UnstableCase(BaseModel):
    """重複執行時風險等級不一致的地址，只計入未出錯的執行。"""

    address: str
    category: str
    expected: Expected
    levels: list[RiskLevel]
    flagged_runs: int
    runs: int
    min_tokens: int
    max_tokens: int
    path_differs: bool


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
    cache_hit_ratio: float | None
    snapshot_misses: int
    snapshots_filled: int
    by_category: dict[str, CategoryMetrics]
    repeat: int = 1
    decision_agreement: float | None = None
    level_agreement: float | None = None
    unstable: list[UnstableCase] = []


def summarize(
    model: str, results: Sequence[CaseResult], snapshots_filled: int = 0, repeat: int = 1
) -> Summary:
    """每次執行都是一個樣本；repeat > 1 時另算同一地址跨次執行的一致性。"""
    ok = [result for result in results if result.error is None]
    compared = _comparable_runs(ok) if repeat > 1 else []
    positives = [result for result in ok if result.expected == "risky"]
    negatives = [result for result in ok if result.expected == "clean"]
    return Summary(
        model=model,
        cases=len(results),
        errors=len(results) - len(ok),
        recall=_rate(positives, _flagged),
        false_positive_rate=_rate(negatives, _flagged),
        baseline_recall=_rate(_with_baseline(positives), _baseline_flagged),
        baseline_false_positive_rate=_rate(_with_baseline(negatives), _baseline_flagged),
        avg_tool_calls=_mean([result.tool_calls for result in ok]),
        avg_tokens=_mean([_tokens(result) for result in ok]),
        avg_clean_tokens=_mean([_tokens(result) for result in negatives]),
        estimated_cost_usd=estimate_cost(model, ok),
        cache_hit_ratio=_cache_hit_ratio(ok),
        snapshot_misses=sum(result.snapshot_misses for result in ok),
        snapshots_filled=snapshots_filled,
        by_category=_by_category(ok),
        repeat=repeat,
        decision_agreement=_agreement(compared, _flagged),
        level_agreement=_agreement(compared, _level),
        unstable=[_unstable(runs) for runs in compared if len({_level(r) for r in runs}) > 1],
    )


def estimate_cost(model: str, results: Sequence[CaseResult]) -> float | None:
    price = PRICING.get(model)
    if price is None:
        return None
    tokens = sum(r.input_tokens * price[0] + r.output_tokens * price[1] for r in results)
    return tokens / 1_000_000


def _cache_hit_ratio(results: Sequence[CaseResult]) -> float | None:
    input_tokens = sum(result.input_tokens for result in results)
    cached = sum(result.cache_read_tokens for result in results)
    return cached / input_tokens if input_tokens else None


def _by_category(results: Sequence[CaseResult]) -> dict[str, CategoryMetrics]:
    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        groups[result.category].append(result)
    return {
        category: CategoryMetrics(
            expected=rows[0].expected,
            cases=len(rows),
            flag_rate=_mean([_flagged(row) for row in rows]),
            baseline_flag_rate=_mean([_baseline_flagged(row) for row in _with_baseline(rows)]),
            avg_tokens=_mean([_tokens(row) for row in rows]),
        )
        for category, rows in sorted(groups.items())
    }


def _comparable_runs(ok: Sequence[CaseResult]) -> list[list[CaseResult]]:
    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for result in ok:
        groups[result.address].append(result)
    return [sorted(runs, key=lambda r: r.run) for runs in groups.values() if len(runs) >= 2]


def _agreement(
    compared: Sequence[list[CaseResult]], key: Callable[[CaseResult], object]
) -> float | None:
    if not compared:
        return None
    return sum(len({key(result) for result in runs}) == 1 for runs in compared) / len(compared)


def _unstable(runs: list[CaseResult]) -> UnstableCase:
    tokens = [_tokens(result) for result in runs]
    return UnstableCase(
        address=runs[0].address,
        category=runs[0].category,
        expected=runs[0].expected,
        levels=[_level(result) for result in runs],
        flagged_runs=sum(_flagged(result) for result in runs),
        runs=len(runs),
        min_tokens=min(tokens),
        max_tokens=max(tokens),
        path_differs=len({frozenset(result.expanded) for result in runs}) > 1,
    )


def _level(result: CaseResult) -> RiskLevel:
    assert result.predicted is not None, "只比較未出錯的執行"
    return result.predicted


def _flagged(result: CaseResult) -> bool:
    return result.predicted in FLAGGED


def _baseline_flagged(result: CaseResult) -> bool:
    return result.baseline in FLAGGED


def _with_baseline(rows: Sequence[CaseResult]) -> list[CaseResult]:
    """即時查詢失敗時 baseline 為 None，不該拉低純規則的召回率／誤報率。"""
    return [row for row in rows if row.baseline is not None]


def _tokens(result: CaseResult) -> int:
    return result.input_tokens + result.output_tokens


def _rate(rows: Sequence[CaseResult], predicate: Callable[[CaseResult], bool]) -> float | None:
    return sum(predicate(row) for row in rows) / len(rows) if rows else None


def _mean(values: Sequence[int]) -> float:
    return sum(values) / len(values) if values else 0.0
