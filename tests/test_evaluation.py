import json

import pytest

from kyt_agent.evaluation.crawler import baseline_level, crawl
from kyt_agent.evaluation.dataset import EvalCase, load_dataset
from kyt_agent.evaluation.metrics import CaseResult, summarize
from kyt_agent.evaluation.runner import run_case, write_eval_result
from kyt_agent.graph.build import build_graph
from kyt_agent.labels import LabelStore
from tests.fakes import (
    EXCHANGE,
    MIXED_CASE,
    MIXER,
    SANCTIONED,
    TARGET,
    UNKNOWN,
    StubChain,
    draft,
    label,
    transfer,
)
from tests.graph_fakes import ai_text, ai_tool_call, fake_deps


def test_load_dataset_skips_blank_lines(tmp_path):
    path = tmp_path / "dataset.jsonl"
    row = {"address": MIXED_CASE, "expected": "risky", "category": "mixer", "source": "https://x"}
    path.write_text(json.dumps(row) + "\n\n", encoding="utf-8")
    assert load_dataset(path)[0].address == MIXED_CASE.lower()


def test_crawl_expands_requested_layers():
    chain = StubChain(
        transfers={
            TARGET: [transfer(1, TARGET, UNKNOWN)],
            UNKNOWN: [transfer(2, UNKNOWN, MIXER)],
        }
    )
    assert crawl(chain, LabelStore([]), TARGET, depth=2, top_n=10) == 3
    assert {("transfers", TARGET), ("transfers", UNKNOWN), ("contract", UNKNOWN)} <= set(
        chain.calls
    )
    assert ("transfers", MIXER) not in chain.calls


def test_baseline_checks_every_direct_counterparty():
    chain = StubChain(
        transfers={
            TARGET: [transfer(n, TARGET, EXCHANGE) for n in range(20)]
            + [transfer(99, SANCTIONED, TARGET)]
        }
    )
    labels = LabelStore([label(SANCTIONED, "sanctioned")])
    assert baseline_level(chain, labels, TARGET) == "HIGH"
    assert baseline_level(chain, labels, SANCTIONED) == "SEVERE"


def test_summarize_metrics():
    results = [
        CaseResult(
            address=TARGET,
            expected="risky",
            predicted="HIGH",
            baseline="LOW",
            tool_calls=4,
            input_tokens=1000,
            output_tokens=100,
        ),
        CaseResult(
            address=MIXER, expected="risky", predicted="MEDIUM", baseline="HIGH", tool_calls=2
        ),
        CaseResult(address=EXCHANGE, expected="clean", predicted="HIGH", baseline="LOW"),
        CaseResult(address=UNKNOWN, expected="clean", predicted=None, baseline="LOW", error="boom"),
    ]
    summary = summarize("openai:gpt-6-luna", results)
    assert (summary.recall, summary.false_positive_rate) == (0.5, 1.0)
    assert (summary.baseline_recall, summary.baseline_false_positive_rate) == (0.5, 0.0)
    assert (summary.errors, summary.avg_tool_calls) == (1, 2.0)
    assert summary.estimated_cost_usd == pytest.approx((1000 * 0.10 + 100 * 0.50) / 1_000_000)


def test_run_case_collects_prediction_and_usage(settings):
    chain = StubChain(transfers={TARGET: [transfer(1, TARGET, MIXER)]})
    labels = LabelStore([label(MIXER, "mixer")])
    script = [ai_tool_call("get_counterparties", "c1", address=TARGET), ai_text("完成")]
    deps = fake_deps(settings, chain, labels, script, [draft("HIGH", [f"label:{MIXER}"])], True)
    case = EvalCase(address=TARGET, expected="risky", category="mixer", source="test")
    result = run_case(build_graph(deps), deps, case)
    assert (result.predicted, result.baseline) == ("HIGH", "MEDIUM")
    assert (result.tool_calls, result.input_tokens, result.output_tokens) == (1, 400, 70)


def test_run_case_records_errors(settings):
    deps = fake_deps(settings, StubChain(), LabelStore([]), [ai_text("完成")], [None, None], True)
    case = EvalCase(address=TARGET, expected="clean", category="normal", source="test")
    result = run_case(build_graph(deps), deps, case)
    assert result.error.startswith("ReportError")


def test_write_eval_result(tmp_path):
    summary = summarize("fake:model", [])
    path = write_eval_result(tmp_path, summary, [])
    assert path.name.endswith("fake_model.json")
    assert json.loads(path.read_text(encoding="utf-8"))["summary"]["model"] == "fake:model"
