import json

import pytest

from kyt_agent.chain.etherscan import EtherscanError
from kyt_agent.chain.snapshot import SnapshotClient
from kyt_agent.evaluation import runner
from kyt_agent.evaluation.crawler import baseline_level, crawl
from kyt_agent.evaluation.dataset import EvalCase, load_dataset
from kyt_agent.evaluation.metrics import CaseResult, CategoryMetrics, UnstableCase, summarize
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
    tx_hash,
)
from tests.graph_fakes import ai_text, ai_tool_call, fake_deps


def _no_flags(item):
    return set()


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
    assert crawl(chain, LabelStore([]), TARGET, depth=2, top_n=10, classify=_no_flags) == 3
    assert {("transfers", TARGET), ("transfers", UNKNOWN), ("contract", UNKNOWN)} <= set(
        chain.calls
    )
    assert ("transfers", MIXER) not in chain.calls
    assert ("transaction", tx_hash(1)) in chain.calls
    assert ("transaction", tx_hash(2)) not in chain.calls


def test_crawl_ranks_counterparties_by_non_spoofed_count():
    spam = "0x" + "0" * 40
    chain = StubChain(
        transfers={
            TARGET: [transfer(n, spam, TARGET, asset="ETH", token_contract=spam) for n in range(3)]
            + [transfer(9, UNKNOWN, TARGET)]
        }
    )

    def classify(item):
        return {"spoofed_token"} if item.token_contract else set()

    crawl(chain, LabelStore([]), TARGET, depth=2, top_n=1, classify=classify)
    assert ("transfers", UNKNOWN) in chain.calls
    assert ("transfers", spam) not in chain.calls


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
            category="mixer_depositor",
            predicted="HIGH",
            baseline="LOW",
            tool_calls=4,
            input_tokens=1000,
            output_tokens=100,
        ),
        CaseResult(
            address=MIXER,
            expected="risky",
            category="indirect_exposure",
            predicted="MEDIUM",
            baseline="HIGH",
            tool_calls=2,
        ),
        CaseResult(
            address=EXCHANGE,
            expected="clean",
            category="exchange_user",
            predicted="HIGH",
            baseline="LOW",
            input_tokens=300,
            output_tokens=100,
        ),
        CaseResult(
            address=UNKNOWN,
            expected="clean",
            category="exchange_user",
            predicted=None,
            baseline="LOW",
            error="boom",
        ),
    ]
    summary = summarize("openai:gpt-6-luna", results)
    assert (summary.recall, summary.false_positive_rate) == (0.5, 1.0)
    assert (summary.baseline_recall, summary.baseline_false_positive_rate) == (0.5, 0.0)
    assert (summary.errors, summary.avg_tool_calls) == (1, 2.0)
    assert summary.estimated_cost_usd == pytest.approx((1300 * 0.10 + 200 * 0.50) / 1_000_000)
    assert summary.avg_clean_tokens == 400.0
    assert list(summary.by_category) == ["exchange_user", "indirect_exposure", "mixer_depositor"]
    assert summary.by_category["indirect_exposure"] == CategoryMetrics(
        expected="risky", cases=1, flag_rate=0.0, baseline_flag_rate=1.0, avg_tokens=0.0
    )
    assert summary.by_category["exchange_user"] == CategoryMetrics(
        expected="clean", cases=1, flag_rate=1.0, baseline_flag_rate=0.0, avg_tokens=400.0
    )


def test_summarize_skips_none_baseline_in_baseline_rates():
    # baseline 為 None 代表當次即時查詢失敗，不該拉低純規則的召回率／誤報率
    results = [
        CaseResult(
            address=TARGET, expected="risky", category="mixer", predicted="HIGH", baseline=None
        ),
        CaseResult(
            address=MIXER, expected="risky", category="mixer", predicted="MEDIUM", baseline="HIGH"
        ),
    ]
    summary = summarize("fake:model", results)
    assert summary.baseline_recall == 1.0
    assert summary.by_category["mixer"].baseline_flag_rate == 1.0


def test_run_case_collects_prediction_and_usage(settings):
    chain = StubChain(transfers={TARGET: [transfer(1, TARGET, MIXER)]})
    labels = LabelStore([label(MIXER, "mixer")])
    script = [ai_tool_call("get_counterparties", "c1", address=TARGET), ai_text("完成")]
    deps = fake_deps(settings, chain, labels, script, [draft("HIGH", [f"label:{MIXER}"])], True)
    case = EvalCase(address=TARGET, expected="risky", category="mixer", source="test")
    result = run_case(build_graph(deps), deps, case)
    assert (result.predicted, result.baseline) == ("HIGH", "HIGH")
    assert (result.tool_calls, result.input_tokens, result.output_tokens) == (1, 400, 70)
    assert result.category == "mixer"


def test_run_case_records_errors(settings):
    deps = fake_deps(settings, StubChain(), LabelStore([]), [ai_text("完成")], [None, None], True)
    case = EvalCase(address=TARGET, expected="clean", category="normal", source="test")
    result = run_case(build_graph(deps), deps, case)
    assert result.error.startswith("ReportError")
    assert result.category == "normal"


def test_write_eval_result(tmp_path):
    summary = summarize("fake:model", [])
    path = write_eval_result(tmp_path, summary, [])
    assert path.name.endswith("fake_model.json")
    assert json.loads(path.read_text(encoding="utf-8"))["summary"]["model"] == "fake:model"


def test_run_eval_fill_missing_records_through_then_replays(settings, monkeypatch):
    live = StubChain(transfers={TARGET: [transfer(1, TARGET, UNKNOWN)]})
    modes: list[str] = []

    def fake_make_deps(eval_settings, *, model, auto_approve):
        modes.append(eval_settings.chain_mode)
        inner = live if eval_settings.chain_mode == "record" else None
        chain = SnapshotClient(eval_settings.data_dir / "snapshots", inner)
        script = [ai_tool_call("get_counterparties", "c1", address=TARGET), ai_text("完成")]
        return fake_deps(eval_settings, chain, LabelStore([]), script, [draft("LOW")], auto_approve)

    monkeypatch.setattr(runner, "make_deps", fake_make_deps)
    case = EvalCase(address=TARGET, expected="clean", category="exchange_user", source="test")

    filled, results = runner.run_eval(settings, [case], None, lambda _: None, fill_missing=True)
    assert (results[0].error, filled.snapshots_filled) == (None, 1)
    assert (settings.data_dir / "snapshots" / "transfers" / f"{TARGET}.json").exists()

    replayed, results = runner.run_eval(settings, [case], None, lambda _: None)
    assert modes == ["record", "replay"]
    assert (results[0].error, results[0].snapshot_misses) == (None, 0)
    assert replayed.snapshots_filled == 0


def test_run_eval_isolates_live_etherscan_failures_per_case(settings, monkeypatch):
    class FlakyChain(StubChain):
        def get_transfers(self, address):
            if address == MIXER:
                raise EtherscanError("HTTP 502")
            return super().get_transfers(address)

    chain = FlakyChain(transfers={TARGET: [transfer(1, TARGET, UNKNOWN)]})
    script = [ai_tool_call("get_counterparties", "c1", address=TARGET), ai_text("完成")]

    def fake_make_deps(eval_settings, *, model, auto_approve):
        return fake_deps(eval_settings, chain, LabelStore([]), script, [draft("LOW")], auto_approve)

    monkeypatch.setattr(runner, "make_deps", fake_make_deps)
    failing = EvalCase(address=MIXER, expected="clean", category="exchange_user", source="test")
    ok = EvalCase(address=TARGET, expected="clean", category="exchange_user", source="test")

    summary, results = runner.run_eval(settings, [failing, ok], None, lambda _: None)

    failing_result, ok_result = results
    assert failing_result.baseline is None
    assert failing_result.error.startswith("EtherscanError")
    assert (ok_result.error, ok_result.baseline) == (None, "LOW")
    assert summary.errors == 1


def test_summary_reports_cache_hit_ratio():
    results = [
        CaseResult(
            address=TARGET,
            expected="risky",
            category="c",
            predicted="HIGH",
            baseline="HIGH",
            input_tokens=1000,
            cache_read_tokens=600,
        ),
        CaseResult(
            address=MIXER,
            expected="clean",
            category="c",
            predicted="LOW",
            baseline="LOW",
            input_tokens=1000,
            cache_read_tokens=200,
        ),
    ]
    assert summarize("fake:model", results).cache_hit_ratio == 0.4
    assert summarize("fake:model", []).cache_hit_ratio is None


def _run(address, level, run, *, tokens=0, expanded=(), error=None, expected="risky"):
    return CaseResult(
        address=address,
        expected=expected,
        category=f"cat-{address[2]}",
        predicted=None if error else level,
        baseline="LOW",
        input_tokens=tokens,
        run=run,
        expanded=list(expanded),
        error=error,
    )


def test_single_run_has_no_consistency_metrics():
    summary = summarize("fake:model", [_run(TARGET, "HIGH", 1)])
    assert summary.repeat == 1
    assert (summary.decision_agreement, summary.level_agreement, summary.unstable) == (
        None,
        None,
        [],
    )


def test_consistency_metrics_compare_non_error_runs():
    results = [
        _run(TARGET, "HIGH", 1, tokens=100, expanded=[TARGET]),
        _run(TARGET, "MEDIUM", 2, tokens=300, expanded=[TARGET, UNKNOWN]),
        _run(TARGET, "HIGH", 3, tokens=200, expanded=[TARGET]),
        _run(MIXER, "HIGH", 1, expanded=[MIXER]),
        _run(MIXER, "SEVERE", 2, expanded=[MIXER]),
        _run(MIXER, "HIGH", 3, expanded=[MIXER]),
        *(_run(EXCHANGE, "LOW", run, expected="clean") for run in (1, 2, 3)),
        _run(UNKNOWN, "LOW", 1, expected="clean"),
        _run(UNKNOWN, None, 2, error="boom", expected="clean"),
        _run(UNKNOWN, "MEDIUM", 3, expected="clean"),
        _run(SANCTIONED, None, 1, error="boom"),
        _run(SANCTIONED, None, 2, error="boom"),
        _run(SANCTIONED, "LOW", 3),
    ]
    summary = summarize("fake:model", results, repeat=3)
    # SANCTIONED 只有一次成功執行，不列入比較
    assert (summary.repeat, summary.cases, summary.errors) == (3, 15, 3)
    assert summary.decision_agreement == 0.75
    assert summary.level_agreement == 0.25
    assert [item.address for item in summary.unstable] == [TARGET, MIXER, UNKNOWN]
    assert summary.unstable[0] == UnstableCase(
        address=TARGET,
        category="cat-1",
        expected="risky",
        levels=["HIGH", "MEDIUM", "HIGH"],
        flagged_runs=2,
        runs=3,
        min_tokens=100,
        max_tokens=300,
        path_differs=True,
    )
    assert (summary.unstable[1].flagged_runs, summary.unstable[1].path_differs) == (3, False)
    assert (summary.unstable[2].levels, summary.unstable[2].runs) == (["LOW", "MEDIUM"], 2)
    assert summary.recall == pytest.approx(5 / 7)


def test_consistency_is_none_when_no_address_has_two_successful_runs():
    results = [_run(TARGET, "HIGH", 1), _run(TARGET, None, 2, error="boom")]
    summary = summarize("fake:model", results, repeat=2)
    assert (summary.decision_agreement, summary.level_agreement) == (None, None)


def test_run_eval_repeats_each_case_and_records_expanded_nodes(settings, monkeypatch):
    chain = StubChain(transfers={TARGET: [transfer(1, TARGET, UNKNOWN)]})
    script = [
        ai_tool_call("get_counterparties", "c1", address=TARGET),
        ai_text("完成"),
        ai_text("完成"),
        ai_tool_call("get_counterparties", "c2", address=TARGET),
        ai_text("完成"),
    ]
    drafts = [draft("HIGH"), draft("MEDIUM"), draft("HIGH")]

    def fake_make_deps(eval_settings, *, model, auto_approve):
        return fake_deps(eval_settings, chain, LabelStore([]), script, drafts, auto_approve)

    monkeypatch.setattr(runner, "make_deps", fake_make_deps)
    case = EvalCase(address=TARGET, expected="risky", category="indirect", source="test")
    seen: list[int] = []

    summary, results = runner.run_eval(
        settings, [case], None, lambda result: seen.append(result.run), repeat=3
    )

    assert seen == [1, 2, 3]
    assert [r.predicted for r in results] == ["HIGH", "MEDIUM", "HIGH"]
    assert [r.expanded for r in results] == [[TARGET], [], [TARGET]]
    assert (summary.repeat, summary.cases, summary.recall) == (3, 3, pytest.approx(2 / 3))
    assert (summary.decision_agreement, summary.level_agreement) == (0.0, 0.0)
    (unstable,) = summary.unstable
    assert (unstable.levels, unstable.flagged_runs, unstable.path_differs) == (
        ["HIGH", "MEDIUM", "HIGH"],
        2,
        True,
    )


def test_write_eval_result_includes_runs_and_consistency(tmp_path):
    results = [_run(TARGET, "HIGH", 1, expanded=[TARGET]), _run(TARGET, "LOW", 2)]
    path = write_eval_result(tmp_path, summarize("fake:model", results, repeat=2), results)
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["summary"]["repeat"] == 2
    assert body["summary"]["unstable"][0]["levels"] == ["HIGH", "LOW"]
    assert [(row["run"], row["expanded"]) for row in body["results"]] == [(1, [TARGET]), (2, [])]
