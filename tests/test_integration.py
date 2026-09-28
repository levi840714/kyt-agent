import pytest
from dotenv import load_dotenv

from kyt_agent.config import PROJECT_ROOT, Settings
from kyt_agent.evaluation.dataset import load_dataset
from kyt_agent.evaluation.runner import run_eval


@pytest.mark.integration
def test_real_llm_completes_recorded_cases():
    load_dotenv(PROJECT_ROOT / ".env")
    cases = load_dataset(PROJECT_ROOT / "eval" / "dataset.jsonl")[:2]
    summary, results = run_eval(Settings(), cases, None, lambda result: None)
    assert summary.errors == 0, [result.error for result in results]
    assert all(result.predicted is not None for result in results)
