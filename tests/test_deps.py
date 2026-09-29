from typing import Any

from kyt_agent.graph import deps
from kyt_agent.labels import LabelStore
from tests.graph_fakes import ScriptedModel


def test_make_deps_sets_llm_timeout_and_retries(monkeypatch, settings):
    captured: dict[str, Any] = {}

    def fake_init_chat_model(model: str, **kwargs: Any) -> ScriptedModel:
        captured.update(model=model, **kwargs)
        return ScriptedModel(messages=iter([]))

    monkeypatch.setattr(deps, "init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(deps.LabelStore, "from_dir", classmethod(lambda cls, path: LabelStore([])))
    deps.make_deps(settings.model_copy(update={"llm_timeout": 45.0}))
    assert captured["timeout"] == 45.0
    assert captured["max_retries"] == deps.LLM_MAX_RETRIES
