from decimal import Decimal

from kyt_agent.config import PROJECT_ROOT, Settings


def test_defaults_match_spec(monkeypatch):
    for name in ("LLM_MODEL", "CHAIN_MODE", "MAX_DEPTH", "MAX_TOOL_CALLS"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None)
    assert settings.llm_model == "google_genai:gemini-3.8-flash"
    assert settings.chain_mode == "live"
    assert (settings.max_depth, settings.max_tool_calls, settings.max_addresses) == (3, 25, 20)
    assert settings.data_dir == PROJECT_ROOT / "data"
    assert settings.wrap_up_hint_after == 6
    assert settings.native_dust_threshold == Decimal("0.0001")


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("CHAIN_MODE", "replay")
    monkeypatch.setenv("MAX_DEPTH", "2")
    monkeypatch.setenv("NATIVE_DUST_THRESHOLD", "0.001")
    settings = Settings(_env_file=None)
    assert (settings.chain_mode, settings.max_depth) == ("replay", 2)
    assert settings.native_dust_threshold == Decimal("0.001")


def test_llm_timeout_default_and_override(monkeypatch):
    monkeypatch.delenv("LLM_TIMEOUT", raising=False)
    assert Settings(_env_file=None).llm_timeout == 120
    monkeypatch.setenv("LLM_TIMEOUT", "30")
    assert Settings(_env_file=None).llm_timeout == 30
