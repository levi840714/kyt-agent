import pytest

from kyt_agent.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        etherscan_api_key="test-key",
        data_dir=tmp_path / "data",
        var_dir=tmp_path / "var",
    )
