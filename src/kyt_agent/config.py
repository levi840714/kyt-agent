from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ChainMode = Literal["live", "record", "replay"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    llm_model: str = "google_genai:gemini-3.8-flash"
    etherscan_api_key: str = ""
    etherscan_min_interval: float = 0.25
    chain_mode: ChainMode = "live"
    max_depth: int = 3
    max_tool_calls: int = 25
    max_addresses: int = 20
    max_tokens: int = 200_000
    top_counterparties: int = 10
    tx_page_size: int = 100
    supplement_tool_calls: int = 10
    max_review_rounds: int = 3
    data_dir: Path = PROJECT_ROOT / "data"
    var_dir: Path = PROJECT_ROOT / "var"
