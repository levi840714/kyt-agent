from pathlib import Path

from pydantic import BaseModel

from kyt_agent.evaluation.metrics import Expected
from kyt_agent.models import LowerStr


class EvalCase(BaseModel):
    address: LowerStr
    expected: Expected
    category: str
    source: str
    note: str = ""


def load_dataset(path: Path) -> list[EvalCase]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [EvalCase.model_validate_json(line) for line in lines if line.strip()]
