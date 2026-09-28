import pytest
from pydantic import ValidationError

from kyt_agent.models import Label, Review, max_risk


def test_max_risk_uses_severity_order():
    assert max_risk("LOW", "HIGH", "MEDIUM") == "HIGH"
    assert max_risk("SEVERE", "LOW") == "SEVERE"


def test_addresses_are_lowercased():
    label = Label(address="0xABC", name="n", category="defi", source="s")
    assert label.address == "0xabc"


def test_review_requires_comment_unless_approved():
    assert Review(decision="approve", reviewer="r", round=1).comment == ""
    with pytest.raises(ValidationError):
        Review(decision="reject", comment=" ", reviewer="r", round=1)
    with pytest.raises(ValidationError):
        Review(decision="request_more", reviewer="r", round=1)
