import httpx

from kyt_agent.labels import (
    LabelStore,
    fetch_ofac_labels,
    parse_ofac_list,
    write_labels_csv,
)
from tests.fakes import EXCHANGE, MIXED_CASE, MIXER, label


def test_lookup_is_case_insensitive():
    store = LabelStore([label(MIXED_CASE, "defi")])
    assert store.get(MIXED_CASE.upper().replace("0X", "0x")) is not None


def test_keeps_most_severe_category():
    store = LabelStore([label(MIXER, "exchange"), label(MIXER, "sanctioned"), label(MIXER, "defi")])
    assert store.get(MIXER).category == "sanctioned"


def test_reads_every_csv_in_directory(tmp_path):
    write_labels_csv(tmp_path / "a.csv", [label(MIXER, "mixer")])
    write_labels_csv(tmp_path / "b.csv", [label(EXCHANGE, "exchange")])
    store = LabelStore.from_dir(tmp_path)
    assert len(store) == 2
    assert store.get(EXCHANGE).category == "exchange"


def test_parse_ofac_list_skips_blank_and_comment_lines():
    labels = parse_ofac_list(f"{MIXED_CASE}\n\n# comment\n")
    assert [item.address for item in labels] == [MIXED_CASE.lower()]
    assert labels[0].category == "sanctioned"


def test_fetch_ofac_labels_downloads_published_list():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=f"{MIXER}\n"))
    labels = fetch_ofac_labels(httpx.Client(transport=transport))
    assert labels[0].address == MIXER
