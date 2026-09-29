import re

import httpx

from kyt_agent.config import PROJECT_ROOT
from kyt_agent.labels import (
    LabelStore,
    fetch_ofac_labels,
    parse_ofac_list,
    read_labels_csv,
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


def test_relayer_is_a_known_category_below_mixer():
    store = LabelStore([label(MIXER, "relayer"), label(MIXER, "mixer")])
    assert store.get(MIXER).category == "mixer"
    assert LabelStore([label(EXCHANGE, "relayer")]).get(EXCHANGE).category == "relayer"


RELAYER_SOURCE = re.compile(
    r"Tornado\.Cash Relayer Registry 0x58e8dcc13be9780fc42e8723d8ead4cf46943df2 "
    r"RelayerRegistered tx 0x[0-9a-f]{64}"
)


def test_relayer_labels_carry_no_registrant_text():
    # ENS 名稱由註冊者自訂，會經由標籤證據送進 LLM，只保留可查證的交易 hash
    relayers = read_labels_csv(PROJECT_ROOT / "data" / "labels" / "relayers.csv")
    assert len(relayers) == 123
    for relayer in relayers:
        assert (relayer.name, relayer.category) == ("Tornado Cash relayer", "relayer")
        assert RELAYER_SOURCE.fullmatch(relayer.source), relayer.source
