from tokenslim.ccr import find_markers
from tokenslim.compressors.tabular import TabularCompressor
from tokenslim.config import Config
from tokenslim.detector import ContentType
from tokenslim.router import ContentRouter
from tokenslim.store import InMemoryCCRStore


def _make_csv(n: int = 60, delim: str = ",", spike_at: int | None = None) -> str:
    lines = [delim.join(("id", "name", "amount"))]
    for i in range(n):
        amount = "9999" if i == spike_at else str(10 + (i % 7))
        lines.append(delim.join((str(i), f"item-{i}", amount)))
    return "\n".join(lines)


def _compress(text: str, store=None, **overrides):
    return TabularCompressor(Config(min_bytes=0).merged(**overrides), store)(text)


def test_wide_csv_compresses():
    text = _make_csv(60)
    out = _compress(text)
    assert len(out) < len(text), "60-row table should shrink"
    assert find_markers(out), "elided rows must leave a CCR marker"


def test_header_head_and_tail_survive():
    text = _make_csv(60)
    out_lines = _compress(text).splitlines()
    assert "id,name,amount" in out_lines
    assert "0,item-0,10" in out_lines  # first data row
    assert "59,item-59,13" in out_lines  # last data row
    assert "30,item-30,12" not in out_lines  # unremarkable middle row is elided


def test_outlier_row_survives():
    text = _make_csv(60, spike_at=30)
    out_lines = _compress(text).splitlines()
    assert "30,item-30,9999" in out_lines, "|z| > 2.5 row must be kept"


def test_stats_summary_per_numeric_column():
    out = _compress(_make_csv(60, spike_at=30))
    assert "# stats id: count=60 min=0 max=59" in out
    assert "# stats amount: count=60 min=10 max=9999" in out
    assert "mean=" in out
    assert "# stats name" not in out, "non-numeric columns get no stats line"


def test_ccr_roundtrip_stores_full_original():
    store = InMemoryCCRStore()
    text = _make_csv(60)
    out = _compress(text, store=store)
    markers = find_markers(out)
    assert len(markers) == 1
    marker = markers[0]
    assert marker.reason == "rows-elided"
    assert marker.count > 0
    assert store.get(marker.hash) == text, "marker must recover the ORIGINAL csv"


def test_marker_auditable_without_store():
    out = _compress(_make_csv(60))
    markers = find_markers(out)
    assert len(markers) == 1 and markers[0].hash


def test_small_table_unchanged():
    text = _make_csv(6)  # 6 data rows <= keep_head + keep_tail
    assert _compress(text) == text


def test_non_tabular_and_ragged_input_unchanged():
    prose = "just some prose\nwith no delimiter\nat all"
    assert _compress(prose) == prose
    ragged = "a,b\n" + "\n".join(f"{i},{i},{i}" for i in range(20))
    assert _compress(ragged) == ragged


def test_garbage_never_raises():
    garbage = 'a,"unclosed\n' * 30
    assert isinstance(_compress(garbage), str)


def test_alternate_delimiters_compress():
    for delim in (";", "\t", "|"):
        text = _make_csv(40, delim=delim)
        out = _compress(text)
        assert len(out) < len(text), repr(delim)
        assert delim.join(("id", "name", "amount")) in out.splitlines()


def test_keep_head_knob_respected():
    out_lines = _compress(_make_csv(60), csv_keep_head=2).splitlines()
    assert "1,item-1,11" in out_lines
    assert "2,item-2,12" not in out_lines


def test_kept_rows_are_byte_identical_to_original_lines():
    # Fields with embedded quotes must never be re-serialised: RFC-4180
    # quote-doubling would corrupt cell content the model then misreads.
    lines = ["id,note,amount"]
    for i in range(40):
        lines.append(f'{i},say "hi-{i}",{10 + (i % 7)}')
    text = "\n".join(lines)
    out_lines = _compress(text).splitlines()
    assert out_lines[0] == lines[0], "header must be verbatim"
    assert lines[1] in out_lines and lines[-1] in out_lines
    original = set(lines)
    kept = [ln for ln in out_lines if not ln.startswith(("#", "[tokenslim:ccr]"))]
    assert kept and all(ln in original for ln in kept), "kept rows were re-serialised"


def test_router_routes_csv_to_tabular():
    router = ContentRouter(config=Config(min_bytes=0))
    result = router.route(_make_csv(60))
    assert result.content_type is ContentType.CSV
    assert result.compressor == "tabular"
    assert result.changed is True
    assert find_markers(result.text)
