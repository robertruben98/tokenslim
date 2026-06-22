from tokenslim.ccr import (
    DROPPED_KEY,
    SENTINEL_KEY,
    content_hash,
    find_markers,
    json_sentinel,
    make_marker,
    parse_marker,
    strip_markers,
    text_marker,
)
from tokenslim.store import InMemoryCCRStore


def test_make_and_parse_marker_roundtrip():
    marker = make_marker("abc123", 42, "middle-elided")
    assert marker == "<<ccr:abc123 42 middle-elided>>"
    parsed = parse_marker(f"prefix {marker} suffix")
    assert parsed.hash == "abc123"
    assert parsed.count == 42
    assert parsed.reason == "middle-elided"


def test_parse_marker_none_when_absent():
    assert parse_marker("no marker here") is None


def test_find_multiple_markers():
    text = f"{make_marker('aa11', 1)} ... {make_marker('bb22', 2, 'lines-elided')}"
    markers = find_markers(text)
    assert [m.hash for m in markers] == ["aa11", "bb22"]
    assert markers[1].reason == "lines-elided"


def test_strip_markers_removes_tokens():
    text = f"before {make_marker('deadbeef', 5)} after"
    stripped = strip_markers(text)
    assert "<<ccr:" not in stripped
    assert "before" in stripped and "after" in stripped


def test_json_sentinel_carries_marker_and_detail():
    dropped = [{"id": i} for i in range(10)]
    sentinel = json_sentinel(dropped, total=18, kept=8)
    assert DROPPED_KEY in sentinel
    assert SENTINEL_KEY in sentinel
    marker = parse_marker(sentinel[DROPPED_KEY])
    assert marker.count == 10
    # Without a store the marker hash is the plain content hash.
    assert marker.hash == sentinel[SENTINEL_KEY]["hash"]


def test_json_sentinel_writes_to_store():
    store = InMemoryCCRStore()
    dropped = [{"id": i} for i in range(10)]
    sentinel = json_sentinel(dropped, total=18, kept=8, store=store)
    marker = parse_marker(sentinel[DROPPED_KEY])
    # The hash is now the store key, and the original is retrievable.
    assert store.get(marker.hash) is not None
    assert len(store) == 1


def test_text_marker_is_parseable_and_storable():
    store = InMemoryCCRStore()
    lines = ["dropped line 1", "dropped line 2", "dropped line 3"]
    marker_line = text_marker(lines, store=store)
    parsed = parse_marker(marker_line)
    assert parsed.count == 3
    assert store.get(parsed.hash) == "\n".join(lines)


def test_content_hash_is_stable():
    assert content_hash("x") == content_hash("x")
    assert content_hash("x") != content_hash("y")
