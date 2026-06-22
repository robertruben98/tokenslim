"""Integration: diff routing + BM25-aware search ranking."""

from tokenslim import compress
from tokenslim.compressors.search import SearchCompressor
from tokenslim.config import Config
from tokenslim.detector import ContentType
from tokenslim.router import build_registry


def _big_diff(n=20):
    blocks = []
    for i in range(n):
        b = [f"diff --git a/m{i}.py b/m{i}.py", f"--- a/m{i}.py", f"+++ b/m{i}.py"]
        for h in range(3):
            b.append(f"@@ -{h * 10 + 1},4 +{h * 10 + 1},4 @@")
            b += [" ctx", f"-old{i}{h}", f"+new{i}{h}", " ctx"]
        blocks.append("\n".join(b))
    return "\n".join(blocks)


def test_registry_wires_diff_compressor():
    reg = build_registry(Config())
    assert reg[ContentType.DIFF][0] == "diff-compressor"


def test_compress_routes_diff_end_to_end():
    diff = _big_diff(20)
    messages = [{"role": "tool", "tool_call_id": "t1", "content": diff}]
    out, stats = compress(messages, options=Config(min_bytes=0, diff_max_files=5))
    assert stats.blocks[0].content_type is ContentType.DIFF
    assert stats.blocks[0].compressor == "diff-compressor"
    assert stats.saved_tokens > 0


def test_query_aware_search_promotes_relevant_file():
    # Two single-hit files; only one matches the query. With a tight 1-file cap
    # the relevant one must be the survivor.
    text = "\n".join(
        [f"noise.py:{i}:    unrelated boilerplate code" for i in range(1, 6)]
        + [f"auth.py:{i}:    validate jwt token signature" for i in range(1, 6)]
    )
    out = SearchCompressor(Config(search_max_files=1, query="jwt token signature"))(text)
    assert "auth.py:" in out
    assert "noise.py:" not in out


def test_search_without_query_unchanged_behaviour():
    # No query -> structural ranking only (regression guard).
    text = "\n".join([f"a.py:{i}:def foo():" for i in range(1, 6)])
    out = SearchCompressor(Config(search_max_files=10))(text)
    assert "a.py:" in out
