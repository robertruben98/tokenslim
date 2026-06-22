from tokenslim.compressors.search import SearchCompressor, parse_search_line
from tokenslim.config import Config


def _compress(text, **cfg):
    return SearchCompressor(Config(**cfg))(text)


# --- line parsing ---------------------------------------------------------


def test_parse_match_line():
    hit = parse_search_line("src/main.py:42:    return x")
    assert hit.path == "src/main.py"
    assert hit.lineno == 42
    assert hit.content == "    return x"
    assert hit.is_match is True


def test_parse_context_line_with_hyphen_separator():
    hit = parse_search_line("src/main.py-41-    setup()")
    assert hit.path == "src/main.py"
    assert hit.lineno == 41
    assert hit.is_match is False


def test_parse_windows_path():
    hit = parse_search_line(r"C:\proj\win.py:10:import os")
    assert hit.path == r"C:\proj\win.py"
    assert hit.lineno == 10


def test_parse_hyphenated_filename():
    # The filename has hyphens AND it's a context line (hyphen separator).
    hit = parse_search_line("my-cool-mod.py-7-    pass")
    assert hit.path == "my-cool-mod.py"
    assert hit.lineno == 7
    assert hit.is_match is False


def test_parse_non_hit_returns_none():
    assert parse_search_line("just some prose") is None
    assert parse_search_line("12:bare") is None  # empty path


# --- compression ----------------------------------------------------------


def test_groups_by_file_removing_path_repetition():
    lines = [f"src/big.py:{i}:    x = {i}" for i in range(40)]
    text = "\n".join(lines)
    out = _compress(text)
    # Path printed once as a header, not 40 times.
    assert out.count("src/big.py") == 1
    assert len(out) < len(text)


def test_caps_number_of_files():
    lines = [f"file_{i}.py:1:hit" for i in range(40)]
    text = "\n".join(lines)
    out = _compress(text, search_max_files=10)
    # Only 10 files kept; the rest summarised.
    headers = [ln for ln in out.splitlines() if ln.endswith(".py:")]
    assert len(headers) == 10
    assert "files / " in out and "elided" in out


def test_definition_lines_rank_above_references():
    # A file whose hits are definitions should outrank a file with bare refs
    # when only one file fits. Use several hits each so grouping is a clear win.
    text = "\n".join(
        [f"refs.py:{i}:    foo()" for i in range(1, 6)]
        + [f"defs.py:{i}:def thing_{i}():" for i in range(1, 6)]
    )
    out = _compress(text, search_max_files=1)
    assert "defs.py:" in out  # the definition file is the one kept
    assert "refs.py:" not in out  # the reference-only file is elided
    assert "elided" in out


def test_non_search_text_passthrough():
    text = "this is not search output at all\njust prose here"
    assert _compress(text) == text
