import time
import types

import pytest

from tokenslim.ccr import find_markers
from tokenslim.compressors import code as code_mod
from tokenslim.compressors.code import CodeCompressor, detect_language
from tokenslim.config import Config
from tokenslim.store import InMemoryCCRStore

# --- Fake tree-sitter AST so the single-parse guarantee is testable even when
#     the tree-sitter grammars aren't installed in the environment. ----------


def _fake_node(node_type, *, children=(), start_byte=0, end_byte=0, column=0, text=b""):
    children = list(children)
    return types.SimpleNamespace(
        type=node_type,
        children=children,
        child_count=len(children),
        start_byte=start_byte,
        end_byte=end_byte,
        start_point=types.SimpleNamespace(row=0, column=column),
        text=text,
        is_error=False,
        has_error=False,
    )


def _fake_python_tree():
    """A module with a single ``def f(): return 1`` for the source below."""
    # src == "def f():\n    return 1\n"  (body spans bytes 9..22)
    body = _fake_node("block", start_byte=9, end_byte=22, column=4, text=b"    return 1\n")
    func = _fake_node("function_definition", children=[body])
    root = _fake_node("module", children=[func])
    return types.SimpleNamespace(root_node=root)


_FAKE_PY_SRC = "def f():\n    return 1\n"


def test_single_parse_per_block_with_spy(monkeypatch):
    """Exactly ONE tree-sitter parse per block, reused across all passes (#121)."""
    calls = {"n": 0}

    def counting_parse(text_bytes, flavor):
        calls["n"] += 1
        assert flavor == "python"
        return _fake_python_tree()

    monkeypatch.setattr(code_mod, "HAS_TREE_SITTER", True)
    monkeypatch.setattr(code_mod, "_parse", counting_parse)

    out = CodeCompressor(Config())(_FAKE_PY_SRC)

    assert calls["n"] == 1  # one parse for the whole block, reused everywhere
    # The single tree drove signature + body passes: the body was elided.
    assert "def f():" in out
    assert "return 1" not in out
    assert find_markers(out)  # a CCR marker was emitted from the same tree


def test_size_cap_skips_ast_and_falls_back(monkeypatch):
    """Over the byte cap the AST path is skipped — zero parses (#121)."""
    calls = {"n": 0}

    def counting_parse(text_bytes, flavor):
        calls["n"] += 1
        return _fake_python_tree()

    monkeypatch.setattr(code_mod, "HAS_TREE_SITTER", True)
    monkeypatch.setattr(code_mod, "_parse", counting_parse)

    big = _FAKE_PY_SRC * 10  # 220 bytes, well over the tiny cap below

    capped = CodeCompressor(Config(code_ast_max_bytes=32))(big)
    assert calls["n"] == 0  # AST path never touched
    assert isinstance(capped, str)  # handed to the text fallback

    # With the cap disabled the same input DOES go through the AST parse.
    calls["n"] = 0
    CodeCompressor(Config(code_ast_max_bytes=0))(big)
    assert calls["n"] >= 1


def test_config_default_code_ast_cap():
    assert Config().code_ast_max_bytes == 256 * 1024


def test_detect_language():
    py_code = """
import os
def hello(a: int) -> str:
    \"\"\"My docstring\"\"\"
    return str(a)
"""
    js_code = """
import { useState } from "react";
function greet(name) {
  console.log(name);
  return `Hello, ${name}`;
}
"""
    assert detect_language(py_code) == "python"
    assert detect_language(js_code) == "javascript"


def test_python_compression_with_docstring():
    code = """def hello(a: int) -> str:
    \"\"\"This is the first line of doc.
    This is the second line.
    \"\"\"
    x = a + 1
    return str(x)
"""
    store = InMemoryCCRStore()
    comp = CodeCompressor(Config(), store=store)
    compressed = comp(code)

    assert "def hello(a: int) -> str:" in compressed
    assert '"""This is the first line of doc."""' in compressed
    assert "x = a + 1" not in compressed
    assert "return str(x)" not in compressed

    markers = find_markers(compressed)
    assert len(markers) == 1
    assert markers[0].reason == "code-elided"

    # Verify CCR store stashed it and can retrieve
    original_elided = store.get(markers[0].hash)
    assert original_elided is not None
    assert "x = a + 1" in original_elided


def test_python_compression_without_docstring():
    code = """def greet(name: str):
    print("hello", name)
    return True
"""
    store = InMemoryCCRStore()
    comp = CodeCompressor(Config(), store=store)
    compressed = comp(code)

    assert "def greet(name: str):" in compressed
    assert "print(" not in compressed

    markers = find_markers(compressed)
    assert len(markers) == 1
    assert markers[0].reason == "code-elided"


def test_javascript_compression():
    code = """function add(a, b) {
  const result = a + b;
  return result;
}
"""
    store = InMemoryCCRStore()
    comp = CodeCompressor(Config(), store=store)
    compressed = comp(code)

    assert "function add(a, b) {" in compressed
    assert "}" in compressed
    assert "const result = a + b;" not in compressed

    markers = find_markers(compressed)
    assert len(markers) == 1
    assert markers[0].reason == "code-elided"

    original_elided = store.get(markers[0].hash)
    assert original_elided is not None
    assert "const result = a + b;" in original_elided


# --- Real-grammar checks (skipped where tree-sitter isn't installed) ---------

requires_ts = pytest.mark.skipif(
    not code_mod.HAS_TREE_SITTER, reason="tree-sitter grammars not installed"
)


@requires_ts
def test_single_parse_real_grammar():
    """With the real grammar, well-formed Python parses exactly once (#121)."""
    import tokenslim.compressors.code as cm

    calls = {"n": 0}
    original = cm._parse

    def counting(text_bytes, flavor):
        calls["n"] += 1
        return original(text_bytes, flavor)

    saved, cm._parse = cm._parse, counting
    try:
        src = "def hello(a: int) -> str:\n    x = a + 1\n    return str(x)\n"
        out = CodeCompressor(Config())(src)
    finally:
        cm._parse = saved

    assert calls["n"] == 1
    assert "return str(x)" not in out


@requires_ts
def test_ast_path_single_parse_scales_linearly():
    """A block with many functions is still parsed exactly ONCE — the real
    O(n^2) -> O(n) guarantee (issue #121). The old code re-parsed the buffer up
    to five times; a constant parse count regardless of function count is the
    deterministic regression guard.

    The input is kept modest on purpose (~7 KB, well under the cap): the native
    tree-sitter parser can segfault on some builds/versions (observed on
    CPython 3.10 + tree-sitter 0.26.0 around ~25 KB of densely-nested defs), so
    the invariant is asserted by counting parses rather than by stressing the C
    parser with a huge buffer.
    """
    import tokenslim.compressors.code as cm

    unit = "def f{i}(a: int, b: int) -> int:\n    total = a + b\n    return total\n\n"
    src = "".join(unit.format(i=i) for i in range(100))
    assert 4_000 < len(src.encode("utf-8")) < 256 * 1024  # many funcs, under the cap

    calls = {"n": 0}
    original = cm._parse

    def counting(text_bytes, flavor):
        calls["n"] += 1
        return original(text_bytes, flavor)

    saved, cm._parse = cm._parse, counting
    try:
        t0 = time.perf_counter()
        out = CodeCompressor(Config())(src)
        elapsed = time.perf_counter() - t0
    finally:
        cm._parse = saved

    assert calls["n"] == 1  # one parse for 100 functions — not 5, not O(n)
    assert elapsed < 20.0, f"AST compression too slow: {elapsed:.1f}s"
    assert "total = a + b" not in out  # every body was elided from the one tree


@requires_ts
def test_oversized_input_fast_fallback():
    """Input over the cap skips the AST parse and returns fast (#121)."""
    import tokenslim.compressors.code as cm

    src = "def f():\n    return 1\n" * 40000  # ~880 KB, well over the 256 KB cap
    assert len(src.encode("utf-8")) > 256 * 1024

    calls = {"n": 0}
    original = cm._parse

    def counting(text_bytes, flavor):
        calls["n"] += 1
        return original(text_bytes, flavor)

    saved, cm._parse = cm._parse, counting
    try:
        t0 = time.perf_counter()
        out = CodeCompressor(Config())(src)
        elapsed = time.perf_counter() - t0
    finally:
        cm._parse = saved

    assert calls["n"] == 0  # AST parse skipped entirely
    assert elapsed < 10.0
    assert isinstance(out, str)
