from tokenslim.ccr import find_markers
from tokenslim.compressors.code import CodeCompressor, detect_language
from tokenslim.config import Config
from tokenslim.store import InMemoryCCRStore


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
