"""CodeCompressor — AST-aware multi-language code compaction using tree-sitter.

Preserves imports, signatures, types, and the first line of docstrings/JSDoc;
elides function and method bodies and caches them in the CCR store if enabled.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from ..ccr import text_marker
from ..config import Config
from ..detector import ContentType

if TYPE_CHECKING:
    from ..store import CCRStore

# Try importing tree-sitter and supported language packages
try:
    import tree_sitter_javascript as tsjavascript
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser

    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False

__all__ = ["CodeCompressor", "detect_language"]


def detect_language(text: str) -> str | None:
    """Detect if the code is Python or JavaScript/TypeScript using tree-sitter."""
    if not HAS_TREE_SITTER:
        return None

    text_bytes = text.encode("utf-8")

    # Quick keyword check to prioritize
    py_hints = len(re.findall(r"\b(def|elif|import|from)\b", text))
    js_hints = len(re.findall(r"\b(function|const|let|console|export)\b", text))

    py_lang = Language(tspython.language())
    js_lang = Language(tsjavascript.language())

    py_parser = Parser(py_lang)
    js_parser = Parser(js_lang)

    if py_hints >= js_hints:
        py_tree = py_parser.parse(text_bytes)
        if not py_tree.root_node.has_error:
            return "python"
        js_tree = js_parser.parse(text_bytes)
        if not js_tree.root_node.has_error:
            return "javascript"
    else:
        js_tree = js_parser.parse(text_bytes)
        if not js_tree.root_node.has_error:
            return "javascript"
        py_tree = py_parser.parse(text_bytes)
        if not py_tree.root_node.has_error:
            return "python"

    # Fallback: choose the one with fewer errors
    def count_errors(node) -> int:
        count = 1 if node.type == "ERROR" or node.is_error else 0
        for child in node.children:
            count += count_errors(child)
        return count

    py_tree = py_parser.parse(text_bytes)
    js_tree = js_parser.parse(text_bytes)
    py_errors = count_errors(py_tree.root_node)
    js_errors = count_errors(js_tree.root_node)

    return "python" if py_errors <= js_errors else "javascript"


class CodeCompressor:
    """AST-aware code compressor using tree-sitter.

    Preserves classes, signatures, imports, types, and the first line of docstrings,
    while eliding function/method bodies. Supports Python and JavaScript.
    """

    name = "code-compressor"

    def __init__(self, config: Config | None = None, store: CCRStore | None = None) -> None:
        self.config = config or Config()
        self.store = store

    def __call__(self, text: str, content_type: ContentType = ContentType.CODE) -> str:
        if not HAS_TREE_SITTER:
            return text

        flavor = detect_language(text)
        if not flavor:
            return text

        text_bytes = text.encode("utf-8")
        if flavor == "python":
            lang = Language(tspython.language())
        else:
            lang = Language(tsjavascript.language())

        parser = Parser(lang)
        tree = parser.parse(text_bytes)

        # Collect function/method bodies to elide
        blocks: list[tuple[Any, str]] = []

        def traverse(node: Any) -> None:
            if flavor == "python":
                if node.type == "function_definition":
                    for child in node.children:
                        if child.type == "block":
                            blocks.append((child, "python"))
                            return
                for child in node.children:
                    traverse(child)
            else:
                if node.type in (
                    "function_declaration",
                    "function_expression",
                    "method_definition",
                    "arrow_function",
                    "generator_function",
                ):
                    for child in node.children:
                        if child.type == "statement_block":
                            blocks.append((child, "javascript"))
                            return
                for child in node.children:
                    traverse(child)

        traverse(tree.root_node)

        if not blocks:
            return text

        # Sort blocks by start byte to apply replacements sequentially
        blocks.sort(key=lambda x: x[0].start_byte)

        chunks: list[bytes] = []
        last_idx = 0

        for block, flav in blocks:
            # Code before the block
            chunks.append(text_bytes[last_idx : block.start_byte])

            if flav == "python":
                docstring_line = None
                if block.child_count > 0:
                    first_child = block.children[0]
                    if first_child.type == "expression_statement" and first_child.child_count > 0:
                        inner = first_child.children[0]
                        if inner.type == "string":
                            doc_bytes = inner.text
                            doc_str = doc_bytes.decode("utf-8", errors="replace")
                            # Strip triple or single quotes
                            m = re.match(r"^(\"\"\"|''')(.*)\1$", doc_str, re.DOTALL)
                            if not m:
                                m = re.match(r"^(\"|')(.*)\1$", doc_str, re.DOTALL)
                            if m:
                                content = m.group(2)
                                first_line = content.split("\n")[0].strip()
                                quotes = m.group(1)
                                docstring_line = f"{quotes}{first_line}{quotes}"
                            else:
                                first_line = doc_str.split("\n")[0].strip()
                                docstring_line = f'"""{first_line}"""'

                block_text = block.text.decode("utf-8", errors="replace")
                block_lines = block_text.splitlines()
                marker = text_marker(block_lines, reason="code-elided", store=self.store)

                indent = " " * block.start_point.column
                if docstring_line:
                    replacement = f"{docstring_line}\n{indent}# {marker}"
                else:
                    replacement = f"# {marker}"

                chunks.append(replacement.encode("utf-8"))
                last_idx = block.end_byte

            elif flav == "javascript":
                # Keep braces and elide the inside
                if (
                    block.child_count >= 2
                    and block.children[0].type == "{"
                    and block.children[-1].type == "}"
                ):
                    start_elide = block.children[0].end_byte
                    end_elide = block.children[-1].start_byte

                    # code between block start and start of elision (i.e. the opening '{')
                    chunks.append(text_bytes[block.start_byte : start_elide])

                    elided_bytes = text_bytes[start_elide:end_elide]
                    elided_str = elided_bytes.decode("utf-8", errors="replace")
                    elided_lines = elided_str.splitlines()
                    marker = text_marker(elided_lines, reason="code-elided", store=self.store)

                    indent = " " * (block.start_point.column + 2)
                    close_indent = " " * block.start_point.column
                    replacement = f"\n{indent}// {marker}\n{close_indent}"

                    chunks.append(replacement.encode("utf-8"))
                    last_idx = end_elide
                else:
                    block_text = block.text.decode("utf-8", errors="replace")
                    block_lines = block_text.splitlines()
                    marker = text_marker(block_lines, reason="code-elided", store=self.store)
                    replacement = f"// {marker}"
                    chunks.append(replacement.encode("utf-8"))
                    last_idx = block.end_byte

        chunks.append(text_bytes[last_idx:])
        return b"".join(chunks).decode("utf-8", errors="replace")
