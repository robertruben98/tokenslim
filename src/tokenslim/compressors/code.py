"""CodeCompressor — AST-aware multi-language code compaction using tree-sitter.

Preserves imports, signatures, types, and the first line of docstrings/JSDoc;
elides function and method bodies and caches them in the CCR store if enabled.

Performance (issue #121): the tree-sitter tree is built exactly ONCE per block
and reused across every pass (language detection, signatures, docstrings,
bodies) instead of re-parsing the buffer up to five times. Blocks larger than
``config.code_ast_max_bytes`` skip the AST path entirely and fall back to the
text compressor, so a pathological payload can never turn into a super-linear
parse / error-recovery.
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

# Node types whose bodies we elide, keyed by language.
_PY_FUNC_TYPES = frozenset({"function_definition"})
_JS_FUNC_TYPES = frozenset(
    {
        "function_declaration",
        "function_expression",
        "method_definition",
        "arrow_function",
        "generator_function",
    }
)
# The body node holding the statements to elide, per language.
_BODY_TYPE = {"python": "block", "javascript": "statement_block"}

_PY_HINT_RE = re.compile(r"\b(def|elif|import|from)\b")
_JS_HINT_RE = re.compile(r"\b(function|const|let|console|export)\b")


def _language(flavor: str) -> Any:
    """Return the tree-sitter ``Language`` for ``flavor``."""
    if flavor == "python":
        return Language(tspython.language())
    return Language(tsjavascript.language())


def _parse(text_bytes: bytes, flavor: str) -> Any:
    """Parse ``text_bytes`` with ``flavor``'s grammar and return the tree.

    This is the single tree-sitter parse choke point for the whole module: every
    parse goes through here, so a spy can assert the tree is built exactly once
    per block (issue #121).
    """
    parser = Parser(_language(flavor))
    return parser.parse(text_bytes)


def _detect_and_parse(text_bytes: bytes) -> tuple[str, Any] | None:
    """Detect the language *and* return its parse tree, parsing at most twice.

    Keyword hints pick the most likely grammar first so a well-formed block
    parses exactly once; the resulting tree is returned for reuse across every
    later pass. A second parse only happens when the first grammar reports
    errors (i.e. the hint was wrong). Returns ``None`` when tree-sitter is
    unavailable, so the compressor degrades to a safe no-op.
    """
    if not HAS_TREE_SITTER:
        return None

    text = text_bytes.decode("utf-8", errors="replace")
    py_hints = len(_PY_HINT_RE.findall(text))
    js_hints = len(_JS_HINT_RE.findall(text))
    order = ("python", "javascript") if py_hints >= js_hints else ("javascript", "python")

    first: tuple[str, Any] | None = None
    for flavor in order:
        tree = _parse(text_bytes, flavor)
        if first is None:
            first = (flavor, tree)
        if not tree.root_node.has_error:
            return flavor, tree
    # Neither grammar parsed cleanly: keep the best-guess (first) tree so we can
    # still elide what we can, without a third parse.
    return first


def detect_language(text: str) -> str | None:
    """Detect if the code is Python or JavaScript/TypeScript using tree-sitter."""
    detected = _detect_and_parse(text.encode("utf-8"))
    return None if detected is None else detected[0]


class CodeCompressor:
    """AST-aware code compressor using tree-sitter.

    Preserves classes, signatures, imports, types, and the first line of docstrings,
    while eliding function/method bodies. Supports Python and JavaScript.
    """

    name = "code-compressor"

    def __init__(self, config: Config | None = None, store: CCRStore | None = None) -> None:
        self.config = config or Config()
        self.store = store

    def _text_fallback(self, text: str, content_type: ContentType) -> str:
        """Compress ``text`` with the cheap text compressor (AST path skipped)."""
        from .text import TextCompressor

        return TextCompressor(self.config, self.store)(text, content_type)

    def __call__(self, text: str, content_type: ContentType = ContentType.CODE) -> str:
        if not HAS_TREE_SITTER:
            return text

        text_bytes = text.encode("utf-8")

        # Size cap for the AST path: on a very large (or malformed) block the
        # tree-sitter parse and its error recovery dominate, so hand oversized
        # inputs to the linear text compressor instead (issue #121). A cap of 0
        # (or less) disables the guard.
        cap = self.config.code_ast_max_bytes
        if cap and cap > 0 and len(text_bytes) > cap:
            return self._text_fallback(text, content_type)

        detected = _detect_and_parse(text_bytes)
        if detected is None:
            return text
        flavor, tree = detected

        # Single tree, reused for every pass below.
        blocks = self._collect_blocks(tree, flavor)
        if not blocks:
            return text

        return self._elide(text_bytes, blocks, flavor)

    def _collect_blocks(self, tree: Any, flavor: str) -> list[Any]:
        """Return function/method body nodes to elide, in source order.

        Linear iterative walk (no per-node re-scan of the buffer, no recursion):
        once a function body is found we do not descend into it, since the whole
        body is elided anyway.
        """
        func_types = _PY_FUNC_TYPES if flavor == "python" else _JS_FUNC_TYPES
        body_type = _BODY_TYPE[flavor]

        blocks: list[Any] = []
        stack: list[Any] = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type in func_types:
                body = next((c for c in node.children if c.type == body_type), None)
                if body is not None:
                    blocks.append(body)
                    continue  # do not descend into the elided body
            stack.extend(node.children)

        blocks.sort(key=lambda b: b.start_byte)
        return blocks

    def _elide(self, text_bytes: bytes, blocks: list[Any], flavor: str) -> str:
        """Rebuild the source with each body replaced by a compact CCR marker.

        Walks the (sorted) body offsets once, copying the spans between them —
        the whole rebuild is linear in the input size.
        """
        chunks: list[bytes] = []
        last_idx = 0

        for block in blocks:
            # Code before the block.
            chunks.append(text_bytes[last_idx : block.start_byte])

            if flavor == "python":
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

            elif flavor == "javascript":
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
