"""HtmlExtractor — boilerplate removal for HTML documents.

Strips page chrome (scripts, styles, navigation, headers/footers, asides,
forms, iframes, SVG, comments) and attribute noise from an HTML document,
keeping the readable content: the page title (as a heading line), headings,
paragraph/list/table/pre/blockquote text and link text, with whitespace
collapsed. The original document is stashed behind a CCR marker so the
compression stays reversible.

Built on :mod:`html.parser` only — no external dependencies. Never raises:
on parse failure, tiny input, or when extraction is not smaller, the original
text is returned unchanged.
"""

from __future__ import annotations

from html.parser import HTMLParser
from typing import TYPE_CHECKING

from ..ccr import text_marker
from ..config import Config
from ..detector import ContentType

if TYPE_CHECKING:
    from ..store import CCRStore

__all__ = ["HtmlExtractor"]

# Subtrees dropped wholesale as boilerplate.
_DROP_TAGS = frozenset(
    {
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "iframe",
        "svg",
        "noscript",
        "template",
    }
)

# Tags that delimit text blocks (lines) in the extracted output.
_BLOCK_TAGS = frozenset(
    {
        "html",
        "head",
        "body",
        "main",
        "section",
        "article",
        "div",
        "p",
        "ul",
        "ol",
        "dl",
        "li",
        "dt",
        "dd",
        "table",
        "thead",
        "tbody",
        "tr",
        "blockquote",
        "pre",
        "br",
        "hr",
        "figure",
        "figcaption",
        "details",
        "summary",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }
)

_HEADING_PREFIX = {f"h{level}": "#" * level + " " for level in range(1, 7)}

# Inputs shorter than this aren't worth extracting.
_MIN_CHARS = 100

_CCR_REASON = "html-boilerplate-removed"


class _TextExtractor(HTMLParser):
    """Collects readable text blocks, skipping boilerplate subtrees."""

    def __init__(self, keep_links: bool = False) -> None:
        super().__init__(convert_charrefs=True)
        self.keep_links = keep_links
        self.title = ""
        self.blocks: list[str] = []
        self._buf: list[str] = []
        self._prefix = ""
        self._skip_tag: str | None = None
        self._skip_depth = 0
        self._in_pre = False
        self._in_title = False
        self._title_parts: list[str] = []
        self._href: str | None = None

    def _flush(self) -> None:
        raw = "".join(self._buf)
        self._buf = []
        text = raw.strip("\n") if self._in_pre else " ".join(raw.split())
        if text:
            self.blocks.append(self._prefix + text)
        self._prefix = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._skip_depth:
            if tag == self._skip_tag:
                self._skip_depth += 1
            return
        if tag in _DROP_TAGS:
            self._skip_tag = tag
            self._skip_depth = 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in ("td", "th"):
            if "".join(self._buf).strip():
                self._buf.append(" | ")
            return
        if tag == "a" and self.keep_links:
            for name, value in attrs:
                if name == "href" and value and not value.startswith(("#", "javascript:")):
                    self._href = value
                    break
            return
        if tag in _BLOCK_TAGS:
            self._flush()
            self._prefix = _HEADING_PREFIX.get(tag, "- " if tag == "li" else "")
            if tag == "pre":
                self._in_pre = True

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if not self._skip_depth:
                    self._skip_tag = None
            return
        if tag == "title":
            self.title = " ".join("".join(self._title_parts).split())
            self._in_title = False
            return
        if tag == "a":
            if self._href:
                self._buf.append(f" ({self._href})")
            self._href = None
            return
        if tag in _BLOCK_TAGS:
            self._flush()
            if tag == "pre":
                self._in_pre = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title_parts.append(data)
            return
        self._buf.append(data)

    def close(self) -> None:
        super().close()
        self._flush()


class HtmlExtractor:
    """Extracts readable content from HTML and drops boilerplate."""

    name = "html-extractor"

    def __init__(self, config: Config | None = None, store: CCRStore | None = None) -> None:
        self.config = config or Config()
        self.store = store

    def __call__(self, text: str, content_type: ContentType = ContentType.HTML) -> str:
        if len(text) < _MIN_CHARS:
            return text
        try:
            parser = _TextExtractor(keep_links=self.config.html_keep_links)
            parser.feed(text)
            parser.close()
        except Exception:
            return text

        pieces: list[str] = []
        if parser.title:
            pieces.append(f"# {parser.title}")
        pieces.extend(parser.blocks)
        if not pieces:
            return text

        marker = text_marker(text.split("\n"), reason=_CCR_REASON, store=self.store)
        result = "\n".join([*pieces, marker])
        return result if len(result) < len(text) else text
