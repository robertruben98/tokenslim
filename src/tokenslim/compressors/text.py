"""TextCompressor — extractive prose and markdown compression.

Splits input text into paragraphs, identifies structural elements (headings, code blocks,
lists, etc.) to keep them intact, splits normal prose into sentences, scores sentences
based on position, length, word rarity, and optional query keyword relevance, and keeps
a target ratio of the sentences. Dropped sentences are stashed in the CCR store.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..ccr import text_marker
from ..config import Config
from ..detector import ContentType

if TYPE_CHECKING:
    from ..store import CCRStore

__all__ = ["TextCompressor"]

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "is", "was", "were", "are", "be", "been", "it", "this", "that",
    "he", "she", "they", "we", "i", "you", "his", "her", "their", "our", "my",
    "your", "them", "us", "him", "me", "as", "by", "from", "about", "into",
    "through", "over", "under", "again", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "can", "will", "just", "should", "would", "now"
}


def _is_structural(paragraph: str) -> bool:
    stripped = paragraph.strip()
    if not stripped:
        return True
    # heading
    if stripped.startswith("#"):
        return True
    # list item
    if re.match(r"^(?:[-*+]\s|\d+\.\s)", stripped):
        return True
    # blockquote
    if stripped.startswith(">"):
        return True
    # code block fence
    if stripped.startswith("```"):
        return True
    # HTML tag or markdown link image/reference/metadata blocks
    return stripped.startswith("<") or stripped.startswith("![")


def _split_into_paragraphs(text: str) -> list[tuple[str, bool]]:
    """Split text into paragraphs, returning list of (text, is_structural)."""
    raw_paragraphs = text.split("\n\n")
    paragraphs: list[tuple[str, bool]] = []

    in_code_block = False
    current_block: list[str] = []

    for raw_p in raw_paragraphs:
        fences = raw_p.count("```")
        if in_code_block:
            current_block.append(raw_p)
            if fences % 2 != 0:
                in_code_block = False
                paragraphs.append(("\n\n".join(current_block), True))
                current_block = []
        else:
            if fences % 2 != 0:
                in_code_block = True
                current_block.append(raw_p)
            else:
                is_struct = _is_structural(raw_p)
                paragraphs.append((raw_p, is_struct))

    if current_block:
        paragraphs.append(("\n\n".join(current_block), True))

    return paragraphs


def _split_sentences(paragraph: str) -> list[str]:
    raw_sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    return [s.strip() for s in raw_sentences if s.strip()]


def _get_content_words(sentence: str) -> list[str]:
    words = re.findall(r"\b[a-zA-Z0-9]+\b", sentence.lower())
    return [w for w in words if w not in _STOPWORDS]


@dataclass
class SentenceInfo:
    paragraph_idx: int
    sentence_idx: int
    text: str
    score: float = 0.0
    keep: bool = False


class TextCompressor:
    """Groups prose sentences, scores them, and filters them."""

    name = "text-compressor"

    def __init__(self, config: Config | None = None, store: CCRStore | None = None) -> None:
        self.config = config or Config()
        self.store = store

    def __call__(self, text: str, content_type: ContentType = ContentType.TEXT) -> str:
        paragraphs = _split_into_paragraphs(text)

        # Count document-wide word frequencies in prose paragraphs
        word_freqs: dict[str, int] = {}
        for p_text, is_struct in paragraphs:
            if not is_struct:
                for word in _get_content_words(p_text):
                    word_freqs[word] = word_freqs.get(word, 0) + 1

        sentences_info: list[SentenceInfo] = []
        sentence_map: dict[tuple[int, int], SentenceInfo] = {}

        for p_idx, (p_text, is_struct) in enumerate(paragraphs):
            if is_struct:
                continue
            p_sentences = _split_sentences(p_text)
            n_sentences = len(p_sentences)
            for s_idx, s_text in enumerate(p_sentences):
                info = SentenceInfo(
                    paragraph_idx=p_idx,
                    sentence_idx=s_idx,
                    text=s_text
                )

                # Heuristic scoring
                score = 1.0

                # Paragraph lead/position bias
                if s_idx == 0:
                    score += 3.0
                elif s_idx == n_sentences - 1:
                    score += 1.5

                c_words = _get_content_words(s_text)
                if not c_words:
                    score = 0.0
                else:
                    score += len(c_words) * 0.01
                    if len(c_words) > 30:
                        score -= (len(c_words) - 30) * 0.05

                    # Rarity heuristic
                    rarity_sum = 0.0
                    for w in c_words:
                        freq = word_freqs.get(w, 0)
                        if freq > 0:
                            rarity_sum += 1.5 / (1.0 + freq)
                    score += rarity_sum / len(c_words)

                # Query relevance
                if self.config.query:
                    query_words = _get_content_words(self.config.query)
                    for qw in query_words:
                        if qw in c_words:
                            score += 3.0

                info.score = score
                sentences_info.append(info)
                sentence_map[(p_idx, s_idx)] = info

        N = len(sentences_info)
        if N == 0:
            return text

        # Select sentences to keep based on target_ratio
        k = max(1, min(N, round(N * self.config.target_ratio)))
        ranked = sorted(sentences_info, key=lambda s: s.score, reverse=True)
        for s in ranked[:k]:
            s.keep = True

        # Reconstruct paragraphs
        out_paragraphs: list[str] = []
        for p_idx, (p_text, is_struct) in enumerate(paragraphs):
            if is_struct:
                out_paragraphs.append(p_text)
                continue

            p_sentences = _split_sentences(p_text)
            reconstructed_sentences: list[str] = []
            dropped_buffer: list[str] = []

            for s_idx, s_text in enumerate(p_sentences):
                info = sentence_map[(p_idx, s_idx)]
                if info.keep:
                    if dropped_buffer:
                        marker = text_marker(
                            dropped_buffer, reason="prose-elided", store=self.store
                        )
                        reconstructed_sentences.append(marker)
                        dropped_buffer = []
                    reconstructed_sentences.append(s_text)
                else:
                    dropped_buffer.append(s_text)

            if dropped_buffer:
                marker = text_marker(
                    dropped_buffer, reason="prose-elided", store=self.store
                )
                reconstructed_sentences.append(marker)

            out_paragraphs.append(" ".join(reconstructed_sentences))

        result = "\n\n".join(out_paragraphs)
        return result if len(result) < len(text) else text
