"""Metrics + savings reporting.

A :class:`MetricsCollector` aggregates token counts and dollar savings across
many ``compress()`` calls and renders a markdown report. The dollar figure comes
from :mod:`tokenslim.pricing`: the tokens *removed* from the prompt would have
been billed at the model's input rate, so saved tokens map directly to saved
input cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .pricing import ModelPrice, estimate_cost

__all__ = ["RunRecord", "MetricsCollector"]


@dataclass(frozen=True)
class RunRecord:
    """One compression run's headline numbers."""

    orig_tokens: int
    new_tokens: int
    model: str | None = None
    label: str = ""

    @property
    def saved_tokens(self) -> int:
        return self.orig_tokens - self.new_tokens

    @property
    def ratio(self) -> float:
        if self.orig_tokens == 0:
            return 0.0
        return 1.0 - (self.new_tokens / self.orig_tokens)


@dataclass
class MetricsCollector:
    """Accumulates compression runs and reports aggregate savings."""

    model: str | None = None
    prices: dict[str, ModelPrice] | None = None
    runs: list[RunRecord] = field(default_factory=list)

    def record(
        self,
        orig_tokens: int,
        new_tokens: int,
        *,
        model: str | None = None,
        label: str = "",
    ) -> RunRecord:
        """Record one run; returns the stored :class:`RunRecord`."""
        rec = RunRecord(orig_tokens, new_tokens, model or self.model, label)
        self.runs.append(rec)
        return rec

    def record_stats(self, stats, *, model: str | None = None, label: str = "") -> RunRecord:
        """Record from a :class:`~tokenslim.compress.CompressionStats`."""
        return self.record(stats.orig_tokens, stats.new_tokens, model=model, label=label)

    # -- aggregates ------------------------------------------------------

    @property
    def total_orig_tokens(self) -> int:
        return sum(r.orig_tokens for r in self.runs)

    @property
    def total_new_tokens(self) -> int:
        return sum(r.new_tokens for r in self.runs)

    @property
    def total_saved_tokens(self) -> int:
        return self.total_orig_tokens - self.total_new_tokens

    @property
    def overall_ratio(self) -> float:
        if self.total_orig_tokens == 0:
            return 0.0
        return 1.0 - (self.total_new_tokens / self.total_orig_tokens)

    def saved_usd(self) -> float:
        """Total USD saved (saved input tokens valued at each run's model rate)."""
        total = 0.0
        for r in self.runs:
            total += estimate_cost(r.model or self.model, r.saved_tokens, 0, prices=self.prices)
        return total

    # -- reporting -------------------------------------------------------

    def generate_report(self, title: str = "tokenslim savings report") -> str:
        """Render a markdown savings report."""
        lines = [f"# {title}", ""]
        if not self.runs:
            lines.append("_No runs recorded._")
            return "\n".join(lines)

        lines += [
            f"- **Runs:** {len(self.runs)}",
            f"- **Original tokens:** {self.total_orig_tokens:,}",
            f"- **Compressed tokens:** {self.total_new_tokens:,}",
            f"- **Saved tokens:** {self.total_saved_tokens:,} ({self.overall_ratio:.1%})",
            f"- **Estimated cost saved:** ${self.saved_usd():,.4f}",
            "",
            "| Run | Model | Orig | New | Saved | Ratio | $ saved |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for i, r in enumerate(self.runs, 1):
            usd = estimate_cost(r.model or self.model, r.saved_tokens, 0, prices=self.prices)
            label = r.label or f"run {i}"
            lines.append(
                f"| {label} | {r.model or 'default'} | {r.orig_tokens:,} | "
                f"{r.new_tokens:,} | {r.saved_tokens:,} | {r.ratio:.1%} | ${usd:,.4f} |"
            )
        return "\n".join(lines)
