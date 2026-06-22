"""Eval harness — ratio + faithfulness over bundled fixtures (offline)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ccr import find_markers
from ..compress import compress
from ..config import Config
from ..metrics import MetricsCollector
from ..tokenizer import count_tokens
from .fixtures import Fixture, all_fixtures

__all__ = ["EvalResult", "run_suite", "perf_report"]


@dataclass
class EvalResult:
    """Outcome of evaluating one fixture."""

    name: str
    orig_tokens: int
    new_tokens: int
    must_keep_ok: bool
    retrievable_ok: bool
    n_markers: int
    missing: list[str] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        if self.orig_tokens == 0:
            return 0.0
        return 1.0 - (self.new_tokens / self.orig_tokens)

    @property
    def faithful(self) -> bool:
        """No answer-bearing content lost: must-keeps survive AND drops recover."""
        return self.must_keep_ok and self.retrievable_ok


def _evaluate(fixture: Fixture, config: Config) -> tuple[EvalResult, str]:
    messages = [{"role": "tool", "tool_call_id": "t", "content": fixture.content}]
    out, stats = compress(messages, options=config)
    visible = out[0]["content"]

    orig_tokens = count_tokens(fixture.content, config.model)
    new_tokens = count_tokens(visible, config.model)

    # 1) Must-keep substrings survive in the *visible* output.
    missing = [s for s in fixture.must_keep if s not in visible]
    must_keep_ok = not missing

    # 2) Faithfulness of drops: every CCR marker's original is retrievable
    #    verbatim from the store (compress -> retrieve round-trip).
    markers = find_markers(visible)
    retrievable_ok = True
    if stats.store is not None:
        for marker in markers:
            if stats.store.get(marker.hash) is None:
                retrievable_ok = False
                break
    else:
        # CCR disabled -> nothing was stored; faithfulness of drops is vacuous
        # only if nothing was dropped.
        retrievable_ok = not markers

    result = EvalResult(
        name=fixture.name,
        orig_tokens=orig_tokens,
        new_tokens=new_tokens,
        must_keep_ok=must_keep_ok,
        retrievable_ok=retrievable_ok,
        n_markers=len(markers),
        missing=missing,
    )
    return result, visible


def run_suite(
    fixtures: list[Fixture] | None = None,
    config: Config | None = None,
) -> list[EvalResult]:
    """Run the eval suite and return per-fixture results.

    Uses ``min_bytes=0`` by default so every fixture is actually compressed.
    """
    fixtures = fixtures if fixtures is not None else all_fixtures()
    config = config or Config(min_bytes=0)
    return [_evaluate(f, config)[0] for f in fixtures]


def perf_report(
    fixtures: list[Fixture] | None = None,
    config: Config | None = None,
    model: str | None = None,
) -> str:
    """Run the bundled workload and render a before/after savings report.

    Combines ratio + faithfulness per fixture with the dollar savings from
    :mod:`tokenslim.metrics` / :mod:`tokenslim.pricing`.
    """
    fixtures = fixtures if fixtures is not None else all_fixtures()
    config = config or Config(min_bytes=0, model=model)
    results = run_suite(fixtures, config)

    collector = MetricsCollector(model=model or config.model)
    for r in results:
        collector.record(r.orig_tokens, r.new_tokens, label=r.name)

    lines = [
        "# tokenslim perf report",
        "",
        f"- **Fixtures:** {len(results)}",
        f"- **Original tokens:** {collector.total_orig_tokens:,}",
        f"- **Compressed tokens:** {collector.total_new_tokens:,}",
        f"- **Saved tokens:** {collector.total_saved_tokens:,} ({collector.overall_ratio:.1%})",
        f"- **Estimated cost saved:** ${collector.saved_usd():,.6f} "
        f"(model: {model or config.model or 'default'})",
        f"- **All faithful:** {all(r.faithful for r in results)}",
        "",
        "| Fixture | Orig | New | Ratio | Faithful | Drops |",
        "| --- | ---: | ---: | ---: | :---: | ---: |",
    ]
    for r in results:
        check = "yes" if r.faithful else "NO"
        lines.append(
            f"| {r.name} | {r.orig_tokens:,} | {r.new_tokens:,} | "
            f"{r.ratio:.1%} | {check} | {r.n_markers} |"
        )
    return "\n".join(lines)
