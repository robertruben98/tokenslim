"""Offline evals — prove compression saves tokens without losing answers.

Without an LLM we can still measure two honest things per fixture:

* **ratio** — how many tokens compression removed.
* **faithfulness** — that nothing answer-bearing was *lost*: the dropped
  material is recovered exactly from the CCR store (compress → retrieve →
  reconstruct → byte-compare), and known must-keep rows (errors) always survive
  in the visible output.

:func:`run_suite` runs the bundled fixtures and returns structured results;
:func:`perf_report` renders a before/after savings report.
"""

from __future__ import annotations

from .harness import EvalResult, perf_report, run_suite

__all__ = ["run_suite", "perf_report", "EvalResult"]
