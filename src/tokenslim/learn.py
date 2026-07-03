"""Mine captured sessions for failure patterns and turn them into agent rules.

Offline pass over the JSONL events written by :mod:`tokenslim.capture` (#41):
:func:`analyze_sessions` runs a set of detectors over ``{ts, session_id,
kind, payload}`` events and returns :class:`Finding` records;
:func:`propose_rules` renders deduplicated findings as a markdown block under
a ``## Learned rules (tokenslim)`` heading; :func:`apply_rules` previews (as a
unified diff) or writes that block into a rules file (``CLAUDE.md`` /
``AGENTS.md``), touching ONLY the tokenslim-managed section between the
``<!-- tokenslim:learn:start -->`` / ``<!-- tokenslim:learn:end -->`` markers.

Detectors (see :func:`analyze_sessions` for the tunable thresholds):

- ``repeated-tool-failure`` — the same tool errored repeatedly (its
  ``tool_call`` payload signals failure, or a failure ``outcome`` immediately
  follows it) → propose checking that tool's preconditions.
- ``user-correction`` — an ``outcome`` marked failure/corrected right after an
  assistant action, carrying correction text → propose a "do X instead of Y"
  rule built from the correction payload.
- ``inefficient-compression`` — ``compress`` events with a savings ratio
  below ``low_ratio`` on payloads of at least ``large_tokens`` tokens →
  propose tuning the tokenslim config.
"""

from __future__ import annotations

import difflib
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "LEARN_END_MARKER",
    "LEARN_START_MARKER",
    "RULES_HEADING",
    "Finding",
    "analyze_sessions",
    "apply_rules",
    "propose_rules",
]

# Markers delimiting the tokenslim-managed section inside a rules file.
LEARN_START_MARKER = "<!-- tokenslim:learn:start -->"
LEARN_END_MARKER = "<!-- tokenslim:learn:end -->"

# Heading of the rules block produced by propose_rules().
RULES_HEADING = "## Learned rules (tokenslim)"

# Statuses (case-insensitive) that mark a payload/outcome as a failure.
_FAILURE_STATUSES = frozenset(
    {"error", "failure", "failed", "fail", "fatal", "timeout", "rejected", "denied"}
)
# Outcome statuses that count as a user correction of the previous action.
_CORRECTION_STATUSES = _FAILURE_STATUSES | {"corrected", "correction"}
# Payload/detail keys searched (in order) for the corrective instruction text.
_CORRECTION_KEYS = ("correction", "instead", "preferred", "fix", "rule")


@dataclass(frozen=True)
class Finding:
    """One mined failure pattern and the rule proposed to prevent it."""

    kind: str
    evidence_count: int
    sessions: tuple[str, ...]
    proposed_rule: str
    confidence: float


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else {}


def _session_id(event: dict[str, Any]) -> str:
    return str(event.get("session_id") or "unknown")


def _is_failure(payload: dict[str, Any]) -> bool:
    """True when a tool_call/outcome payload signals an error."""
    if str(payload.get("status", "")).strip().lower() in _FAILURE_STATUSES:
        return True
    if payload.get("error"):
        return True
    if payload.get("success") is False:
        return True
    exit_code = payload.get("exit_code")
    return isinstance(exit_code, int) and exit_code != 0


def _correction_text(payload: dict[str, Any]) -> str | None:
    """Extract the corrective instruction ("do X") from an outcome payload."""
    sources: list[dict[str, Any]] = [payload]
    detail = payload.get("detail")
    if isinstance(detail, dict):
        sources.append(detail)
    for source in sources:
        for key in _CORRECTION_KEYS:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _confidence(evidence_count: int) -> float:
    """Deterministic, monotone confidence in (0, 0.95] from evidence volume."""
    return round(min(0.95, evidence_count / (evidence_count + 3)), 2)


def analyze_sessions(
    events: Iterable[dict[str, Any]],
    *,
    min_tool_failures: int = 3,
    min_corrections: int = 2,
    min_inefficient: int = 2,
    low_ratio: float = 0.05,
    large_tokens: int = 1000,
) -> list[Finding]:
    """Run all detectors over captured ``events`` and return findings.

    ``events`` is any iterable of ``{ts, session_id, kind, payload}`` dicts —
    typically ``tokenslim.capture.read_sessions(path)``. Malformed events are
    skipped; the pass never raises on odd payloads. Findings are sorted by
    descending confidence, then kind, then rule text (deterministic).
    """
    # tool name -> session ids of each observed failure (one entry per failure).
    tool_failures: defaultdict[str, list[str]] = defaultdict(list)
    # (action label, correction text) -> session ids of each correction.
    corrections: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    # session ids of each inefficient compress event.
    inefficient: list[str] = []
    # Per session: (kind, payload, failure_already_counted) of the previous event.
    prev: dict[str, tuple[str, dict[str, Any], bool]] = {}

    for event in events:
        if not isinstance(event, dict):
            continue
        kind = event.get("kind")
        payload = _payload(event)
        session = _session_id(event)

        if kind == "tool_call":
            tool = str(payload.get("tool") or "unknown")
            failed = _is_failure(payload)
            if failed:
                tool_failures[tool].append(session)
            prev[session] = ("tool_call", payload, failed)
        elif kind == "compress":
            ratio = payload.get("ratio")
            orig = payload.get("orig_tokens")
            if (
                isinstance(ratio, (int, float))
                and isinstance(orig, (int, float))
                and ratio < low_ratio
                and orig >= large_tokens
            ):
                inefficient.append(session)
            prev[session] = ("compress", payload, False)
        elif kind == "outcome":
            status = str(payload.get("status", "")).strip().lower()
            prev_kind, prev_payload, counted = prev.get(session, ("", {}, False))
            failed = _is_failure(payload)
            # Attribute a failure outcome to the tool_call right before it,
            # unless that call already counted as a failure by itself.
            if failed and prev_kind == "tool_call" and not counted:
                tool = str(prev_payload.get("tool") or "unknown")
                tool_failures[tool].append(session)
            # A failure/corrected outcome right after an assistant action that
            # carries correction text is a user correction of that action.
            if status in _CORRECTION_STATUSES and prev_kind in ("tool_call", "compress"):
                text = _correction_text(payload)
                if text:
                    if prev_kind == "tool_call":
                        action = f"`{prev_payload.get('tool') or 'unknown'}`"
                    else:
                        action = "the current `compress` settings"
                    corrections[(action, text)].append(session)
            prev[session] = ("outcome", payload, False)
        else:
            # Unknown kinds still break tool_call -> outcome adjacency.
            prev[session] = (str(kind), payload, False)

    findings: list[Finding] = []
    for tool, sessions in tool_failures.items():
        count = len(sessions)
        if count >= min_tool_failures:
            rule = (
                f"Check `{tool}` preconditions (inputs, paths, permissions) before "
                f"calling it; verify the failure mode from past sessions is handled."
            )
            findings.append(
                Finding(
                    kind="repeated-tool-failure",
                    evidence_count=count,
                    sessions=tuple(sorted(set(sessions))),
                    proposed_rule=rule,
                    confidence=_confidence(count),
                )
            )
    for (action, text), sessions in corrections.items():
        count = len(sessions)
        if count >= min_corrections:
            rule = f"Do {text} instead of {action}."
            findings.append(
                Finding(
                    kind="user-correction",
                    evidence_count=count,
                    sessions=tuple(sorted(set(sessions))),
                    proposed_rule=rule,
                    confidence=_confidence(count),
                )
            )
    if len(inefficient) >= min_inefficient:
        count = len(inefficient)
        rule = (
            f"Tune the tokenslim config for large payloads (lower `min_bytes`, raise "
            f"crush aggressiveness, or enable more compressors): compression saved "
            f"under {low_ratio:.0%} on payloads of {large_tokens}+ tokens."
        )
        findings.append(
            Finding(
                kind="inefficient-compression",
                evidence_count=count,
                sessions=tuple(sorted(set(inefficient))),
                proposed_rule=rule,
                confidence=_confidence(count),
            )
        )

    findings.sort(key=lambda f: (-f.confidence, f.kind, f.proposed_rule))
    return findings


def propose_rules(findings: Iterable[Finding]) -> str:
    """Render ``findings`` as a deduplicated markdown rules block.

    One imperative rule per line with a short "why", under the
    ``## Learned rules (tokenslim)`` heading. Duplicate rule texts are
    emitted once (highest-confidence occurrence wins). Output is
    deterministic for a given set of findings; empty findings yield ``""``.
    """
    ordered = sorted(
        findings,
        key=lambda f: (-f.confidence, f.kind, f.proposed_rule, -f.evidence_count, f.sessions),
    )
    lines = [RULES_HEADING, ""]
    seen: set[str] = set()
    for finding in ordered:
        if finding.proposed_rule in seen:
            continue
        seen.add(finding.proposed_rule)
        n_sessions = len(finding.sessions)
        plural = "s" if n_sessions != 1 else ""
        lines.append(
            f"- {finding.proposed_rule} (why: {finding.kind} seen "
            f"{finding.evidence_count}x across {n_sessions} session{plural})"
        )
    if not seen:
        return ""
    return "\n".join(lines) + "\n"


def apply_rules(
    markdown_block: str,
    target_path: str | os.PathLike[str],
    dry_run: bool = True,
) -> str:
    """Preview or write ``markdown_block`` into the managed section of a file.

    Returns a unified diff of the change (empty string when the file is
    already up to date). With ``dry_run=True`` (the default) nothing is
    written. With ``dry_run=False`` the block is written between the
    ``LEARN_START_MARKER`` / ``LEARN_END_MARKER`` comments: an existing
    section is replaced in place (idempotent), otherwise the section is
    appended; the rest of the file is never touched. A missing target file
    is treated as empty and created on write.
    """
    path = os.fspath(target_path)
    try:
        with open(path, encoding="utf-8") as fh:
            old = fh.read()
    except FileNotFoundError:
        old = ""

    section = f"{LEARN_START_MARKER}\n{markdown_block.strip()}\n{LEARN_END_MARKER}\n"
    pattern = re.compile(
        re.escape(LEARN_START_MARKER) + r".*?" + re.escape(LEARN_END_MARKER) + r"\n?",
        re.DOTALL,
    )
    if pattern.search(old):
        new = pattern.sub(lambda _m: section, old, count=1)
    else:
        if not old or old.endswith("\n\n"):
            separator = ""
        elif old.endswith("\n"):
            separator = "\n"
        else:
            separator = "\n\n"
        new = old + separator + section

    if new == old:
        return ""

    diff = "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    if not dry_run:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new)
    return diff
