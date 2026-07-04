"""Bundled eval fixtures — representative agent tool-outputs.

Each fixture is a realistic, oversized tool output (a JSON list endpoint, an SRE
log, a code-search dump) plus the *must-keep* substrings that an answer would
depend on. The harness checks both that the fixture compresses well and that
every must-keep string is still present in the visible compressed output.

Fixtures are generated programmatically so the file stays small while the
payloads are large enough to trigger real compression.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Fixture:
    name: str
    content: str
    # Substrings that must survive in the *visible* compressed output.
    must_keep: tuple[str, ...] = ()
    # Human note on what an answer over this fixture would need.
    question: str = ""
    extras: dict = field(default_factory=dict)


def _json_rows_fixture() -> Fixture:
    # A list endpoint: 200 "ok" orders plus one failed payment — the answer-
    # bearing row that must never be dropped.
    rows = [
        {"id": i, "status": "ok", "amount": 10 + (i % 7), "currency": "USD"} for i in range(200)
    ]
    rows[123] = {
        "id": 123,
        "status": "error",
        "amount": 999,
        "currency": "USD",
        "detail": "payment declined by issuer",
    }
    return Fixture(
        name="json-orders",
        content=json.dumps(rows),
        must_keep=("payment declined by issuer", '"status":"error"'),
        question="Which order failed and why?",
    )


def _json_cyclic_fixture() -> Fixture:
    # #122 regression: a realistic list endpoint whose numeric ``price`` field
    # cycles through 37 repeated values (10..46). Before the per-column
    # uniformity guard, every row looked "rare" (each price recurs ~13x, below
    # the absolute rare cutoff) so the array crushed ~0%. The lone genuine
    # anomaly — a refunded, out-of-range, error-tagged row — must still survive.
    rows = [
        {
            "id": i,
            "name": f"item-{i}",
            "price": 10 + (i % 37),
            "status": "ok",
            "desc": "x" * 40,
        }
        for i in range(500)
    ]
    rows[137] = {
        "id": 137,
        "name": "item-137",
        "price": 99999,
        "status": "error",
        "desc": "refund failed: gateway timeout",
    }
    return Fixture(
        name="json-cyclic",
        content=json.dumps(rows),
        must_keep=('"status":"error"', "refund failed: gateway timeout"),
        question="Which row failed and why?",
    )


def _log_fixture() -> Fixture:
    lines = [
        f"2026-01-02 10:00:{i % 60:02d} INFO request {1000 + i} served 200" for i in range(120)
    ]
    lines.insert(60, "2026-01-02 10:01:00 ERROR db connection refused: pool exhausted")
    lines.append("2026-01-02 10:02:00 INFO shutting down: 1 error, 120 ok")
    return Fixture(
        name="sre-log",
        content="\n".join(lines),
        must_keep=("ERROR db connection refused: pool exhausted",),
        question="What error occurred during the run?",
    )


def _search_fixture() -> Fixture:
    # A ripgrep dump: 40 boilerplate reference files (capped away) plus a couple
    # of files with many hits each (path repetition the grouping kills). One of
    # them holds the real definition the answer needs.
    lines = [f"src/refs/mod_{i}.py:{i + 1}:    return helper()" for i in range(40)]
    for ln in range(1, 30):
        lines.append(f"src/big_module.py:{ln}:    log.debug('processing item %d', {ln})")
    lines.append("src/auth.py:42:def validate_token(token: str) -> bool:")
    lines.append("src/auth.py:43:    return verify(token)")
    return Fixture(
        name="code-search",
        content="\n".join(lines),
        must_keep=("def validate_token", "src/auth.py"),
        question="Where is validate_token defined?",
    )


def all_fixtures() -> list[Fixture]:
    """Return the bundled fixture suite."""
    return [_json_rows_fixture(), _json_cyclic_fixture(), _log_fixture(), _search_fixture()]
