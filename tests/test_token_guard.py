"""Token-inflation guard tests (#117): compression must never cost tokens.

The audit measured TextCompressor inflating a real git log by +7.8%: CCR markers
cost tokens, and the old guard compared *characters*, so a marker-bearing block
could shrink in chars while growing in tokens. The fix is a per-block token
guard in ``compress()``: a routed block is kept only when it saves more than
``config.min_token_savings`` tokens net, else it reverts to a passthrough.

Invariant under test: ``stats.new_tokens <= stats.orig_tokens`` for every block
and therefore in aggregate.
"""

from __future__ import annotations

import json

from tokenslim import compress
from tokenslim.config import Config
from tokenslim.evals.fixtures import all_fixtures


def _inflating_prose() -> str:
    """A prose shape that inflated tokens before the guard (+7.6% measured).

    Many short paragraphs whose dropped sentences are individually replaced by a
    verbose CCR marker — fewer characters, more tokens.
    """
    return "\n\n".join(
        f"Topic {i}: this lead sentence carries the main point clearly and stays. "
        f"Secondary detail alpha {i}. Tertiary aside beta {i}. Extra note gamma {i}."
        for i in range(40)
    )


def test_inflating_prose_is_not_inflated():
    text = _inflating_prose()
    out, stats = compress([{"role": "user", "content": text}], min_bytes=0, target_ratio=0.25)
    # Before the fix this was new/orig = 1394/1295 (+7.6%).
    assert stats.new_tokens <= stats.orig_tokens


def test_reverted_block_is_recorded_as_passthrough():
    text = _inflating_prose()
    out, stats = compress([{"role": "user", "content": text}], min_bytes=0, target_ratio=0.25)
    # The block gave no net token win, so it reverts: unchanged output + a
    # BlockStat that reflects the passthrough (not changed).
    assert out[0]["content"] == text
    assert stats.blocks, "expected at least one recorded block"
    block = stats.blocks[0]
    assert block.changed is False
    assert block.new_tokens == block.orig_tokens


def test_invariant_holds_across_eval_corpus():
    for fixture in all_fixtures():
        out, stats = compress([{"role": "user", "content": fixture.content}], min_bytes=0)
        assert stats.new_tokens <= stats.orig_tokens, f"{fixture.name} inflated tokens"


def test_min_token_savings_floor_reverts_marginal_wins():
    """A block that genuinely compresses is reverted when the savings floor is
    set above its net win — proving the floor is configurable (#117)."""
    payload = json.dumps({"items": [{"id": i, "status": "ok"} for i in range(200)]})
    # Baseline: this compresses (net token win) with the default floor.
    _, baseline = compress([{"role": "user", "content": payload}], min_bytes=0)
    assert baseline.new_tokens < baseline.orig_tokens

    # With an unreachable savings floor, the same block reverts to passthrough.
    out, stats = compress(
        [{"role": "user", "content": payload}],
        options=Config(min_bytes=0, min_token_savings=10**9),
    )
    assert stats.new_tokens == stats.orig_tokens
    assert all(not b.changed for b in stats.blocks)


def test_real_compression_is_still_kept():
    """Regression guard: the token guard must not over-revert genuine wins."""
    rows = json.dumps([{"id": i, "status": "ok", "amount": i} for i in range(300)])
    out, stats = compress([{"role": "user", "content": rows}], min_bytes=0)
    assert stats.new_tokens < stats.orig_tokens
    assert stats.saved_tokens > 0
