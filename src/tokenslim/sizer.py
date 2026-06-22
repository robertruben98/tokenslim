"""Adaptive sizer — how many items to keep under a budget.

Every selective compressor (log, search, diff, SmartCrusher) faces the same
question: given ``n`` candidate items and a target compression ratio, how many
should survive? :func:`compute_optimal_k` answers it with an exponential-decay
budget — keep proportionally more of a small input and aggressively fewer of a
large one — clamped to sane bounds.

Design notes
------------
* Monotonic: ``k`` never decreases as ``n`` grows, and never increases as the
  target ratio shrinks (a tighter ratio keeps fewer items).
* No overshoot: a budget that resolves to 1 keeps exactly 1 (the historical
  ``k=1`` bug kept 2).
* ``k`` is capped at ``n`` — you can't keep more than you have.
"""

from __future__ import annotations

import math

__all__ = ["compute_optimal_k"]


def compute_optimal_k(
    n_items: int,
    target_ratio: float = 0.2,
    *,
    min_k: int = 1,
    max_k: int | None = None,
    decay: float = 0.5,
) -> int:
    """Return how many of ``n_items`` to keep.

    Args:
        n_items: Number of candidate items.
        target_ratio: Fraction of items to keep at the reference size
            (0 < ratio <= 1). Lower keeps fewer.
        min_k: Never keep fewer than this (when ``n_items`` allows).
        max_k: Optional hard ceiling on the result.
        decay: Exponential-decay strength in [0, 1]. 0 = linear
            (``k = ratio * n``); higher pulls large inputs down harder so the
            kept count grows sub-linearly.

    The kept fraction is ``target_ratio`` scaled down as ``n`` grows::

        effective_ratio = target_ratio * n ** (-decay)
        k = ceil(effective_ratio * n) = ceil(target_ratio * n ** (1 - decay))
    """
    if n_items <= 0:
        return 0
    target_ratio = max(0.0, min(1.0, target_ratio))
    decay = max(0.0, min(1.0, decay))

    raw = target_ratio * (n_items ** (1.0 - decay))
    k = math.ceil(raw)

    # Clamp: at least min_k (but not more than we have), at most n_items / max_k.
    k = max(k, min(min_k, n_items))
    if max_k is not None:
        k = min(k, max_k)
    return min(k, n_items)
