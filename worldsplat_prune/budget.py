"""Segment-budget allocation helper.

A small, dependency-free allocator retained for the future per-segment pruning
option (see ``docs/worldsplat_prune_spec.md`` §7.3): it distributes an integer
budget across weighted buckets by largest-remainder, so the shares sum exactly.
Pure ``torch`` and device-agnostic.
"""
from __future__ import annotations

import torch


def allocate(delta_total: int, p):
    """Split ``delta_total`` integer slots across ids by weight ``p``.

    Largest-remainder allocation: each id gets its floor share, the remaining
    slots go to the ids with the largest fractional parts. Returns ``long`` with
    the same shape as ``p``.
    """
    p = p.reshape(-1)
    total = p.sum()
    long = torch.zeros_like(p, dtype=torch.long)
    delta_total = int(delta_total)
    if delta_total <= 0 or total <= 0:
        return long
    exact = p / total * delta_total
    long = torch.floor(exact).to(torch.long)
    rem = delta_total - int(long.sum())
    if rem > 0:
        frac = exact - long.to(exact.dtype)
        k = min(rem, frac.numel())
        top = torch.topk(frac, k).indices
        long[top] += 1
    return long
