"""Tests for the post-reconstruction pruning operator (worldsplat_prune.prune, §4-5).

CPU-only: the operator is pure torch and never imports the base 3DGS model.

    pytest tests/test_prune.py
    python tests/test_prune.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worldsplat_prune import prune as P


def _inputs(n, seg=None, purity=None, d=None, alpha=None, scaling=None):
    seg = torch.zeros(n, dtype=torch.long) if seg is None else torch.tensor(seg)
    purity = torch.ones(n) if purity is None else torch.tensor(purity, dtype=torch.float32)
    d = torch.full((n,), 0.5) if d is None else torch.tensor(d, dtype=torch.float32)
    alpha = torch.ones(n) if alpha is None else torch.tensor(alpha, dtype=torch.float32)
    scaling = torch.ones(n, 3) if scaling is None else torch.tensor(scaling, dtype=torch.float32)
    return seg, purity, d, alpha, scaling


def _plan(seg, purity, d, alpha, scaling, **kw):
    return P.plan_pruning(alpha, scaling, boundary_dist=d, seg=seg, purity=purity,
                          extent=kw.pop("extent", 10.0), **kw)


# --------------------------------------------------------------------------- #
# score                                                                         #
# --------------------------------------------------------------------------- #
def test_geometry_backbone_and_confidence():
    alpha = torch.tensor([2.0, 1.0])
    scaling = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    g = P.geometry_backbone(alpha, scaling)
    assert abs(g[0].item() - 12.0) < 1e-6 and abs(g[1].item() - 1.0) < 1e-6

    # bounded confidence: floor 1 - strength, floor not crossed
    conf = P.semantic_confidence(torch.tensor([0.0, 0.5, 1.0]), strength=0.5, curve=1.0)
    assert abs(conf[0].item() - 0.5) < 1e-6      # rho=0 -> floor
    assert abs(conf[1].item() - 0.75) < 1e-6
    assert abs(conf[2].item() - 1.0) < 1e-6


# --------------------------------------------------------------------------- #
# gates                                                                         #
# --------------------------------------------------------------------------- #
def test_gates_band_overrides_dead():
    # idx0 dead by opacity; idx4 dead by opacity but on the band -> survives
    seg, purity, d, alpha, scaling = _inputs(
        6, seg=[1, 1, 2, 2, 3, 3], d=[0.5, 0.5, 0.01, 0.5, 0.01, 0.5],
        alpha=[0.001, 1, 1, 1, 0.001, 1])
    plan = _plan(seg, purity, d, alpha, scaling, keep_frac=1.0)

    assert plan.dead.tolist() == [True, False, False, False, False, False]
    assert plan.band.tolist() == [False, False, True, False, True, False]
    assert plan.k_dead == 1
    assert plan.n_protected == 2                  # band {2,4} alive
    # ceiling: target >= survivors -> nothing contested removed
    assert plan.n_after == 5 and plan.k_contested == 0


def test_target_protection_is_budget_scope_not_dead_override():
    # seg 3 = target; idx3 is dead (no band) -> still removed (target != dead)
    seg, purity, d, alpha, scaling = _inputs(
        6, seg=[1, 1, 2, 2, 3, 3], d=[0.5, 0.5, 0.01, 0.5, 0.01, 0.5],
        alpha=[0.001, 1, 1, 0.001, 1, 1])
    plan = _plan(seg, purity, d, alpha, scaling, keep_frac=0.5, target=3)

    assert plan.dead.tolist() == [True, False, False, True, False, False]
    assert plan.protected.tolist() == [False, False, True, False, True, True]
    assert plan.contested.tolist() == [False, True, False, False, False, False]
    assert plan.n_protected == 3 and plan.k_dead == 2
    assert plan.n_after == 3 and plan.k_contested == 1


# --------------------------------------------------------------------------- #
# selection                                                                     #
# --------------------------------------------------------------------------- #
def test_selection_deletes_lowest_confidence():
    # equal geometry; purity 0 -> conf 0.5, so the two low-purity ones go first
    seg, purity, d, alpha, scaling = _inputs(4, purity=[1.0, 1.0, 0.0, 0.0])
    plan = _plan(seg, purity, d, alpha, scaling, keep_frac=0.5)

    assert plan.n_target == 2 and plan.k_contested == 2
    assert plan.keep.tolist() == [True, True, False, False]
    assert plan.n_after == 2


def test_geometry_only_control_ignores_purity():
    seg, purity, d, alpha, scaling = _inputs(4, purity=[1.0, 1.0, 0.0, 0.0])
    plan = _plan(seg, purity, d, alpha, scaling, keep_frac=0.5,
                 use_band=False, use_purity=False)
    # geometry is identical everywhere -> some 2 of 4 are dropped, not purity-driven
    assert plan.k_contested == 2 and plan.n_after == 2


def test_reachability_floor_and_ceiling():
    # band protects 4 of 6; keep_frac 0.2 -> target 1 < protected -> floor breached
    seg, purity, d, alpha, scaling = _inputs(6)
    d = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.5, 0.5])
    plan = _plan(seg, purity, d, alpha, scaling, keep_frac=0.2)
    assert not plan.reachable_floor
    assert plan.n_after == plan.n_protected == 4
    # above the ceiling: dead removal already undershoots the target
    seg2, p2, d2, a2, s2 = _inputs(6)
    a2[0] = 0.001
    plan2 = _plan(seg2, p2, d2, a2, s2, n_target=6)
    assert not plan2.reachable_ceiling and plan2.n_after == 5


def test_per_segment_allocation_spreads_the_quota():
    seg, purity, d, alpha, scaling = _inputs(
        6, seg=[1, 1, 1, 2, 2, 2])
    plan = _plan(seg, purity, d, alpha, scaling, keep_frac=0.5,
                 allocation="per_segment")
    assert plan.k_contested == 3
    lost1 = int((plan.delete & (seg == 1)).sum().item())
    lost2 = int((plan.delete & (seg == 2)).sum().item())
    assert lost1 >= 1 and lost2 >= 1 and lost1 + lost2 == 3


def test_random_control_runs_and_hits_budget():
    seg, purity, d, alpha, scaling = _inputs(20)
    plan = _plan(seg, purity, d, alpha, scaling, keep_frac=0.5, random_control=True)
    assert plan.n_after == 10


def test_jaccard():
    a = torch.tensor([True, True, False, False])
    b = torch.tensor([True, False, True, False])
    assert abs(P.jaccard(a, b) - 1 / 3) < 1e-6
    assert P.jaccard(torch.zeros(3, dtype=torch.bool),
                     torch.zeros(3, dtype=torch.bool)) == 1.0


def test_report_shape():
    seg, purity, d, alpha, scaling = _inputs(4, purity=[1.0, 1.0, 0.0, 0.0])
    rep = _plan(seg, purity, d, alpha, scaling, keep_frac=0.5).report()
    for k in ("n_before", "n_after", "k_dead", "k_contested", "contested_share",
              "effectively_geometry_only", "reachable_floor", "reachable_ceiling"):
        assert k in rep
    assert rep["n_before"] == 4 and rep["removed"] == 2


# --------------------------------------------------------------------------- #
# class-conditioned δ (D20)                                                     #
# --------------------------------------------------------------------------- #
STRIDE = P.DEFAULT_STRIDE


def test_band_class_mapping_matches_scalar_when_equal():
    # a table whose entries equal the scalar must reproduce the scalar band,
    # so supplying a mapping is a no-op when it agrees with the default (D1).
    seg = torch.tensor([1 * STRIDE + 1, 1 * STRIDE + 2,
                        2 * STRIDE + 1, 2 * STRIDE + 2])
    d = torch.tensor([0.03, 0.05, 0.03, 0.05])
    ones = torch.ones(4)
    base = _plan(seg, ones, d, ones, torch.ones(4, 3), keep_frac=0.5, band=0.04)
    mapped = _plan(seg, ones, d, ones, torch.ones(4, 3), keep_frac=0.5,
                   band={1: 0.04, 2: 0.04})
    assert base.band.tolist() == mapped.band.tolist()
    assert base.keep.tolist() == mapped.keep.tolist()
    assert base.n_after == mapped.n_after


def test_band_class_protects_named_class():
    # class 1 gets a larger δ; class 2 falls back to the 0.04 default. Both
    # centres sit at the same d_i, so only the protected class is banded.
    seg = torch.tensor([1 * STRIDE + 1, 2 * STRIDE + 1])
    d = torch.tensor([0.06, 0.06])
    plan = _plan(seg, torch.ones(2), d, torch.ones(2), torch.ones(2, 3),
                 keep_frac=0.0, band={1: 0.1})
    assert plan.band.tolist() == [True, False]
    assert plan.protected.tolist() == [True, False]
    assert plan.contested.tolist() == [False, True]
    # floor breach: the protected class alone is kept
    assert not plan.reachable_floor
    assert plan.keep.tolist() == [True, False]


def test_band_class_class_of_hook():
    # `class_of` recovers class ids directly (e.g. an already class-only seg).
    seg = torch.tensor([1, 2])
    d = torch.tensor([0.06, 0.06])
    plan = _plan(seg, torch.ones(2), d, torch.ones(2), torch.ones(2, 3),
                 keep_frac=0.0, band={1: 0.1}, class_of=lambda s: s)
    assert plan.band.tolist() == [True, False]


def test_band_class_recorded_in_meta_and_report():
    seg = torch.tensor([1 * STRIDE + 1, 2 * STRIDE + 1])
    d = torch.tensor([0.06, 0.06])
    plan = _plan(seg, torch.ones(2), d, torch.ones(2), torch.ones(2, 3),
                 band={1: 0.1, 2: 0.05})
    assert plan.meta["band_class"] == {1: 0.1, 2: 0.05}
    assert plan.meta["band"] == P.DEFAULT_BAND
    rep = plan.report()
    assert rep["band_class"] == {1: 0.1, 2: 0.05}


def test_parse_band_class_pairs_and_json():
    import json
    import tempfile

    assert P.parse_band_class(["1:0.05", "3:0.1"]) == {1: 0.05, 3: 0.1}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"2": 0.07, "5": 0.12}, f)
        path = f.name
    try:
        assert P.parse_band_class([path]) == {2: 0.07, 5: 0.12}
        assert P.parse_band_class(["1:0.05", path]) == {1: 0.05, 2: 0.07, 5: 0.12}
    finally:
        os.remove(path)


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)
