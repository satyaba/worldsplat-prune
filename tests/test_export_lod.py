"""Tests for the LOD-family export (worldsplat_prune.export_lod, C3).

CPU-only: the planning/histogram/manifest halves are pure torch and never import
a trained model or the CUDA rasteriser.

    pytest tests/test_export_lod.py
    python tests/test_export_lod.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worldsplat_prune import export_lod as E

STRIDE = E.DEFAULT_STRIDE


def _tensors(n=100, seg=None, d=None, purity=None, alpha=None, scaling=None):
    seg = torch.zeros(n, dtype=torch.long) if seg is None else torch.tensor(seg)
    purity = torch.ones(n) if purity is None else torch.tensor(purity, dtype=torch.float32)
    d = torch.full((n,), 0.5) if d is None else torch.tensor(d, dtype=torch.float32)
    alpha = torch.ones(n) if alpha is None else torch.tensor(alpha, dtype=torch.float32)
    scaling = torch.ones(n, 3) if scaling is None else torch.tensor(scaling, dtype=torch.float32)
    return alpha, scaling, d, seg, purity


def _family(n=100, budgets=E.DEFAULT_BUDGETS, plan=None, **kw):
    alpha, scaling, d, seg, purity = _tensors(n, **kw)
    return E.plan_lod_family(alpha, scaling, boundary_dist=d, seg=seg,
                             purity=purity, extent=10.0, budgets=budgets,
                             **(plan or {}))


# --------------------------------------------------------------------------- #
# planning                                                                      #
# --------------------------------------------------------------------------- #
def test_plan_lod_family_matched_counts():
    fam = _family(n=100, budgets=(0.25, 0.5, 0.75))
    assert [b for b, _ in fam] == [0.25, 0.5, 0.75]
    counts = {b: plan.n_after for b, plan in fam}
    assert counts == {0.25: 25, 0.5: 50, 0.75: 75}
    for b, plan in fam:
        assert plan.n_target == round(b * 100)
        assert plan.reachable_floor and plan.reachable_ceiling


def test_plan_lod_family_reports_unreachable_floor():
    # a wide band protects most Gaussians -> the low budget is unreachable
    n = 10
    d = torch.tensor([0.01] * 8 + [0.5, 0.5])
    alpha, scaling, _, seg, purity = _tensors(n)
    fam = E.plan_lod_family(alpha, scaling, boundary_dist=d, seg=seg,
                            purity=purity, extent=10.0, budgets=(0.1,))
    b, plan = fam[0]
    assert not plan.reachable_floor
    assert plan.n_after == plan.n_protected == 8


def test_manifest_deterministic_random_control():
    f1 = _family(n=40, plan={"random_control": True, "seed": 0})
    f2 = _family(n=40, plan={"random_control": True, "seed": 0})
    assert [p.keep.tolist() for _, p in f1] == [p.keep.tolist() for _, p in f2]


# --------------------------------------------------------------------------- #
# histogram                                                                     #
# --------------------------------------------------------------------------- #
def test_kept_class_histogram():
    seg = torch.tensor([1 * STRIDE, 1 * STRIDE, 2 * STRIDE, 0])
    keep = torch.tensor([True, False, True, True])
    assert E.kept_class_histogram(seg, keep) == {0: 1, 1: 1, 2: 1}
    assert E.kept_class_histogram(torch.zeros(0, dtype=torch.long),
                                  torch.zeros(0, dtype=torch.bool)) == {}


# --------------------------------------------------------------------------- #
# manifest                                                                      #
# --------------------------------------------------------------------------- #
def test_family_manifest_structure():
    n = 100
    seg = [1 * STRIDE] * 60 + [2 * STRIDE] * 40
    alpha, scaling, d, seg, purity = _tensors(n, seg=seg)
    fam = E.plan_lod_family(alpha, scaling, boundary_dist=d, seg=seg,
                            purity=purity, extent=10.0, budgets=(0.25, 0.5, 0.75))
    man = E.family_manifest(fam, base_model="/m", iteration=30000, n_before=100,
                            extent=10.0, band=0.04, band_class={1: 0.05}, seg=seg,
                            vote_meta={"n_labeled": 100, "mean_purity": 0.9})

    assert man["n_before"] == 100 and man["iteration"] == 30000
    assert man["budgets"] == [0.25, 0.5, 0.75]
    assert man["band"] == 0.04 and man["band_class"] == {1: 0.05}
    assert len(man["levels"]) == 3

    by_b = {lv["keep_frac"]: lv for lv in man["levels"]}
    assert by_b[0.25]["n_after"] == 25
    assert by_b[0.25]["path"].replace("\\", "/") == \
        "lod_0.25/point_cloud/iteration_30000/point_cloud.ply"
    # class histogram sums to the kept count
    assert sum(by_b[0.25]["class_hist"].values()) == 25
    assert set(by_b[0.5]["class_hist"]) <= {1, 2}
    assert by_b[0.5]["report"]["n_after"] == 50
    assert man["vote"]["n_labeled"] == 100


def test_family_manifest_deterministic():
    seg = [1 * STRIDE] * 60 + [2 * STRIDE] * 40
    alpha, scaling, d, seg, purity = _tensors(100, seg=seg)

    def build():
        fam = E.plan_lod_family(alpha, scaling, boundary_dist=d, seg=seg,
                                purity=purity, extent=10.0, budgets=(0.25, 0.5, 0.75))
        return E.family_manifest(fam, base_model="/m", iteration=30000, n_before=100,
                                 extent=10.0, band=0.04, band_class={}, seg=seg)

    a = json.dumps(build(), sort_keys=True)
    b = json.dumps(build(), sort_keys=True)
    assert a == b


def test_write_manifest_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "nested", "lod_manifest.json")
        E.write_manifest(path, {"a": 1, "b": [0.25, 0.5]})
        with open(path, encoding="utf-8") as f:
            assert json.load(f) == {"a": 1, "b": [0.25, 0.5]}


def test_lod_naming():
    assert E.lod_dirname(0.5) == "lod_0.5"
    assert E.lod_dirname(0.25) == "lod_0.25"
    assert E.lod_ply_relpath(0.75, 30000).replace("\\", "/") == \
        "lod_0.75/point_cloud/iteration_30000/point_cloud.ply"


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
