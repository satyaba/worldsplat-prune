"""LOD-family export (C3, spec §6) — one frozen asset → N budget-matched PLYs.

The budget-controllable "semantic LoD" family is the same post-reconstruction
pipeline run at several *global* budgets. Instead of re-loading and re-voting per
budget, this module runs the heavy path **once** (load + §3 vote, via
:func:`prune_model.load_and_vote`), plans one
:class:`~worldsplat_prune.prune.PrunePlan` per budget, and writes each plan as a
standalone model directory plus a single deterministic manifest::

    <out>/
      lod_0.25/point_cloud/iteration_<n>/point_cloud.ply
      lod_0.25/cfg_args
      lod_0.5/point_cloud/iteration_<n>/point_cloud.ply
      lod_0.5/cfg_args
      lod_0.75/...
      lod_manifest.json

Every ``lod_<frac>/`` directory is a valid model dir, so the family is directly
scorable through ``eval_worldsplat`` / ``experiments.metrics`` without any
re-planning. The operator is **never forked**: budgets only change ``keep_frac``
on the shared :func:`worldsplat_prune.prune.plan_pruning` call, and the subset/
save path is the driver's own.

The planning, histogram and manifest halves are pure ``torch`` and CPU-testable
(``tests/test_export_lod.py``); only :func:`export_lod_family` touches the model.
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Optional, Sequence

import torch

from .prune import DEFAULT_BAND, DEFAULT_STRIDE, plan_pruning

DEFAULT_BUDGETS = (0.25, 0.5, 0.75)


# --------------------------------------------------------------------------- #
# naming / layout                                                               #
# --------------------------------------------------------------------------- #
def lod_dirname(budget: float) -> str:
    """Model-dir name for one budget, e.g. ``lod_0.5``."""
    return f"lod_{float(budget):g}"


def lod_ply_relpath(budget: float, iteration: int) -> str:
    """The PLY path of one budget relative to the family root (POSIX-ish)."""
    return os.path.join(lod_dirname(budget), "point_cloud",
                        f"iteration_{int(iteration)}", "point_cloud.ply")


# --------------------------------------------------------------------------- #
# planning (pure)                                                               #
# --------------------------------------------------------------------------- #
def plan_lod_family(alpha, scaling, *, boundary_dist, seg, purity, extent,
                    budgets: Sequence[float] = DEFAULT_BUDGETS,
                    **plan_kwargs) -> list:
    """Plan one :class:`PrunePlan` per budget from a single vote bundle (pure).

    ``plan_kwargs`` are the ordinary :func:`plan_pruning` switches (``band``,
    ``strength``, ``use_band``, ``allocation``, ``seed``, …); only ``keep_frac``
    varies per budget. Returns ``[(budget, plan), …]`` in ``budgets`` order.
    """
    return [(float(b), plan_pruning(
                alpha, scaling, boundary_dist=boundary_dist, seg=seg,
                purity=purity, extent=extent, keep_frac=float(b), **plan_kwargs))
            for b in budgets]


def kept_class_histogram(seg, keep, *, stride: int = DEFAULT_STRIDE) -> dict:
    """Kept-Gaussian count per class id (``seg // stride``; void = class 0)."""
    seg = seg.reshape(-1)
    keep = keep.reshape(-1).to(torch.bool)
    if int(seg.numel()) == 0:
        return {}
    cls = torch.div(seg, int(stride), rounding_mode="floor")[keep]
    uniq, counts = torch.unique(cls, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq.tolist(), counts.tolist())}


def family_manifest(family, *, base_model: str, iteration: int, n_before: int,
                    extent: float, band, band_class,
                    stride: int = DEFAULT_STRIDE, seg=None,
                    vote_meta: Optional[dict] = None) -> dict:
    """The deterministic ``lod_manifest.json`` payload (pure).

    Per budget: the matched count, the full :meth:`PrunePlan.report`, the floor/
    ceiling reachability flags, the kept-class histogram and the PLY path.
    """
    levels = []
    for b, plan in family:
        rep = plan.report()
        entry = {
            "keep_frac": float(b),
            "n_after": int(plan.n_after),
            "n_target": int(plan.n_target),
            "reachable_floor": bool(plan.reachable_floor),
            "reachable_ceiling": bool(plan.reachable_ceiling),
            "path": lod_ply_relpath(b, iteration),
            "report": rep,
        }
        if seg is not None:
            entry["class_hist"] = kept_class_histogram(seg, plan.keep, stride=stride)
        levels.append(entry)
    return {
        "base_model": str(base_model),
        "iteration": int(iteration),
        "n_before": int(n_before),
        "stride": int(stride),
        "extent": float(extent),
        "band": band,
        "band_class": {int(k): float(v) for k, v in dict(band_class or {}).items()},
        "budgets": [float(b) for b, _ in family],
        "vote": dict(vote_meta or {}),
        "levels": levels,
    }


def write_manifest(path, manifest) -> None:
    """Write the manifest as sorted, indented JSON (deterministic across runs)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


# --------------------------------------------------------------------------- #
# export (heavy: model path)                                                    #
# --------------------------------------------------------------------------- #
def export_lod_family(dataset, iteration, pipe, a, *, out_dir,
                      budgets: Sequence[float] = DEFAULT_BUDGETS) -> dict:
    """Load + vote once, plan per budget, write the PLY family + manifest (C3).

    Reuses :func:`prune_model.load_and_vote` and the driver's subset/save path, so
    the operator sees the identical bundle the single-budget driver would produce.
    Returns the manifest dict (also written to ``<out_dir>/lod_manifest.json``).
    """
    # Lazy import: prune_model imports this module, so avoid the top-level cycle.
    from prune_model import _plan_kwargs, _resolve_band, _subset_model, load_and_vote

    gaussians, scene, bundle = load_and_vote(dataset, iteration, pipe, a)
    n_before = int(bundle.n_gaussians)
    it = int(scene.loaded_iter)

    family = plan_lod_family(
        gaussians.get_opacity, gaussians.get_scaling,
        boundary_dist=bundle.boundary_dist, seg=bundle.seg, purity=bundle.purity,
        extent=scene.cameras_extent, budgets=budgets, **_plan_kwargs(a))

    band_spec = _resolve_band(a)
    if isinstance(band_spec, dict):
        band_scalar = DEFAULT_BAND
        band_class = band_spec
    else:
        band_scalar = float(band_spec)
        band_class = {}
    manifest = family_manifest(
        family, base_model=dataset.model_path, iteration=it, n_before=n_before,
        extent=scene.cameras_extent, band=band_scalar, band_class=band_class,
        seg=bundle.seg,
        vote_meta={k: bundle.meta[k] for k in
                   ("visibility", "vote", "occlusion", "min_views",
                    "n_visible", "n_labeled", "mean_purity")})

    if a.dry_run:
        print(f"[lod] dry run: budgets "
              f"{[round(float(b), 3) for b, _ in family]} planned, nothing written")
        return manifest

    out_dir = str(out_dir)
    cfg_src = os.path.join(dataset.model_path, "cfg_args")
    for b, plan in family:
        level_dir = os.path.join(out_dir, lod_dirname(b))
        ply_path = os.path.join(out_dir, lod_ply_relpath(b, it))
        os.makedirs(os.path.dirname(ply_path), exist_ok=True)
        pruned = _subset_model(gaussians, plan.keep)
        pruned.save_ply(ply_path)
        # a valid model dir needs the trained cfg for eval_worldsplat to read
        if os.path.exists(cfg_src):
            shutil.copyfile(cfg_src, os.path.join(level_dir, "cfg_args"))
        print(f"[lod] {lod_dirname(b)}: {plan.n_after} gaussians "
              f"(target {plan.n_target}) -> {ply_path}")

    write_manifest(os.path.join(out_dir, "lod_manifest.json"), manifest)
    print(f"[lod] wrote {len(family)} budgets -> {out_dir}/lod_manifest.json")
    return manifest


__all__ = [
    "DEFAULT_BUDGETS",
    "lod_dirname",
    "lod_ply_relpath",
    "plan_lod_family",
    "kept_class_histogram",
    "family_manifest",
    "write_manifest",
    "export_lod_family",
]
