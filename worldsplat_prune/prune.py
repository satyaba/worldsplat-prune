"""Post-reconstruction semantic-aware pruning operator (§4–5).

Given a loaded (frozen) 3DGS model and the per-Gaussian :class:`VoteBundle`
(§3), this decides **which Gaussians to delete** to hit a real-time render
budget while preserving semantic structure. It is a pure planning step — it
produces a keep mask + a report; applying it (subsetting and writing a PLY) and
scoring the result are separate (see the ``prune_model.py`` driver).

The pipeline (``docs/worldsplat_prune_spec.md`` §4):

0. **Gates**, in precedence ``band > dead > contested``:
   ``band_i = d_i < δ_i`` ([D1] absolute protect; [D20] class-conditioned
   ``δ_i``); ``target_i = seg_i ==
   s_target`` ([D13], budget-scope); ``protected = band or target``;
   ``dead_i = (α_i < ε) or (max_j s_ij > 0.1·extent)`` **and not band_i** ([D5]);
   ``contested = not protected and not dead``.
1. **Hard-remove** every ``dead`` Gaussian, outside the quota.
2. From **contested only**, delete ``K = (N − |dead|) − N_target`` by score,
   keeping the higher-scoring ones.

Score (§5.1), higher = keep:

    score_i = (α_i · Π_j s_ij) · conf_i
    conf_i  = 1 − σ·(1 − ρ_i^a)     ∈ [1−σ, 1]        (default σ=0.5, a=1)

``α·Πs`` is the geometry backbone (volume-weighted opacity); ``conf`` is a
**bounded** semantic tiebreak, so a low-purity (or never-labelled, ``ρ=0``)
Gaussian keeps at least ``1−σ`` of its keep-priority — void is never a hard
kill ([D12]). Void (``ν``) and support (``n_views``) are deliberately *off*.

The **control matrix** (§5.2) is expressed by flags: the geometry-only control
is ``use_band=False, use_purity=False``; the random structure bar is
``random_control=True``; per-segment allocation is the ``allocation=
"per_segment"`` ablation. All produce the same keep-mask/report interface.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import torch

from .budget import allocate

# Hyperparameters fixed by the design doc ([D1]/[D11]/[D12]).
DEFAULT_BAND = 0.04          # normalized `d_i` threshold == 4 px at dt_norm=100
DEFAULT_EPSILON = 0.005      # alpha floor (the trainer's prune call)
DEFAULT_STRENGTH = 0.5       # σ, semantic strength
DEFAULT_CURVE = 1.0          # a, purity curve
DEFAULT_EXTENT_FRAC = 0.1    # max(s) > 0.1 * cameras_extent -> dead
# Composite `(class, instance)` stride; keep in sync with
# ``worldsplat_prune.datasets.composite.COMPOSITE_STRIDE``. Defined here so this
# module stays a pure-torch import (no cv2/numpy via the dataset package).
DEFAULT_STRIDE = 10_000

# The band threshold δ: a scalar (D1, all classes) or a `{class_id: δ}` table
# ([D20], class-conditioned; unspecified classes fall back to DEFAULT_BAND).
BandSpec = Union[float, Mapping[int, float]]


# --------------------------------------------------------------------------- #
# score                                                                         #
# --------------------------------------------------------------------------- #
def geometry_backbone(alpha: torch.Tensor, scaling: torch.Tensor) -> torch.Tensor:
    """``α · Π s`` — volume-weighted opacity (§5.1). ``[N]``."""
    return alpha.reshape(-1).to(torch.float32) * scaling.to(torch.float32).prod(dim=-1)


def semantic_confidence(purity: torch.Tensor, strength: float = DEFAULT_STRENGTH,
                        curve: float = DEFAULT_CURVE) -> torch.Tensor:
    """Bounded ``conf = 1 − σ·(1 − ρ^a) ∈ [1−σ, 1]`` (§5.1, [D12])."""
    p = purity.reshape(-1).to(torch.float32).clamp(0.0, 1.0)
    return 1.0 - float(strength) * (1.0 - p.pow(float(curve)))


def score_gaussians(alpha, scaling, purity, *, strength=DEFAULT_STRENGTH,
                    curve=DEFAULT_CURVE, use_purity=True) -> torch.Tensor:
    """The keep-score (§5.1): geometry, optionally times the confidence."""
    geom = geometry_backbone(alpha, scaling)
    if not use_purity:
        return geom
    return geom * semantic_confidence(purity, strength, curve)


# --------------------------------------------------------------------------- #
# band (δ) — [D1]/[D20]                                                          #
# --------------------------------------------------------------------------- #
def parse_band_class(values) -> dict:
    """Parse ``--band_class`` tokens into a ``{class_id: δ}`` table.

    Each token is either a ``class:delta`` pair (repeatable) or the path of a
    JSON object ``{class_id: delta}``. A token that names an existing ``.json``
    file is always read as JSON (so Windows paths containing ``:`` still work);
    otherwise a ``class:delta`` pair is expected.
    """
    table: dict = {}
    for tok in values or []:
        tok = str(tok).strip()
        if not tok:
            continue
        if tok.lower().endswith(".json") or os.path.exists(tok):
            with open(tok, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, Mapping):
                raise ValueError(f"band_class JSON must be an object: {tok}")
            for k, v in data.items():
                table[int(k)] = float(v)
        else:
            if ":" not in tok:
                raise ValueError(
                    f"band_class token must be 'class:delta' or a JSON path: {tok!r}")
            k, v = tok.split(":", 1)
            table[int(k)] = float(v)
    return table


def band_delta(seg: torch.Tensor, table: Mapping[int, float], default: float,
               *, stride: int = DEFAULT_STRIDE,
               class_of: Optional[Callable] = None,
               device=None) -> torch.Tensor:
    """Per-Gaussian δ ``[N]`` from a ``{class_id: δ}`` table ([D20]).

    The class id is ``class_of(seg)`` when a hook is given, else ``seg // stride``
    (recovering the composite ``(class, instance)`` half, §4.1). Classes absent
    from ``table`` get ``default`` (the scalar D1 δ).
    """
    if class_of is not None:
        cls = class_of(seg)
    else:
        if int(stride) <= 0:
            raise ValueError("stride must be positive when class_of is None")
        cls = torch.div(seg, int(stride), rounding_mode="floor")
    delta = torch.full(cls.shape, float(default), dtype=torch.float32,
                       device=device if device is not None else cls.device)
    for c, v in table.items():
        delta[cls == int(c)] = float(v)
    return delta


# --------------------------------------------------------------------------- #
# plan                                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class PrunePlan:
    """The gate membership, the keep/delete decision and the count contract."""

    keep: torch.Tensor                        # bool [N]
    delete: torch.Tensor                      # bool [N]
    band: torch.Tensor                        # bool [N]
    dead: torch.Tensor                        # bool [N]
    protected: torch.Tensor                   # bool [N]
    contested: torch.Tensor                   # bool [N]
    score: torch.Tensor                       # float [N]
    n_before: int
    n_after: int
    k_dead: int
    k_contested: int
    n_protected: int
    n_target: int
    target_id: Optional[int]
    reachable_floor: bool
    reachable_ceiling: bool
    allocation: str = "global"
    meta: dict = field(default_factory=dict)

    def report(self) -> dict:
        removed = self.n_before - self.n_after
        contested_share = self.k_contested / removed if removed else 0.0
        return {
            "n_before": self.n_before,
            "n_after": self.n_after,
            "n_target": self.n_target,
            "k_dead": self.k_dead,
            "k_contested": self.k_contested,
            "contested_share": contested_share,
            "effectively_geometry_only": self.k_contested == 0,
            "n_protected": self.n_protected,
            "n_band": int(self.band.sum().item()),
            "target_id": self.target_id,
            "allocation": self.allocation,
            "reachable_floor": self.reachable_floor,
            "reachable_ceiling": self.reachable_ceiling,
            "removed": removed,
            "band": self.meta.get("band"),
            "band_class": self.meta.get("band_class", {}),
        }


def _select_lowest(contested: torch.Tensor, score: torch.Tensor, k: int,
                   seg: Optional[torch.Tensor], allocation: str) -> torch.Tensor:
    """Return a ``delete`` bool mask over contested Gaussians (``k`` of them)."""
    device = contested.device
    delete = torch.zeros_like(contested)
    if k <= 0:
        return delete
    idx = torch.nonzero(contested, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return delete
    k = min(int(k), int(idx.numel()))
    sc = score[idx]

    if allocation == "global":
        chosen = torch.topk(sc, k, largest=False).indices
        delete[idx[chosen]] = True
        return delete

    if allocation == "per_segment":
        if seg is None:
            raise ValueError("per_segment allocation needs `seg`")
        seg_c = seg[idx]
        uniq = torch.unique(seg_c)
        weight = torch.stack([(seg_c == u).sum() for u in uniq]).to(torch.float32)
        quota = allocate(k, weight)
        for u, q in zip(uniq.tolist(), quota.tolist()):
            if q <= 0:
                continue
            mask_u = seg_c == u
            local = torch.nonzero(mask_u, as_tuple=False).squeeze(1)
            take = torch.topk(sc[local], min(int(q), local.numel()),
                              largest=False).indices
            delete[idx[local[take]]] = True
        return delete

    raise ValueError(f"unknown allocation {allocation!r} (global|per_segment)")


def plan_pruning(alpha, scaling, *, boundary_dist, seg, purity,
                 extent: float,
                 keep_frac: float = 0.5,
                 n_target: Optional[int] = None,
                 band: BandSpec = DEFAULT_BAND,
                 epsilon: float = DEFAULT_EPSILON,
                 strength: float = DEFAULT_STRENGTH,
                 curve: float = DEFAULT_CURVE,
                 use_band: bool = True,
                 use_purity: bool = True,
                 allocation: str = "global",
                 target: Optional[int] = None,
                 dead_extent_frac: float = DEFAULT_EXTENT_FRAC,
                 random_control: bool = False,
                 seed: int = 0,
                 stride: int = DEFAULT_STRIDE,
                 class_of: Optional[Callable] = None) -> PrunePlan:
    """Plan a pruning pass (§4–5). Pure tensors; device-agnostic.

    Parameters
    ----------
    alpha, scaling
        Loaded geometry: opacity ``α [N]`` (post-sigmoid) and scales ``s [N, 3]``
        (post-exp).
    boundary_dist, seg, purity
        The :class:`~worldsplat_prune.vote.VoteBundle` fields (§3.4).
    extent
        ``scene.cameras_extent`` — the scale-prune reference ([D11]).
    keep_frac or n_target
        Global budget: keep ``round(keep_frac·N)`` (or exactly ``n_target``).
    band, epsilon, strength, curve
        Hyperparameters ([D1]/[D11]/[D12]). ``band`` is a scalar δ (default) or a
        ``{class_id: δ}`` table for class-conditioned protection ([D20]).
    stride, class_of
        How to recover the class id from a composite ``seg``: ``class_of(seg)`` if
        given, else ``seg // stride`` ([D20]).
    use_band, use_purity
        Control-matrix switches: ``False/False`` is the geometry-only control.
    allocation
        ``"global"`` (primary [D4]) or ``"per_segment"`` (ablation).
    target
        Consistent instance id to keep whole ([D13] mode (b)); ``None`` = off.
    random_control
        Replace the score by noise (the random structure bar, §5.2).
    """
    alpha = alpha.reshape(-1).to(torch.float32)
    scaling = scaling.to(torch.float32)
    d = boundary_dist.reshape(-1).to(torch.float32)
    seg = seg.reshape(-1)
    purity = purity.reshape(-1).to(torch.float32)
    N = int(alpha.shape[0])
    device = alpha.device

    # ---- (0) gates ------------------------------------------------------- #
    if isinstance(band, Mapping):
        band_table = {int(k): float(v) for k, v in band.items()}
        band_default = float(DEFAULT_BAND)
        delta = band_delta(seg, band_table, band_default, stride=stride,
                           class_of=class_of, device=device)
        band_mask = (d < delta) if use_band else torch.zeros(N, dtype=torch.bool, device=device)
    else:
        band_table = {}
        band_default = float(band)
        band_mask = (d < band_default) if use_band else torch.zeros(N, dtype=torch.bool, device=device)
    if target is not None:
        target_mask = seg == int(target)
    else:
        target_mask = torch.zeros(N, dtype=torch.bool, device=device)
    protected = band_mask | target_mask

    max_scale = scaling.max(dim=-1).values
    dead_raw = (alpha < float(epsilon)) | (max_scale > float(dead_extent_frac) * float(extent))
    dead = dead_raw & ~band_mask                              # band overrides dead
    contested = (~protected) & (~dead)

    k_dead = int(dead.sum().item())
    n_protected = int((protected & ~dead).sum().item())
    alive = ~dead
    n_rem = N - k_dead

    # ---- budget ---------------------------------------------------------- #
    if n_target is None:
        n_target = int(round(float(keep_frac) * N))
    n_target = max(0, min(int(n_target), N))

    reachable_floor = n_target >= n_protected
    reachable_ceiling = n_target <= n_rem

    # ---- (2) selection --------------------------------------------------- #
    if random_control:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.rand(N, generator=gen).to(device)
        score = noise
    else:
        score = score_gaussians(alpha, scaling, purity, strength=strength,
                                curve=curve, use_purity=use_purity)

    keep = alive.clone()
    if n_target >= n_rem:
        # above the ceiling: keep every survivor, delete no contested
        pass
    elif n_target <= n_protected:
        # below the floor: the protected set alone exceeds the budget
        keep = protected & alive
    else:
        k = n_rem - n_target
        delete = _select_lowest(contested, score, k, seg, allocation)
        keep = alive & ~delete

    k_contested = int((contested & ~keep).sum().item())
    n_after = int(keep.sum().item())
    delete_mask = ~keep

    return PrunePlan(
        keep=keep, delete=delete_mask, band=band_mask, dead=dead,
        protected=protected, contested=contested, score=score,
        n_before=N, n_after=n_after, k_dead=k_dead, k_contested=k_contested,
        n_protected=n_protected, n_target=n_target, target_id=target,
        reachable_floor=reachable_floor, reachable_ceiling=reachable_ceiling,
        allocation=allocation,
        meta={"use_band": use_band, "use_purity": use_purity,
              "strength": float(strength), "curve": float(curve),
              "band": band_default, "band_class": band_table,
              "epsilon": float(epsilon),
              "extent": float(extent), "random_control": bool(random_control)})


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    """Jaccard of two boolean masks (the §6 kept-set-movement diagnostic)."""
    a = a.reshape(-1).to(torch.bool)
    b = b.reshape(-1).to(torch.bool)
    union = int((a | b).sum().item())
    if union == 0:
        return 1.0
    return int((a & b).sum().item()) / union


__all__ = [
    "PrunePlan",
    "plan_pruning",
    "score_gaussians",
    "geometry_backbone",
    "semantic_confidence",
    "band_delta",
    "parse_band_class",
    "jaccard",
    "DEFAULT_BAND",
    "DEFAULT_EPSILON",
    "DEFAULT_STRENGTH",
    "DEFAULT_CURVE",
    "DEFAULT_STRIDE",
]
