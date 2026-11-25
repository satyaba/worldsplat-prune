"""Post-hoc label lifting onto 3D Gaussians — the class-vote pass (§3).

The prompt-driven front-end (§2.2) produces a per-view canonical label map
``L_v`` (a composite ``(class, instance)`` id, ``0 == void``). This module
re-projects those 2D maps onto the **Gaussian centres** of an already-trained
model and reduces them into a per-Gaussian semantic signature, the *vote
bundle* consumed by the pruning operator (§4–5).

This is the post-hoc replacement for the training-time online vote
(``GaussianModel.add_semantic_stats``), whose buffers were never persisted to
the PLY (``docs/worldsplat_prune_spec.md`` §3): on a loaded model they are
absent, so the vote must be recomputed in one deterministic pass.

Mechanism (mirrors the trainer's convention, [D6]/[D8]):

* membership — ``render`` ([D6] default): a Gaussian is a member of a view when
  the CUDA rasteriser gives it a footprint (``radii > 0``); ``project`` (cheap
  preview, CPU-ok): its centre lands inside the image rectangle;
* pixel lookup — the centre is projected through ``cam.full_proj_transform``
  (the same transform the rasteriser uses) and ``L_v``/``dt`` are sampled at the
  nearest pixel, exactly as ``add_semantic_stats`` did after the render;
* **occlusion** — a centre-projection test is depth-blind, so geometry *behind*
  the visible surface inherits the surface's label, painting a ghost of the
  object onto the background (spec Risk 1). When ``occlusion`` is on (default),
  a sample only counts if the Gaussian is not behind the rendered surface:
  its inverse depth ``1/z`` must be within ``depth_tol`` of the pixel's
  reference inverse depth — the rasteriser's α-weighted ``1/z`` under ``render``,
  or a per-pixel nearest-``z`` z-buffer of the centres under ``project``;
* vote — ``first`` ([D8] default): the first non-void id a Gaussian sees is
  frozen (view order is deterministic, sorted by ``image_name``); ``majority``:
  argmax over the per-Gaussian id histogram, ties → void.

Output bundle (§3.4) — detached tensors, *not* model state:

============== ================= ===============================================
field          shape             meaning
============== ================= ===============================================
``seg``        ``long [N]``      assigned id (0 = void/unassigned)
``purity``     ``float [N]``     ``seg_cnt / tot`` (first) or ``hist_max / tot``
``void_frac``  ``float [N]``     ``void / (labeled + void)`` ∈ [0, 1]
``boundary_dist`` ``float [N]``  ``mean(dt) / dt_norm`` ∈ [0, 1] (0 = on edge)
``n_views``    ``long [N]``      #views that sampled the Gaussian (occlusion-aware)
``seg_hist``   ``float [N, K]``  id distribution (optional, ``dense_hist``)
``hist_labels`` ``long [K]``     the id behind each ``seg_hist`` column
``meta``       ``dict``          visibility, vote, occlusion, view order, ...
============== ================= ===============================================

Finalization: ``min_views`` ([D7] default 3) → below it ``seg = 0`` and
``purity = 0``; never-seen (``tot == 0``) or seen-only-in-void → ``seg = 0``,
``purity = 0``, ``boundary_dist = 1.0`` (conservative "interior", so the band
does not protect it).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

import torch

# Hyperparameters fixed by the design doc ([D7], [D11]).
DEFAULT_DT_NORM = 100.0      # px scaling for `d_i`: dt_norm px -> d_i = 1
DEFAULT_MIN_VIEWS = 3        # [D7]
DEFAULT_DEPTH_TOL = 0.02     # relative tolerance on inverse depth for occlusion
_OFFSET = 1 << 32            # packing base for (gaussian, label) keys


# --------------------------------------------------------------------------- #
# bundle                                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class VoteBundle:
    """Per-Gaussian semantic signature (§3.4). All tensors are detached."""

    seg: torch.Tensor                     # long [N]
    purity: torch.Tensor                  # float [N]
    void_frac: torch.Tensor               # float [N]
    boundary_dist: torch.Tensor           # float [N]
    n_views: torch.Tensor                 # long [N]
    seg_hist: Optional[torch.Tensor] = None        # float [N, K] or None
    hist_labels: Optional[torch.Tensor] = None     # long [K] or None
    meta: dict = field(default_factory=dict)

    @property
    def n_gaussians(self) -> int:
        return int(self.seg.shape[0])

    def summary(self) -> str:
        """One-line diagnostic (the 'is this scene usable at all' readout)."""
        m = self.meta
        occl = "occl" if m.get("occlusion") else "no-occl"
        return (f"vote[{m.get('visibility')}/{m.get('vote')}/{occl}] "
                f"N={self.n_gaussians} visible={m.get('n_visible')} "
                f"labeled={m.get('n_labeled')} "
                f"mean_purity={m.get('mean_purity'):.3f} "
                f"mean_void_frac={m.get('mean_void_frac'):.3f}")


# --------------------------------------------------------------------------- #
# camera order + projection                                                     #
# --------------------------------------------------------------------------- #
def order_cameras(cameras: Sequence[Any], order: str = "sorted") -> List[Any]:
    """Deterministic view order for the vote ([D8] — 'first' depends on order).

    ``sorted`` sorts by ``image_name``; ``given`` keeps the caller's order.
    """
    if order == "sorted":
        return sorted(cameras, key=lambda c: c.image_name)
    if order == "given":
        return list(cameras)
    raise ValueError(f"unknown camera order {order!r} (sorted|given)")


def project_centers(xyz: torch.Tensor, cam: Any):
    """Project Gaussian centres into ``cam`` pixels.

    Uses ``cam.full_proj_transform`` (4x4, world -> clip) exactly as the
    rasteriser does, then the perspective divide and the NDC -> pixel map.
    Returns ``(u, v, depth, in_bounds)``; ``u``/``v`` are float pixels,
    ``depth`` is the clip-space ``w`` (positive in front of the camera, equal to
    the camera-space ``z`` the rasteriser stores), and ``in_bounds`` is
    ``depth > 0`` **and** the centre inside the image.
    """
    M = cam.full_proj_transform.to(device=xyz.device, dtype=xyz.dtype)
    ones = xyz.new_ones((xyz.shape[0], 1))
    h = torch.cat([xyz, ones], dim=1)              # [N, 4]
    # row-vector convention matching the trainer: `full_proj_transform` is
    # already (P @ W2V)^T, so points are applied as `h @ M` (equivalently
    # `M.T @ h_col`), exactly as `geom_transform_points` / the rasteriser do.
    p = h @ M                                       # [N, 4]
    w = p[:, 3]
    in_front = w > 1e-6
    safe = torch.where(in_front, w, torch.ones_like(w))
    ndc = p[:, :3] / safe[:, None]
    W = float(cam.image_width)
    H = float(cam.image_height)
    u = (ndc[:, 0] * 0.5 + 0.5) * W
    v = (ndc[:, 1] * 0.5 + 0.5) * H
    in_bounds = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, w, in_bounds


def _pixel_index(u, v, W, H):
    """Clamped nearest-pixel flat index and a validity mask for ``(u, v)``."""
    ui = torch.round(u).long().clamp_(0, W - 1)
    vi = torch.round(v).long().clamp_(0, H - 1)
    return vi * W + ui


def _gather_flat(flat: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor,
                 dtype: torch.dtype) -> torch.Tensor:
    """``flat[idx]`` where ``mask``; 0 elsewhere. Returns ``[N]`` of ``dtype``."""
    out = torch.zeros(idx.shape, dtype=dtype, device=idx.device)
    if mask.any():
        out[mask] = flat[idx[mask]].to(dtype)
    return out


def pixel_min_invdepth(u, v, w, mask, H: int, W: int) -> torch.Tensor:
    """Per-pixel **maximum inverse depth** (nearest surface) from centres.

    A cheap z-buffer over the projected Gaussian centres: for every pixel, the
    largest ``1/z`` among the masked Gaussians. Used as the occlusion reference
    under ``project`` visibility (or whenever a rendered depth map is
    unavailable). Returns a float ``[H, W]`` map (0 where no member projects).
    """
    Npix = H * W
    idx = _pixel_index(u, v, W, H)
    minz = torch.full((Npix,), float("inf"), dtype=torch.float32, device=u.device)
    if mask.any():
        minz.scatter_reduce_(0, idx[mask], w[mask].to(torch.float32), reduce="amin")
    invd = torch.where(torch.isfinite(minz),
                       1.0 / minz.clamp_min(1e-12),
                       torch.zeros_like(minz))
    return invd.reshape(H, W)


# --------------------------------------------------------------------------- #
# visibility                                                                    #
# --------------------------------------------------------------------------- #
class _RenderPipe:
    """Minimal stand-in for ``arguments.PipelineParams`` for ``render()``."""

    debug = False
    antialiasing = False
    compute_cov3D_python = False
    convert_SHs_python = False


def make_render_fn(gaussians, pipe=None, bg=None) -> Callable[[Any], tuple]:
    """Build the ``render``-visibility provider: ``cam -> (radii, invdepth)``.

    ``invdepth`` is the rasteriser's α-weighted inverse-depth map (``[H, W]``),
    used for the occlusion test. Imported lazily so this module stays importable
    without the CUDA rasteriser (the ``project`` path and the pure helpers are
    CPU-ok).
    """
    from gaussian_renderer import render as _render

    if pipe is None:
        pipe = _RenderPipe()
    if bg is None:
        bg = torch.zeros(3, dtype=torch.float32, device=gaussians.get_xyz.device)

    def fn(cam: Any):
        pkg = _render(cam, gaussians, pipe, bg)
        return pkg["radii"], pkg.get("depth")

    return fn


# --------------------------------------------------------------------------- #
# reduction                                                                     #
# --------------------------------------------------------------------------- #
def _reduce_majority(vote_idx: torch.Tensor, vote_lab: torch.Tensor, n: int):
    """Argmax id per Gaussian (ties -> void). Returns ``(seg, topcount)``."""
    seg = torch.zeros(n, dtype=torch.long, device=vote_idx.device)
    topcount = torch.zeros(n, dtype=torch.long, device=vote_idx.device)
    if vote_idx.numel() == 0:
        return seg, topcount
    keys = vote_idx.to(torch.int64) * _OFFSET + vote_lab.to(torch.int64)
    uniq, counts = torch.unique(keys, return_counts=True)
    gi = uniq // _OFFSET
    gl = uniq % _OFFSET
    # per-gaussian maximum count, then the pair(s) that reach it
    maxc = torch.zeros(n, dtype=torch.long, device=gi.device)
    maxc.scatter_reduce_(0, gi, counts, reduce="amax")
    topcount = maxc
    is_max = counts == maxc[gi]
    n_max = torch.zeros(n, dtype=torch.long, device=gi.device)
    n_max.index_add_(0, gi, is_max.to(torch.long))
    # a gaussian is decided only when its top count is unique (no tie)
    winner = is_max & (n_max[gi] == 1)
    seg[gi[winner]] = gl[winner]
    return seg, topcount


def _build_hist(vote_idx, vote_lab, n) -> tuple:
    """Dense ``[N, K]`` id histogram from unique (gaussian, id) pairs."""
    device = vote_idx.device
    labels = torch.unique(vote_lab)
    hist = torch.zeros((n, labels.numel()), dtype=torch.float32, device=device)
    if labels.numel() == 0:
        return hist, labels
    keys = vote_idx.to(torch.int64) * _OFFSET + vote_lab.to(torch.int64)
    uniq, counts = torch.unique(keys, return_counts=True)
    gi = uniq // _OFFSET
    gl = uniq % _OFFSET
    col = torch.searchsorted(labels, gl)
    hist[gi, col] = counts.to(torch.float32)
    return hist, labels


# --------------------------------------------------------------------------- #
# main pass                                                                     #
# --------------------------------------------------------------------------- #
def vote_classes_over_views(gaussians, cameras, store, *,
                            visibility: str = "render",
                            vote: str = "first",
                            dt_norm: float = DEFAULT_DT_NORM,
                            min_views: int = DEFAULT_MIN_VIEWS,
                            occlusion: bool = True,
                            depth_tol: float = DEFAULT_DEPTH_TOL,
                            mask_margin: float = 0.0,
                            dense_hist: bool = False,
                            chunk: int = 200_000,
                            camera_order: str = "sorted",
                            render_fn: Optional[Callable[[Any], Any]] = None,
                            pipe=None, bg=None) -> VoteBundle:
    """Recompute the per-Gaussian semantic signature from per-view maps (§3).

    Parameters
    ----------
    gaussians
        A loaded ``GaussianModel`` (only ``get_xyz`` is read).
    cameras
        The view list the vote may see — **train-only** ([D10]).
    store
        A :class:`~worldsplat_prune.semantic_store.SemanticStore`; ``get``
        yields ``{"label", "dt", ...}`` on device.
    visibility
        ``"render"`` ([D6] default) or ``"project"`` (cheap/preview).
    vote
        ``"first"`` ([D8] default) or ``"majority"``.
    occlusion
        Drop samples whose Gaussian is behind the visible surface (default on;
        fixes the background-ghost failure — spec Risk 1).
    depth_tol
        Relative tolerance on inverse depth: a Gaussian is occluded when its
        ``1/z`` is below ``(1 - depth_tol) x`` the pixel reference.
    mask_margin
        Drop samples whose pixel is within this many px of a label boundary
        (``dt < mask_margin``). Front-end masks are a pixel or two generous at
        silhouettes, so rim samples are where background geometry wrongly
        inherits the object id; ``0`` disables the filter.
    dt_norm, min_views
        See §3.4 ([D11], [D7]).
    dense_hist
        Also materialise the ``[N, K]`` id histogram (and its labels).
    chunk
        Gaussian-chunk size for the sample/accumulate step (memory bound).
    camera_order
        ``"sorted"`` (deterministic, default) or ``"given"``.
    render_fn, pipe, bg
        For ``visibility="render"``: an injected ``cam -> (radii, invdepth)``
        provider (a bare ``radii`` tensor is also accepted), or a
        ``PipelineParams`` and background colour to build one.
    """
    if visibility not in ("render", "project"):
        raise ValueError(f"unknown visibility {visibility!r} (render|project)")
    if vote not in ("first", "majority"):
        raise ValueError(f"unknown vote {vote!r} (first|majority)")

    xyz_all = gaussians.get_xyz.detach()
    N = int(xyz_all.shape[0])
    device = xyz_all.device

    cams = order_cameras(cameras, camera_order)

    # O(N) streaming accumulators.
    tot = torch.zeros(N, dtype=torch.long, device=device)
    void = torch.zeros(N, dtype=torch.long, device=device)
    dt_sum = torch.zeros(N, dtype=torch.float32, device=device)
    top1 = torch.zeros(N, dtype=torch.long, device=device)
    top1_cnt = torch.zeros(N, dtype=torch.long, device=device)

    if visibility == "render" and render_fn is None:
        render_fn = make_render_fn(gaussians, pipe=pipe, bg=bg)

    collect = (vote == "majority") or dense_hist
    vote_idx: List[torch.Tensor] = []
    vote_lab: List[torch.Tensor] = []

    n_processed = 0
    n_occluded = 0
    n_masked = 0
    for cam in cams:
        sem = store.get(cam.image_name, int(cam.image_height), int(cam.image_width))
        if sem is None:
            continue
        n_processed += 1
        label_map = sem["label"]
        dt_map = sem["dt"]
        H, W = int(label_map.shape[0]), int(label_map.shape[1])

        # Project every centre once per view (also the z-buffer for `project`).
        u_all, v_all, w_all, inb_all = project_centers(xyz_all, cam)

        if visibility == "render":
            out = render_fn(cam)
            if isinstance(out, tuple):
                radii_all, invdepth = out
            else:
                radii_all, invdepth = out, None
            counted_all = radii_all > 0
        else:
            invdepth = None
            counted_all = inb_all

        # A sample has a pixel only where the centre is in frame.
        has_pixel_all = counted_all & inb_all

        ref_map = None
        if occlusion:
            if invdepth is not None:
                ref_map = invdepth
                if ref_map.dim() == 3:               # rasteriser returns [1,H,W]
                    ref_map = ref_map[0]
                ref_map = ref_map.to(torch.float32)
            else:
                ref_map = pixel_min_invdepth(u_all, v_all, w_all,
                                             has_pixel_all, H, W)

        flat_label = label_map.reshape(-1)
        flat_dt = dt_map.reshape(-1)
        flat_ref = ref_map.reshape(-1) if ref_map is not None else None

        for start in range(0, N, chunk):
            stop = min(start + chunk, N)
            u = u_all[start:stop]
            counted = counted_all[start:stop]
            if not counted.any():
                continue
            idx = _pixel_index(u, v_all[start:stop], W, H)
            sample_ok = counted & inb_all[start:stop]

            lab = _gather_flat(flat_label, idx, sample_ok, torch.long)
            dtv = _gather_flat(flat_dt, idx, sample_ok, torch.float32)

            if mask_margin > 0.0:
                # silhouettes are where front-end masks are generous; drop rim
                # samples rather than let background inherit the object id.
                rim = sample_ok & (dtv < mask_margin)
                if rim.any():
                    n_masked += int(rim.sum().item())
                    counted = counted & ~rim
                    sample_ok = sample_ok & ~rim
                if not counted.any():
                    continue

            if flat_ref is not None:
                ref = _gather_flat(flat_ref, idx, sample_ok, torch.float32)
                gi = 1.0 / w_all[start:stop].clamp_min(1e-12)
                # behind the surface (smaller 1/z) -> occluded, dropped entirely
                occluded = sample_ok & (ref > 0) & (gi < ref * (1.0 - depth_tol))
                if occluded.any():
                    n_occluded += int(occluded.sum().item())
                    counted = counted & ~occluded
                    sample_ok = sample_ok & ~occluded
                if not counted.any():
                    continue

            tot[start:stop] += counted.to(torch.long)
            lab_sel = counted & (lab > 0)
            void_sel = counted & (lab == 0)
            void[start:stop] += void_sel.to(torch.long)
            dt_sum[start:stop] += torch.where(lab_sel, dtv, torch.zeros_like(dtv))

            if vote == "first":
                t1 = top1[start:stop]
                new = lab_sel & (t1 == 0)
                t1[new] = lab[new]
                match = lab_sel & (lab == t1)
                top1_cnt[start:stop] += match.to(torch.long)

            if collect and lab_sel.any():
                nz = torch.nonzero(lab_sel, as_tuple=False).squeeze(1)
                vote_idx.append((nz + start).to(torch.long))
                vote_lab.append(lab[nz].to(torch.long))

    n_lab = tot - void
    seen = tot > 0
    keep = tot >= int(min_views)

    # ---- vote reduction -------------------------------------------------- #
    if vote == "majority":
        if vote_idx:
            vi = torch.cat(vote_idx)
            vl = torch.cat(vote_lab)
            seg_raw, topcount = _reduce_majority(vi, vl, N)
        else:
            seg_raw = torch.zeros(N, dtype=torch.long, device=device)
            topcount = torch.zeros(N, dtype=torch.long, device=device)
        numer = topcount.to(torch.float32)
    else:
        seg_raw = top1
        numer = top1_cnt.to(torch.float32)

    seg = torch.where(keep, seg_raw, torch.zeros_like(seg_raw))
    denom = tot.to(torch.float32).clamp_min(1.0)
    purity = torch.where(seg > 0, numer / denom, torch.zeros_like(denom))
    void_frac = torch.where(seen, void.to(torch.float32) / denom,
                            torch.zeros_like(denom))
    d = torch.where(n_lab > 0,
                    dt_sum / (n_lab.to(torch.float32) * float(dt_norm)),
                    torch.ones(N, dtype=torch.float32, device=device))
    d = d.clamp_(0.0, 1.0)

    seg_hist = hist_labels = None
    if dense_hist:
        if vote_idx:
            seg_hist, hist_labels = _build_hist(torch.cat(vote_idx),
                                                torch.cat(vote_lab), N)
        else:
            seg_hist = torch.zeros((N, 0), dtype=torch.float32, device=device)
            hist_labels = torch.zeros(0, dtype=torch.long, device=device)

    n_labeled = int((seg > 0).sum().item())
    n_seen = int(seen.sum().item())
    meta = {
        "visibility": visibility,
        "vote": vote,
        "occlusion": bool(occlusion),
        "depth_tol": float(depth_tol),
        "mask_margin": float(mask_margin),
        "camera_order": camera_order,
        "view_order": [c.image_name for c in cams],
        "n_views_processed": n_processed,
        "n_occluded": n_occluded,
        "n_masked": n_masked,
        "dt_norm": float(dt_norm),
        "min_views": int(min_views),
        "n_gaussians": N,
        "n_visible": n_seen,
        "n_labeled": n_labeled,
        "mean_purity": float(purity[seg > 0].mean()) if n_labeled else 0.0,
        "mean_void_frac": float(void_frac[seen].mean()) if n_seen else 0.0,
    }

    return VoteBundle(seg=seg, purity=purity, void_frac=void_frac,
                      boundary_dist=d, n_views=tot,
                      seg_hist=seg_hist, hist_labels=hist_labels, meta=meta)


__all__ = [
    "VoteBundle",
    "vote_classes_over_views",
    "order_cameras",
    "project_centers",
    "pixel_min_invdepth",
    "make_render_fn",
    "DEFAULT_DT_NORM",
    "DEFAULT_MIN_VIEWS",
    "DEFAULT_DEPTH_TOL",
]
