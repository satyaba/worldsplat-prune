"""Tests for the post-hoc label-lifting vote pass (worldsplat_prune.vote, §3).

CPU-only: the ``project`` visibility path is pure torch, and the ``render`` path
is exercised through an injected membership function, so no CUDA rasteriser is
needed. The label source is the real :class:`SemanticStore` over a tiny in-memory
adapter, so the label/dt pipeline is the production one.

    pytest tests/test_vote.py
    python tests/test_vote.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worldsplat_prune.datasets.base import LabelAdapter
from worldsplat_prune.semantic_store import SemanticStore
from worldsplat_prune import vote as V


# --------------------------------------------------------------------------- #
# fixtures                                                                      #
# --------------------------------------------------------------------------- #
class _MapAdapter(LabelAdapter):
    """In-memory label adapter: stem -> int32 [H, W] map."""

    name = "fake"

    def __init__(self, maps):
        super().__init__(".", {})
        self._maps = {k: np.asarray(v, dtype=np.int32) for k, v in maps.items()}

    def resolve(self, image_name):
        return Path(image_name).stem

    def load(self, image_name):
        return self._maps.get(Path(image_name).stem)


class _Cam:
    """Minimal camera with an identity clip transform (world -> pixel)."""

    def __init__(self, name, w, h):
        self.image_name = name
        self.image_width = w
        self.image_height = h
        self.full_proj_transform = torch.eye(4)


class _Gaussians:
    def __init__(self, xyz):
        self._xyz = torch.as_tensor(xyz, dtype=torch.float32)

    @property
    def get_xyz(self):
        return self._xyz


# g0 -> pixel (row 2, col 2); g1 -> (row 2, col 3); g2 -> off-frame
_XYZ = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [10.0, 0.0, 0.0]]


def _maps():
    a = np.zeros((4, 4), dtype=np.int32); a[2, 2] = 5
    b = np.zeros((4, 4), dtype=np.int32); b[2, 2] = 9; b[2, 3] = 7
    c = np.zeros((4, 4), dtype=np.int32); c[2, 2] = 5; c[2, 3] = 7
    return {"A": a, "B": b, "C": c}


def _setup(views=("A", "B", "C")):
    adapter = _MapAdapter(_maps())
    store = SemanticStore(adapter, device="cpu")
    cams = [_Cam(v, 4, 4) for v in views]
    return _Gaussians(_XYZ), cams, store


# --------------------------------------------------------------------------- #
# projection + ordering (pure helpers)                                          #
# --------------------------------------------------------------------------- #
def test_project_centers_identity_and_bounds():
    g, cams, _ = _setup()
    u, v, _w, inb = V.project_centers(g.get_xyz, _Cam("A", 4, 4))
    assert inb.tolist() == [True, True, False]      # g2 is off-frame
    assert (u[0].item(), v[0].item()) == (2.0, 2.0)
    assert (u[1].item(), v[1].item()) == (3.0, 2.0)


def _real_cam(name, w, h, fov=1.0):
    """A camera whose ``full_proj_transform`` is built exactly as the trainer's."""
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix

    w2v = torch.tensor(getWorld2View2(np.eye(3), np.zeros(3))).transpose(0, 1)
    proj = getProjectionMatrix(0.01, 100.0, fov, fov).transpose(0, 1)
    cam = _Cam(name, w, h)
    cam.full_proj_transform = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    return cam


def test_projection_matches_trainer_convention():
    # Camera at the origin looking down +z; the NDC->pixel map must match the
    # trainer's: an on-axis point lands at the image centre, a point behind the
    # camera is out, and a point on the frustum edge is just out of frame.
    import math

    cam = _real_cam("A", 100, 100, fov=1.0)
    tan_half = math.tan(0.5)
    xyz = torch.tensor([[0.0, 0.0, 1.0],          # on axis
                        [0.0, 0.0, -1.0],         # behind
                        [tan_half, 0.0, 1.0]])    # exactly on the right edge
    u, v, _w, inb = V.project_centers(xyz, cam)
    assert abs(u[0].item() - 50.0) < 1e-3 and abs(v[0].item() - 50.0) < 1e-3
    assert inb[0].item()
    assert not inb[1].item()
    assert not inb[2].item()


def test_order_cameras():
    cams = [_Cam("C", 4, 4), _Cam("A", 4, 4), _Cam("B", 4, 4)]
    assert [c.image_name for c in V.order_cameras(cams, "sorted")] == ["A", "B", "C"]
    assert [c.image_name for c in V.order_cameras(cams, "given")] == ["C", "A", "B"]
    try:
        V.order_cameras(cams, "nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown order")


# --------------------------------------------------------------------------- #
# vote: first vs majority                                                       #
# --------------------------------------------------------------------------- #
def test_vote_first_matches_spec():
    g, cams, store = _setup()
    b = V.vote_classes_over_views(g, cams, store, visibility="project",
                                  vote="first", min_views=1)
    assert b.seg.tolist() == [5, 7, 0]              # first non-void wins; g2 unseen
    assert b.n_views.tolist() == [3, 3, 0]
    # g0: 5 in 2/3 views -> purity 2/3; g1: 7 in 2/3 (void in A)
    assert abs(b.purity[0].item() - 2 / 3) < 1e-6
    assert abs(b.purity[1].item() - 2 / 3) < 1e-6
    assert b.purity[2].item() == 0.0
    assert b.void_frac[0].item() == 0.0
    assert abs(b.void_frac[1].item() - 1 / 3) < 1e-6
    # isolated pixels sit on a boundary -> d = 0; unseen -> d = 1 (interior)
    assert b.boundary_dist[0].item() == 0.0
    assert b.boundary_dist[2].item() == 1.0
    assert b.meta["n_visible"] == 2 and b.meta["n_labeled"] == 2


def test_vote_majority():
    g, cams, store = _setup()
    b = V.vote_classes_over_views(g, cams, store, visibility="project",
                                  vote="majority", min_views=1)
    assert b.seg.tolist() == [5, 7, 0]              # 5:2/9:1 and 7:2/void:1



def test_majority_tie_is_void():
    # two views disagree -> top count not unique -> void
    a = np.zeros((4, 4), dtype=np.int32); a[2, 2] = 5
    b_ = np.zeros((4, 4), dtype=np.int32); b_[2, 2] = 9
    adapter = _MapAdapter({"A": a, "B": b_})
    store = SemanticStore(adapter, device="cpu")
    g = _Gaussians(_XYZ)
    res = V.vote_classes_over_views(g, [_Cam("A", 4, 4), _Cam("B", 4, 4)], store,
                                    visibility="project", vote="majority",
                                    min_views=1)
    assert res.seg.tolist() == [0, 0, 0]


def test_min_views_finalizes_to_void():
    g, cams, store = _setup()
    b = V.vote_classes_over_views(g, cams, store, visibility="project",
                                  vote="first", min_views=4)   # seen only 3x
    assert b.seg.tolist() == [0, 0, 0]
    assert b.purity.tolist() == [0.0, 0.0, 0.0]
    assert b.n_views.tolist() == [3, 3, 0]           # support is still recorded


# --------------------------------------------------------------------------- #
# vote: render visibility (injected membership)                                 #
# --------------------------------------------------------------------------- #
def test_render_visibility_counts_offframe_as_void():
    g, cams, store = _setup()
    N = 3

    def render_fn(_cam):
        return torch.ones(N, dtype=torch.int32)      # every Gaussian visible

    b = V.vote_classes_over_views(g, cams, store, visibility="render",
                                  vote="first", min_views=1, render_fn=render_fn)
    assert b.seg.tolist() == [5, 7, 0]
    # g2 is visible (radii>0) but its centre is off-frame -> void every view
    assert b.n_views.tolist() == [3, 3, 3]
    assert b.void_frac[2].item() == 1.0
    assert b.boundary_dist[2].item() == 1.0


def test_render_visibility_excludes_zero_radius():
    g, cams, store = _setup()

    def render_fn(_cam):
        return torch.tensor([1, 0, 0], dtype=torch.int32)   # only g0 visible

    b = V.vote_classes_over_views(g, cams, store, visibility="render",
                                  vote="first", min_views=1, render_fn=render_fn)
    assert b.n_views.tolist() == [3, 0, 0]
    assert b.seg.tolist() == [5, 0, 0]


# --------------------------------------------------------------------------- #
# occlusion (depth-aware lifting)                                               #
# --------------------------------------------------------------------------- #
def test_occlusion_rejects_gaussians_behind_surface():
    # three Gaussians on the optical axis at z = 1 (front), 2, 5; the label sits
    # on their shared pixel. Without occlusion all inherit the label; with
    # occlusion only the front one keeps it (the others are behind the surface).
    cam = _real_cam("A", 100, 100)
    m = np.zeros((100, 100), dtype=np.int32)
    m[50, 50] = 7
    store = SemanticStore(_MapAdapter({"A": m}), device="cpu")
    g = _Gaussians([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 5.0]])

    no_occl = V.vote_classes_over_views(g, [cam], store, visibility="project",
                                        vote="first", min_views=1, occlusion=False)
    assert no_occl.seg.tolist() == [7, 7, 7]          # the background ghost

    occl = V.vote_classes_over_views(g, [cam], store, visibility="project",
                                     vote="first", min_views=1, occlusion=True)
    assert occl.seg.tolist() == [7, 0, 0]
    assert occl.meta["n_occluded"] == 2


def test_pixel_min_invdepth():
    u = torch.tensor([50.0, 50.0, 60.0])
    v = torch.tensor([50.0, 50.0, 60.0])
    w = torch.tensor([5.0, 1.0, 2.0])                 # nearest at (50,50) is z=1
    mask = torch.ones(3, dtype=torch.bool)
    ref = V.pixel_min_invdepth(u, v, w, mask, H=100, W=100)
    assert abs(ref[50, 50].item() - 1.0) < 1e-6
    assert abs(ref[60, 60].item() - 0.5) < 1e-6


def test_occlusion_via_rendered_depth():
    cam = _real_cam("A", 100, 100)
    m = np.zeros((100, 100), dtype=np.int32)
    m[50, 50] = 7
    store = SemanticStore(_MapAdapter({"A": m}), device="cpu")
    g = _Gaussians([[0.0, 0.0, 1.0], [0.0, 0.0, 5.0]])

    depth = torch.zeros(1, 100, 100)                  # alpha-weighted inverse depth
    depth[0, 50, 50] = 1.0                            # front surface at z = 1

    def render_fn(_cam):
        return torch.ones(2, dtype=torch.int32), depth

    res = V.vote_classes_over_views(g, [cam], store, visibility="render",
                                    vote="first", min_views=1, render_fn=render_fn)
    assert res.seg.tolist() == [7, 0]                 # the z=5 Gaussian is occluded
    assert res.n_views.tolist() == [1, 0]


# --------------------------------------------------------------------------- #
# mask-rim filter                                                               #
# --------------------------------------------------------------------------- #
def test_mask_margin_drops_rim_samples():
    a = np.zeros((8, 8), dtype=np.int32)
    a[2:6, 2:6] = 5                                   # object block; edge at row/col 2
    store = SemanticStore(_MapAdapter({"A": a}), device="cpu")
    g = _Gaussians([[-0.5, -0.5, 0.0],                # -> (row 2, col 2): boundary
                    [-0.25, -0.25, 0.0]])             # -> (row 3, col 3): interior
    cam = _Cam("A", 8, 8)

    off = V.vote_classes_over_views(g, [cam], store, visibility="project",
                                    vote="first", min_views=1, mask_margin=0.0)
    assert off.seg.tolist() == [5, 5]

    on = V.vote_classes_over_views(g, [cam], store, visibility="project",
                                   vote="first", min_views=1, mask_margin=0.5)
    assert on.seg.tolist() == [0, 5]                  # rim sample dropped
    assert on.meta["n_masked"] == 1


# --------------------------------------------------------------------------- #
# boundary distance + histogram                                                 #
# --------------------------------------------------------------------------- #
def test_boundary_dist_interior_block_and_normalizer():
    # 4x6 frame: left half id 5, right half id 9 -> a vertical boundary at col
    # 2/3. A centre at col 0 sits a few px from that boundary; d = dt / dt_norm.
    from worldsplat_prune import semantic_utils as su

    a = np.zeros((4, 6), dtype=np.int32)
    a[:, :3] = 5
    a[:, 3:] = 9
    dt_true = float(su.distance_transform(a)[0][2, 0])
    adapter = _MapAdapter({"A": a})
    store = SemanticStore(adapter, device="cpu")
    g = _Gaussians([[-1.0, 0.0, 0.0]])               # -> pixel (row 2, col 0)
    res = V.vote_classes_over_views(g, [_Cam("A", 6, 4)], store,
                                    visibility="project", vote="first",
                                    min_views=1, dt_norm=100.0)
    assert abs(res.boundary_dist[0].item() - dt_true / 100.0) < 1e-6
    # d saturates at 1: a normalizer smaller than the raw distance clips
    res1 = V.vote_classes_over_views(g, [_Cam("A", 6, 4)], store,
                                     visibility="project", vote="first",
                                     min_views=1, dt_norm=1.0)
    assert res1.boundary_dist[0].item() == 1.0


def test_dense_hist_majority():
    g, cams, store = _setup()
    b = V.vote_classes_over_views(g, cams, store, visibility="project",
                                  vote="majority", min_views=1, dense_hist=True)
    assert b.seg_hist is not None and b.hist_labels is not None
    labels = b.hist_labels.tolist()
    assert labels == [5, 7, 9]                        # sorted unique ids
    row0 = b.seg_hist[0]
    assert row0[labels.index(5)].item() == 2 and row0[labels.index(9)].item() == 1
    row1 = b.seg_hist[1]
    assert row1[labels.index(7)].item() == 2


def test_never_seen_and_empty_cameras():
    g, cams, store = _setup(views=())
    b = V.vote_classes_over_views(g, [], store, visibility="project", vote="first")
    assert b.seg.tolist() == [0, 0, 0]
    assert b.boundary_dist.tolist() == [1.0, 1.0, 1.0]
    assert b.meta["n_views_processed"] == 0


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
