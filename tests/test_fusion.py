"""Tests for the minimal 3D instance fusion (worldsplat_prune.prep).

Covers the pure association logic (union-find over shared 3D points, colour
gate, support filtering) and the COLMAP IO / projection / label-splatting helpers.
No torch / GPU / rasterizer required.

    pytest tests/test_fusion.py
    python tests/test_fusion.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worldsplat_prune.prep import scene_io as sio
from worldsplat_prune.prep import fuse_instances as fi


# --------------------------------------------------------------------------- #
# association                                                                   #
# --------------------------------------------------------------------------- #
def test_union_find_basic():
    uf = fi.UnionFind(5)
    uf.union(0, 1)
    uf.union(3, 4)
    uf.union(1, 3)
    roots = uf.roots()
    assert roots[0] == roots[1] == roots[3] == roots[4]
    assert roots[2] != roots[0]


def test_associate_shares_points():
    # obs 0 and 1 share points {10, 11}; obs 2 is disjoint
    obs = [np.array([1, 2, 10, 11]), np.array([10, 11, 12]), np.array([99, 100])]
    roots = fi.associate_observations(obs, min_shared_points=2)
    assert roots[0] == roots[1]
    assert roots[2] != roots[0]


def test_associate_min_shared_points():
    obs = [np.array([10]), np.array([10, 11])]  # share only 1 point
    assert fi.associate_observations(obs, min_shared_points=2)[0] != \
        fi.associate_observations(obs, min_shared_points=2)[1]
    # ... but they merge when the bar is 1
    r = fi.associate_observations(obs, min_shared_points=1)
    assert r[0] == r[1]


def test_associate_color_gate():
    obs = [np.array([10, 11, 12]), np.array([10, 11, 12])]
    colors = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])]  # orthogonal
    assert fi.associate_observations(obs, colors, min_shared_points=1,
                                     color_sim=0.9)[0] != \
        fi.associate_observations(obs, colors, min_shared_points=1,
                                  color_sim=0.9)[1]
    same = [np.array([1.0, 0.0, 0.0]), np.array([0.9, 0.0, 0.1])]
    r = fi.associate_observations(obs, same, min_shared_points=1, color_sim=0.5)
    assert r[0] == r[1]


def test_assign_global_ids_support_filter():
    # two clusters; cluster B (obs 2,3) is one frame -> dropped at min_views=2
    roots = [0, 0, 1, 1]
    obs_cam = [0, 1, 5, 5]
    gid = fi.assign_global_ids(roots, obs_cam, min_views=2)
    assert gid[0] == gid[1] == 1
    assert gid[2] == 0 and gid[3] == 0


def test_assign_global_ids_dense_and_stable():
    roots = [0, 0, 0, 1, 1]
    obs_cam = [0, 1, 2, 3, 4]
    gid = fi.assign_global_ids(roots, obs_cam, min_views=1)
    assert set(gid) == {1, 2}          # dense ids, no holes


# --------------------------------------------------------------------------- #
# overlap-based association (the fix)                                           #
# --------------------------------------------------------------------------- #
def test_overlap_assoc_merges_same_object_only():
    # A,B = same point set (same object across views); C = disjoint
    coords = np.array([[0., 0, 0], [0.1, 0, 0], [0.2, 0, 0],
                       [5., 0, 0], [5.1, 0, 0], [5.2, 0, 0]], dtype=float)
    obs = [np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([3, 4, 5])]
    roots = fi.associate_by_overlap(obs, coords, metric="jaccard", min_score=0.25)
    assert roots[0] == roots[1]
    assert roots[2] != roots[0]


def test_overlap_assoc_resists_hub_point_collapse():
    # three distinct masks that share only ONE hub point: must NOT all collapse
    coords = np.array([[i * 0.1, 0, 0] for i in range(12)] + [[0.0, 0, 0]], dtype=float)
    hub = 12
    obs = [np.array([0, 1, 2, 3, hub]),
           np.array([4, 5, 6, 7, hub]),
           np.array([8, 9, 10, 11, hub])]
    roots = fi.associate_by_overlap(obs, coords, metric="jaccard", min_score=0.25)
    assert roots[0] != roots[1] != roots[2] != roots[0]
    # the old shared-count rule *would* collapse them
    old = fi.associate_observations(obs, min_shared_points=1)
    assert old[0] == old[1] == old[2]


def test_overlap_metric_merges_subset_fragment():
    coords = np.array([[0., 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]], dtype=float)
    full = np.array([0, 1, 2, 3])
    frag = np.array([0, 1])                      # a sub-part of the same object
    # jaccard = 2/4 = 0.5; overlap = 2/2 = 1.0
    r_j = fi.associate_by_overlap([full, frag], coords, metric="jaccard", min_score=0.75)
    assert r_j[0] != r_j[1]
    r_o = fi.associate_by_overlap([full, frag], coords, metric="overlap", min_score=0.75)
    assert r_o[0] == r_o[1]


def test_overlap_assoc_color_gate():
    coords = np.array([[0., 0, 0], [0.1, 0, 0]], dtype=float)
    obs = [np.array([0, 1]), np.array([0, 1])]            # identical point sets
    colors = [np.array([1., 0, 0]), np.array([0., 0, 1])]  # orthogonal colours
    r = fi.associate_by_overlap(obs, coords, colors, metric="jaccard",
                                min_score=0.5, color_sim=0.9)
    assert r[0] != r[1]
    same = [np.array([1., 0, 0]), np.array([0.9, 0, 0.1])]
    r2 = fi.associate_by_overlap(obs, coords, same, metric="jaccard",
                                 min_score=0.5, color_sim=0.5)
    assert r2[0] == r2[1]


# --------------------------------------------------------------------------- #
# projection / splatting                                                        #
# --------------------------------------------------------------------------- #
def _cam(name="rgb_0.png", W=100, H=100, f=100.0, R=None, t=None):
    return sio.Camera(camera_id=1, image_id=1, name=name, width=W, height=H,
                      fx=f, fy=f, cx=W / 2.0, cy=H / 2.0,
                      R_wc=np.eye(3) if R is None else R,
                      t_wc=np.zeros(3) if t is None else t)


def test_project_points_identity_pose():
    cam = _cam()
    xyz = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0], [0.0, 0.0, -1.0]])
    u, v, z, valid = sio.project_points(xyz, cam)
    assert np.allclose([u[0], v[0]], [50, 50])
    assert np.allclose([u[1], v[1]], [100, 50])
    assert valid[2] is np.False_ or valid[2] == False  # behind camera


def test_points_at_pixels():
    cam = _cam(W=4, H=4, f=4.0, R=np.eye(3), t=np.zeros(3))
    lab = np.zeros((4, 4), dtype=np.int32)
    lab[2, 2] = 7
    # a point that projects exactly to pixel (2,2)
    xyz = np.array([[(2 - 2) / 4.0, (2 - 2) / 4.0, 1.0]])
    out = sio.points_at_pixels(lab, cam, xyz)
    assert out[0] == 7


def test_assign_within_radius_only_fills_nearby():
    m = np.zeros((5, 5), dtype=np.int32)
    m[2, 2] = 3
    out = sio.assign_within_radius(m, radius=1)
    assert out[2, 2] == 3
    assert out[1, 2] == 3 and out[3, 2] == 3 and out[2, 1] == 3 and out[2, 3] == 3
    assert out[0, 0] == 0                    # diagonal (dist ~1.41) stays void
    assert sio.assign_within_radius(m, 0)[2, 2] == 3


def test_id_map_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.png"
        ids = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int32)
        sio.write_id_map(p, ids)
        back = sio.read_id_map(p)
        assert np.array_equal(back, ids)


# --------------------------------------------------------------------------- #
# COLMAP text loader + splatting                                                #
# --------------------------------------------------------------------------- #
def _write_text_model(root: Path):
    sp = root / "sparse" / "0"
    sp.mkdir(parents=True)
    (sp / "cameras.txt").write_text(
        "# Camera list\n"
        "1 SIMPLE_PINHOLE 4 4 4.0 2.0 2.0\n")
    # two images: identity pose (identity quaternion = [1,0,0,0])
    (sp / "images.txt").write_text(
        "# Image list\n"
        "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 rgb_0.png\n"
        "0.0 0.0 -1\n"
        "2 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 rgb_1.png\n"
        "0.0 0.0 -1\n")
    (sp / "points3D.txt").write_text(
        "# 3D points\n"
        "1 0.0 0.0 1.0 255 0 0 0.1\n"
        "2 0.0 0.0 1.0 0 255 0 0.1\n")
    return sp


def test_load_colmap_scene_text():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        sp = _write_text_model(root)
        scene = sio.load_colmap_scene(sp)
        assert len(scene.cameras) == 2
        assert scene.xyz.shape == (2, 3)
        c = scene.by_stem["rgb_0"]
        assert c.fx == 4.0 and c.width == 4


def test_splat_point_labels_to_views():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        sp = _write_text_model(root)
        scene = sio.load_colmap_scene(sp)
        labels = np.array([1, 2], dtype=np.int32)
        out = Path(d) / "out"
        n = sio.splat_point_labels_to_views(scene, labels, out, radius=0)
        assert n == 2
        m = sio.read_id_map(out / "rgb_0.png")
        # both points project to (2,2); the lower label wins the pixel
        assert m[2, 2] in (1, 2)
        assert (m > 0).sum() == 1


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
