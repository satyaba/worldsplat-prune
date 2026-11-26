"""Tests for the reference-geometry label adapter + driver (C2, §2.3, [D19]).

CPU-only: the cores are numpy/cv2; no torch or trained model is imported.

    pytest tests/test_tum2twin.py
    python tests/test_tum2twin.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worldsplat_prune.datasets.composite import COMPOSITE_STRIDE as S
from worldsplat_prune.datasets.tum2twin import TUM2TWINLabelAdapter
from worldsplat_prune.prep import scene_io as sio
from worldsplat_prune.prep import label_from_reference as L


def _cam(w=4, h=4, fx=1.0, fy=1.0, cx=2.0, cy=2.0):
    return sio.Camera(0, 0, "cam", w, h, fx, fy, cx, cy,
                      np.eye(3), np.zeros(3))


# --------------------------------------------------------------------------- #
# route (a): rasterisation                                                      #
# --------------------------------------------------------------------------- #
def test_rasterize_nearest_surface_wins_and_unmatched_is_void():
    cam = _cam()
    xyz = np.array([[0, 0, 1.0], [0, 0, 5.0], [1, 1, 1.0], [-10, -10, 1.0]])
    cls = [10, 20, 30, 40]
    inst = [1, 1, 1, 1]
    m = L.rasterize_composite(cam, xyz, cls, inst)
    assert m[2, 2] == 10 * S + 1          # nearest surface at the pixel
    assert m[3, 3] == 30 * S + 1
    assert m[0, 0] == 0                   # no reference surface -> void
    assert (m // S)[m > 0].tolist() == sorted((m // S)[m > 0].tolist())


def test_rasterize_void_class_or_instance_is_void():
    cam = _cam()
    # (0,0) has no instance -> composite void, even though the class is set
    xyz = np.array([[0, 0, 1.0], [1, 1, 1.0]])
    m = L.rasterize_composite(cam, xyz, [10, 0], [0, 1])
    assert m[2, 2] == 0 and m[3, 3] == 0


def test_rasterize_radius_dilates_labels():
    cam = _cam()
    xyz = np.array([[0, 0, 1.0]])
    m0 = L.rasterize_composite(cam, xyz, [10], [1], radius=0)
    m1 = L.rasterize_composite(cam, xyz, [10], [1], radius=1)
    assert m1[2, 2] == 10 * S + 1
    assert m1[1, 2] == 10 * S + 1         # within 1 px
    assert m0[1, 2] == 0
    assert m1[0, 0] == 0                  # beyond the radius from any label


# --------------------------------------------------------------------------- #
# route (b): 3D nearest-surface assignment                                      #
# --------------------------------------------------------------------------- #
def test_assign_reference_labels_3d_tolerance():
    q = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    ref = np.array([[0.0, 0, 0], [0.2, 0, 0]])
    out = L.assign_reference_labels_3d(q, ref, [3, 4], [1, 2], radius=0.5)
    assert out[0] == 3 * S + 1            # nearest reference point
    assert out[1] == 0                    # outside the tolerance -> void


def test_assign_reference_labels_3d_empty_reference_is_void():
    q = np.zeros((3, 3))
    out = L.assign_reference_labels_3d(q, np.zeros((0, 3)), [], [], radius=1.0)
    assert out.tolist() == [0, 0, 0]


# --------------------------------------------------------------------------- #
# reference cloud loading                                                       #
# --------------------------------------------------------------------------- #
def test_load_reference_cloud_npz_and_npy():
    with tempfile.TemporaryDirectory() as td:
        npz = os.path.join(td, "ref.npz")
        np.savez(npz, xyz=np.array([[0, 0, 0], [1, 2, 3]], float),
                 **{"class": np.array([5, 6]), "instance": np.array([1, 2])})
        xyz, cls, inst = L.load_reference_cloud(npz)
        assert xyz.shape == (2, 3) and cls.tolist() == [5, 6] and inst.tolist() == [1, 2]

        npy = os.path.join(td, "ref5.npy")
        np.save(npy, np.array([[0, 0, 0, 7, 3], [1, 1, 1, 8, 4]], float))
        xyz, cls, inst = L.load_reference_cloud(npy)
        assert cls.tolist() == [7, 8] and inst.tolist() == [3, 4]

        # bare xyz + sidecar class/instance
        bare = os.path.join(td, "xyz.npy")
        cp = os.path.join(td, "c.npy"); ip = os.path.join(td, "i.npy")
        np.save(bare, np.zeros((2, 3)))
        np.save(cp, np.array([1, 2])); np.save(ip, np.array([9, 9]))
        xyz, cls, inst = L.load_reference_cloud(bare, class_path=cp, instance_path=ip)
        assert cls.tolist() == [1, 2] and inst.tolist() == [9, 9]


def test_load_reference_cloud_rejects_bad_shape():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "bad.npy")
        np.save(p, np.zeros((2, 3)))
        try:
            L.load_reference_cloud(p)
        except ValueError:
            return
        raise AssertionError("expected ValueError for a bare [M,3] .npy")


# --------------------------------------------------------------------------- #
# validation confusion summary                                                  #
# --------------------------------------------------------------------------- #
def test_confusion_summary():
    s = L.confusion_summary([1, 1, 2, 2], [1, 2, 2, 2])
    assert s["n"] == 4 and s["labels"] == [1, 2]
    assert abs(s["accuracy"] - 0.75) < 1e-9
    assert abs(s["iou"][1] - 0.5) < 1e-9
    assert abs(s["iou"][2] - 2 / 3) < 1e-9
    assert s["matrix"] == [[1, 0], [1, 2]]


# --------------------------------------------------------------------------- #
# adapter                                                                       #
# --------------------------------------------------------------------------- #
def _write_dataset(td):
    d = Path(td) / "semantic_reference"
    d.mkdir(parents=True, exist_ok=True)
    ids = np.zeros((4, 4), dtype=np.int32)
    ids[1, 1] = 5 * S + 2
    ids[2, 2] = 6 * S + 1
    np.save(d / "rgb_0.npy", ids)
    return d


def test_adapter_load_class_instance_and_meta():
    with tempfile.TemporaryDirectory() as td:
        _write_dataset(td)
        ad = TUM2TWINLabelAdapter(td, {})
        got = ad.load("rgb_0")
        assert got[1, 1] == 5 * S + 2 and got[2, 2] == 6 * S + 1
        cls, inst = ad.load_class_instance("rgb_0")
        assert cls[1, 1] == 5 and inst[1, 1] == 2
        assert ad.resolve("rgb_0") is not None
        assert ad.load("missing") is None

        meta = ad.meta()
        assert meta["fuse_instances"] is False
        assert meta["stride"] == S
        assert meta["class_taxonomy"]                 # reports the taxonomy
        assert "3D reference" in meta["instance_consistency"]


def test_adapter_registration():
    from worldsplat_prune.datasets import get_adapter

    with tempfile.TemporaryDirectory() as td:
        _write_dataset(td)
        for name in ("reference", "tum2twin"):
            assert isinstance(get_adapter(name, td, {}), TUM2TWINLabelAdapter)


def test_adapter_missing_dir_raises():
    with tempfile.TemporaryDirectory() as td:
        try:
            TUM2TWINLabelAdapter(td, {})
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError without semantic_reference/")


def test_label_from_reference_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        ds = Path(td)
        sparse = ds / "sparse" / "0"
        sparse.mkdir(parents=True)
        # a pinhole camera at the identity, 4x4, fx=fy=1, cx=cy=2
        (sparse / "cameras.txt").write_text(
            "# Camera list\n1 PINHOLE 4 4 1 1 2 2\n")
        (sparse / "images.txt").write_text(
            "# Image list\n1 1 0 0 0 0 0 0 1 rgb_0.png\n\n")
        (sparse / "points3D.txt").write_text("# 3D points\n")

        ref = ds / "ref.npz"
        np.savez(ref, xyz=np.array([[0.0, 0.0, 1.0]]),
                 **{"class": np.array([5]), "instance": np.array([2])})

        meta = L.label_from_reference(ds, ref, sparse_dir=sparse, radius=0)
        assert meta["num_frames"] == 1
        assert meta["consistency"].startswith("cross-view consistent")
        assert (ds / "semantic_reference" / "rgb_0.npy").exists()

        ad = TUM2TWINLabelAdapter(ds, {})
        got = ad.load("rgb_0")
        assert got[2, 2] == 5 * S + 2       # world (0,0,1) -> pixel (row 2, col 2)
        assert ad.meta()["num_frames"] == 1

        # the reference maps feed the production store/vote unchanged (n_labeled > 0)
        import torch

        from worldsplat_prune.semantic_store import SemanticStore
        from worldsplat_prune import vote as V

        class _G:
            def __init__(self, xyz):
                self._xyz = torch.as_tensor(xyz, dtype=torch.float32)

            @property
            def get_xyz(self):
                return self._xyz

        class _C:
            image_name = "rgb_0"
            image_width = 4
            image_height = 4
            full_proj_transform = torch.eye(4)

        store = SemanticStore(ad, device="cpu")
        b = V.vote_classes_over_views(_G([[0.0, 0.0, 0.0]]), [_C()], store,
                                      visibility="project", vote="first", min_views=1)
        assert b.seg.tolist() == [5 * S + 2]
        assert b.meta["n_labeled"] == 1 and b.meta["mean_purity"] > 0.0


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
