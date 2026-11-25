"""Tests for the composite ``(class, instance)`` labels and per-class fusion.

Covers the pure encoding (:mod:`worldsplat_prune.datasets.composite`), the
``CompositeLabelAdapter`` (int32 ``.npy`` above the 16-bit range, and integer
images), and the ``fuse_instances_per_class`` front-end driver end-to-end on a
tiny synthetic COLMAP scene. No GPU / torch / SAM2 required.

    pytest tests/test_composite.py
    python tests/test_composite.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worldsplat_prune.datasets import get_adapter
from worldsplat_prune.datasets import composite as comp
from worldsplat_prune.prep import fuse_instances as fi


# --------------------------------------------------------------------------- #
# pure encoding                                                                 #
# --------------------------------------------------------------------------- #
def test_encode_decode_roundtrip():
    cls = np.array([[0, 1, 2], [3, 1, 0]], dtype=np.int32)
    inst = np.array([[0, 5, 7], [9, 4, 0]], dtype=np.int32)   # 0 only where class is 0
    ids = comp.encode(cls, inst)
    assert ids.dtype == np.int32
    # void where either half is void
    assert ids[0, 0] == 0 and ids[1, 2] == 0
    assert ids[0, 1] == 1 * comp.COMPOSITE_STRIDE + 5
    c2, i2 = comp.decode(ids)
    assert np.array_equal(c2, cls)
    assert np.array_equal(i2, inst)


def test_encode_rejects_instance_overflow():
    try:
        comp.encode([1], [comp.COMPOSITE_STRIDE])
    except ValueError:
        return
    raise AssertionError("expected ValueError for instance id >= stride")


def test_compose_maps_disjoint_and_overlap_first_wins():
    a = np.array([[1, 0], [0, 0]], dtype=np.int32)   # class 1
    b = np.array([[0, 0], [1, 0]], dtype=np.int32)   # class 2
    out = comp.compose_maps({1: a, 2: b})
    S = comp.COMPOSITE_STRIDE
    assert out[0, 0] == 1 * S + 1
    assert out[1, 0] == 2 * S + 1
    # overlap: first class (in insertion order) wins
    c = np.array([[1, 0], [0, 0]], dtype=np.int32)
    d = np.array([[1, 0], [0, 0]], dtype=np.int32)
    out2 = comp.compose_maps({1: c, 2: d})
    assert out2[0, 0] == 1 * S + 1


# --------------------------------------------------------------------------- #
# adapter                                                                       #
# --------------------------------------------------------------------------- #
def test_composite_adapter_npy_above_uint16():
    with tempfile.TemporaryDirectory() as d:
        ds = Path(d)
        dir_ = ds / "semantic_composite"
        dir_.mkdir(parents=True)
        # class 7 -> id > 65535, only representable as int32 .npy
        ids = np.zeros((4, 4), dtype=np.int32)
        ids[:, :2] = 7 * comp.COMPOSITE_STRIDE + 1
        ids[:, 2:] = 1 * comp.COMPOSITE_STRIDE + 2
        comp.write_composite_map(dir_ / "rgb_0.npy", ids)

        ad = get_adapter("composite", ds)
        out = ad.load("rgb_0.png")          # resolved by stem
        assert out is not None
        assert np.array_equal(out, ids)
        cls, inst = ad.load_class_instance("rgb_0")
        assert set(np.unique(cls).tolist()) == {1, 7}
        assert inst[0, 0] == 1 and inst[0, 3] == 2
        assert ad.meta()["num_frames"] == 1


def test_composite_adapter_png_and_min_area():
    with tempfile.TemporaryDirectory() as d:
        ds = Path(d)
        dir_ = ds / "semantic_composite"
        ids = np.zeros((3, 3), dtype=np.int32)
        ids[:2, :] = 1 * comp.COMPOSITE_STRIDE + 4     # 6-px instance (fits uint16)
        ids[2, 2] = 1 * comp.COMPOSITE_STRIDE + 9     # 1-px speckle
        comp.write_composite_map(dir_ / "a.png", ids)
        assert (dir_ / "a.png").exists()

        kept = get_adapter("composite", ds, {"min_area": 3}).load("a.png")
        assert set(np.unique(kept).tolist()) == {0, comp.COMPOSITE_STRIDE + 4}


def test_composite_adapter_missing_dir_raises():
    with tempfile.TemporaryDirectory() as d:
        try:
            get_adapter("composite", Path(d))
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError without composite labels")


# --------------------------------------------------------------------------- #
# per-class fusion end-to-end                                                   #
# --------------------------------------------------------------------------- #
def _write_model(root: Path):
    sp = root / "sparse" / "0"
    sp.mkdir(parents=True)
    (sp / "cameras.txt").write_text(
        "# Camera list\n"
        "1 SIMPLE_PINHOLE 4 4 4.0 2.0 2.0\n")
    (sp / "images.txt").write_text(
        "# Image list\n"
        "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 rgb_0.png\n"
        "0.0 0.0 -1\n"
        "2 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 rgb_1.png\n"
        "0.0 0.0 -1\n")
    # point 1 -> pixel (2,2); point 2 -> pixel (1,2)
    (sp / "points3D.txt").write_text(
        "# 3D points\n"
        "1 0.0 0.0 1.0 255 0 0 0.1\n"
        "2 -0.25 0.0 1.0 0 255 0 0.1\n")


def _write_ids(path: Path, ids: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ids.astype(np.uint16)).save(path)


def test_fuse_instances_per_class_composes():
    with tempfile.TemporaryDirectory() as d:
        ds = Path(d)
        _write_model(ds)
        # class "chair" (logged first): instance 1 at pixel (2,2) -> point 1
        chair = np.zeros((4, 4), dtype=np.int32)
        chair[2, 2] = 1
        _write_ids(ds / "sem_chair" / "rgb_0.png", chair)
        _write_ids(ds / "sem_chair" / "rgb_1.png", chair)
        # class "table": instance 1 at pixel (1,2) -> point 2
        table = np.zeros((4, 4), dtype=np.int32)
        table[2, 1] = 1
        _write_ids(ds / "sem_table" / "rgb_0.png", table)
        _write_ids(ds / "sem_table" / "rgb_1.png", table)

        summary = fi.fuse_instances_per_class(
            ds, {"chair": "sem_chair", "table": "sem_table"},
            min_views=2, metric="jaccard", min_score=0.0)

        S = comp.COMPOSITE_STRIDE
        assert summary["class_ids"] == {"chair": 1, "table": 2}
        assert summary["composite_views"] == 2

        m0 = comp.read_composite_map(ds / "semantic_composite" / "rgb_0.npy")
        assert m0[2, 2] == 1 * S + 1
        assert m0[2, 1] == 2 * S + 1

        pts = np.load(ds / "semantic_composite" / "composite_point_labels.npy")
        assert pts[0] == 1 * S + 1     # point 1 -> chair instance
        assert pts[1] == 2 * S + 1     # point 2 -> table instance

        # the composite map is consumable by the adapter
        ad = get_adapter("composite", ds)
        assert np.array_equal(ad.load("rgb_0.png"), m0)
        assert ad.meta()["num_frames"] == 2       # composite_point_labels.npy is not a frame


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
