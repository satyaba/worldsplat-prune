"""Tests for the 3D instance-metric harness (worldsplat_prune.prep.eval_instances) and the
point-label lifting in worldsplat_prune.prep.scene_io. CPU only.

    pytest tests/test_eval_instances.py
    python tests/test_eval_instances.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worldsplat_prune.prep import eval_instances as ei
from worldsplat_prune.prep import scene_io as sio
from worldsplat_prune.datasets.composite import COMPOSITE_STRIDE


# --------------------------------------------------------------------------- #
# metrics                                                                       #
# --------------------------------------------------------------------------- #
def test_perfect_prediction():
    gt = np.array([1, 1, 2, 2, 3, 0], dtype=int)
    res = ei.evaluate_instance_labels(gt.copy(), gt)
    assert res["ap"]["AP25"] == 1.0 and res["ap"]["AP50"] == 1.0
    assert res["map"] == 1.0 and res["coverage"] == 1.0
    assert abs(res["miou_matched"] - 1.0) < 1e-9


def test_disjoint_prediction():
    # pred instances cover different points than gt -> IoU 1/3 < 0.5 -> no match @0.5
    gt = np.array([1, 1, 2, 2])
    pred = np.array([5, 0, 5, 0])
    res = ei.evaluate_instance_labels(pred, gt, thresholds=(0.5,))
    assert res["map"] == 0.0 and res["coverage"] == 0.0
    # ...but the loose threshold does credit it
    res25 = ei.evaluate_instance_labels(pred, gt, thresholds=(0.25,))
    assert res25["ap"]["AP25"] > 0.0


def test_merged_prediction_half_recall():
    # gt has two 2-point instances; pred merges them into one 4-point mask.
    # IoU(pred, gt1) = 2/4 = 0.5 -> matches gt1 at tau<=0.5; gt2 missed.
    gt = np.array([1, 1, 2, 2])
    pred = np.array([9, 9, 9, 9])
    res = ei.evaluate_instance_labels(pred, gt, thresholds=(0.5,))
    assert abs(res["ap"]["AP50"] - 0.5) < 1e-9   # precision 1, recall 0.5
    assert abs(res["coverage"] - 0.5) < 1e-9
    assert abs(res["miou_matched"] - 0.5) < 1e-9


def test_split_prediction_full_ap():
    # gt is one 4-point instance; pred splits it in two 2-point masks (IoU 0.5 each).
    # The first matches (recall 1.0 immediately) -> AP = 1 at tau 0.5.
    gt = np.array([1, 1, 1, 1])
    pred = np.array([1, 1, 2, 2])
    res = ei.evaluate_instance_labels(pred, gt, thresholds=(0.5,))
    assert abs(res["ap"]["AP50"] - 1.0) < 1e-9
    assert abs(res["coverage"] - 1.0) < 1e-9


def test_min_area_drops_speckle():
    gt = np.array([1, 1, 1, 1, 2])
    pred = np.array([1, 1, 1, 1, 2])
    res_full = ei.evaluate_instance_labels(pred, gt, thresholds=(0.5,))
    res_min = ei.evaluate_instance_labels(pred, gt, thresholds=(0.5,), min_area=2)
    assert res_full["n_gt"] == 2 and res_min["n_gt"] == 1


def test_iou_matrix_symmetry_and_void_excluded():
    pred = np.array([1, 1, 2, 0])
    gt = np.array([1, 2, 2, 2])
    M, sp, sg = ei.contingency(ei._dense(pred), ei._dense(gt))
    iou = ei.iou_matrix(M, sp, sg)
    # pred1 vs gt1: inter 1, union 2 -> 0.5
    assert abs(iou[1, 1] - 0.5) < 1e-9
    # pred2 (1 pt) vs gt2 (3 pts): inter 1, union 1+3-1 = 3 -> 1/3
    assert abs(iou[2, 2] - 1.0 / 3.0) < 1e-9


# --------------------------------------------------------------------------- #
# point-label lifting                                                           #
# --------------------------------------------------------------------------- #
def _scene_2views(root: Path):
    sp = root / "sparse" / "0"
    sp.mkdir(parents=True)
    (sp / "cameras.txt").write_text("1 SIMPLE_PINHOLE 4 4 4.0 2.0 2.0\n")
    (sp / "images.txt").write_text(
        "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 rgb_0.png\n0.0 0.0 -1\n"
        "2 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 rgb_1.png\n0.0 0.0 -1\n")
    (sp / "points3D.txt").write_text(
        "1 0.0 0.0 1.0 255 0 0 0.1\n2 0.0 0.0 1.0 0 255 0 0.1\n")
    return sio.load_colmap_scene(sp)


def test_point_labels_first_vs_majority_tie():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        scene = _scene_2views(root)
        lab_dir = root / "lbl"
        lab_dir.mkdir()
        m0 = np.zeros((4, 4), dtype=np.int32); m0[2, 2] = 5
        m1 = np.zeros((4, 4), dtype=np.int32); m1[2, 2] = 9
        sio.write_id_map(lab_dir / "rgb_0.png", m0)
        sio.write_id_map(lab_dir / "rgb_1.png", m1)

        first = sio.point_labels_from_views(scene, lab_dir, mode="first")
        assert set(first.tolist()) == {5}

        # 2 views disagree -> tie -> majority leaves the points void
        maj = sio.point_labels_from_views(scene, lab_dir, mode="majority")
        assert set(maj.tolist()) == {0}

        # make it a 3rd agreeing view -> majority resolves to 5
        m2 = np.zeros((4, 4), dtype=np.int32); m2[2, 2] = 5
        sio.write_id_map(lab_dir / "rgb_2.png", m2)
        (root / "sparse" / "0" / "images.txt").write_text(
            "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 rgb_0.png\n0.0 0.0 -1\n"
            "2 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 rgb_1.png\n0.0 0.0 -1\n"
            "3 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 rgb_2.png\n0.0 0.0 -1\n")
        scene3 = sio.load_colmap_scene(root / "sparse" / "0")
        maj3 = sio.point_labels_from_views(scene3, lab_dir, mode="majority")
        assert set(maj3.tolist()) == {5}


# --------------------------------------------------------------------------- #
# class validation of the front-end composite maps                              #
# --------------------------------------------------------------------------- #
def test_level_transform_decodes_composite():
    S = COMPOSITE_STRIDE
    ids = np.array([[2 * S + 3, 0]], dtype=np.int32)
    assert ei._level_transform("composite") is None
    assert ei._level_transform("class")(ids)[0, 0] == 2
    assert ei._level_transform("instance")(ids)[0, 0] == 3


def test_evaluate_class_labels_accuracy():
    gt = np.array([1, 1, 2, 2, 0])
    assert ei.evaluate_class_labels(gt.copy(), gt)["accuracy"] == 1.0
    pred = np.array([1, 2, 2, 2, 1])     # point 2 mislabelled
    res = ei.evaluate_class_labels(pred, gt)
    assert res["points"] == 4
    assert abs(res["accuracy"] - 0.75) < 1e-9
    assert ei.evaluate_class_labels(np.zeros(3), np.zeros(3))["points"] == 0


def test_run_class_eval_on_composite_dir():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _scene_2views(root)                       # points project to pixel (2,2)
        S = COMPOSITE_STRIDE
        cdir = root / "semantic_composite"
        cdir.mkdir()
        for stem in ("rgb_0", "rgb_1"):
            m = np.zeros((4, 4), dtype=np.int32)
            m[2, 2] = 2 * S + 3                   # class 2, instance 3
            np.save(cdir / f"{stem}.npy", m)
        gdir = root / "semantic_class"
        gdir.mkdir()
        for stem in ("rgb_0", "rgb_1"):
            m = np.zeros((4, 4), dtype=np.int32)
            m[2, 2] = 2
            sio.write_id_map(gdir / f"{stem}.png", m)

        res = ei.run_class_eval(root, class_gt="semantic_class",
                                methods={"frontend": "semantic_composite"},
                                level="class")
        assert res["methods"]["frontend"]["accuracy"] == 1.0


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
