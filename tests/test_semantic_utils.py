"""Tests for the depth/geometry-edge helpers (T2, [D20]).

CPU-only: ``semantic_utils`` is cv2/numpy, no torch or model needed.

    pytest tests/test_semantic_utils.py
    python tests/test_semantic_utils.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worldsplat_prune import semantic_utils as su


def test_depth_edge_mask_thin_ridge():
    depth = np.full((7, 7), 5.0, dtype=np.float32)
    depth[3, :] = 1.0                       # a 1-px horizontal ridge, much nearer
    edge = su.depth_edge_mask(depth)
    assert edge[3, 3]                       # on the ridge
    assert edge[2, 3] and edge[4, 3]        # the two flanking rows are edges too
    assert not edge[0, 0] and not edge[6, 6]


def test_geometry_distance_transform_ridge_is_zero():
    depth = np.full((7, 7), 5.0, dtype=np.float32)
    depth[3, :] = 1.0
    dt, edge = su.geometry_distance_transform(depth)
    assert edge.any()
    assert dt[3, 3] == 0.0                  # on the ridge -> on a boundary
    assert dt[0, 0] > 0.0                   # far corner -> interior


def test_geometry_distance_transform_invalid_depth_is_inf():
    depth = np.zeros((4, 4), dtype=np.float32)   # all invalid
    dt, edge = su.geometry_distance_transform(depth)
    assert not edge.any()
    assert np.all(np.isinf(dt))


def test_depth_edge_mask_occlusion_only_marks_near_side():
    depth = np.full((1, 5), 5.0, dtype=np.float32)
    depth[0, 2] = 1.0
    both = su.depth_edge_mask(depth)
    near = su.depth_edge_mask(depth, occlusion_only=True)
    assert both.tolist() == [[False, True, True, True, False]]
    # only the near (ridge) pixel survives the occlusion filter
    assert near.tolist() == [[False, False, True, False, False]]


def test_depth_edge_mask_ignores_invalid_neighbours():
    depth = np.full((3, 3), 5.0, dtype=np.float32)
    depth[1, 1] = 0.0                       # invalid hole: no edges around it
    assert not su.depth_edge_mask(depth).any()


def test_distance_transform_from_mask():
    empty = su.distance_transform_from_mask(np.zeros((3, 3), dtype=bool))
    assert np.all(np.isinf(empty))
    dt = su.distance_transform_from_mask(np.eye(3, dtype=bool))
    # 0 on the mask; the distance grows away from it (cv2 L2 is approximate)
    assert dt[0, 0] == 0.0
    assert 0.0 < dt[0, 1] < dt[0, 2]


def test_resize_depth_nearest():
    d = np.arange(4, dtype=np.float32).reshape(2, 2)
    r = su.resize_depth(d, 4, 4)
    assert r.shape == (4, 4) and r[0, 0] == 0 and r[3, 3] == 3
    assert su.resize_depth(d, 2, 2) is d or np.array_equal(su.resize_depth(d, 2, 2), d)


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
