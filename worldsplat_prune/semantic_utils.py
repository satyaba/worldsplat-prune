"""Geometric utilities over canonical segment maps (cv2/numpy only)."""
from __future__ import annotations

import cv2
import numpy as np


def resize_ids(label: np.ndarray, H: int, W: int) -> np.ndarray:
    """Nearest-neighbor resize of an id map. Ids must never be interpolated."""
    label = np.asarray(label)
    if label.shape == (H, W):
        return label.astype(np.int32)
    return cv2.resize(label.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)


def boundary_mask(label: np.ndarray) -> np.ndarray:
    """True where a pixel is adjacent to a different segment id."""
    lab = np.asarray(label)
    b = np.zeros(lab.shape, dtype=bool)
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:, 1:] |= lab[:, :-1] != lab[:, 1:]
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    b[1:, :] |= lab[:-1, :] != lab[1:, :]
    return b


def distance_transform(label: np.ndarray):
    """Euclidean distance (px) to the nearest segment boundary; 0 on boundary.

    Returns ``(dt, boundary)`` with ``dt`` float32 ``[H, W]``.
    """
    b = boundary_mask(label)
    src = (~b).astype(np.uint8)  # distanceTransform measures to nearest zero pixel
    dt = cv2.distanceTransform(src, cv2.DIST_L2, 3).astype(np.float32)
    return dt, b


def normal_field(dt: np.ndarray) -> np.ndarray:
    """Unit boundary normal ``[H, W, 2]`` pointing away from the nearest boundary."""
    gy, gx = np.gradient(dt.astype(np.float32))
    n = np.stack([gx, gy], axis=-1)
    norm = np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6, None)
    return (n / norm).astype(np.float32)


def signed_distance(label: np.ndarray, dt: np.ndarray, seg_ids: np.ndarray) -> np.ndarray:
    """Signed distance for pixels relative to ``seg_ids`` (>=0 inside, <0 outside)."""
    inside = np.asarray(label) == np.asarray(seg_ids)
    return np.where(inside, dt, -dt).astype(np.float32)


def read_ids(path) -> np.ndarray:
    """Read a label image as ``int32`` ids.

    Uses PIL (not cv2) so palette (``mode='P'``) PNGs yield palette *indices*
    rather than BGR colours, and 16-bit PNGs (e.g. SAM2 instance maps with more
    than 255 masks) round-trip without truncation.
    """
    from PIL import Image

    arr = np.array(Image.open(path))
    if arr.ndim == 3:  # defensive: collapse unexpected multi-channel labels
        arr = arr[..., 0]
    return arr.astype(np.int32)


def fill_void_nearest(ids: np.ndarray) -> np.ndarray:
    """Assign every void (``0``) pixel the id of the nearest non-void pixel.

    SAM2 (and other class-agnostic segmenters) leave unlabelled pixels wherever
    no mask fires. Left as void, those pixels would spawn spurious "boundaries"
    against every neighbouring instance and inflate the boundary band. Nearest
    fill removes them without touching real instance edges.

    Uses OpenCV's per-pixel distance labels (``DIST_LABEL_PIXEL``), so there is
    no scipy dependency. An all-void (or empty) map is returned unchanged.
    """
    ids = np.asarray(ids)
    void = ids == 0
    if not void.any() or void.all():
        return ids.astype(np.int32)
    # void=1, non-void=0: the transform then measures, for every void pixel, the
    # distance to the nearest non-void pixel and labels it with that pixel.
    src = void.astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    lut[labels[~void]] = ids[~void]  # each non-void pixel owns its label
    out = ids.astype(np.int32).copy()
    out[void] = lut[labels[void]]
    return out


def segment_area_perimeter(label: np.ndarray):
    """Return ``(area, perimeter)`` dicts keyed by segment id."""
    ids, counts = np.unique(label, return_counts=True)
    area = {int(i): int(c) for i, c in zip(ids, counts)}

    b = boundary_mask(label)
    bids, bcounts = np.unique(label[b], return_counts=True)
    perim = {int(i): int(c) for i, c in zip(bids, bcounts)}
    return area, perim
