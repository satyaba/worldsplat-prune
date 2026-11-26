"""Geometric utilities over canonical segment maps (cv2/numpy only)."""
from __future__ import annotations

import cv2
import numpy as np

# [D20]/T2 geometry-edge term defaults.
DEFAULT_EDGE_TOL = 0.05       # relative depth-discontinuity threshold
DEFAULT_EDGE_ABS_TOL = 0.02   # absolute floor for the threshold


def resize_ids(label: np.ndarray, H: int, W: int) -> np.ndarray:
    """Nearest-neighbor resize of an id map. Ids must never be interpolated."""
    label = np.asarray(label)
    if label.shape == (H, W):
        return label.astype(np.int32)
    return cv2.resize(label.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)


def resize_depth(depth: np.ndarray, H: int, W: int) -> np.ndarray:
    """Nearest-neighbor resize of a depth map (no interpolation across edges)."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape == (H, W):
        return depth
    return cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32)


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


def depth_edge_mask(depth: np.ndarray, *, edge_tol: float = DEFAULT_EDGE_TOL,
                    abs_tol: float = DEFAULT_EDGE_ABS_TOL,
                    occlusion_only: bool = False) -> np.ndarray:
    """Boolean ``[H, W]`` mask of depth discontinuities ([D20]/T2).

    A pixel is an edge when its depth differs from a 4-neighbour by more than
    ``max(abs_tol, edge_tol * min(|d_p|, |d_q|))`` — a relative threshold so one
    tolerance works across depth scales. Invalid pixels (``depth <= 0``) never
    form edges. With ``occlusion_only`` only the *nearer* side of a jump is
    marked (the occluding silhouette: wires, railings, canopies), which is the
    signal the class map misses.
    """
    d = np.asarray(depth, dtype=np.float32)
    valid = d > 0
    edge = np.zeros(d.shape, dtype=bool)

    def _acc(a_idx, b_idx):
        a = d[a_idx]
        b = d[b_idx]
        va = valid[a_idx]
        vb = valid[b_idx]
        ok = va & vb
        thr = np.maximum(float(abs_tol), float(edge_tol) * np.minimum(a, b))
        diff = b - a
        if occlusion_only:
            edge[a_idx] |= (diff > thr) & ok          # a nearer, b farther
            edge[b_idx] |= (-diff > thr) & ok         # b nearer, a farther
        else:
            m = (np.abs(diff) > thr) & ok
            edge[a_idx] |= m
            edge[b_idx] |= m

    _acc((slice(None), slice(None, -1)), (slice(None), slice(1, None)))   # x
    _acc((slice(None, -1), slice(None)), (slice(1, None), slice(None)))   # y
    return edge


def distance_transform_from_mask(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance (px) to the nearest ``True`` pixel of ``mask``.

    ``0`` on the mask; ``inf`` where the mask is empty (no boundary to be near).
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    src = (~mask).astype(np.uint8)  # distanceTransform measures to nearest zero
    return cv2.distanceTransform(src, cv2.DIST_L2, 3).astype(np.float32)


def geometry_distance_transform(depth: np.ndarray, *,
                                edge_tol: float = DEFAULT_EDGE_TOL,
                                abs_tol: float = DEFAULT_EDGE_ABS_TOL,
                                occlusion_only: bool = False):
    """``(dt_geom, edge_mask)`` from a per-view depth map ([D20]/T2).

    ``dt_geom`` is the distance (px) to the nearest depth discontinuity; invalid
    pixels (``depth <= 0``) are ``inf`` so they never protect. A thin depth ridge
    therefore gives ``dt_geom = 0`` along it, letting the band catch structures
    the class map misses.
    """
    d = np.asarray(depth, dtype=np.float32)
    valid = d > 0
    edge = depth_edge_mask(d, edge_tol=edge_tol, abs_tol=abs_tol,
                           occlusion_only=occlusion_only)
    if not edge.any():
        return np.full(d.shape, np.inf, dtype=np.float32), edge
    dt = distance_transform_from_mask(edge)
    dt = np.where(valid, dt, np.inf).astype(np.float32)
    return dt, edge


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
