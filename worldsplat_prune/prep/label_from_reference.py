"""Reference-geometry labelling driver (C2, §2.3, [D19]) — route (a).

Turns the twin's **LoD / LiDAR / ALS** reference geometry into the per-view
canonical composite maps the pruning pipeline consumes::

    reference geometry ─► project/rasterise per camera ─► L_v : int32 [H, W]
        (xyz + class + instance)                              id = class*STRIDE + instance

Route (a) — preferred: rasterise labelled reference surfaces into per-view maps
(drop-in for ``SemanticStore``/``vote``/``prune``; zero operator changes). The
z-buffer keeps the nearest surface, so occlusion is respected. Instances come
from the 3D geometry, so the labels are **cross-view consistent by
construction** and no ``fuse_instances`` step is needed.

Route (b) — direct 3D assignment — is provided by
:func:`assign_reference_labels_3d` for callers that want per-Gaussian labels
without a 2D round-trip.

    python -m worldsplat_prune.prep.label_from_reference \\
        --dataset_dir "$DATASET_ROOT/tum2twin/tile_000" \\
        --reference ref/als_points.npz --radius 2

Then run ``prune_model.py --label_adapter reference``.

The core (``load_reference_cloud``, ``rasterize_composite``,
``assign_reference_labels_3d``, ``confusion_summary``) is numpy/cv2-only and
CPU-tested (``tests/test_tum2twin.py``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..datasets.composite import COMPOSITE_STRIDE, encode, write_composite_map
from ..datasets.tum2twin import DEFAULT_TAXONOMY
from . import scene_io as sio


# --------------------------------------------------------------------------- #
# reference cloud loading                                                       #
# --------------------------------------------------------------------------- #
def _stack(xyz, cls, inst):
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    cls = np.asarray(cls).reshape(-1).astype(np.int64)
    inst = np.asarray(inst).reshape(-1).astype(np.int64)
    if not (xyz.shape[0] == cls.shape[0] == inst.shape[0]):
        raise ValueError(
            f"xyz/class/instance length mismatch: {xyz.shape[0]}, "
            f"{cls.shape[0]}, {inst.shape[0]}")
    cls = np.where(cls > 0, cls, 0)
    inst = np.where(inst > 0, inst, 0)
    return xyz, cls, inst


def _load_ply(path: Path):
    try:
        from plyfile import PlyData
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "reading .ply reference clouds needs `plyfile`; export an .npz/.npy "
            "(xyz, class, instance) instead") from e
    v = PlyData.read(str(path))["vertex"].data
    names = set(v.dtype.names or ())

    def pick(*cands):
        for c in cands:
            if c in names:
                return np.asarray(v[c])
        return None

    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
    cls = pick("class", "scalar_class", "classification", "label", "cls")
    inst = pick("instance", "instance_id", "object", "object_id", "inst")
    if cls is None or inst is None:
        raise ValueError(
            "a .ply reference cloud must carry per-vertex 'class' and 'instance' "
            f"fields (found {sorted(names)})")
    return xyz, cls, inst


def load_reference_cloud(path, *, class_path=None, instance_path=None):
    """Load reference geometry into ``(xyz [M,3] float64, class [M], instance [M])``.

    Accepted layouts:

    * ``.npz`` with ``xyz``/``class``/``instance`` arrays;
    * ``.npy`` of shape ``[M, 3]`` (xyz; then ``class_path``/``instance_path`` are
      required) or ``[M, >=5]`` (``x y z class instance`` in the first five cols);
    * ``.ply`` with per-vertex ``x/y/z`` + ``class`` + ``instance`` (needs plyfile).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        z = np.load(path)
        cls = z["class"] if "class" in z else z.get("classes")
        inst = z["instance"] if "instance" in z else z.get("instances")
        if cls is None or inst is None:
            raise ValueError(".npz must contain 'class' and 'instance' arrays")
        xyz, cls, inst = z["xyz"], cls, inst
    elif suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        if class_path or instance_path:
            if not (class_path and instance_path):
                raise ValueError("provide BOTH --class and --instance with a bare xyz .npy")
            xyz = arr
            cls = np.load(class_path)
            inst = np.load(instance_path)
        elif arr.ndim == 2 and arr.shape[1] >= 5:
            xyz, cls, inst = arr[:, :3], arr[:, 3], arr[:, 4]
        else:
            raise ValueError(
                "a .npy reference cloud must be [M, >=5] (x y z class instance) or a "
                "bare [M, 3] xyz alongside --class / --instance")
    elif suffix == ".ply":
        xyz, cls, inst = _load_ply(path)
    else:
        raise ValueError(f"unsupported reference cloud format: {path.suffix}")
    return _stack(xyz, cls, inst)


# --------------------------------------------------------------------------- #
# route (a): per-view rasterisation                                             #
# --------------------------------------------------------------------------- #
def rasterize_composite(cam, xyz, class_ids, instance_ids, *,
                        stride: int = COMPOSITE_STRIDE, radius: int = 0) -> np.ndarray:
    """Rasterise labelled reference points into one camera's composite id map.

    Points are projected with :func:`scene_io.project_points` and painted
    far-to-near so the **nearest surface wins** (occlusion). Pixels with no
    reference surface — or a point that projects out of frame — stay void (``0``).
    ``radius`` dilates labels into a small neighbourhood (depth-free splat).
    Returns ``int32 [H, W]``.
    """
    H, W = int(cam.height), int(cam.width)
    out = np.zeros((H, W), dtype=np.int32)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] == 0:
        return out
    ids = encode(class_ids, instance_ids, stride=stride)      # 0 where either void
    u, v, z, valid = sio.project_points(xyz, cam)
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)
    inb = valid & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (ids > 0)
    sel = np.nonzero(inb)[0]
    if sel.size:
        order = sel[np.argsort(-z[sel])]                      # far first, near last
        out[vi[order], ui[order]] = ids[order]
    if radius > 0:
        out = sio.assign_within_radius(out, radius)
    return out


def write_reference_maps(scene, xyz, class_ids, instance_ids, out_dir, *,
                         stride: int = COMPOSITE_STRIDE, radius: int = 0) -> int:
    """Write ``<stem>.npy`` composite maps for every camera; returns the count."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for cam in scene.cameras:
        m = rasterize_composite(cam, xyz, class_ids, instance_ids,
                                stride=stride, radius=radius)
        write_composite_map(out_dir / f"{cam.stem}.npy", m)
        written += 1
    return written


# --------------------------------------------------------------------------- #
# route (b): direct 3D nearest-surface assignment                              #
# --------------------------------------------------------------------------- #
def assign_reference_labels_3d(query_xyz, ref_xyz, ref_class, ref_instance, *,
                               radius: float, stride: int = COMPOSITE_STRIDE,
                               chunk: int = 65_536) -> np.ndarray:
    """Per-query composite id from the nearest reference point within ``radius``.

    Route (b): labels Gaussians directly in 3D. A query with no reference surface
    inside the tolerance ``radius`` is **void** (``0``) — unmatched points are
    never protected. Brute-force within ``chunk`` query blocks (modest reference
    clouds; route (a) is the primary path). Returns ``int32 [M]``.
    """
    q = np.asarray(query_xyz, dtype=np.float64).reshape(-1, 3)
    r = np.asarray(ref_xyz, dtype=np.float64).reshape(-1, 3)
    cls = np.asarray(ref_class).reshape(-1).astype(np.int64)
    inst = np.asarray(ref_instance).reshape(-1).astype(np.int64)
    out = np.zeros(q.shape[0], dtype=np.int32)
    if q.shape[0] == 0 or r.shape[0] == 0:
        return out
    radius = float(radius)
    for s in range(0, q.shape[0], int(chunk)):
        stop = min(s + int(chunk), q.shape[0])
        # squared-distance matrix [block, R]
        d2 = ((q[s:stop, None, :] - r[None, :, :]) ** 2).sum(-1)
        j = d2.argmin(axis=1)
        dmin = np.sqrt(d2[np.arange(stop - s), j])
        hit = dmin <= radius
        if hit.any():
            ids = encode(cls[j], inst[j], stride=stride)
            out[s:stop][hit] = ids[hit]
    return out


# --------------------------------------------------------------------------- #
# validation (confusion against held-out ALS class labels)                      #
# --------------------------------------------------------------------------- #
def confusion_summary(pred_class, gt_class, *, labels=None) -> dict:
    """Confusion summary of predicted vs GT class ids (row = GT, col = pred)."""
    pred = np.asarray(pred_class).reshape(-1).astype(np.int64)
    gt = np.asarray(gt_class).reshape(-1).astype(np.int64)
    if labels is None:
        labels = np.unique(np.concatenate([pred, gt])) if pred.size else np.array([], int)
    labels = np.asarray(labels).astype(np.int64)
    idx = {int(l): i for i, l in enumerate(labels.tolist())}
    k = labels.size
    M = np.zeros((k, k), dtype=np.int64)
    for p, g in zip(pred.tolist(), gt.tolist()):
        if g in idx and p in idx:
            M[idx[g], idx[p]] += 1
    diag = np.diag(M)
    total = int(M.sum())
    iou = {}
    for i, l in enumerate(labels.tolist()):
        tp = int(M[i, i]); fp = int(M[:, i].sum()) - tp; fn = int(M[i, :].sum()) - tp
        denom = tp + fp + fn
        iou[int(l)] = (tp / denom) if denom else 0.0
    return {
        "labels": [int(l) for l in labels.tolist()],
        "matrix": M.tolist(),          # matrix[gt_idx][pred_idx]
        "accuracy": (float(diag.sum()) / total) if total else 0.0,
        "iou": iou,
        "n": total,
    }


def _validate_against(scene, pred_dir, gt, stride: int):
    gt = Path(gt)
    preds, gts = [], []
    for cam in scene.cameras:
        pm = sio.read_view_map(pred_dir, cam.stem)
        if pm is None:
            continue
        if gt.is_dir():
            gm = sio.read_view_map(gt, cam.stem)
        elif gt.suffix.lower() == ".npy":
            gm = np.load(gt)
        else:
            gm = sio.read_id_map(gt)
        if gm is None:
            continue
        preds.append((np.asarray(pm, dtype=np.int64) // int(stride)).reshape(-1))
        gts.append(np.asarray(gm, dtype=np.int64).reshape(-1))
    if not preds:
        return {"error": "no overlapping frames between predictions and GT"}
    return confusion_summary(np.concatenate(preds), np.concatenate(gts))


# --------------------------------------------------------------------------- #
# driver                                                                        #
# --------------------------------------------------------------------------- #
def _load_taxonomy(spec) -> dict:
    if spec is None:
        return dict(DEFAULT_TAXONOMY)
    return {int(k): str(v) for k, v in json.loads(Path(spec).read_text()).items()}


def label_from_reference(dataset_dir, reference, *, sparse_dir=None, out_dir=None,
                         stride: int = COMPOSITE_STRIDE, radius: int = 2,
                         class_path=None, instance_path=None, taxonomy=None,
                         validate=None, transform=None) -> dict:
    """Rasterise ``reference`` geometry into ``<dataset>/semantic_reference/*.npy``.

    Returns the metadata dict (also written to ``reference_meta.json``).
    """
    dataset_dir = Path(dataset_dir)
    sparse = Path(sparse_dir) if sparse_dir else dataset_dir / "sparse" / "0"
    scene = sio.load_colmap_scene(sparse)
    xyz, cls, inst = load_reference_cloud(reference, class_path=class_path,
                                          instance_path=instance_path)
    if transform is not None:
        xyz = transform(xyz)

    out_dir = Path(out_dir) if out_dir else dataset_dir / "semantic_reference"
    out_dir.mkdir(parents=True, exist_ok=True)

    class_hist: dict = {}
    for cam in scene.cameras:
        m = rasterize_composite(cam, xyz, cls, inst, stride=stride, radius=radius)
        write_composite_map(out_dir / f"{cam.stem}.npy", m)
        cid = (m // int(stride))[m > 0]
        u, c = np.unique(cid, return_counts=True)
        for k, v in zip(u.tolist(), c.tolist()):
            class_hist[int(k)] = class_hist.get(int(k), 0) + int(v)

    meta = {
        "generated_by": "worldsplat_prune.prep.label_from_reference",
        "reference_path": str(reference),
        "camera_source": str(sparse),
        "num_frames": len(scene.cameras),
        "stride": int(stride),
        "radius": int(radius),
        "class_hist": class_hist,
        "taxonomy": _load_taxonomy(taxonomy),
        "instance_source": "LoD object / building id",
        "consistency": "cross-view consistent by construction (3D reference)",
    }
    if validate is not None:
        meta["validation"] = _validate_against(scene, out_dir, validate, stride)
    (out_dir / "reference_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[reference] wrote {len(scene.cameras)} maps -> {out_dir}")
    return meta


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rasterise LoD/LiDAR/ALS reference geometry into per-view labels (C2)")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--reference", required=True,
                    help="reference cloud: .npz (xyz/class/instance), .npy, or .ply")
    ap.add_argument("--class", dest="class_path", default=None,
                    help="class array for a bare xyz .npy reference")
    ap.add_argument("--instance", dest="instance_path", default=None,
                    help="instance array for a bare xyz .npy reference")
    ap.add_argument("--sparse_dir", default=None, help="default <dataset_dir>/sparse/0")
    ap.add_argument("--out_dir", default=None,
                    help="default <dataset_dir>/semantic_reference")
    ap.add_argument("--stride", type=int, default=COMPOSITE_STRIDE)
    ap.add_argument("--radius", type=int, default=2, help="label dilation in px")
    ap.add_argument("--taxonomy", default=None, help="JSON {class_id: name}")
    ap.add_argument("--validate", default=None,
                    help="held-out ALS class map (file or per-stem dir) -> confusion summary")
    args = ap.parse_args(argv)

    meta = label_from_reference(
        args.dataset_dir, args.reference, sparse_dir=args.sparse_dir,
        out_dir=args.out_dir, stride=args.stride, radius=args.radius,
        class_path=args.class_path, instance_path=args.instance_path,
        taxonomy=args.taxonomy, validate=args.validate)
    if "validation" in meta:
        print(f"[reference] validation: acc={meta['validation'].get('accuracy')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
