"""3D instance-segmentation metrics for the label-fusion comparison.

Scores a per-3D-point **predicted** instance labeling against a per-3D-point
**ground-truth** labeling: AP / AP50 / AP25, matched mean-IoU, and coverage.

The GT is built the same way for every method: Replica's per-view
``semantic_instance`` maps are lifted onto the COLMAP points by **majority vote**
(:func:`worldsplat_prune.prep.scene_io.point_labels_from_views`). Predictions follow the same
substrate -- either a method's ``point_labels.npy`` (our fusion, SAMPro3D,
MaskClustering, ...) or its per-view maps. Because every method is scored on the
identical point set with the identical GT, the comparison is apples-to-apples.

Metrics follow the standard class-agnostic 3D-instance protocol: predictions are
ranked by mask size (no confidence scores are available zero-shot), greedily
matched to the best-IoU unmatched GT instance at threshold ``tau``, and AP is the
area under the interpolated precision-recall curve.

    python -m worldsplat_prune.prep.eval_instances --dataset_dir <replica_scene> \
        --gt semantic_instance_gt \
        --method fused=semantic_instance_fused \
        --method raw=semantic_instance:first
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import scene_io as sio


# --------------------------------------------------------------------------- #
# metrics                                                                       #
# --------------------------------------------------------------------------- #
def _dense(labels: np.ndarray) -> np.ndarray:
    """Relabel ``>0`` ids to a dense ``1..K`` range (0 stays void)."""
    labels = np.asarray(labels)
    out = np.zeros(labels.shape, dtype=np.int64)
    u = np.unique(labels)
    u = u[u > 0]
    if u.size == 0:
        return out
    remap = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    remap[u] = np.arange(1, u.size + 1)
    return remap[labels]


def _drop_small(labels: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return labels
    labels = np.asarray(labels)
    u, c = np.unique(labels[labels > 0], return_counts=True)
    small = u[c < min_area]
    if small.size:
        labels = labels.copy()
        labels[np.isin(labels, small)] = 0
    return labels


def contingency(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersection table ``M[p, g]`` = #points with pred ``p`` and gt ``g``.

    Index 0 is void/unlabeled. ``pred_sizes``/``gt_sizes`` are the instance sizes
    in the *full* labelings (a point labelled by only one side still counts toward
    that side's size, so the IoU union is correct). Returns ``(M, pred_sizes, gt_sizes)``.
    """
    n_p = int(pred.max()) if pred.size else 0
    n_g = int(gt.max()) if gt.size else 0
    M = np.zeros((n_p + 1, n_g + 1), dtype=np.int64)
    valid = (pred > 0) & (gt > 0)
    if valid.any():
        flat = pred[valid].astype(np.int64) * (n_g + 1) + gt[valid]
        M += np.bincount(flat, minlength=(n_p + 1) * (n_g + 1)).reshape(n_p + 1, n_g + 1)
    pred_sizes = np.bincount(pred[pred > 0].astype(np.int64), minlength=n_p + 1)
    gt_sizes = np.bincount(gt[gt > 0].astype(np.int64), minlength=n_g + 1)
    return M, pred_sizes, gt_sizes


def iou_matrix(M: np.ndarray, pred_sizes: np.ndarray, gt_sizes: np.ndarray) -> np.ndarray:
    inter = M.astype(np.float64)
    union = pred_sizes[:, None] + gt_sizes[None, :] - M
    return inter / np.clip(union, 1.0, None)


def _interp_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def average_precision(iou: np.ndarray, pred_sizes: np.ndarray, gt_sizes: np.ndarray,
                      thresh: float) -> dict:
    """Greedy size-ranked matching AP at ``thresh``. Returns AP + matched pairs."""
    n_p = iou.shape[0] - 1
    n_gt = int((gt_sizes > 0).sum())
    if n_gt == 0:
        return {"ap": 0.0, "pairs": [], "ious": [], "precision": [], "recall": []}

    order = np.argsort(-pred_sizes[1:])            # rank preds by size desc
    matched = np.zeros(iou.shape[1], dtype=bool)
    tp: List[int] = []
    fp: List[int] = []
    pairs: List[Tuple[int, int]] = []
    mious: List[float] = []
    for pi in order:
        lab = pi + 1
        if pred_sizes[lab] <= 0:
            continue
        cand = np.nonzero(~matched[1:])[0]
        if cand.size == 0:
            tp.append(0); fp.append(1); continue
        row = iou[lab, 1:][cand]
        k = int(np.argmax(row))
        j = int(cand[k])
        if row[k] >= thresh:
            matched[1 + j] = True
            tp.append(1); fp.append(0)
            pairs.append((lab, 1 + j)); mious.append(float(row[k]))
        else:
            tp.append(0); fp.append(1)

    tp_a = np.asarray(tp); fp_a = np.asarray(fp)
    if tp_a.size == 0:
        return {"ap": 0.0, "pairs": pairs, "ious": mious, "precision": [], "recall": []}
    tp_c = np.cumsum(tp_a); fp_c = np.cumsum(fp_a)
    recall = tp_c / n_gt
    precision = tp_c / np.maximum(tp_c + fp_c, 1)
    return {"ap": _interp_ap(recall, precision),
            "pairs": pairs, "ious": mious,
            "precision": precision.tolist(), "recall": recall.tolist()}


def evaluate_instance_labels(pred: np.ndarray, gt: np.ndarray,
                             thresholds: Sequence[float] = (0.25, 0.5),
                             min_area: int = 0) -> dict:
    """AP@taus, mAP, matched mIoU and coverage for one predicted labeling."""
    pred = _dense(_drop_small(np.asarray(pred), min_area))
    gt = _dense(_drop_small(np.asarray(gt), min_area))
    M, sp, sg = contingency(pred, gt)
    iou = iou_matrix(M, sp, sg)
    n_gt = int((sg > 0).sum())
    n_pred = int((sp > 0).sum())

    per_thresh = {}
    for t in thresholds:
        r = average_precision(iou, sp, sg, t)
        per_thresh[t] = r
    aps = {f"AP{int(round(t * 100))}": per_thresh[t]["ap"] for t in thresholds}
    r50 = per_thresh[max(thresholds)]
    matched = r50["pairs"]
    return {
        "n_pred": n_pred,
        "n_gt": n_gt,
        "ap": aps,
        "map": float(np.mean([per_thresh[t]["ap"] for t in thresholds])),
        "coverage": len(matched) / max(n_gt, 1),
        "miou_matched": float(np.mean(r50["ious"])) if r50["ious"] else 0.0,
    }


# --------------------------------------------------------------------------- #
# driver                                                                        #
# --------------------------------------------------------------------------- #
def _load_labels(scene: sio.ColmapScene, spec: str, base: Optional[Path] = None) -> np.ndarray:
    """``spec`` is ``/path/to/point_labels.npy`` or ``<dir>[:first|:majority]``.

    Relative paths resolve against ``base`` (the dataset dir). The mode suffix is
    recognised by *suffix* (not the first colon) so Windows drive letters work.
    """
    mode = "majority"
    explicit = False
    for m in ("first", "majority"):
        if spec.endswith(":" + m):
            path = spec[: -(len(m) + 1)]
            mode = m
            explicit = True
            break
    else:
        path = spec
    p = Path(path)
    if not p.is_absolute() and base is not None:
        p = base / p
    if p.suffix == ".npy":
        return np.load(p).astype(np.int32)
    npy = p / "point_labels.npy"
    if npy.exists() and not explicit:
        return np.load(npy).astype(np.int32)
    return sio.point_labels_from_views(scene, p, mode=mode)


def run_eval(dataset_dir: str | Path, *, gt: str, methods: Dict[str, str],
             sparse_subdir: str = "sparse/0", thresholds: Sequence[float] = (0.25, 0.5),
             min_area: int = 0, gt_npy: Optional[str] = None) -> dict:
    dataset_dir = Path(dataset_dir)
    scene = sio.load_colmap_scene(dataset_dir / sparse_subdir)

    if gt_npy:
        gt_labels = np.load(gt_npy).astype(np.int32)
    else:
        gt_p = Path(gt)
        gt_p = gt_p if gt_p.is_absolute() else dataset_dir / gt_p
        gt_labels = sio.point_labels_from_views(scene, gt_p, mode="majority")
    if (gt_labels > 0).sum() == 0:
        raise SystemExit("GT has no labeled points (check --gt path / mode)")
    print(f"[eval] {scene.xyz.shape[0]} points | GT: "
          f"{len(np.unique(gt_labels[gt_labels > 0]))} instances, "
          f"{(gt_labels > 0).sum()} labeled points", flush=True)

    header = (f"{'method':<16} {'n_pred':>7} {'AP25':>6} {'AP50':>6} {'mAP':>6} "
              f"{'mIoU*':>6} {'cov@50':>7}")
    print("\n" + header)
    print("-" * len(header))
    report = {"points": int(scene.xyz.shape[0]), "gt_instances":
              int(len(np.unique(gt_labels[gt_labels > 0]))), "methods": {}}
    for name, spec in methods.items():
        pred = _load_labels(scene, spec, base=dataset_dir)
        res = evaluate_instance_labels(pred, gt_labels, thresholds=thresholds,
                                       min_area=min_area)
        print(f"{name:<16} {res['n_pred']:>7} {res['ap'].get('AP25', 0):>6.3f} "
              f"{res['ap'].get('AP50', 0):>6.3f} {res['map']:>6.3f} "
              f"{res['miou_matched']:>6.3f} {res['coverage']:>7.3f}")
        report["methods"][name] = res
    print("\n* mIoU over matched pairs at AP50. Predictions ranked by mask size.")
    return report


def _parse_methods(items: Sequence[str]) -> Dict[str, str]:
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--method must be name=path (got {it!r})")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="3D instance metrics for the fusion comparison")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--gt", default="semantic_instance_gt",
                    help="GT point-labels .npy, or a dir of per-view instance maps")
    ap.add_argument("--gt_npy", default=None, help="explicit GT point-labels .npy")
    ap.add_argument("--method", action="append", default=[],
                    metavar="NAME=PATH[:mode]",
                    help="repeatable; PATH is a point_labels.npy or a per-view dir "
                         "(optional :first|:majority)")
    ap.add_argument("--sparse_subdir", default="sparse/0")
    ap.add_argument("--iou", nargs="+", type=float, default=[0.25, 0.5])
    ap.add_argument("--min_area", type=int, default=0)
    ap.add_argument("--json_out", default="")
    args = ap.parse_args(argv)
    methods = _parse_methods(args.method)
    if not methods:
        raise SystemExit("pass at least one --method NAME=PATH")
    rep = run_eval(args.dataset_dir, gt=args.gt, methods=methods,
                   sparse_subdir=args.sparse_subdir, thresholds=tuple(args.iou),
                   min_area=args.min_area, gt_npy=args.gt_npy)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rep, indent=2))
        print(f"[eval] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
