"""Offline SAM2 pseudo-labelling for non-segmented datasets.

Mip-NeRF 360, Tanks & Temples, Deep Blending and custom captures ship no
ground-truth semantics. WorldSplat-Prune only needs *a* per-view segment map, so we run a
class-agnostic segmenter once, offline, and cache the result on disk in the
layout :class:`worldsplat_prune.datasets.instance_dir.InstanceDirLabelAdapter` consumes::

    <dataset>/images/<stem>.{png,jpg,jpeg,...}      # input RGB
    <dataset>/semantic_instance/<stem>.png          # uint16 instance ids, 0=void

SAM2 is class-agnostic, so the ids are **instances**, not classes -- exactly the
unit Component B budgets over (see ``docs/worldsplat_prune_spec.md`` §2.0). Masks are
composed greedily by predicted IoU and stability, then uncovered pixels are
nearest-filled so they do not masquerade as boundaries.

This module is deliberately split so the *pure* post-processing
(:func:`masks_to_instance_map`) is importable and testable without SAM2/torch;
only :func:`label_dataset` touches the heavy dependencies (lazy imports).

Colab (run from the repo root):

    !pip install -q "git+https://github.com/facebookresearch/sam2.git"
    # weights: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
    python -m worldsplat_prune.segment_sam2 \\
        --dataset_dir "$DATASET_ROOT/mipnerf360/garden" \\
        --sam2_ckpt   "$HOME_DIR/checkpoints/sam2.1_hiera_large.pt" \\
        --sam2_cfg    configs/sam2.1/sam2.1_hiera_l.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image

from . import semantic_utils as su

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# pure post-processing (no SAM2 / torch)                                        #
# --------------------------------------------------------------------------- #
def masks_to_instance_map(
    masks: Sequence[np.ndarray],
    scores: Optional[Sequence[float]] = None,
    *,
    min_area: int = 0,
    max_area_frac: float = 1.0,
    fill_void: bool = True,
    out_shape: Optional[tuple] = None,
) -> np.ndarray:
    """Compose SAM2 masks into a single ``int32 [H, W]`` instance map.

    Masks are painted in **descending score** order, so where several masks
    overlap the most confident one wins. Each surviving mask becomes a distinct
    id starting at 1 (``0 == void``).

    Parameters
    ----------
    masks       : boolean ``[H, W]`` masks (already the chosen SAM2 candidates).
    scores      : per-mask confidence (defaults to input order).
    min_area    : drop masks with fewer than this many pixels.
    max_area_frac : drop masks covering more than this fraction of the frame
                    (SAM2 occasionally emits a whole-image "background" mask).
    fill_void   : nearest-fill pixels no mask covered (recommended; see module
                    docstring).
    out_shape   : ``(H, W)`` to resize each mask to (nearest), else inferred.
    """
    masks = [np.asarray(m, dtype=bool) for m in masks]
    if out_shape is None:
        if not masks:
            raise ValueError("no masks and no out_shape to infer H/W from")
        out_shape = masks[0].shape
    H, W = int(out_shape[0]), int(out_shape[1])
    if scores is None:
        scores = list(range(len(masks), 0, -1))  # stable: earlier masks first
    order = sorted(range(len(masks)), key=lambda i: -float(scores[i]))

    area_cap = max_area_frac * H * W
    inst = np.zeros((H, W), dtype=np.int32)
    next_id = 1
    for i in order:
        m = masks[i]
        if m.shape != (H, W):
            m = su.resize_ids(m.astype(np.int32), H, W).astype(bool)
        area = int(m.sum())
        if area < max(min_area, 1) or area > area_cap:
            continue
        paint = m & (inst == 0)          # higher-score masks keep their pixels
        if not paint.any():
            continue
        inst[paint] = next_id
        next_id += 1

    if fill_void:
        inst = su.fill_void_nearest(inst)
    return inst


def _save_ids(path: Path, ids: np.ndarray) -> None:
    """Write an id map as a 16-bit PNG (supports > 255 instances/frame)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(ids).astype(np.uint16)
    Image.fromarray(arr).save(path)


# --------------------------------------------------------------------------- #
# SAM2 driver (heavy deps imported lazily)                                      #
# --------------------------------------------------------------------------- #
def _build_generator(ckpt: str, cfg: str, device: str,
                     points_per_side: int, pred_iou_thresh: float,
                     stability_score_thresh: float, points_per_batch: int = 64):
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    model = build_sam2(cfg, ckpt, device=device)
    return SAM2AutomaticMaskGenerator(
        model,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
    )


def label_dataset(
    dataset_dir: str | Path,
    *,
    images_subdir: str = "images",
    out_subdir: str = "semantic_instance",
    sam2_ckpt: str,
    sam2_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    device: str = "cuda",
    points_per_side: int = 32,
    points_per_batch: int = 64,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    min_area: int = 100,
    max_area_frac: float = 0.9,
    fill_void: bool = True,
    resize_long_side: Optional[int] = None,
    stride: int = 1,
    overwrite: bool = False,
) -> int:
    """Label every frame under ``dataset_dir/images`` into ``semantic_instance``.

    Returns the number of frames written. Existing labels are skipped unless
    ``overwrite``.

    Memory knobs (SAM2 AMG is the peak): ``resize_long_side`` caps the working
    resolution (ids are written back at native size), ``points_per_side`` sets the
    prompt grid (cost ~ O(n^2): 64 is 4x 32), and ``points_per_batch`` caps how
    many prompts are decoded at once. CUDA cache is released after every frame.
    """
    dataset_dir = Path(dataset_dir)
    img_dir = dataset_dir / images_subdir
    out_dir = dataset_dir / out_subdir
    imgs = sorted(p for p in img_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not imgs:
        raise FileNotFoundError(f"no images under {img_dir}")

    gen = _build_generator(sam2_ckpt, sam2_cfg, device, points_per_side,
                           pred_iou_thresh, stability_score_thresh, points_per_batch)

    import torch

    written = 0
    for k, p in enumerate(imgs[::stride]):
        outp = out_dir / f"{p.stem}.png"
        if outp.exists() and not overwrite:
            continue
        rgb = np.array(Image.open(p).convert("RGB"))
        H, W = rgb.shape[:2]
        work = rgb
        if resize_long_side and max(H, W) > resize_long_side:
            s = resize_long_side / max(H, W)
            work = np.array(Image.fromarray(rgb).resize((int(round(W * s)), int(round(H * s)))))

        with torch.inference_mode():
            raw = gen.generate(work)
        masks = [m["segmentation"] for m in raw]
        scores = [float(m.get("predicted_iou", 0.0)) + float(m.get("stability_score", 0.0))
                  for m in raw]
        del raw
        inst = masks_to_instance_map(
            masks, scores, min_area=min_area, max_area_frac=max_area_frac,
            fill_void=fill_void, out_shape=(H, W))
        _save_ids(outp, inst)
        written += 1
        if device.startswith("cuda"):
            torch.cuda.empty_cache()      # AMG peaks per image; release before the next
        print(f"[sam2] {k + 1}/{len(imgs[::stride])} {p.name}: "
              f"{int(inst.max())} instances -> {outp}", flush=True)

    print(f"[sam2] wrote {written} label map(s) to {out_dir}")
    return written


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="SAM2 offline instance pseudo-labels for WorldSplat-Prune")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--images_subdir", default="images")
    ap.add_argument("--out_subdir", default="semantic_instance")
    ap.add_argument("--sam2_ckpt", required=True)
    ap.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--points_per_side", type=int, default=32)
    ap.add_argument("--points_per_batch", type=int, default=64,
                    help="AMG prompt batch size; lower it if SAM2 OOMs")
    ap.add_argument("--pred_iou_thresh", type=float, default=0.88)
    ap.add_argument("--stability_score_thresh", type=float, default=0.95)
    ap.add_argument("--min_area", type=int, default=100)
    ap.add_argument("--max_area_frac", type=float, default=0.9)
    ap.add_argument("--resize_long_side", type=int, default=None,
                    help="cap the working resolution (e.g. 1024) to save memory")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no_fill_void", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    label_dataset(
        args.dataset_dir,
        images_subdir=args.images_subdir, out_subdir=args.out_subdir,
        sam2_ckpt=args.sam2_ckpt, sam2_cfg=args.sam2_cfg, device=args.device,
        points_per_side=args.points_per_side, points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_area=args.min_area, max_area_frac=args.max_area_frac,
        fill_void=not args.no_fill_void, resize_long_side=args.resize_long_side,
        stride=args.stride, overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
