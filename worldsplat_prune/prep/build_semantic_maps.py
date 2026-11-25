"""End-to-end prompt-driven label front-end driver (§2.2).

Chains the front-end pieces into the canonical composite ``L_v``:

    phrases ─► GroundingDINO ─► boxes ─► SAM2 (box-prompted) ─► per-frame masks
        ─► per-class fuse_instances ─► consistent instance ids
            ─► composite (class, instance) id map == L_v

Every stage is skippable so the intermediate artifacts (detections JSON,
per-class masks, fused maps) can be inspected or reused:

* ``skip_detect=True``  reuses an existing ``detections.json``;
* ``detect_only=True``  stops after detection (no SAM2 required).

    python -m worldsplat_prune.prep.build_semantic_maps \\
        --dataset_dir "$DATASET_ROOT/mipnerf360/garden" \\
        --phrases phrases.txt \\
        --sam2_ckpt "$HOME_DIR/checkpoints/sam2.1_hiera_large.pt"

Then train/eval with ``--label_adapter composite``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from ..frontend.grounding import DEFAULT_MODEL_ID, detect_dataset, parse_phrases
from ..frontend.sam2_box import label_dataset_boxes
from .fuse_instances import fuse_instances_per_class


def build_semantic_maps(
    dataset_dir: str | Path,
    phrases,
    *,
    images_subdir: str = "images",
    detections_json: str | Path = "detections.json",
    class_subdir_fmt: str = "semantic_instance_{cls}",
    fused_dir: str | Path = "semantic_instance_fused",
    composite_dir: str | Path = "semantic_composite",
    sam2_ckpt: str,
    sam2_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    device: str = "cuda",
    det_model_id: str = DEFAULT_MODEL_ID,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    min_area: int = 100,
    max_area_frac: float = 0.9,
    fill_void: bool = True,
    resize_long_side: Optional[int] = None,
    metric: str = "jaccard",
    min_score: float = 0.25,
    min_views: int = 2,
    color_sim: float = 0.0,
    stride: int = 1,
    overwrite: bool = False,
    skip_detect: bool = False,
    detect_only: bool = False,
) -> dict:
    """Run the front-end and write composite maps. Returns a summary dict."""
    dataset_dir = Path(dataset_dir)
    phrases = parse_phrases(phrases)
    if not phrases:
        raise ValueError("empty phrase list")

    det_path = detections_json if Path(detections_json).is_absolute() else dataset_dir / detections_json
    if skip_detect and det_path.exists():
        detections = json.loads(det_path.read_text())
        print(f"[frontend] detections <- {det_path}", flush=True)
    else:
        detections = detect_dataset(
            dataset_dir, phrases, det_path, images_subdir=images_subdir,
            model_id=det_model_id, device=device, box_threshold=box_threshold,
            text_threshold=text_threshold, stride=stride,
            resize_long_side=resize_long_side)

    if detect_only:
        print("[frontend] detect_only: stopping after detection", flush=True)
        return {"detections_json": str(det_path), "phrases": phrases,
                "detect_only": True}

    class_dirs = label_dataset_boxes(
        dataset_dir, detections, images_subdir=images_subdir,
        out_subdir_fmt=class_subdir_fmt, sam2_ckpt=sam2_ckpt, sam2_cfg=sam2_cfg,
        device=device, min_area=min_area, max_area_frac=max_area_frac,
        fill_void=fill_void, resize_long_side=resize_long_side, stride=stride,
        overwrite=overwrite)
    if not class_dirs:
        raise SystemExit("no class maps were produced (empty detections?)")

    # Class ids follow the curated phrase order, so they are stable even when a
    # phrase produced no detections this run.
    class_ids = {p: i + 1 for i, p in enumerate(phrases)}
    summary = fuse_instances_per_class(
        dataset_dir, class_dirs, class_ids=class_ids, out_dir=fused_dir,
        composite_out_dir=composite_dir, images_subdir=images_subdir,
        metric=metric, min_score=min_score, min_views=min_views,
        color_sim=color_sim)

    summary["detections_json"] = str(det_path)
    summary["phrases"] = phrases
    print(f"[frontend] done -> {summary['composite_out_dir']} "
          f"(use --label_adapter composite)", flush=True)
    return summary


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Prompt-driven label front-end (Grounded-SAM-2)")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--phrases", required=True,
                    help="phrase file (one per line) or an 'a. b. c.' string")
    ap.add_argument("--images_subdir", default="images")
    ap.add_argument("--detections_json", default="detections.json")
    ap.add_argument("--class_subdir_fmt", default="semantic_instance_{cls}")
    ap.add_argument("--fused_dir", default="semantic_instance_fused")
    ap.add_argument("--composite_dir", default="semantic_composite")
    ap.add_argument("--sam2_ckpt", default=None)
    ap.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--det_model_id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--box_threshold", type=float, default=0.3)
    ap.add_argument("--text_threshold", type=float, default=0.25)
    ap.add_argument("--min_area", type=int, default=100)
    ap.add_argument("--max_area_frac", type=float, default=0.9)
    ap.add_argument("--resize_long_side", type=int, default=None)
    ap.add_argument("--metric", default="jaccard", choices=["jaccard", "overlap", "shared"])
    ap.add_argument("--min_score", type=float, default=0.25)
    ap.add_argument("--min_views", type=int, default=2)
    ap.add_argument("--color_sim", type=float, default=0.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no_fill_void", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip_detect", action="store_true")
    ap.add_argument("--detect_only", action="store_true")
    args = ap.parse_args(argv)

    if not args.detect_only and not args.sam2_ckpt:
        ap.error("--sam2_ckpt is required unless --detect_only")

    build_semantic_maps(
        args.dataset_dir, args.phrases, images_subdir=args.images_subdir,
        detections_json=args.detections_json, class_subdir_fmt=args.class_subdir_fmt,
        fused_dir=args.fused_dir, composite_dir=args.composite_dir,
        sam2_ckpt=args.sam2_ckpt or "", sam2_cfg=args.sam2_cfg, device=args.device,
        det_model_id=args.det_model_id, box_threshold=args.box_threshold,
        text_threshold=args.text_threshold, min_area=args.min_area,
        max_area_frac=args.max_area_frac, fill_void=not args.no_fill_void,
        resize_long_side=args.resize_long_side, metric=args.metric,
        min_score=args.min_score, min_views=args.min_views, color_sim=args.color_sim,
        stride=args.stride, overwrite=args.overwrite,
        skip_detect=args.skip_detect, detect_only=args.detect_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
