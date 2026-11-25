"""SAM2 **box-prompted** labeler — the masking half of the front-end (§2.2).

``segment_sam2`` runs SAM2's automatic mask generator (AMG): class-agnostic
masks with no class tag. The prompt-driven front-end instead feeds
GroundingDINO's boxes to SAM2 (box-prompt mode, *same weights*) and groups the
returned masks by the box's class phrase. Each frame therefore yields one
instance map **per class**; instance ids restart per frame and are made
cross-view consistent — and composed into the canonical ``(class, instance)``
map — by :func:`worldsplat_prune.prep.fuse_instances.fuse_instances_per_class`.

The **pure** composition (:func:`masks_to_class_maps`) is importable and testable
without SAM2/torch; only :class:`SAM2BoxLabeler` touches the heavy dependencies
(imported lazily).

Colab (run from the repo root)::

    !pip install -q "git+https://github.com/facebookresearch/sam2.git"
    # weights: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
    python -m worldsplat_prune.frontend.sam2_box \\
        --dataset_dir "$DATASET_ROOT/mipnerf360/garden" \\
        --detections  detections.json \\
        --sam2_ckpt   "$HOME_DIR/checkpoints/sam2.1_hiera_large.pt"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..segment_sam2 import IMAGE_EXTS, masks_to_instance_map


# --------------------------------------------------------------------------- #
# pure composition                                                              #
# --------------------------------------------------------------------------- #
def masks_to_class_maps(
    masks_by_class: Mapping[Hashable, Sequence[np.ndarray]],
    scores_by_class: Optional[Mapping[Hashable, Sequence[float]]] = None,
    *,
    min_area: int = 0,
    max_area_frac: float = 1.0,
    fill_void: bool = False,
    out_shape: Optional[Tuple[int, int]] = None,
) -> Dict[Hashable, np.ndarray]:
    """Compose per-class SAM2 masks into one instance map **per class**.

    Each class is composed independently with the greedy, score-ordered
    :func:`worldsplat_prune.segment_sam2.masks_to_instance_map`, so instance ids
    never collide across classes (they are re-based by class later, via the
    composite id). Returns ``{class_key: int32 [H, W] instance map}``; classes
    with no surviving mask are omitted.

    ``fill_void`` defaults to **False** here (unlike the AMG path): box-prompted
    masks are selective, so undetected pixels must stay void — nearest-filling
    them would tile the whole frame with the nearest object of the class.
    """
    scores_by_class = scores_by_class or {}
    out: Dict[Hashable, np.ndarray] = {}
    for cls, masks in masks_by_class.items():
        if masks is None or len(masks) == 0:
            continue
        scores = scores_by_class.get(cls)
        inst = masks_to_instance_map(
            masks, scores, min_area=min_area, max_area_frac=max_area_frac,
            fill_void=fill_void, out_shape=out_shape)
        if inst.max() > 0:
            out[cls] = inst
    return out


# --------------------------------------------------------------------------- #
# SAM2 box-prompt driver (heavy deps imported lazily)                            #
# --------------------------------------------------------------------------- #
def _build_predictor(ckpt: str, cfg: str, device: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(cfg, ckpt, device=device)
    return SAM2ImagePredictor(model)


def _slug(cls: Hashable) -> str:
    """Filesystem-safe directory name for a class phrase."""
    s = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(cls).strip())
    s = "_".join(part for part in s.split("_") if part)
    return s or "class"


class SAM2BoxLabeler:
    """Wrap ``SAM2ImagePredictor`` for repeated box-prompted masks on one frame.

    Usage::

        labeler = SAM2BoxLabeler(ckpt, cfg, device)
        maps = labeler.predict_class_maps(rgb, {"chair": [box_xyxy, ...]}, ...)

    The image embedding is computed once per frame (``set_image``) and reused for
    every box, so N boxes cost one encoder pass plus N mask decodes.
    """

    def __init__(self, sam2_ckpt: str, sam2_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
                 device: str = "cuda", multimask_output: bool = False):
        self.predictor = _build_predictor(sam2_ckpt, sam2_cfg, device)
        self.device = device
        self.multimask_output = multimask_output
        self._image_set = False

    def set_image(self, rgb: np.ndarray) -> None:
        self.predictor.set_image(rgb)
        self._image_set = True

    def predict_boxes(self, boxes: Sequence[Sequence[float]]) -> Tuple[List[np.ndarray], List[float]]:
        """Return ``(masks, scores)`` for xyxy boxes on the current image."""
        if not self._image_set:
            raise RuntimeError("call set_image(rgb) before predict_boxes()")
        masks: List[np.ndarray] = []
        scores: List[float] = []
        for box in boxes:
            m, s, _ = self.predictor.predict(
                box=np.asarray(box, dtype=np.float32),
                multimask_output=self.multimask_output)
            if self.multimask_output:        # pick the best of the 3 proposals
                k = int(np.argmax(s))
                masks.append(np.asarray(m[k], dtype=bool))
                scores.append(float(s[k]))
            else:
                masks.append(np.asarray(m.squeeze(0) if m.ndim == 3 else m, dtype=bool))
                scores.append(float(s) if np.ndim(s) == 0 else float(np.max(s)))
        return masks, scores

    def predict_class_maps(
        self,
        rgb: np.ndarray,
        boxes_by_class: Mapping[Hashable, Sequence[Sequence[float]]],
        *,
        min_area: int = 0,
        max_area_frac: float = 0.9,
        fill_void: bool = False,
    ) -> Dict[Hashable, np.ndarray]:
        """Box-prompt every class's boxes and compose one instance map per class."""
        H, W = rgb.shape[:2]
        self.set_image(rgb)
        masks_by_class: Dict[Hashable, List[np.ndarray]] = {}
        scores_by_class: Dict[Hashable, List[float]] = {}
        for cls, boxes in boxes_by_class.items():
            if boxes is None or len(boxes) == 0:
                continue
            masks, scores = self.predict_boxes(boxes)
            masks_by_class[cls] = masks
            scores_by_class[cls] = scores
        return masks_to_class_maps(
            masks_by_class, scores_by_class, min_area=min_area,
            max_area_frac=max_area_frac, fill_void=fill_void, out_shape=(H, W))


# --------------------------------------------------------------------------- #
# driver over precomputed detections                                            #
# --------------------------------------------------------------------------- #
def label_dataset_boxes(
    dataset_dir: str | Path,
    detections: Mapping[str, Mapping[str, Sequence[Sequence[float]]]],
    *,
    images_subdir: str = "images",
    out_subdir_fmt: str = "semantic_instance_{cls}",
    sam2_ckpt: str,
    sam2_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    device: str = "cuda",
    min_area: int = 100,
    max_area_frac: float = 0.9,
    fill_void: bool = False,
    resize_long_side: Optional[int] = None,
    stride: int = 1,
    overwrite: bool = False,
) -> dict:
    """Label frames from **precomputed** per-frame detections.

    ``detections`` maps an image name (or stem) to ``{class_phrase: [box, ...]}``
    (xyxy, pixels). SAM2 turns each box into a mask and each class's masks become
    a per-frame instance map written to ``semantic_instance_<cls>/<stem>.png``
    (class phrases are slugified for the directory name). Returns a mapping
    ``{class_phrase: relative_out_dir}`` — exactly the ``class_dirs`` argument
    :func:`worldsplat_prune.prep.fuse_instances.fuse_instances_per_class` expects.
    """
    from PIL import Image
    import torch

    dataset_dir = Path(dataset_dir)
    img_dir = dataset_dir / images_subdir
    labeler = SAM2BoxLabeler(sam2_ckpt, sam2_cfg, device)

    class_dirs: Dict[str, str] = {}
    imgs = sorted(p for p in img_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)[::stride]
    for p in imgs:
        det = detections.get(p.name) or detections.get(p.stem)
        if not det:
            continue
        rgb = np.array(Image.open(p).convert("RGB"))
        H, W = rgb.shape[:2]
        work = rgb
        if resize_long_side and max(H, W) > resize_long_side:
            s = resize_long_side / max(H, W)
            work = np.array(Image.fromarray(rgb).resize((int(round(W * s)), int(round(H * s)))))
        # scale boxes into the (possibly resized) working frame
        sx, sy = work.shape[1] / W, work.shape[0] / H
        boxes_scaled = {cls: [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
                              for b in boxes]
                        for cls, boxes in det.items()}
        with torch.inference_mode():
            maps = labeler.predict_class_maps(
                work, boxes_scaled, min_area=min_area,
                max_area_frac=max_area_frac, fill_void=fill_void)
        from .. import semantic_utils as su
        for cls, inst in maps.items():
            if inst.shape != (H, W):        # write maps at native frame resolution
                inst = su.resize_ids(inst, H, W)
            rel = out_subdir_fmt.format(cls=_slug(cls))
            class_dirs.setdefault(str(cls), rel)
            out = dataset_dir / rel / f"{p.stem}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists() and not overwrite:
                continue
            from ..prep.scene_io import write_id_map
            write_id_map(out, inst)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[sam2-box] {p.name}: {len(maps)} class map(s)", flush=True)

    (dataset_dir / "class_dirs.json").write_text(json.dumps(class_dirs, indent=2))
    return class_dirs


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="SAM2 box-prompted labels from detections")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--detections", required=True,
                    help="JSON: {image_name: {class: [[x0,y0,x1,y1], ...]}}")
    ap.add_argument("--images_subdir", default="images")
    ap.add_argument("--out_subdir_fmt", default="semantic_instance_{cls}")
    ap.add_argument("--sam2_ckpt", required=True)
    ap.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min_area", type=int, default=100)
    ap.add_argument("--max_area_frac", type=float, default=0.9)
    ap.add_argument("--resize_long_side", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fill_void", action="store_true",
                    help="nearest-fill undetected pixels (off by default: keep void)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    detections = json.loads(Path(args.detections).read_text())
    label_dataset_boxes(
        args.dataset_dir, detections, images_subdir=args.images_subdir,
        out_subdir_fmt=args.out_subdir_fmt, sam2_ckpt=args.sam2_ckpt,
        sam2_cfg=args.sam2_cfg, device=args.device, min_area=args.min_area,
        max_area_frac=args.max_area_frac, fill_void=args.fill_void,
        resize_long_side=args.resize_long_side, stride=args.stride,
        overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
