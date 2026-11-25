"""GroundingDINO wrapper — text phrases → boxes (open-vocabulary detector).

The front-end (§2.2) is **prompt-driven / open-vocabulary**: a curated phrase
list is run through GroundingDINO (text→boxes), and each box is handed to SAM2 in
box-prompt mode (:mod:`worldsplat_prune.frontend.sam2_box`). Class ids are the
phrase-list order — there is no fixed taxonomy.

The **pure** helpers (phrase parsing / prompt formatting / label matching) are
importable and testable without torch/transformers; only
:class:`GroundingDinoDetector` imports the model (lazily).

Backend: HuggingFace ``transformers`` (``IDEA-Research/grounding-dino-tiny`` or
``-base``). The post-processing API moved between transformers releases, so the
call site tolerates both the ``box_threshold=`` and older ``threshold=`` keyword.

    python -m worldsplat_prune.frontend.grounding \\
        --dataset_dir "$DATASET_ROOT/mipnerf360/garden" \\
        --phrases phrases.txt --out detections.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..segment_sam2 import IMAGE_EXTS

DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


# --------------------------------------------------------------------------- #
# pure helpers                                                                  #
# --------------------------------------------------------------------------- #
def normalize_phrase(p: str) -> str:
    """Lower-case, trim, and drop a trailing period (GroundingDINO style)."""
    return str(p).strip().lower().rstrip(".").strip()


def parse_phrases(phrases) -> List[str]:
    """De-duplicate phrases (order-preserving) and normalise them.

    Accepts a list/tuple of phrases, a newline-separated string, or a path to a
    phrase file (one phrase per line; ``#`` comments allowed).
    """
    if isinstance(phrases, (list, tuple)):
        raw = list(phrases)
    else:
        s = str(phrases)
        p = Path(s)
        text = p.read_text() if (len(s) < 4096 and p.exists() and p.is_file()) else s
        raw = text.splitlines()
    out: List[str] = []
    seen = set()
    for line in raw:
        line = line.split("#", 1)[0]
        norm = normalize_phrase(line)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def format_text_prompt(phrases: Sequence[str]) -> str:
    """GroundingDINO text prompt: ``"chair. table. lamp."``."""
    return ". ".join(normalize_phrase(p) for p in phrases if str(p).strip()) + "."


def match_label_to_phrase(label: str, phrases: Sequence[str]) -> Optional[str]:
    """Map a detector label back to a phrase: exact, else substring, else ``None``."""
    key = normalize_phrase(label)
    norm = {normalize_phrase(p): p for p in phrases}
    if key in norm:
        return norm[key]
    for pn, p in norm.items():
        if pn and (pn in key or key in pn):
            return p
    return None


# --------------------------------------------------------------------------- #
# detector (heavy deps imported lazily)                                         #
# --------------------------------------------------------------------------- #
def _load_detector(model_id: str, device: str):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


class GroundingDinoDetector:
    """Open-vocabulary detector: ``detect(rgb, phrases) -> {phrase: [box_xyxy]}``."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "cuda",
                 box_threshold: float = 0.3, text_threshold: float = 0.25):
        self.model_id = model_id
        self.device = device
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.processor, self.model = _load_detector(model_id, device)

    def detect(self, rgb: np.ndarray, phrases: Sequence[str]) -> Dict[str, List[List[float]]]:
        """Return ``{phrase: [[x0, y0, x1, y1], ...]}`` (pixel xyxy) for one frame."""
        import torch
        from PIL import Image

        phrases = parse_phrases(list(phrases))
        out: Dict[str, List[List[float]]] = {p: [] for p in phrases}
        if not phrases:
            return out

        image = Image.fromarray(rgb)
        text = format_text_prompt(phrases)
        inputs = self.processor(images=image, text=text,
                                return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        H, W = rgb.shape[:2]
        kwargs = dict(input_ids=inputs.input_ids,
                      text_threshold=self.text_threshold, target_sizes=[(H, W)])
        try:
            res = self.processor.post_process_grounded_object_detection(
                outputs, box_threshold=self.box_threshold, **kwargs)[0]
        except TypeError:      # older transformers used `threshold=`
            res = self.processor.post_process_grounded_object_detection(
                outputs, threshold=self.box_threshold, **kwargs)[0]

        labels = res.get("labels", res.get("text_labels", []))
        for box, score, label in zip(res["boxes"], res["scores"], labels):
            phrase = match_label_to_phrase(label, phrases)
            if phrase is None:
                continue
            out[phrase].append([float(v) for v in box.tolist()])
        return out


# --------------------------------------------------------------------------- #
# dataset driver                                                                #
# --------------------------------------------------------------------------- #
def detect_dataset(
    dataset_dir: str | Path,
    phrases: Sequence[str],
    out_json: str | Path,
    *,
    images_subdir: str = "images",
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cuda",
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    stride: int = 1,
    resize_long_side: Optional[int] = None,
) -> dict:
    """Run the detector over every frame; write ``{image_name: {phrase: boxes}}``.

    Returns the detections dict (also written to ``out_json``). Boxes are in the
    frame's native pixel coordinates (scaled back up if ``resize_long_side`` was
    used for inference).
    """
    from PIL import Image

    dataset_dir = Path(dataset_dir)
    phrases = parse_phrases(phrases)
    detector = GroundingDinoDetector(model_id, device, box_threshold, text_threshold)

    img_dir = dataset_dir / images_subdir
    imgs = sorted(p for p in img_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)[::stride]
    detections: Dict[str, dict] = {}
    for i, p in enumerate(imgs):
        rgb = np.array(Image.open(p).convert("RGB"))
        H, W = rgb.shape[:2]
        work, sx, sy = rgb, 1.0, 1.0
        if resize_long_side and max(H, W) > resize_long_side:
            s = resize_long_side / max(H, W)
            work = np.array(Image.fromarray(rgb).resize((int(round(W * s)), int(round(H * s)))))
            sx, sy = W / work.shape[1], H / work.shape[0]
        det = detector.detect(work, phrases)
        det = {ph: [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes]
               for ph, boxes in det.items()}
        detections[p.name] = det
        print(f"[grounding] {i + 1}/{len(imgs)} {p.name}: "
              f"{sum(len(v) for v in det.values())} boxes", flush=True)

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(detections, indent=2))
    print(f"[grounding] wrote {out_json}", flush=True)
    return detections


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="GroundingDINO phrase->boxes over a dataset")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--phrases", required=True,
                    help="phrase file (one per line) or a 'a. b. c.' string")
    ap.add_argument("--out", required=True, help="detections JSON output path")
    ap.add_argument("--images_subdir", default="images")
    ap.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--box_threshold", type=float, default=0.3)
    ap.add_argument("--text_threshold", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--resize_long_side", type=int, default=None)
    args = ap.parse_args(argv)

    detect_dataset(
        args.dataset_dir, args.phrases, args.out,
        images_subdir=args.images_subdir, model_id=args.model_id, device=args.device,
        box_threshold=args.box_threshold, text_threshold=args.text_threshold,
        stride=args.stride, resize_long_side=args.resize_long_side)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
