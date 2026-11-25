"""Captioner → phrase-list tool (open-vocabulary seed, §2.2).

D15: the phrase list is seeded by an **offline** MLLM auto-captioner over a
sample of frames and then **curated by a human** — there is no hand-written
taxonomy. This module provides:

* a pluggable :class:`Captioner` interface with a small HuggingFace backend as
  the default (:class:`HFCaptioner`) and a wrapper for any callable/API
  (:class:`CallableCaptioner`);
* :func:`extract_phrases`, a pure heuristic that turns a free-text caption into
  candidate noun phrases (unigrams + adjacent bigrams, stop-word filtered);
* :func:`seed_phrase_list`, which samples frames, captions them, merges the
  candidates and (optionally) writes the seed list for human review.

The result is a de-duplicated, ordered phrase list. It is a *seed*: the curated
list is what :mod:`worldsplat_prune.frontend.grounding` consumes.

    python -m worldsplat_prune.frontend.caption \\
        --dataset_dir "$DATASET_ROOT/mipnerf360/garden" \\
        --out phrases_seed.txt --n 16
"""
from __future__ import annotations

import argparse
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from ..segment_sam2 import IMAGE_EXTS

DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"

# Minimal stop-word set; only content words become candidate phrases.
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "by", "with",
    "for", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "there", "here", "which",
    "who", "whom", "whose", "what", "where", "when", "why", "how", "some",
    "any", "each", "both", "all", "no", "not", "very", "then", "than", "into",
    "over", "under", "above", "below", "between", "next", "near", "front",
    "left", "right", "top", "bottom", "standing", "sitting", "large", "small",
    "his", "her", "their", "our", "your", "my", "image", "photo", "picture",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------- #
# pure helper                                                                   #
# --------------------------------------------------------------------------- #
def extract_phrases(text: str, *, max_words: int = 2) -> List[str]:
    """Candidate noun phrases from a caption: content unigrams + adjacent bigrams.

    Deterministic and order-preserving; ``max_words`` caps n-gram size (1 or 2).
    The output is a *seed* — noisy terms are expected and curated by hand.
    """
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t]
    out: List[str] = []
    seen = set()

    def add(phrase: str) -> None:
        phrase = phrase.strip()
        if phrase and phrase not in seen:
            seen.add(phrase)
            out.append(phrase)

    for i, tok in enumerate(tokens):
        if tok in STOPWORDS or len(tok) < 3 or tok.isdigit():
            continue
        add(tok)
        if max_words >= 2 and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt not in STOPWORDS and len(nxt) >= 3 and not nxt.isdigit():
                add(f"{tok} {nxt}")
    return out


def merge_phrase_lists(lists: Sequence[Sequence[str]]) -> List[str]:
    """Union phrase lists, de-duplicated, order of first appearance."""
    out: List[str] = []
    seen = set()
    for lst in lists:
        for p in lst:
            p = str(p).strip().lower()
            if p and p not in seen:
                seen.add(p)
                out.append(p)
    return out


# --------------------------------------------------------------------------- #
# captioner interface                                                           #
# --------------------------------------------------------------------------- #
class Captioner(ABC):
    """Anything that turns an RGB frame into a caption string."""

    @abstractmethod
    def caption(self, rgb: np.ndarray) -> str:
        raise NotImplementedError


class CallableCaptioner(Captioner):
    """Adapt a plain ``fn(rgb) -> str`` (e.g. an API client) to :class:`Captioner`."""

    def __init__(self, fn: Callable[[np.ndarray], str]):
        self.fn = fn

    def caption(self, rgb: np.ndarray) -> str:
        return str(self.fn(rgb))


class HFCaptioner(Captioner):
    """HuggingFace ``image-to-text`` pipeline backend (default: BLIP)."""

    def __init__(self, model_id: str = DEFAULT_CAPTION_MODEL, device: str = "cuda",
                 prompt: Optional[str] = None):
        from transformers import pipeline

        self.model_id = model_id
        self.prompt = prompt
        self.pipe = pipeline("image-to-text", model=model_id,
                             device=0 if str(device).startswith("cuda") else -1)

    def caption(self, rgb: np.ndarray) -> str:
        from PIL import Image

        kw = {"prompt": self.prompt} if self.prompt else {}
        res = self.pipe(Image.fromarray(rgb), **kw)
        if isinstance(res, list) and res:
            return str(res[0].get("generated_text", ""))
        return ""


# --------------------------------------------------------------------------- #
# seeding driver                                                                #
# --------------------------------------------------------------------------- #
def seed_phrase_list(
    dataset_dir: str | Path,
    captioner: Captioner,
    *,
    images_subdir: str = "images",
    n: int = 16,
    stride: Optional[int] = None,
    max_words: int = 2,
    out: Optional[str | Path] = None,
) -> List[str]:
    """Caption a frame sample and return the merged candidate phrase list.

    ``stride=None`` samples ``n`` frames evenly across the sorted image list;
    ``stride`` overrides the sampling step (and then ``n`` is ignored).
    """
    from PIL import Image

    dataset_dir = Path(dataset_dir)
    img_dir = dataset_dir / images_subdir
    imgs = sorted(p for p in img_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not imgs:
        raise FileNotFoundError(f"no images under {img_dir}")
    if stride and stride > 0:
        sample = imgs[::stride]
    else:
        step = max(1, len(imgs) // max(1, n))
        sample = imgs[::step][:n]

    per_frame: List[List[str]] = []
    for i, p in enumerate(sample):
        rgb = np.array(Image.open(p).convert("RGB"))
        cap = captioner.caption(rgb)
        per_frame.append(extract_phrases(cap, max_words=max_words))
        print(f"[caption] {i + 1}/{len(sample)} {p.name}: {cap!r}", flush=True)

    phrases = merge_phrase_lists(per_frame)
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(phrases) + "\n")
        print(f"[caption] {len(phrases)} seed phrases -> {out} (curate before use)",
              flush=True)
    return phrases


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Seed a phrase list from frame captions")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out", required=True, help="phrase seed file to write")
    ap.add_argument("--images_subdir", default="images")
    ap.add_argument("--model_id", default=DEFAULT_CAPTION_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--prompt", default=None,
                    help="optional captioning prompt for VLM backends")
    args = ap.parse_args(argv)

    captioner = HFCaptioner(args.model_id, args.device, args.prompt)
    seed_phrase_list(args.dataset_dir, captioner, images_subdir=args.images_subdir,
                     n=args.n, stride=args.stride, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
