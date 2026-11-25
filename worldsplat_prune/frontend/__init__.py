"""Prompt-driven segmentation front-end (Grounded-SAM-2).

Modules
-------
``sam2_box``   : SAM2 **box-prompted** labeler (the masking half of §2.2). The
                 existing :mod:`worldsplat_prune.segment_sam2` uses SAM2's
                 automatic generator (AMG); this uses the same weights in
                 box-prompt mode.

The class detector (GroundingDINO), the caption→phrase tool and the pipeline
driver are added as sibling modules. None of them are imported here, so this
package stays importable without ``sam2`` / ``transformers`` / ``torch``.
"""
from __future__ import annotations

from .caption import (
    CallableCaptioner,
    Captioner,
    HFCaptioner,
    extract_phrases,
    merge_phrase_lists,
    seed_phrase_list,
)
from .grounding import (
    GroundingDinoDetector,
    detect_dataset,
    format_text_prompt,
    match_label_to_phrase,
    parse_phrases,
)
from .sam2_box import SAM2BoxLabeler, masks_to_class_maps

__all__ = [
    "SAM2BoxLabeler", "masks_to_class_maps",
    "GroundingDinoDetector", "detect_dataset", "format_text_prompt",
    "match_label_to_phrase", "parse_phrases",
    "Captioner", "CallableCaptioner", "HFCaptioner",
    "extract_phrases", "merge_phrase_lists", "seed_phrase_list",
]
