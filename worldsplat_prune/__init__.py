"""WorldSplat-Prune — semantic labelling machinery for 3D Gaussian Splatting.

Shared components reused by the post-reconstruction, semantic-aware pruning
pipeline (see ``docs/worldsplat_prune_spec.md``): per-view label adapters, the
semantic store, the SAM2 pseudo-labeller, and the dataset-prep / fusion tools.
Imported by ``eval_worldsplat.py`` and the prep tools; independent of the base
3DGS trainer.
"""

from __future__ import annotations

__all__ = [
    "datasets",
    "semantic_utils",
    "semantic_store",
    "segment_sam2",
    "prep",
    "budget",
    "verify_alignment",
    "eval_boundary",
    "vote",
    "prune",
]
