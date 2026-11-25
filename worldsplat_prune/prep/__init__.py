"""Dataset preparation helpers for WorldSplat-Prune (SfM + pseudo-labels).

Nothing here is imported by training; these are one-off, Colab/GPU-time tools
that turn a raw capture into the layout the trainer expects::

    <dataset>/
      images/            # undistorted RGB frames
      sparse/0/          # COLMAP model, PINHOLE camera (required by the rasterizer)
      semantic_instance/ # optional SAM2 pseudo-labels (worldsplat_prune.segment_sam2)

See ``Prepare_Datasets.ipynb`` for the end-to-end driver.
"""
from __future__ import annotations

__all__ = ["hloc_sfm", "scene_io", "fuse_instances", "eval_instances"]
