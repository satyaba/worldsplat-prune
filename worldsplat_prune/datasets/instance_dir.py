"""Directory-of-instance-labels adapter (SAM2 pseudo-labels, and similar).

Non-segmented datasets -- Mip-NeRF 360, Tanks & Temples, Deep Blending, custom
captures -- ship no ground-truth semantics. We run an offline, class-agnostic
segmenter (SAM2 automatic mask generation; see :mod:`worldsplat_prune.segment_sam2`) over
each training image and write one instance-id map per frame. This adapter
consumes that directory, turning it into the canonical per-view segment map
``L_v`` consumed by WorldSplat-Prune (``0 == void``).

Layout (``label_dir`` from ``cfg`` or ``<DATASET_DIR>/semantic_instance``)::

    <DATASET_DIR>/images/<stem>.{png,jpg,jpeg,...}
    <DATASET_DIR>/semantic_instance/<stem>.png        # uint16 instance ids

Ids are matched by filename **stem**, so extension differences between the RGB
frames and the label maps are fine. Ids are **per frame**; they are deliberately
*not* required to be consistent across views:

* the boundary geometry (``dt`` / normals) that
  :class:`~worldsplat_prune.semantic_store.SemanticStore` derives from ``L_v``
  is well defined per frame;
* cross-view instance identity (the vote pass, target-instance protection) needs
  the fused maps from :mod:`worldsplat_prune.prep.fuse_instances`.

``min_area`` drops instances smaller than N pixels (set to void, then optionally
nearest-filled by ``fill_void``), which is the cheap defence against SAM2's
many tiny speckle masks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import semantic_utils as su
from .base import LabelAdapter


class InstanceDirLabelAdapter(LabelAdapter):
    name = "instance_dir"

    def __init__(self, dataset_dir, cfg=None):
        super().__init__(dataset_dir, cfg)

        label_dir = self.cfg.get("label_dir")
        if label_dir:
            p = Path(label_dir)
            # a relative label dir is resolved against the dataset, matching the
            # fusion tools (otherwise it would be relative to the cwd)
            self.label_dir = p if p.is_absolute() else (self.dataset_dir / p)
        else:
            self.label_dir = self.dataset_dir / "semantic_instance"
        self.void_id = int(self.cfg.get("void_id", 0))
        self.min_area = int(self.cfg.get("min_area", 0) or 0)
        self.fill_void = bool(self.cfg.get("fill_void", False))

        if not self.label_dir.exists():
            raise FileNotFoundError(f"instance labels not found: {self.label_dir}")

        self._index = {}
        for p in sorted(self.label_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in {".png", ".tif", ".tiff", ".bmp"}:
                self._index.setdefault(p.stem, p)

    def resolve(self, image_name: str):
        return self._index.get(Path(image_name).stem)

    def _drop_small(self, ids: np.ndarray) -> np.ndarray:
        """Set instances with fewer than ``min_area`` pixels to void."""
        if self.min_area <= 1:
            return ids
        uniq, counts = np.unique(ids, return_counts=True)
        small = uniq[(counts < self.min_area) & (uniq != self.void_id)]
        if small.size:
            ids = np.where(np.isin(ids, small), self.void_id, ids)
        return ids

    def load(self, image_name: str):
        p = self.resolve(image_name)
        if p is None:
            return None

        ids = su.read_ids(p)
        if self.void_id != 0:
            ids = np.where(ids == self.void_id, 0, ids)
        ids = self._drop_small(ids.astype(np.int32))
        if self.fill_void:
            ids = su.fill_void_nearest(ids)
        return ids.astype(np.int32)

    def meta(self) -> dict:
        return {
            "name": self.name,
            "source": "sam2/instance" if self.label_dir.name.startswith("semantic") else str(self.label_dir),
            "num_frames": len(self._index),
            "label_dir": str(self.label_dir),
            "min_area": self.min_area,
            "fill_void": self.fill_void,
            "num_classes": None,  # class-agnostic: ids are instances, not classes
        }
