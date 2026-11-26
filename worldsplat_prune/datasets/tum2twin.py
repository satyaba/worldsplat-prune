"""TUM2TWIN reference-geometry label adapter (C2, §2.3, [D19]).

The urban label source is the twin's **reference geometry** — LoD models, LiDAR
and ALS semantic point clouds — not per-view 2D image maps. The labelling driver
(:mod:`worldsplat_prune.prep.label_from_reference`) rasterises that geometry into
per-view labels::

    <DATASET_DIR>/semantic_reference/<stem>.npy     # int32 composite ids

with the **same** composite ``(class, instance)`` encoding as
:mod:`worldsplat_prune.datasets.composite`:

    id = class_id * STRIDE + instance_id           (0 == void)

so ``SemanticStore`` / ``vote`` / ``prune`` are unchanged. Because the ids come
from 3D reference geometry, instances are **cross-view consistent by
construction** — the driver reports this and there is no ``fuse_instances`` step.

Registered as both ``tum2twin`` and ``reference`` (see ``datasets/__init__.py``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .. import semantic_utils as su
from .base import LabelAdapter
from .composite import COMPOSITE_STRIDE, decode, read_composite_map

# A generic urban ALS/LoD taxonomy used when no dataset-specific table is given.
# Real TUM2TWIN class ids are supplied via ``cfg["taxonomy"]`` (dict or JSON path)
# or the ``reference_meta.json`` the driver writes next to the maps.
DEFAULT_TAXONOMY = {
    1: "ground", 2: "road", 3: "sidewalk", 4: "vegetation",
    5: "building", 6: "facade", 7: "roof", 8: "vehicle",
    9: "pole", 10: "wire", 11: "water", 12: "other",
}


def _load_taxonomy(spec) -> dict:
    if spec is None:
        return dict(DEFAULT_TAXONOMY)
    if isinstance(spec, dict):
        return {int(k): str(v) for k, v in spec.items()}
    p = Path(spec)
    if p.exists():
        data = json.loads(p.read_text())
        return {int(k): str(v) for k, v in data.items()}
    raise FileNotFoundError(f"taxonomy file not found: {spec}")


class TUM2TWINLabelAdapter(LabelAdapter):
    """Consume per-view composite maps rasterised from reference geometry."""

    name = "tum2twin"

    def __init__(self, dataset_dir, cfg=None):
        super().__init__(dataset_dir, cfg)

        label_dir = self.cfg.get("label_dir")
        if label_dir:
            p = Path(label_dir)
            self.label_dir = p if p.is_absolute() else (self.dataset_dir / p)
        else:
            self.label_dir = self.dataset_dir / "semantic_reference"
        self.stride = int(self.cfg.get("stride", COMPOSITE_STRIDE))
        self.void_id = int(self.cfg.get("void_id", 0))
        self.min_area = int(self.cfg.get("min_area", 0) or 0)
        self.fill_void = bool(self.cfg.get("fill_void", False))
        self.taxonomy = _load_taxonomy(self.cfg.get("taxonomy"))

        if not self.label_dir.exists():
            raise FileNotFoundError(
                f"reference labels not found: {self.label_dir}\n"
                "run: python -m worldsplat_prune.prep.label_from_reference "
                "--dataset_dir <DATASET_DIR> --reference <geometry>")

        self._index: Dict[str, Path] = {}
        for p in sorted(self.label_dir.iterdir()):
            if not p.is_file():
                continue
            if p.name in ("reference_meta.json",):
                continue
            if p.suffix.lower() == ".npy" or p.suffix.lower() in {".png", ".tif", ".tiff", ".bmp"}:
                self._index.setdefault(p.stem, p)

        self._meta_file = self.label_dir / "reference_meta.json"

    def resolve(self, image_name: str):
        return self._index.get(Path(image_name).stem)

    def _drop_small(self, ids: np.ndarray) -> np.ndarray:
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
        ids = read_composite_map(p)
        if self.void_id != 0:
            ids = np.where(ids == self.void_id, 0, ids)
        ids = self._drop_small(ids.astype(np.int32))
        if self.fill_void:
            ids = su.fill_void_nearest(ids)
        return ids.astype(np.int32)

    def load_class_instance(self, image_name: str):
        """Return ``(class_ids, instance_ids)`` int32 maps, or ``None``."""
        ids = self.load(image_name)
        if ids is None:
            return None
        return decode(ids, stride=self.stride)

    def meta(self) -> dict:
        out = {
            "name": self.name,
            "source": "reference geometry (LoD / LiDAR / ALS)",
            "num_frames": len(self._index),
            "label_dir": str(self.label_dir),
            "stride": self.stride,
            "min_area": self.min_area,
            "fill_void": self.fill_void,
            "class_taxonomy": self.taxonomy,
            "instance_source": "LoD object / building id (baked into the ids)",
            "instance_consistency": "cross-view consistent by construction (3D reference)",
            "fuse_instances": False,
        }
        if self._meta_file.exists():
            try:
                extra = json.loads(self._meta_file.read_text())
            except (json.JSONDecodeError, OSError):
                extra = {}
            for k in ("reference_path", "radius", "class_hist", "consistency",
                      "instance_source", "generated_by"):
                if k in extra:
                    out.setdefault(k, extra[k])
        return out


__all__ = ["TUM2TWINLabelAdapter", "DEFAULT_TAXONOMY"]
