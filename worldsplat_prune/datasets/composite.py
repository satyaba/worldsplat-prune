"""Composite ``(class, instance)`` label adapter — the front-end's ``L_v``.

The prompt-driven front-end (``docs/worldsplat_prune_spec.md`` §2.2) is
**per-object** segmentation with **per-class** tags: SAM2 masks are the objects
(instances), GroundingDINO supplies the class phrase. The canonical per-view map
is therefore the *composite* id

    composite_id = class_id * STRIDE + instance_id          (0 == void)

where ``class_id`` indexes the curated phrase list (``1..K``) and ``instance_id``
is the per-class fused instance id (``1..STRIDE-1``). The encoding is
fixed-stride and reversible: :func:`decode` recovers the two halves, and
:func:`encode` builds the id from them.

Because ``class_id * STRIDE`` quickly exceeds the 16-bit range of the PNG id maps
the rest of the pipeline writes, composite maps are stored as int32 ``.npy`` per
view; the adapter also reads integer image maps (``.png``/``.tif``) for
round-tripping smaller labelings.

Layout (``label_dir`` from ``cfg`` or ``<DATASET_DIR>/semantic_composite``)::

    <DATASET_DIR>/images/<stem>.{png,jpg,...}
    <DATASET_DIR>/semantic_composite/<stem>.npy       # int32 composite ids

Class ids are consistent by construction; instance ids are made cross-view
consistent by :func:`worldsplat_prune.prep.fuse_instances.fuse_instances_per_class`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from .. import semantic_utils as su
from .base import LabelAdapter

# Instance ids ``1..STRIDE-1`` fit under each class; class ids start at 1.
COMPOSITE_STRIDE = 10_000


# --------------------------------------------------------------------------- #
# pure encoding                                                                 #
# --------------------------------------------------------------------------- #
def encode(class_ids, instance_ids, *, stride: int = COMPOSITE_STRIDE) -> np.ndarray:
    """Pack ``(class_id, instance_id)`` into a composite id map.

    A composite is void (``0``) where either half is void, so a Gaussian is
    labelled only when both the class and the instance are known.
    """
    cls = np.asarray(class_ids, dtype=np.int64)
    inst = np.asarray(instance_ids, dtype=np.int64)
    if cls.shape != inst.shape:
        raise ValueError(f"class/instance shape mismatch: {cls.shape} vs {inst.shape}")
    valid = (cls > 0) & (inst > 0)
    if (inst >= stride).any():
        raise ValueError(f"instance ids must be < stride ({stride})")
    out = np.zeros(cls.shape, dtype=np.int32)
    out[valid] = (cls[valid] * stride + inst[valid]).astype(np.int32)
    return out


def decode(ids, *, stride: int = COMPOSITE_STRIDE) -> Tuple[np.ndarray, np.ndarray]:
    """Split a composite id map back into ``(class_ids, instance_ids)``.

    Void stays ``(0, 0)`` on both halves.
    """
    ids = np.asarray(ids, dtype=np.int64)
    cls = np.where(ids > 0, ids // stride, 0)
    inst = np.where(ids > 0, ids % stride, 0)
    return cls.astype(np.int32), inst.astype(np.int32)


def compose_maps(class_maps: Mapping[int, np.ndarray], *,
                 stride: int = COMPOSITE_STRIDE) -> np.ndarray:
    """Union per-class instance maps into one composite id map.

    ``class_maps`` maps ``class_id -> instance map`` (each ``0 == void``). Where
    maps overlap the **first** non-void class (in insertion order) wins, so the
    result is deterministic. Classes are normally disjoint; overlaps are the
    front-end's detection ambiguity and are resolved conservatively.
    """
    out: Optional[np.ndarray] = None
    for class_id, inst in class_maps.items():
        cid = int(class_id)
        if cid <= 0:
            raise ValueError(f"class ids must be > 0 (got {cid})")
        inst = np.asarray(inst, dtype=np.int64)
        if out is None:
            out = np.zeros(inst.shape, dtype=np.int32)
        elif inst.shape != out.shape:
            raise ValueError(f"instance map shape {inst.shape} != {out.shape}")
        take = (inst > 0) & (out == 0)
        out[take] = (cid * stride + inst[take]).astype(np.int32)
    if out is None:
        out = np.zeros((0, 0), dtype=np.int32)
    return out


# --------------------------------------------------------------------------- #
# id-map IO (int32 ``.npy`` or integer image)                                   #
# --------------------------------------------------------------------------- #
def read_composite_map(path: str | Path) -> np.ndarray:
    """Read a composite map from ``.npy`` (int32) or an integer image."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.int32)
    return su.read_ids(path)


def write_composite_map(path: str | Path, ids: np.ndarray) -> None:
    """Write a composite map; ``.npy`` is lossless for ids beyond 16 bits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(ids, dtype=np.int32)
    if path.suffix.lower() == ".npy":
        np.save(path, ids)
    else:  # integer image (only safe while every id fits uint16)
        from PIL import Image

        if ids.size and int(ids.max()) > np.iinfo(np.uint16).max:
            raise ValueError(
                "composite ids exceed uint16; write a .npy instead of an image")
        Image.fromarray(ids.astype(np.uint16)).save(path)


# --------------------------------------------------------------------------- #
# adapter                                                                       #
# --------------------------------------------------------------------------- #
class CompositeLabelAdapter(LabelAdapter):
    """Consume per-frame composite ``(class, instance)`` maps."""

    name = "composite"

    def __init__(self, dataset_dir, cfg=None):
        super().__init__(dataset_dir, cfg)

        label_dir = self.cfg.get("label_dir")
        if label_dir:
            p = Path(label_dir)
            self.label_dir = p if p.is_absolute() else (self.dataset_dir / p)
        else:
            self.label_dir = self.dataset_dir / "semantic_composite"
        self.stride = int(self.cfg.get("stride", COMPOSITE_STRIDE))
        self.void_id = int(self.cfg.get("void_id", 0))
        self.min_area = int(self.cfg.get("min_area", 0) or 0)
        self.fill_void = bool(self.cfg.get("fill_void", False))

        if not self.label_dir.exists():
            raise FileNotFoundError(f"composite labels not found: {self.label_dir}")

        self._index: Dict[str, Path] = {}
        for p in sorted(self.label_dir.iterdir()):
            if p.is_file() and (p.suffix.lower() == ".npy"
                                or p.suffix.lower() in {".png", ".tif", ".tiff", ".bmp"}):
                self._index.setdefault(p.stem, p)

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
        return {
            "name": self.name,
            "source": "composite (class, instance)",
            "num_frames": len(self._index),
            "label_dir": str(self.label_dir),
            "stride": self.stride,
            "min_area": self.min_area,
            "fill_void": self.fill_void,
        }
