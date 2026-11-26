"""Per-view semantic store.

Wraps a :class:`~worldsplat_prune.datasets.base.LabelAdapter` and, for a given
``image_name`` at the camera's training resolution, returns the canonical
segment map plus its distance transform and normal field, cached on device.

The store is keyed by ``image_name`` -- exactly the field stored on
``scene.cameras.Camera`` (``cameras.py:33``) -- so **no change to the Camera
constructor is required**.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import semantic_utils as su
from .datasets.base import LabelAdapter


class SemanticStore:
    def __init__(self, adapter: LabelAdapter, device: str = "cuda", cache_dir=None,
                 edge_tol: float = 0.0, edge_abs_tol: float = su.DEFAULT_EDGE_ABS_TOL,
                 edge_occlusion_only: bool = False):
        self.adapter = adapter
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._mem: dict = {}
        # [D20]/T2 geometry-edge term: off (0.0) unless the adapter has depth.
        self.edge_tol = float(edge_tol)
        self.edge_abs_tol = float(edge_abs_tol)
        self.edge_occlusion_only = bool(edge_occlusion_only)

    def get(self, image_name: str, H: int, W: int):
        """Return ``{"label", "dt", "normal"[, "dt_geom"]}`` tensors on device.

        ``label`` is ``int32 [H, W]`` (0 == void); ``dt`` is ``float32 [H, W]``;
        ``normal`` is ``float32 [H, W, 2]``. ``dt_geom`` (the depth-discontinuity
        distance transform, ``inf`` on invalid depth) is present only when the
        edge term is enabled and the adapter supplies a depth map.
        """
        key = (image_name, int(H), int(W))
        if key in self._mem:
            return self._mem[key]

        lab = self.adapter.load(image_name)
        if lab is None:
            return None

        lab = su.resize_ids(lab, H, W)
        dt, _ = su.distance_transform(lab)
        nrm = su.normal_field(dt)

        out = {
            "label": torch.as_tensor(lab, device=self.device),
            "dt": torch.as_tensor(dt, device=self.device),
            "normal": torch.as_tensor(nrm, device=self.device),
        }
        if self.edge_tol > 0.0:
            load_depth = getattr(self.adapter, "load_depth", None)
            depth = load_depth(image_name) if load_depth is not None else None
            if depth is not None:
                depth = su.resize_depth(depth, H, W)
                dt_geom, _ = su.geometry_distance_transform(
                    depth, edge_tol=self.edge_tol, abs_tol=self.edge_abs_tol,
                    occlusion_only=self.edge_occlusion_only)
                out["dt_geom"] = torch.as_tensor(dt_geom, device=self.device)
        self._mem[key] = out
        return out

    def label(self, image_name: str) -> np.ndarray | None:
        """Native-resolution canonical ids, for precompute/verification."""
        return self.adapter.load(image_name)

    def meta(self) -> dict:
        return self.adapter.meta()
