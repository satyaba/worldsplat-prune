"""Label adapter registry."""
from __future__ import annotations

from .base import LabelAdapter
from .composite import CompositeLabelAdapter
from .instance_dir import InstanceDirLabelAdapter
from .replica import ReplicaLabelAdapter

_ADAPTERS = {
    "replica": ReplicaLabelAdapter,
    # Class-agnostic instance maps (SAM2 pseudo-labels, ...). "sam2" is an alias
    # so a run reads naturally: --label_adapter sam2.
    "instance_dir": InstanceDirLabelAdapter,
    "sam2": InstanceDirLabelAdapter,
    # Prompt-driven front-end output: composite (class, instance) maps (§2.2).
    "composite": CompositeLabelAdapter,
    "grounded_sam2": CompositeLabelAdapter,
}


def get_adapter(name: str, dataset_dir, cfg=None) -> LabelAdapter:
    key = (name or "replica").lower()
    if key not in _ADAPTERS:
        raise KeyError(f"unknown label adapter '{name}'. available: {sorted(_ADAPTERS)}")
    return _ADAPTERS[key](dataset_dir, cfg)


__all__ = ["LabelAdapter", "ReplicaLabelAdapter", "InstanceDirLabelAdapter",
           "CompositeLabelAdapter", "get_adapter"]
