#!/usr/bin/env python3
"""Portable PSMA 1--4 voxel component gate used by the final candidate."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


FEATURES = ("size_voxels", "volume_mm3", "bbox_fill_fraction")
SEMANTICS = "connectivity18_psma5_plus_shape_gate_1to4"


def load_gate(path: str | Path) -> dict:
    document = json.loads(Path(path).read_text())
    if document.get("status") != "frozen_psma_shape_component_gate":
        raise RuntimeError("invalid PSMA shape-gate artifact")
    if document.get("candidate_semantics") != SEMANTICS:
        raise RuntimeError("PSMA shape-gate semantics mismatch")
    if tuple(document.get("features", ())) != FEATURES:
        raise RuntimeError("PSMA shape-gate feature schema mismatch")
    for key in ("scaler_mean", "scaler_scale", "coefficient"):
        values = np.asarray(document.get(key), dtype=np.float64)
        if values.shape != (len(FEATURES),) or not np.isfinite(values).all():
            raise RuntimeError(f"invalid PSMA shape-gate field {key}")
    if not math.isfinite(float(document.get("intercept", math.nan))):
        raise RuntimeError("invalid PSMA shape-gate intercept")
    if float(document.get("decision_threshold", -1)) != 0.5:
        raise RuntimeError("PSMA shape-gate threshold must remain frozen at 0.5")
    return document


def component_score(features: tuple[float, float, float], gate: dict) -> float:
    values = np.asarray(features, dtype=np.float64)
    mean = np.asarray(gate["scaler_mean"], dtype=np.float64)
    scale = np.asarray(gate["scaler_scale"], dtype=np.float64)
    coefficient = np.asarray(gate["coefficient"], dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all() or np.any(scale <= 0):
        raise RuntimeError("invalid PSMA component features or scaler")
    logit = float(np.dot((values - mean) / scale, coefficient) + gate["intercept"])
    if logit >= 0:
        return float(1.0 / (1.0 + math.exp(-logit)))
    exp_logit = math.exp(logit)
    return float(exp_logit / (1.0 + exp_logit))


def apply_psma_shape_gate(
    raw_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    gate: dict,
) -> tuple[np.ndarray, dict]:
    """Filter with official connectivity and restore selected small components."""
    import cc3d
    from scipy.ndimage import find_objects

    raw = np.asarray(raw_mask, dtype=bool)
    if raw.ndim != 3:
        raise ValueError(f"expected a 3D raw mask, got {raw.shape}")
    spacing = np.asarray(spacing_xyz, dtype=np.float64)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError(f"invalid spacing {spacing_xyz}")
    if not np.any(raw):
        return raw.astype(np.uint8), {
            "raw_components": 0, "selected_components": 0, "selected_voxels": 0,
        }
    labels, count = cc3d.connected_components(
        raw.astype(np.uint8, copy=False), connectivity=18, return_N=True
    )
    sizes = np.bincount(labels.ravel(), minlength=int(count) + 1)
    base = cc3d.dust(
        raw.astype(np.uint8, copy=False), threshold=5, connectivity=18
    ) > 0
    boxes = find_objects(labels, max_label=int(count))
    selected_ids = []
    voxel_volume = float(np.prod(spacing))
    for component_id in range(1, int(count) + 1):
        size = int(sizes[component_id])
        if not 1 <= size <= 4:
            continue
        region = boxes[component_id - 1]
        if region is None:
            raise RuntimeError(f"missing component bounding box {component_id}")
        bbox_volume = int(np.prod([axis.stop - axis.start for axis in region]))
        score = component_score(
            (float(size), size * voxel_volume, float(size / bbox_volume)), gate
        )
        if score >= 0.5:
            selected_ids.append(component_id)
    if selected_ids:
        base |= np.isin(labels, np.asarray(selected_ids, dtype=labels.dtype))
    selected_voxels = int(sum(int(sizes[index]) for index in selected_ids))
    return base.astype(np.uint8), {
        "raw_components": int(count),
        "selected_components": len(selected_ids),
        "selected_voxels": selected_voxels,
    }
