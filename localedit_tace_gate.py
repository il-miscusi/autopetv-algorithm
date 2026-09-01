#!/usr/bin/env python3
"""Frozen LocalEdit--TACE HoleGuard Fusion v5 transaction gate.

Technical telemetry preserves the historical ``gaussian_v6`` identifier so
that the strict full-OOF manifests remain traceable; the public method name is
TACE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_fill_holes, generate_binary_structure, label


@dataclass(frozen=True)
class LocalEditGaussianGateResult:
    mask: np.ndarray
    source: str
    reason: str
    active_positive_voxels: int
    active_negative_voxels: int
    local_add_voxels: int
    local_remove_voxels: int
    gaussian_add_voxels: int
    gaussian_remove_voxels: int
    accepted_add_voxels: int
    accepted_remove_voxels: int
    rejected_split_remove_voxels: int
    new_hole_voxels_candidate: int = 0
    new_hole_voxels_gaussian: int = 0
    hole_fallback_level: str = "none"


def _enclosed_hole_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return binary_fill_holes(mask) & ~mask


def _new_enclosed_hole_voxels(previous: np.ndarray, candidate: np.ndarray) -> int:
    previous_holes = _enclosed_hole_mask(previous)
    candidate_holes = _enclosed_hole_mask(candidate)
    return int(np.count_nonzero(candidate_holes & ~previous_holes))


def _rollback_component_splitting_removals(
    previous: np.ndarray,
    proposed_remove: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Reject a partial REMOVE transaction that splits a 3D component."""
    safe = np.asarray(proposed_remove, dtype=bool).copy()
    if not np.any(safe):
        return safe, 0
    structure = generate_binary_structure(3, 3)
    previous_labels, _ = label(previous, structure=structure)
    rejected_voxels = 0
    affected_ids = np.unique(previous_labels[safe])
    for component_id in affected_ids[affected_ids != 0]:
        component = previous_labels == component_id
        component_remove = safe & component
        surviving = component & ~component_remove
        if not np.any(surviving):
            continue
        _, surviving_count = label(surviving, structure=structure)
        if int(surviving_count) <= 1:
            continue
        rejected_voxels += int(component_remove.sum())
        safe[component] = False
    return safe, rejected_voxels


def select_localedit_gaussian_candidate(
    domain: str,
    m0_mask: np.ndarray,
    previous_mask: np.ndarray,
    local_mask: np.ndarray,
    gaussian_mask: np.ndarray,
    positive_core: np.ndarray,
    negative_core: np.ndarray,
    hole_guarded_gaussian_fallback: bool = False,
) -> LocalEditGaussianGateResult:
    """Fuse sign-consistent proposals as one transaction of ``previous_mask``."""
    normalized = str(domain).lower().rstrip("_")
    if normalized not in ("fdg", "psma"):
        raise ValueError(f"unsupported tracer domain {domain!r}")

    m0 = np.asarray(m0_mask, dtype=bool)
    previous = np.asarray(previous_mask, dtype=bool)
    local = np.asarray(local_mask, dtype=bool)
    gaussian = np.asarray(gaussian_mask, dtype=bool)
    positive = np.asarray(positive_core, dtype=bool)
    negative = np.asarray(negative_core, dtype=bool)
    if not (
        m0.shape
        == previous.shape
        == local.shape
        == gaussian.shape
        == positive.shape
        == negative.shape
    ):
        raise ValueError("M0, previous, candidate and click-core masks must have identical shapes")
    if np.any(positive & negative):
        raise ValueError("positive and negative click cores must be disjoint")

    active_positive = positive & ~previous
    active_negative = negative & previous
    local_add = local & ~previous
    local_remove = previous & ~local
    gaussian_add = gaussian & ~previous
    gaussian_remove = previous & ~gaussian

    if normalized == "psma" and not np.any(m0):
        gaussian_new_holes = (
            _new_enclosed_hole_voxels(previous, gaussian)
            if hole_guarded_gaussian_fallback
            else 0
        )
        selected = previous.copy() if gaussian_new_holes else gaussian
        selected_add = selected & ~previous
        selected_remove = previous & ~selected
        return LocalEditGaussianGateResult(
            mask=selected,
            source="previous_mask" if gaussian_new_holes else "gaussian_v6",
            reason=(
                "gaussian_new_hole_previous_fallback"
                if gaussian_new_holes
                else "psma_empty_m0_strict_oof_fallback"
            ),
            active_positive_voxels=int(active_positive.sum()),
            active_negative_voxels=int(active_negative.sum()),
            local_add_voxels=int(local_add.sum()),
            local_remove_voxels=int(local_remove.sum()),
            gaussian_add_voxels=int(gaussian_add.sum()),
            gaussian_remove_voxels=int(gaussian_remove.sum()),
            accepted_add_voxels=int(selected_add.sum()),
            accepted_remove_voxels=int(selected_remove.sum()),
            rejected_split_remove_voxels=0,
            new_hole_voxels_gaussian=gaussian_new_holes,
            hole_fallback_level="previous" if gaussian_new_holes else "none",
        )

    fused = previous.copy()
    accepted_add = np.zeros(previous.shape, dtype=bool)
    accepted_remove = np.zeros(previous.shape, dtype=bool)
    rejected_split_remove_voxels = 0

    if np.any(active_positive):
        accepted_add = (local_add | gaussian_add | active_positive) & ~negative
        fused |= accepted_add

    if np.any(active_negative):
        accepted_remove = ((local_remove & gaussian_remove) | active_negative) & ~positive
        accepted_remove, rejected_split_remove_voxels = (
            _rollback_component_splitting_removals(previous, accepted_remove)
        )
        fused &= ~accepted_remove

    active_add_count = int(active_positive.sum())
    active_remove_count = int(active_negative.sum())
    candidate_new_holes = 0
    gaussian_new_holes = 0
    hole_fallback_level = "none"
    if hole_guarded_gaussian_fallback:
        candidate_new_holes = _new_enclosed_hole_voxels(previous, fused)
        if candidate_new_holes:
            gaussian_new_holes = _new_enclosed_hole_voxels(
                previous, gaussian
            )
            if gaussian_new_holes:
                fused = previous.copy()
                source = "previous_mask"
                reason = "fusion_and_gaussian_new_hole_previous_fallback"
                hole_fallback_level = "previous"
            else:
                fused = gaussian.copy()
                source = "gaussian_v6"
                reason = "fusion_new_hole_gaussian_fallback"
                hole_fallback_level = "gaussian"
            accepted_add = fused & ~previous
            accepted_remove = previous & ~fused
    if hole_fallback_level == "none":
        if active_add_count and active_remove_count:
            source = "action_consistent_fusion"
            reason = "mixed_unsatisfied_clicks_add_union_consensus_remove"
        elif active_add_count:
            source = "action_consistent_fusion"
            reason = "positive_transaction_union"
        elif active_remove_count:
            source = "gaussian_v6"
            reason = "negative_transaction_consensus_dmm_guard"
        else:
            source = "previous_mask"
            reason = "all_cumulative_clicks_satisfied"

    return LocalEditGaussianGateResult(
        mask=fused,
        source=source,
        reason=reason,
        active_positive_voxels=active_add_count,
        active_negative_voxels=active_remove_count,
        local_add_voxels=int(local_add.sum()),
        local_remove_voxels=int(local_remove.sum()),
        gaussian_add_voxels=int(gaussian_add.sum()),
        gaussian_remove_voxels=int(gaussian_remove.sum()),
        accepted_add_voxels=int(accepted_add.sum()),
        accepted_remove_voxels=int(accepted_remove.sum()),
        rejected_split_remove_voxels=rejected_split_remove_voxels,
        new_hole_voxels_candidate=candidate_new_holes,
        new_hole_voxels_gaussian=gaussian_new_holes,
        hole_fallback_level=hole_fallback_level,
    )
