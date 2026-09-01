#!/usr/bin/env python3
"""Click-local Gaussian threshold modulation over a fixed M0 probability map.

The probability map remains fixed across interaction rounds. Each call edits
the accepted previous mask only inside a sign-specific Gaussian support and
keeps every voxel outside that support bit-identical to the previous mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, generate_binary_structure, label


@dataclass(frozen=True)
class GaussianClickConfig:
    base_threshold: float = 0.5
    add_strength: float = 0.25
    remove_strength: float = 0.25
    add_sigma_mm: float = 12.0
    remove_sigma_mm: float = 8.0
    add_plateau_mm: float = 3.0
    remove_plateau_mm: float = 0.0
    support_sigmas: float = 3.0
    threshold_min: float = 0.0
    threshold_max: float = 1.0
    adaptive_core_margin: float = 1e-6
    max_add_voxels: int = 20_000
    max_remove_voxels: int = 20_000
    topology_connectivity: int = 2


@dataclass(frozen=True)
class GaussianClickResult:
    mask: np.ndarray
    add: np.ndarray
    remove: np.ndarray
    threshold_field: np.ndarray | None
    effective_add_strength: float
    effective_remove_strength: float
    rejected_oversize_components: int
    rejected_add_merge_components: int
    rejected_remove_split_voxels: int


def autopet_v6_config(domain: str) -> GaussianClickConfig:
    """Return the frozen TACE configuration for one tracer domain."""
    normalized = str(domain).lower().rstrip("_")
    if normalized == "psma":
        return GaussianClickConfig(
            add_strength=0.20,
            remove_strength=0.30,
            add_sigma_mm=10.0,
            remove_sigma_mm=6.0,
            max_add_voxels=10_000,
            max_remove_voxels=5_000,
        )
    if normalized == "fdg":
        return GaussianClickConfig(
            add_strength=0.25,
            remove_strength=0.25,
            add_sigma_mm=12.0,
            remove_sigma_mm=8.0,
            max_add_voxels=20_000,
            max_remove_voxels=10_000,
        )
    raise ValueError(f"unsupported AutoPET tracer domain: {domain!r}")


def _nonzero_slices(
    mask: np.ndarray,
    margin: int | tuple[int, int, int] = 0,
) -> tuple[slice, slice, slice] | None:
    """Return a clipped sparse bounding box without changing mask semantics."""
    occupied_axes = [
        np.flatnonzero(np.any(mask, axis=tuple(other for other in range(3) if other != axis)))
        for axis in range(3)
    ]
    if occupied_axes[0].size == 0:
        return None
    margins = (margin, margin, margin) if isinstance(margin, int) else margin
    return tuple(
        slice(
            max(int(axis_coordinates.min()) - int(axis_margin), 0),
            min(int(axis_coordinates.max()) + int(axis_margin) + 1, mask.shape[axis]),
        )
        for axis, (axis_coordinates, axis_margin) in enumerate(zip(occupied_axes, margins))
    )


def _sparse_label(
    mask: np.ndarray,
    structure: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Label a sparse mask inside its exact nonzero box and embed the result."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    slices = _nonzero_slices(mask)
    if slices is None:
        return labels, 0
    local_labels, count = label(mask[slices], structure=structure)
    labels[slices] = local_labels
    return labels, int(count)


def _sparse_binary_dilation(mask: np.ndarray, structure: np.ndarray) -> np.ndarray:
    """Dilate only the one-voxel halo that can differ from background."""
    output = np.zeros(mask.shape, dtype=bool)
    slices = _nonzero_slices(mask, margin=1)
    if slices is None:
        return output
    output[slices] = binary_dilation(mask[slices], structure=structure)
    return output


def _validate_inputs(
    lesion_probability: np.ndarray,
    previous_mask: np.ndarray,
    positive_core: np.ndarray,
    negative_core: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]]:
    probability = np.asarray(lesion_probability, dtype=np.float32)
    previous = np.asarray(previous_mask, dtype=bool)
    positive = np.asarray(positive_core, dtype=bool)
    negative = np.asarray(negative_core, dtype=bool)
    if probability.ndim != 3:
        raise ValueError(f"expected a 3D probability map, got {probability.shape}")
    if previous.shape != probability.shape or positive.shape != probability.shape or negative.shape != probability.shape:
        raise ValueError("probability, mask and click cores must have identical shapes")
    if np.any(positive & negative):
        raise ValueError("positive and negative click cores must not overlap")
    if not np.all(np.isfinite(probability)) or probability.min() < 0 or probability.max() > 1:
        raise ValueError("lesion_probability must be finite and within [0, 1]")
    spacing = tuple(float(value) for value in spacing_mm)
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError(f"invalid spacing_mm={spacing_mm}")
    return probability, previous, positive, negative, spacing


def gaussian_click_field(
    click_core: np.ndarray,
    spacing_mm: tuple[float, float, float],
    sigma_mm: float,
    support_sigmas: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a max-over-clicks Gaussian field and its finite support."""
    core = np.asarray(click_core, dtype=bool)
    if sigma_mm <= 0 or support_sigmas <= 0:
        raise ValueError("sigma_mm and support_sigmas must be positive")
    if not np.any(core):
        return np.zeros(core.shape, dtype=np.float32), np.zeros(core.shape, dtype=bool)
    support_radius_mm = float(sigma_mm * support_sigmas)
    margins = tuple(
        int(np.ceil(support_radius_mm / float(spacing))) + 1
        for spacing in spacing_mm
    )
    slices = _nonzero_slices(core, margin=margins)
    if slices is None:
        raise RuntimeError("nonempty click core produced no sparse bounding box")
    local_distance_mm = distance_transform_edt(~core[slices], sampling=spacing_mm)
    local_support = local_distance_mm <= support_radius_mm
    support = np.zeros(core.shape, dtype=bool)
    support[slices] = local_support
    field = np.zeros(core.shape, dtype=np.float32)
    local_field = np.zeros(local_support.shape, dtype=np.float32)
    local_field[local_support] = np.exp(
        -0.5 * np.square(local_distance_mm[local_support] / sigma_mm)
    )
    field[slices] = local_field
    return field, support


def _seeded_bounded_components(
    candidate: np.ndarray,
    seed: np.ndarray,
    max_voxels: int,
) -> tuple[np.ndarray, int]:
    accepted = np.zeros(candidate.shape, dtype=bool)
    if max_voxels <= 0:
        raise ValueError("max_voxels must be positive")
    if not np.any(candidate) or not np.any(seed):
        return accepted, 0
    components, _ = _sparse_label(
        candidate, structure=generate_binary_structure(3, 3)
    )
    component_ids = np.unique(components[seed])
    rejected = 0
    for component_id in component_ids[component_ids != 0]:
        component = components == component_id
        if int(component.sum()) > max_voxels:
            rejected += 1
        else:
            accepted |= component
    return accepted, rejected


def _adaptive_strengths(
    probability: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    config: GaussianClickConfig,
) -> tuple[float, float]:
    add_strength = config.add_strength
    remove_strength = config.remove_strength
    # Make the action capable of crossing the clicked core while retaining a
    # configured lower bound. Caps follow from the legal threshold interval.
    if np.any(positive):
        add_core_reference = float(np.median(probability[positive]))
        required = config.base_threshold - add_core_reference + config.adaptive_core_margin
        add_strength = max(add_strength, required)
        # A high-probability positive click normally belongs to a raw M0
        # component removed by tracer dust filtering. Restore that probability
        # basin with tau=0 at the click, but do not apply the low-evidence
        # physical plateau that caused the observed add/remove oscillation.
        if add_core_reference >= config.base_threshold:
            add_strength = max(add_strength, config.base_threshold - config.threshold_min)
    if np.any(negative):
        remove_core_reference = float(np.median(probability[negative]))
        required = remove_core_reference - config.base_threshold + config.adaptive_core_margin
        remove_strength = max(remove_strength, required)
    # A strength capped at base_threshold only makes tau==0 at the exact click
    # and degenerates to scribble-pixel painting when M0 probability is tiny.
    # Overdrive the Gaussian just enough to create a physically defined flat
    # threshold plateau. The plateau is one resolvable neighborhood, not a
    # parameter sweep, and clipping still bounds tau to [min, max].
    # Only low-evidence ADD clicks need a minimum volume. If the fixed ADD
    # modulation already satisfies the median scribble voxel, a low-probability
    # outlier must not inflate the entire action (the observed oscillation case).
    if (
        config.add_plateau_mm > 0
        and np.any(positive)
        and add_core_reference < config.base_threshold - config.add_strength
    ):
        add_strength = max(
            add_strength,
            (config.base_threshold - config.threshold_min)
            * np.exp(config.add_plateau_mm ** 2 / (2.0 * config.add_sigma_mm ** 2)),
        )
    if config.remove_plateau_mm > 0:
        remove_strength = max(
            remove_strength,
            (config.threshold_max - config.base_threshold)
            * np.exp(config.remove_plateau_mm ** 2 / (2.0 * config.remove_sigma_mm ** 2)),
        )
    return float(max(add_strength, 0.0)), float(max(remove_strength, 0.0))


def _safe_to_remove_voxel(mask: np.ndarray, index: tuple[int, int, int], structure: np.ndarray) -> bool:
    """Conservative simple-point check that prevents a REMOVE split."""
    z, y, x = index
    z0, z1 = max(z - 1, 0), min(z + 2, mask.shape[0])
    y0, y1 = max(y - 1, 0), min(y + 2, mask.shape[1])
    x0, x1 = max(x - 1, 0), min(x + 2, mask.shape[2])
    local = mask[z0:z1, y0:y1, x0:x1].copy()
    local[z - z0, y - y0, x - x0] = False
    _, count = label(local, structure=structure)
    return count <= 1


def _topology_safe_actions(
    previous: np.ndarray,
    add: np.ndarray,
    remove: np.ndarray,
    spacing_mm: tuple[float, float, float],
    connectivity: int,
    existing_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Project Gaussian actions onto component-preserving local edits.

    REMOVE candidates are peeled from the current component boundary inward;
    any voxel that would split the foreground is retained. ADD components may
    create one prompted lesion or expand one existing lesion, but a component
    touching two existing lesions is rejected because it would merge them.
    """
    if connectivity not in (1, 2, 3):
        raise ValueError(f"topology_connectivity must be 1, 2 or 3, got {connectivity}")
    structure = generate_binary_structure(3, connectivity)

    surviving = previous.copy()
    accepted_remove = np.zeros(previous.shape, dtype=bool)
    rejected_remove_split = 0
    if np.any(remove):
        # Boundary-first peeling shrinks a component instead of drilling a hole
        # through its center. Stable lexicographic ordering keeps runs exact.
        coordinates = np.argwhere(remove)
        if existing_labels is None:
            existing_labels, _ = label(previous, structure=structure)
        coordinate_component_ids = existing_labels[tuple(coordinates.T)]
        boundary_depth = np.zeros(coordinates.shape[0], dtype=np.float64)
        for component_id in np.unique(coordinate_component_ids):
            if component_id == 0:
                continue
            component = existing_labels == component_id
            slices = _nonzero_slices(component, margin=1)
            if slices is None:
                continue
            local_depth = distance_transform_edt(
                component[slices], sampling=spacing_mm
            )
            selected = np.flatnonzero(coordinate_component_ids == component_id)
            starts = np.asarray([axis_slice.start for axis_slice in slices])
            local_coordinates = coordinates[selected] - starts[None, :]
            boundary_depth[selected] = local_depth[tuple(local_coordinates.T)]
        order = np.lexsort((
            coordinates[:, 2],
            coordinates[:, 1],
            coordinates[:, 0],
            boundary_depth,
        ))
        for coordinate in coordinates[order]:
            index = tuple(int(value) for value in coordinate)
            if not surviving[index]:
                continue
            if _safe_to_remove_voxel(surviving, index, structure):
                surviving[index] = False
                accepted_remove[index] = True
            else:
                rejected_remove_split += 1

    accepted_add = np.zeros(previous.shape, dtype=bool)
    rejected_add_merge = 0
    if np.any(add):
        if existing_labels is None:
            existing_labels, _ = label(surviving, structure=structure)
        add_labels, count = _sparse_label(add, structure=structure)
        for component_id in range(1, count + 1):
            component = add_labels == component_id
            touching = np.unique(existing_labels[_sparse_binary_dilation(component, structure)])
            touching = touching[touching != 0]
            if touching.size > 1:
                rejected_add_merge += 1
            else:
                accepted_add |= component
    return accepted_add, accepted_remove, rejected_add_merge, rejected_remove_split


def apply_gaussian_probability_clicks(
    lesion_probability: np.ndarray,
    previous_mask: np.ndarray,
    positive_core: np.ndarray,
    negative_core: np.ndarray,
    spacing_mm: tuple[float, float, float],
    config: GaussianClickConfig = GaussianClickConfig(),
    *,
    materialize_threshold_field: bool = True,
    foreground_labels: np.ndarray | None = None,
) -> GaussianClickResult:
    """Apply ADD/REMOVE deltas without resegmenting the full probability map."""
    probability, previous, positive, negative, spacing = _validate_inputs(
        lesion_probability, previous_mask, positive_core, negative_core, spacing_mm
    )
    # Grand Challenge supplies cumulative scribbles. A positive core already
    # inside the accepted mask, or a negative core already outside it, is
    # satisfied and must not keep competing with newer opposite-sign clicks.
    active_positive = positive & ~previous
    active_negative = negative & previous
    add_field, add_support = gaussian_click_field(
        active_positive, spacing, config.add_sigma_mm, config.support_sigmas
    )
    remove_field, remove_support = gaussian_click_field(
        active_negative, spacing, config.remove_sigma_mm, config.support_sigmas
    )
    add_strength, remove_strength = _adaptive_strengths(
        probability, active_positive, active_negative, config
    )
    threshold = None
    if materialize_threshold_field:
        threshold = np.clip(
            config.base_threshold - add_strength * add_field + remove_strength * remove_field,
            config.threshold_min,
            config.threshold_max,
        ).astype(np.float32, copy=False)

    # A zero probability contains no M0 evidence and therefore remains
    # fail-closed even when the adaptive ADD threshold reaches zero. At the
    # opposite extreme, a negative click is allowed to cross p==1 once the
    # local threshold reaches one; component seeding and size caps still bound
    # the edit.
    add_candidate = np.zeros(previous.shape, dtype=bool)
    add_slices = _nonzero_slices(add_support)
    if add_slices is not None:
        add_threshold = (
            threshold[add_slices]
            if threshold is not None
            else np.clip(
                config.base_threshold
                - add_strength * add_field[add_slices]
                + remove_strength * remove_field[add_slices],
                config.threshold_min,
                config.threshold_max,
            ).astype(np.float32, copy=False)
        )
        add_candidate[add_slices] = (
            (~previous[add_slices])
            & add_support[add_slices]
            & ~negative[add_slices]
            & (probability[add_slices] > 0)
            & (probability[add_slices] >= add_threshold)
        )
    remove_candidate = np.zeros(previous.shape, dtype=bool)
    remove_slices = _nonzero_slices(remove_support)
    if remove_slices is not None:
        remove_threshold = (
            threshold[remove_slices]
            if threshold is not None
            else np.clip(
                config.base_threshold
                - add_strength * add_field[remove_slices]
                + remove_strength * remove_field[remove_slices],
                config.threshold_min,
                config.threshold_max,
            ).astype(np.float32, copy=False)
        )
        remove_candidate[remove_slices] = (
            previous[remove_slices]
            & remove_support[remove_slices]
            & ~positive[remove_slices]
            & (probability[remove_slices] <= remove_threshold)
        )
    add, add_rejected = _seeded_bounded_components(
        add_candidate, active_positive, config.max_add_voxels
    )
    remove, remove_rejected = _seeded_bounded_components(
        remove_candidate, active_negative, config.max_remove_voxels
    )
    add, remove, add_merge_rejected, remove_split_rejected = _topology_safe_actions(
        previous,
        add,
        remove,
        spacing,
        config.topology_connectivity,
        foreground_labels,
    )
    updated = (previous | add) & ~remove
    return GaussianClickResult(
        mask=updated,
        add=add,
        remove=remove,
        threshold_field=threshold,
        effective_add_strength=add_strength,
        effective_remove_strength=remove_strength,
        rejected_oversize_components=add_rejected + remove_rejected,
        rejected_add_merge_components=add_merge_rejected,
        rejected_remove_split_voxels=remove_split_rejected,
    )
