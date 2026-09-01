"""Grand Challenge runtime adapters for frozen TACE (technical policy v6).

The audited Gaussian editor operates on z-y-x arrays. Grand Challenge click
coordinates are x-y-z, and the only portable transaction carrier between
interactive invocations is the previous segmentation in the mounted output
directory. This module keeps those interface concerns separate from the
frozen edit policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class PreviousMaskState:
    mask: np.ndarray
    source: str
    fallback_reason: str | None


def clicks_to_zyx_cores(clicks: dict, shape: tuple[int, int, int]):
    """Convert cumulative official x-y-z clicks into disjoint z-y-x cores.

    Exact sign conflicts are fail-closed but deterministic: the most recent
    click at a voxel wins. Official evaluation clicks are correct and should
    not conflict; this rule only protects malformed or corrected user input.
    """
    positive = np.zeros(shape, dtype=bool)
    negative = np.zeros(shape, dtype=bool)
    valid = out_of_bounds = overwritten_conflicts = 0
    for item in clicks.get("points", []):
        try:
            x, y, z = (int(round(float(value))) for value in item["point"])
            name = item["name"]
        except (KeyError, TypeError, ValueError):
            continue
        index = (z, y, x)
        if not all(0 <= value < size for value, size in zip(index, shape)):
            out_of_bounds += 1
            continue
        if name == "tumor":
            overwritten_conflicts += int(negative[index])
            negative[index] = False
            positive[index] = True
        elif name == "background":
            overwritten_conflicts += int(positive[index])
            positive[index] = False
            negative[index] = True
        else:
            continue
        valid += 1
    stats = {
        "valid_clicks": valid,
        "positive_core_voxels": int(positive.sum()),
        "negative_core_voxels": int(negative.sum()),
        "out_of_bounds_clicks": out_of_bounds,
        "overwritten_exact_conflicts": overwritten_conflicts,
    }
    return positive, negative, stats


def _geometry_matches(image: sitk.Image, reference: sitk.Image) -> bool:
    return (
        image.GetSize() == reference.GetSize()
        and np.allclose(image.GetSpacing(), reference.GetSpacing(), atol=1e-6, rtol=0)
        and np.allclose(image.GetOrigin(), reference.GetOrigin(), atol=1e-5, rtol=0)
        and np.allclose(image.GetDirection(), reference.GetDirection(), atol=1e-6, rtol=0)
    )


def load_previous_or_m0(
    output_path: str | Path,
    reference_path: str | Path,
    m0: np.ndarray,
) -> PreviousMaskState:
    """Load the previous accepted mask, or fail closed to immutable M0."""
    output = Path(output_path)
    if not output.is_file():
        return PreviousMaskState(np.asarray(m0, dtype=bool).copy(), "m0_fallback", "missing_previous_output")
    try:
        previous_image = sitk.ReadImage(str(output))
        reference = sitk.ReadImage(str(reference_path))
        if not _geometry_matches(previous_image, reference):
            return PreviousMaskState(np.asarray(m0, dtype=bool).copy(), "m0_fallback", "previous_geometry_mismatch")
        previous = sitk.GetArrayFromImage(previous_image)
        if previous.shape != m0.shape:
            return PreviousMaskState(np.asarray(m0, dtype=bool).copy(), "m0_fallback", "previous_shape_mismatch")
        if not np.all(np.isin(np.unique(previous), (0, 1))):
            return PreviousMaskState(np.asarray(m0, dtype=bool).copy(), "m0_fallback", "previous_not_binary")
        return PreviousMaskState(previous.astype(bool, copy=False), "previous_output", None)
    except Exception as error:
        return PreviousMaskState(
            np.asarray(m0, dtype=bool).copy(),
            "m0_fallback",
            f"previous_read_error:{type(error).__name__}",
        )
