"""Scribble conformance for autoPET V interactive segmentation.

All arrays are in NIfTI/nibabel index order (x, y, z) — the same space the
Grand Challenge lesion-clicks.json point coordinates live in (the evaluator
generates them with np.argwhere on nibabel arrays). SimpleITK callers must
transpose (2, 1, 0) before and after.

Design: the previous iteration's prediction is the base. Each new scribble is
a guaranteed-correct annotation of the largest remaining error component of
that prediction (the evaluator draws it inside pred!=gt), so:
  - background scribble  -> the predicted component it touches is (mostly)
    false: remove it. Oversized components lose only a local geodesic chunk.
  - foreground scribble  -> a lesion was missed there: add a PET-thresholded
    connected region grown from the scribble, volume-capped.
Scribble voxels themselves are ground-truth certain (fg inside gt=1, bg inside
gt=0) and are enforced unconditionally at the end.
"""

import numpy as np
from scipy import ndimage

_STRUCT26 = ndimage.generate_binary_structure(3, 3)


# ---------------------------------------------------------------------------
# Click parsing
# ---------------------------------------------------------------------------

def split_clicks(gc_json):
    """GC 'Multiple points' json -> (fg_points, bg_points) as [x,y,z] tuples."""
    pts = (gc_json or {}).get("points", [])
    fg = [tuple(int(c) for c in p["point"]) for p in pts if p.get("name") == "tumor"]
    bg = [tuple(int(c) for c in p["point"]) for p in pts if p.get("name") == "background"]
    return fg, bg


def new_suffix(cur, prev):
    """Points accumulate append-only per class; the new scribble is the suffix.
    If the prefix does not match (unexpected), treat everything as new."""
    if prev and cur[: len(prev)] == prev:
        return cur[len(prev):]
    return cur if not prev else cur


def group_by_slice(points):
    """One scribble lives on one axial slice (z = index 2). Group points."""
    groups = {}
    for p in points:
        groups.setdefault(p[2], []).append(p)
    return list(groups.values())


def _inside(p, shape):
    return all(0 <= c < s for c, s in zip(p, shape))


# ---------------------------------------------------------------------------
# Background: component removal
# ---------------------------------------------------------------------------

def remove_bg_components(pred, bg_points, spacing_xyz=(2.0, 2.0, 3.0),
                         pet=None, tol=2, small_ml=30.0, whole_cap_ml=1500.0,
                         local_mm=25.0, suv_ratio=0.5):
    """Remove prediction touching background scribble points.

    A bg scribble marks the largest false-positive component of the previous
    prediction — but that component may be a rim around a mostly-TRUE lesion
    mass (whole-component removal there is catastrophic; measured Dice
    0.87 -> 0.10 on the sample case). Whole-component removal only when the
    component is small (small_ml) or, with PET, when the scribble sits at the
    component's own uptake level (wholly-false physiological organ: bladder,
    kidney, brain are uniformly hot; a rim scribble sits in cold overseg
    around a hot true center). Otherwise chip locally within local_mm."""
    pred = pred.astype(bool, copy=True)
    if not bg_points or not pred.any():
        return pred
    vox_ml = float(np.prod(spacing_xyz)) / 1000.0
    lab, _ = ndimage.label(pred, structure=_STRUCT26)
    seeds_by_label = {}
    for p in bg_points:
        if not _inside(p, pred.shape):
            continue
        l = int(lab[p])
        if l == 0:
            sl = tuple(slice(max(0, c - tol), min(s, c + tol + 1))
                       for c, s in zip(p, pred.shape))
            w = lab[sl]
            nz = w[w > 0]
            if nz.size:
                l = int(np.bincount(nz).argmax())
        if l:
            seeds_by_label.setdefault(l, []).append(p)
    objects = ndimage.find_objects(lab)
    for l, seeds in seeds_by_label.items():
        box = objects[l - 1]
        comp_box = lab[box] == l
        vol_ml = comp_box.sum() * vox_ml
        whole = vol_ml <= small_ml
        if not whole and pet is not None and vol_ml <= whole_cap_ml:
            scr_suv = float(np.median([pet[p] for p in seeds]))
            comp_p75 = float(np.percentile(pet[box][comp_box], 75))
            whole = comp_p75 <= 0 or scr_suv >= suv_ratio * comp_p75
        if whole:
            pred[box][comp_box] = False
        else:
            # Conservative chip commensurate with the scribble itself: the
            # scribble traces the overseg region's largest cross-section, so
            # the plausible false region is an in-plane neighbourhood of the
            # scribble extended a few slices — NOT a ball (measured: a 25mm
            # ball around a 15-voxel rim sliver deleted 2013 true voxels for
            # 229 false ones). Depth-gating keeps the deep true core.
            seed_mask = np.zeros(comp_box.shape, dtype=bool)
            for p in seeds:
                seed_mask[tuple(p[a] - box[a].start for a in range(3))] = True
            inplane_mm = 4.0
            z_mm = 6.0
            # scale z so the single threshold allows inplane_mm laterally
            # and z_mm axially (ellipsoidal neighbourhood)
            dist_scr = ndimage.distance_transform_edt(
                ~seed_mask,
                sampling=(spacing_xyz[0], spacing_xyz[1],
                          spacing_xyz[2] * inplane_mm / z_mm))
            depth = ndimage.distance_transform_edt(
                comp_box, sampling=spacing_xyz)
            scr_depth = float(np.median(depth[seed_mask]))
            pred[box][comp_box & (dist_scr <= inplane_mm)
                      & (depth <= scr_depth + 4.0)] = False
    return pred


# ---------------------------------------------------------------------------
# Foreground: PET-adaptive region growing
# ---------------------------------------------------------------------------

def grow_fg_region(pred, pet, scribble_points, spacing_xyz=(2.0, 2.0, 3.0),
                   pad_mm=80.0, fracs=(0.45, 0.55, 0.65, 0.8),
                   cap_ml=400.0, fallback_r=2):
    """Union a PET-thresholded connected region grown from one fg scribble.

    pet=None (or growth failure) degrades to a small dilation of the scribble,
    which is still guaranteed net-positive (scribble voxels are true tumor)."""
    pred = pred.astype(bool, copy=True)
    pts = [p for p in scribble_points if _inside(p, pred.shape)]
    if not pts:
        return pred
    region = None
    if pet is not None:
        region = _grow(pet, pts, spacing_xyz, pad_mm, fracs, cap_ml)
    if region is None:
        region = _dilated_scribble(pred.shape, pts, fallback_r)
    return pred | region


def _grow(pet, pts, spacing_xyz, pad_mm, fracs, cap_ml):
    shape = pet.shape
    pad = [max(1, int(round(pad_mm / s))) for s in spacing_xyz]
    lo = [max(0, min(p[a] for p in pts) - pad[a]) for a in range(3)]
    hi = [min(shape[a], max(p[a] for p in pts) + pad[a] + 1) for a in range(3)]
    box = tuple(slice(lo[a], hi[a]) for a in range(3))
    pet_box = pet[box]
    seeds_box = [tuple(p[a] - lo[a] for a in range(3)) for p in pts]
    seed_vals = np.array([pet_box[s] for s in seeds_box], dtype=np.float32)
    ref = float(np.median(seed_vals))
    if not np.isfinite(ref) or ref <= 0:
        return None
    vox_ml = float(np.prod(spacing_xyz)) / 1000.0
    for frac in fracs:
        mask = pet_box >= frac * ref
        lab, _ = ndimage.label(mask, structure=_STRUCT26)
        keep = {int(lab[s]) for s in seeds_box} - {0}
        if not keep:
            continue
        region_box = np.isin(lab, list(keep))
        if region_box.sum() * vox_ml <= cap_ml:
            region = np.zeros(shape, dtype=bool)
            region[box] = region_box
            return region
    return None


def _dilated_scribble(shape, pts, r):
    m = np.zeros(shape, dtype=bool)
    for p in pts:
        m[p] = True
    return ndimage.binary_dilation(m, structure=_STRUCT26, iterations=r)


# ---------------------------------------------------------------------------
# Top-level per-iteration application
# ---------------------------------------------------------------------------

def apply_scribbles(pred, pet, clicks, prev_clicks=None,
                    spacing_xyz=(2.0, 2.0, 3.0)):
    """Conform pred to the clicks json. Incremental when prev_clicks given
    (only the new scribble reshapes regions); all points are hard-enforced."""
    fg, bg = split_clicks(clicks)
    if prev_clicks is not None:
        pfg, pbg = split_clicks(prev_clicks)
        new_fg, new_bg = new_suffix(fg, pfg), new_suffix(bg, pbg)
    else:
        new_fg, new_bg = fg, bg

    pred = pred.astype(bool, copy=True)
    if new_bg:
        pred = remove_bg_components(pred, new_bg, spacing_xyz, pet=pet)
    for scribble in group_by_slice(new_fg):
        pred = grow_fg_region(pred, pet, scribble, spacing_xyz)

    # hard enforcement: scribble voxels are ground-truth certain
    for p in fg:
        if _inside(p, pred.shape):
            pred[p] = True
    for p in bg:
        if _inside(p, pred.shape):
            pred[p] = False
    return pred
