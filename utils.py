"""
Utility helpers for the AutoPET V 2026 interactive-segmentation GC container.

设计原则 (Karpathy: simplicity first):
- click 的 EDT dense 编码 *不在这里重写*。它由 nnunetv2 包内的
  `sparse_to_dense_point_nnInteractive` (+ `PointInteraction_stub`, +
  `preprocess_point`) 负责，容器会 bake 整个 autoPET-interactive 包，所以
  process.py 直接 import 复用，确保推理 click 编码与训练逐字一致 (EDT，sigma=point_width)。
- 本文件只保留两件容器特有的轻量逻辑：
    1. 把 GC 的 lesion-clicks.json 透传成 predict_single_npy_array 期望的 dict
       (其实就是原样 {"points": [...]}, 这里做格式校验 + 容错)。
    2. scribble 强制后处理 (FG click -> 1, BG click -> 0)，在原始图像 voxel 空间。
"""
import json

import numpy as np
from scipy import ndimage


def load_gc_clicks(json_path):
    """
    读取 GC 的 lesion-clicks.json，返回 predict_single_npy_array 直接可用的 dict。

    GC 格式 (累积追加):
      {
        "version": {...},
        "type": "Multiple points",
        "points": [
          {"point": [x, y, z], "name": "tumor"},
          {"point": [x, y, z], "name": "background"},
          ...
        ]
      }

    - iter 0 时 points 为空 -> 返回 {"points": []}，predict_single_npy_array
      会得到空 click，模型出无 scribble 的初始预测。
    - point 坐标是原始图像的 voxel 坐标，顺序 [x, y, z] (与 sparse_to_dense_point_nnInteractive
      / preprocess_point 约定一致，后者内部做 x,y,z=point 解包)。
    """
    if json_path is None:
        return {"points": []}
    try:
        with open(json_path, "r") as f:
            gc_dict = json.load(f)
    except FileNotFoundError:
        print(f"[WARN] click json not found at {json_path}, treating as iter-0 (no clicks)")
        return {"points": []}

    points = gc_dict.get("points", [])
    # 校验 name 只允许 tumor / background；非法点直接丢弃并告警，不让推理崩
    clean = []
    for p in points:
        name = p.get("name")
        if name in ("tumor", "background"):
            clean.append({"point": [float(c) for c in p["point"]], "name": name})
        else:
            print(f"[WARN] dropping click with unknown name={name!r}: {p}")
    return {"points": clean}


def apply_scribble_override(seg, clicks, radius=2, verbose=True):
    """
    Scribble 强制后处理 (提交路线图 Step 2，无需重训，直接抬高 iter1-4 的 AUC-Dice)。

    seg:    np.ndarray，二值 lesion mask，原始图像形状。
            ⚠️ seg 来自 SimpleITKIO/nnUNet 管线，轴序为 [z, y, x] (SimpleITK 约定)。
    clicks: load_gc_clicks 的返回，point 顺序为原始图像 voxel [x, y, z]。

    因此索引时必须把 [x, y, z] 反转成 [z, y, x] 才能命中 seg 的对应体素。
    - tumor click 命中体素 -> 强制 1
    - background click 命中体素 -> 强制 0

    ⚠️ 需实测复核的点：seg 的轴序是否确为 [z,y,x]，以及 click 坐标系
    是否就是 SimpleITK 读图后的 voxel index (而不是物理坐标 / nibabel 轴序)。
    相关轴序与几何对齐由容器端到端测试验证。
    """
    seg = seg.astype(np.uint8, copy=True)
    Z, Y, X = seg.shape  # SimpleITK 轴序 [z, y, x]
    r = int(radius)
    # 半径 r 的球偏移：FG->1 / BG->0 在 click 周围画小球，而非单体素。
    # 对齐训练 EDT point_width=2；单体素对 Dice 几乎无益、对 DMM(CC 级 F1@IoU≥0.1)
    # 可能造成有害的 1-voxel FP 组件。
    ball = [(dz, dy, dx)
            for dz in range(-r, r + 1)
            for dy in range(-r, r + 1)
            for dx in range(-r, r + 1)
            if dz * dz + dy * dy + dx * dx <= r * r]
    n_fg = n_bg = n_skip = 0
    for p in clicks.get("points", []):
        x, y, z = [int(round(c)) for c in p["point"]]  # click 为原图 voxel [x, y, z]
        if not (0 <= z < Z and 0 <= y < Y and 0 <= x < X):
            n_skip += 1
            continue
        val = 1 if p["name"] == "tumor" else 0
        for dz, dy, dx in ball:
            zz, yy, xx = z + dz, y + dy, x + dx  # [x,y,z] -> seg 索引 [z,y,x]
            if 0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X:
                seg[zz, yy, xx] = val
        if val == 1:
            n_fg += 1
        else:
            n_bg += 1
    if verbose:
        print(f"[scribble override r={r}] fg={n_fg} bg={n_bg} skipped(center OOB)={n_skip}")
    return seg


def _click_anchor_mask(shape, clicks, name, radius):
    """Return a z-y-x mask around valid clicks of one semantic type."""
    anchors = np.zeros(shape, dtype=bool)
    z_size, y_size, x_size = shape
    radius = int(radius)
    offsets = [
        (dz, dy, dx)
        for dz in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dz * dz + dy * dy + dx * dx <= radius * radius
    ]
    for item in clicks.get("points", []):
        if item.get("name") != name:
            continue
        x, y, z = [int(round(value)) for value in item["point"]]
        if not (0 <= z < z_size and 0 <= y < y_size and 0 <= x < x_size):
            continue
        for dz, dy, dx in offsets:
            zz, yy, xx = z + dz, y + dy, x + dx
            if 0 <= zz < z_size and 0 <= yy < y_size and 0 <= xx < x_size:
                anchors[zz, yy, xx] = True
    return anchors


def _components_touching(delta, anchors):
    """Keep only 26-connected delta components touched by an anchor mask."""
    if not np.any(delta) or not np.any(anchors):
        return np.zeros_like(delta, dtype=bool)
    structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
    labels, _ = ndimage.label(delta, structure=structure)
    selected = np.unique(labels[anchors & (labels > 0)])
    if selected.size == 0:
        return np.zeros_like(delta, dtype=bool)
    return np.isin(labels, selected)


def apply_click_local_delta_fusion(base_seg, prompted_seg, clicks, anchor_radius=2, verbose=True):
    """Fuse only click-supported interactive deltas into the champion mask.

    Positive and negative corrections are intentionally asymmetric:
    - tumor clicks may add only prompted-vs-base foreground components touching a tumor click;
    - background clicks may remove only base-vs-prompted foreground components touching a background click.

    Every unrelated prompted-model change is discarded, so an empty click list returns the
    champion mask bit-for-bit. ``apply_scribble_override`` should run afterwards to enforce
    the click centers even when the prompted network offers no usable connected delta.
    """
    base = np.asarray(base_seg) > 0
    prompted = np.asarray(prompted_seg) > 0
    if base.shape != prompted.shape or base.ndim != 3:
        raise ValueError(f"expected matching 3D masks, got base={base.shape} prompted={prompted.shape}")

    positive_anchors = _click_anchor_mask(base.shape, clicks, "tumor", anchor_radius)
    negative_anchors = _click_anchor_mask(base.shape, clicks, "background", anchor_radius)
    positive_delta = prompted & ~base
    negative_delta = base & ~prompted
    accepted_additions = _components_touching(positive_delta, positive_anchors)
    accepted_removals = _components_touching(negative_delta, negative_anchors)

    fused = (base & ~accepted_removals) | accepted_additions
    if verbose:
        rejected_changes = np.count_nonzero((base ^ prompted) & ~(accepted_additions | accepted_removals))
        print(
            "[click-local delta fusion] "
            f"added={np.count_nonzero(accepted_additions)} "
            f"removed={np.count_nonzero(accepted_removals)} "
            f"rejected_unprompted_changes={rejected_changes}"
        )
    return fused.astype(np.uint8)


def prune_small_new_components(base_seg, fused_seg, minimum_size, verbose=True):
    """Drop only small isolated components newly introduced beyond the champion base.

    Champion components are never removed here. This applies the deployed tracer-specific
    CC prior to click additions without reprocessing or damaging the immutable base mask.
    """
    base = np.asarray(base_seg) > 0
    fused = np.asarray(fused_seg) > 0
    if base.shape != fused.shape or base.ndim != 3:
        raise ValueError(f"expected matching 3D masks, got base={base.shape} fused={fused.shape}")
    minimum_size = int(minimum_size)
    if minimum_size <= 1:
        return fused.astype(np.uint8)
    structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
    labels, count = ndimage.label(fused, structure=structure)
    output = fused.copy()
    removed_components = removed_voxels = 0
    for component_id in range(1, count + 1):
        component = labels == component_id
        if np.any(component & base):
            continue
        size = int(np.count_nonzero(component))
        if size < minimum_size:
            output[component] = False
            removed_components += 1
            removed_voxels += size
    if verbose:
        print(
            "[new-component guard] "
            f"minimum_size={minimum_size} removed_components={removed_components} "
            f"removed_voxels={removed_voxels}"
        )
    return output.astype(np.uint8)
