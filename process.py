"""AutoPET V 2026 LocalEdit--TACE HoleGuard Fusion v5 container.

Every invocation obtains the hard M0 and float32 lesion probability from one
Dataset222 AutoPET-III five-fold forward pass. Iter0 remains bit-identical to
the deployed champion. Later invocations transact cumulative clicks against
the previous accepted mask using frozen tracer-specific TACE and LocalEdit
proposals. HoleGuard fails closed to TACE, then to the previous mask.
"""
import os
import gc
import json
import subprocess
import sys
import tempfile

import cc3d
import numpy as np
import SimpleITK

from tracer_router import predict_tracer
# ── 路径常量 (GC 接口 + bake 进镜像的权重目录) ──────────────────────────────
INPUT_PATH = "/input"
OUTPUT_PATH = "/output/images/tumor-lesion-segmentation"
CLICK_JSON = "/input/lesion-clicks.json"


def postprocess_champion_mask(raw_base, tracer):
    """Apply the tracer-specific iteration-0 component floor."""
    base = np.asarray(raw_base, dtype=np.uint8)
    before = int(base.sum())
    if tracer == "fdg":
        threshold = 25
    elif tracer == "psma":
        threshold = 5
    else:
        raise RuntimeError(f"unsupported tracer route {tracer!r}")
    if before:
        base = (
            cc3d.dust(base, threshold=threshold, connectivity=18) > 0
        ).astype(np.uint8)
    return base, f"connectivity18_{tracer}{threshold}"


def _load_gc_clicks_lightweight(json_path):
    """Read clicks without importing the interactive model or CUDA stack."""
    if json_path is None:
        return {"points": []}
    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        print(f"[WARN] click json not found at {json_path}, treating as iter-0 (no clicks)")
        return {"points": []}

    clean = []
    for item in payload.get("points", []):
        name = item.get("name")
        if name not in ("tumor", "background"):
            print(f"[WARN] dropping click with unknown name={name!r}: {item}")
            continue
        clean.append(
            {"point": [float(value) for value in item["point"]], "name": name}
        )
    return {"points": clean}

def _resolve_champion_root():
    for path in (
        "/opt/algorithm/champion",
        "/opt/app/champion",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "champion"),
    ):
        if os.path.isfile(os.path.join(path, "predict_wrap.py")) and os.path.isdir(os.path.join(path, "model")):
            return path
    raise FileNotFoundError("native AutoPET-III champion runtime was not baked into the image")

class AutopetInteractive:
    def __init__(self):
        os.makedirs(OUTPUT_PATH, exist_ok=True)

    @staticmethod
    def check_gpu():
        # Interactive-only import: iter0 must leave the parent CUDA context
        # untouched before the champion five-fold subprocess starts.
        import torch

        print("Checking GPU availability")
        avail = torch.cuda.is_available()
        print(f"Available: {avail}")
        print(f"Device count: {torch.cuda.device_count()}")
        if avail:
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
            print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory}")

    @staticmethod
    def _predict_champion(ct_path, pet_path, tracer, *, save_probability):
        """Return the champion hard M0 and, when requested, its probability.

        Isolation is required because the champion and interactive models ship two
        incompatible forks under the same ``nnunetv2`` package name. Iter0 never
        consumes a probability map, so omitting ``--save_probabilities`` there
        avoids a full two-channel float32 export without changing the hard mask.
        """
        champion_root = _resolve_champion_root()
        with tempfile.TemporaryDirectory(prefix="autopet_champion_") as temporary:
            input_dir = os.path.join(temporary, "input")
            output_dir = os.path.join(temporary, "output")
            os.makedirs(input_dir)
            os.makedirs(output_dir)
            ct_image = SimpleITK.ReadImage(ct_path)
            pet_image = SimpleITK.ReadImage(pet_path)
            SimpleITK.WriteImage(ct_image, os.path.join(input_dir, "case_0000.nii.gz"), True)
            SimpleITK.WriteImage(pet_image, os.path.join(input_dir, "case_0001.nii.gz"), True)

            command = [
                sys.executable,
                "-u",
                os.path.join(champion_root, "predict_wrap.py"),
                "-i", input_dir,
                "-o", output_dir,
                "-d", "222",
                "-tr", "autoPET3_Trainer",
                "-p", "nnUNetResEncUNetLPlansMultiTalent",
                "-c", "3d_fullres_bs3",
                "-f", "0", "1", "2", "3", "4",
                "--disable_tta",
            ]
            if save_probability:
                command.append("--save_probabilities")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.path.join(champion_root, "autopet-3-submission")
            environment["nnUNet_results"] = os.path.join(champion_root, "model")
            environment["nnUNet_compile"] = "0"
            print("Running native champion five-fold base", flush=True)
            subprocess.run(command, env=environment, check=True)
            prediction_path = os.path.join(output_dir, "case.nii.gz")
            if not os.path.isfile(prediction_path):
                raise FileNotFoundError(f"champion did not produce {prediction_path}")
            prediction_image = SimpleITK.ReadImage(prediction_path)
            raw_base = (SimpleITK.GetArrayFromImage(prediction_image) > 0).astype(np.uint8)
            lesion_probability = None
            if save_probability:
                probability_path = os.path.join(output_dir, "case.npz")
                if not os.path.isfile(probability_path):
                    raise FileNotFoundError(f"champion did not produce {probability_path}")
                with np.load(probability_path) as archive:
                    probabilities = np.asarray(archive["probabilities"], dtype=np.float32)
                if probabilities.ndim != 4 or probabilities.shape[0] != 2:
                    raise RuntimeError(
                        f"expected champion probabilities [2,z,y,x], got {probabilities.shape}"
                    )
                # Detach channel 1 so the two-channel export buffer can be released
                # before CPU-side TACE allocates its Gaussian working arrays.
                lesion_probability = probabilities[1].copy()
                if lesion_probability.shape != raw_base.shape:
                    raise RuntimeError(
                        f"probability/base shape mismatch {lesion_probability.shape} != {raw_base.shape}"
                    )
                probability_argmax = np.argmax(probabilities, axis=0).astype(np.uint8)
                if not np.array_equal(probability_argmax, raw_base):
                    disagreement = int(np.count_nonzero(probability_argmax != raw_base))
                    raise RuntimeError(
                        f"champion probability/hard-mask mismatch at {disagreement} voxels"
                    )
                del probabilities, probability_argmax

        before = int(raw_base.sum())
        base, postprocess = postprocess_champion_mask(raw_base, tracer)
        print(
            f"champion tracer={tracer} postprocess={postprocess} "
            f"foreground={before}->{int(base.sum())} "
        )
        return base, lesion_probability

    @staticmethod
    def _predict_champion_base_and_probability(ct_path, pet_path, tracer):
        """Return the champion hard M0 and native-grid lesion probability."""
        base, lesion_probability = AutopetInteractive._predict_champion(
            ct_path, pet_path, tracer, save_probability=True
        )
        assert lesion_probability is not None
        return base, lesion_probability

    @staticmethod
    def _predict_champion_base(ct_path, pet_path, tracer):
        """Run the identical champion ensemble without an unused probability export."""
        base, lesion_probability = AutopetInteractive._predict_champion(
            ct_path, pet_path, tracer, save_probability=False
        )
        assert lesion_probability is None
        return base

    @staticmethod
    def _find_single(subdir):
        d = os.path.join(INPUT_PATH, subdir)
        files = [f for f in os.listdir(d) if f.endswith(".mha")]
        assert len(files) == 1, f"expected exactly 1 .mha in {d}, found {files}"
        return os.path.join(d, files[0]), os.path.splitext(files[0])[0]

    def process(self):
        # ── 1. 先读路径和 clicks；iter0 在任何 Torch/CUDA import 前分流 ──
        ct_path, uuid = self._find_single("images/ct")
        pet_path, _ = self._find_single("images/pet")
        print(f"CT={ct_path}\nPET={pet_path}\nuuid={uuid}")
        clicks = _load_gc_clicks_lightweight(
            CLICK_JSON if os.path.isfile(CLICK_JSON) else None
        )
        print(f"num clicks: {len(clicks['points'])}")

        # Tracer routing is CPU-only and uses native z-y-x arrays. Reading with
        # SimpleITK avoids importing either nnUNet fork in the iter0 parent.
        ct_image = SimpleITK.ReadImage(ct_path)
        pet_image = SimpleITK.ReadImage(pet_path)
        ct_native = SimpleITK.GetArrayFromImage(ct_image)
        pet_native = SimpleITK.GetArrayFromImage(pet_image)
        tracer, tracer_details = predict_tracer(
            ct_native, pet_native, return_details=True
        )
        print(
            f"tracer={tracer} psma_votes={tracer_details['psma_votes']}/3 "
            f"unanimous={tracer_details['unanimous']} "
            f"votes={tracer_details['votes']} "
            f"invalid_pet={tracer_details['pet_invalid_count']} "
            f"invalid_ct={tracer_details['ct_invalid_count']}"
        )
        # ── 2. iter0：干净父进程 + 原生冠军五折子进程 ───────────────────
        if not clicks["points"]:
            # No Torch import or CUDA call has occurred in this parent process.
            del ct_native, pet_native, ct_image, pet_image
            gc.collect()
            champion_base = self._predict_champion_base(ct_path, pet_path, tracer)
            seg = champion_base
            print(
                "iter0: clean parent; writing native champion base without "
                "probability export, parent CUDA initialization, or interactive inference"
            )
            self._write_output(seg, None, ct_path, uuid)
            print("Done.")
            return

        # ── 3. 有 clicks：冠军子进程仍先获得干净 CUDA 上下文 ─────────────
        del ct_native, pet_native, ct_image, pet_image
        gc.collect()
        champion_base, lesion_probability = self._predict_champion_base_and_probability(
            ct_path, pet_path, tracer
        )

        # Only after the champion child has exited may the parent initialize
        # CUDA and import the interactive stack.
        self.check_gpu()
        from gaussian_probability_click import (
            apply_gaussian_probability_clicks,
            autopet_v6_config,
        )
        from localedit_tace_gate import select_localedit_gaussian_candidate
        from pf_tace_runtime import clicks_to_zyx_cores, load_previous_or_m0

        # ── 4. 两个冻结分支提出局部候选，再由 V5 事务门裁决 ──────────
        output_path = os.path.join(OUTPUT_PATH, uuid + ".mha")
        previous_state = load_previous_or_m0(output_path, ct_path, champion_base)
        positive_core, negative_core, click_stats = clicks_to_zyx_cores(
            clicks, champion_base.shape
        )
        # SimpleITK spacing is x-y-z; all runtime arrays are z-y-x.
        spacing_zyx = tuple(float(value) for value in reversed(SimpleITK.ReadImage(ct_path).GetSpacing()))
        tace_result = apply_gaussian_probability_clicks(
            lesion_probability,
            previous_state.mask,
            positive_core,
            negative_core,
            spacing_zyx,
            autopet_v6_config(tracer),
        )
        # TACE-only fusion: the LocalEdit candidate slot is filled with the
        # previous accepted mask, which is a no-op proposal — the transaction
        # gate then adjudicates the TACE proposals alone, with the split/merge
        # and HoleGuard checks unchanged.
        fusion = select_localedit_gaussian_candidate(
            tracer,
            champion_base,
            previous_state.mask,
            previous_state.mask,
            tace_result.mask,
            positive_core,
            negative_core,
            hole_guarded_gaussian_fallback=True,
        )
        seg = fusion.mask.astype(np.uint8)
        telemetry = {
            **click_stats,
            "state_source": previous_state.source,
            "fallback_reason": previous_state.fallback_reason,
            "tace_added_voxels": int(tace_result.add.sum()),
            "tace_removed_voxels": int(tace_result.remove.sum()),
            "tace_effective_add_strength": tace_result.effective_add_strength,
            "tace_effective_remove_strength": tace_result.effective_remove_strength,
            "tace_rejected_oversize_components": tace_result.rejected_oversize_components,
            "tace_rejected_add_merge_components": tace_result.rejected_add_merge_components,
            "tace_rejected_remove_split_voxels": tace_result.rejected_remove_split_voxels,
            "selected_source": fusion.source,
            "selection_reason": fusion.reason,
            "accepted_add_voxels": fusion.accepted_add_voxels,
            "accepted_remove_voxels": fusion.accepted_remove_voxels,
            "rejected_split_remove_voxels": fusion.rejected_split_remove_voxels,
            "new_hole_voxels_candidate": fusion.new_hole_voxels_candidate,
            "new_hole_voxels_tace": fusion.new_hole_voxels_gaussian,
            "hole_fallback_level": fusion.hole_fallback_level,
        }
        print(f"LocalEdit--TACE HoleGuard Fusion v5 telemetry: {telemetry}")

        # ── 5. 写出 uuid.mha (几何对齐到原始 PET/CT) ──────────────────────
        self._write_output(seg, None, ct_path, uuid)
        print("Done.")

    @staticmethod
    def _write_output(seg, props, ref_mha_path, uuid):
        """
        seg: np.ndarray [z,y,x] 二值。用 SimpleITKIO 写出，并把几何信息
        (spacing/origin/direction) 对齐到原始输入图像，保证 GC 能配准 GT。

        ⚠️ 用参考图像 (CT) 拷贝几何最稳：SimpleITKIO 的 props 里有 sitk_stuff，
        但直接 CopyInformation 自参考图最不易错。需复核 props 写出是否
        已自带正确几何 (若是则可去掉 CopyInformation)。
        """
        out_path = os.path.join(OUTPUT_PATH, uuid + ".mha")
        # seg 轴序 [z,y,x] -> SimpleITK GetImageFromArray 期望 [z,y,x] (它内部转 [x,y,z])，一致。
        seg_img = SimpleITK.GetImageFromArray(seg.astype(np.uint8))
        ref = SimpleITK.ReadImage(ref_mha_path)
        # 加固：seg resample 回原图后必须和参考 CT 同 size，否则 CopyInformation 几何错配。
        assert seg_img.GetSize() == ref.GetSize(), \
            f"seg size {seg_img.GetSize()} != ref CT size {ref.GetSize()}"
        seg_img.CopyInformation(ref)
        SimpleITK.WriteImage(seg_img, out_path, useCompression=True)
        print(f"Output written to: {out_path}")


if __name__ == "__main__":
    print("START")
    AutopetInteractive().process()
