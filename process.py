"""autoPET V interactive lesion segmentation — conformance algorithm.

Per GC invocation (= one interactive iteration of one case):
  - iteration 0 (no clicks): run the pretrained nnU-Net baseline (4 channels,
    zero scribble heatmaps), cache the prediction in /cache.
  - iteration t>0, warm cache: load the cached prediction and apply the NEW
    scribble via conformance (PET region-grow for tumor scribbles, component
    removal for background scribbles). No re-inference — seconds, monotone.
  - cold cache with clicks (defensive): run nnU-Net with scribble heatmap
    channels, then conform ALL scribbles.

Output mask copies the PET image geometry. State is keyed by input filename
uuid; /cache is cleared by the evaluator between cases.
"""

import json
import os
import shutil
import subprocess
import traceback

import numpy as np
import SimpleITK

from conformance import apply_scribbles, split_clicks

CACHE_DIR = "/cache"


class AutopetInteractive:

    def __init__(self):
        self.input_path = "/input/"
        self.output_path = "/output/images/tumor-lesion-segmentation/"
        self.nii_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs"
        )
        self.result_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result"
        )
        self.nii_seg_file = "TCIA_001.nii.gz"

    # ------------------------------------------------------------------
    # IO helpers (all volumes handled in nib order (x,y,z) via transpose)
    # ------------------------------------------------------------------

    def _read_xyz(self, path):
        img = SimpleITK.ReadImage(path)
        arr = SimpleITK.GetArrayFromImage(img).transpose(2, 1, 0)
        return img, arr

    def _write_mask(self, arr_xyz, ref_img, out_path):
        m = SimpleITK.GetImageFromArray(
            arr_xyz.astype(np.uint8).transpose(2, 1, 0))
        m.CopyInformation(ref_img)
        SimpleITK.WriteImage(m, out_path, True)

    def _near_mask(self, shape, points, spacing_xyz, radius_mm=60.0):
        import scipy.ndimage as ndi
        seed = np.zeros(shape, dtype=bool)
        for p in points:
            if all(0 <= c < s for c, s in zip(p, shape)):
                seed[p] = True
        dist = ndi.distance_transform_edt(~seed, sampling=spacing_xyz)
        return dist <= radius_mm

    def _write_heatmap(self, ref_nii_path, points_xyz, out_path):
        """Binary click heatmap channel (sigma=0, as the baseline trains
        with), written on the PET grid. Points are (x,y,z) indices."""
        ref = SimpleITK.ReadImage(ref_nii_path)
        arr = np.zeros(SimpleITK.GetArrayFromImage(ref).shape,
                       dtype=np.float32)  # (z,y,x)
        for p in points_xyz:
            x, y, z = p
            if 0 <= z < arr.shape[0] and 0 <= y < arr.shape[1] \
                    and 0 <= x < arr.shape[2]:
                arr[z, y, x] = 1.0
        img = SimpleITK.GetImageFromArray(arr)
        img.CopyInformation(ref)
        SimpleITK.WriteImage(img, out_path, True)

    def load_clicks(self):
        p = os.path.join(self.input_path, "lesion-clicks.json")
        if not os.path.exists(p):
            return {"points": []}
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return {"points": []}

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def cache_paths(self, uuid):
        return (os.path.join(CACHE_DIR, f"{uuid}_pred.npz"),
                os.path.join(CACHE_DIR, f"{uuid}_clicks.json"))

    def load_cache(self, uuid):
        pred_p, clicks_p = self.cache_paths(uuid)
        try:
            if os.path.exists(pred_p) and os.path.exists(clicks_p):
                pred = np.load(pred_p)["pred"].astype(bool)
                with open(clicks_p) as f:
                    clicks = json.load(f)
                return pred, clicks
        except Exception:
            traceback.print_exc()
        return None, None

    def save_cache(self, uuid, pred, clicks):
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            pred_p, clicks_p = self.cache_paths(uuid)
            np.savez_compressed(pred_p, pred=pred.astype(np.uint8))
            with open(clicks_p, "w") as f:
                json.dump(clicks, f)
        except Exception:
            traceback.print_exc()

    # ------------------------------------------------------------------
    # nnU-Net
    # ------------------------------------------------------------------

    def run_nnunet(self, ct_path, pet_path, clicks):
        os.makedirs(self.nii_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)
        for f in os.listdir(self.nii_path):
            os.remove(os.path.join(self.nii_path, f))
        for f in os.listdir(self.result_path):
            os.remove(os.path.join(self.result_path, f))

        ct_nii = os.path.join(self.nii_path, "TCIA_001_0000.nii.gz")
        pet_nii = os.path.join(self.nii_path, "TCIA_001_0001.nii.gz")
        SimpleITK.WriteImage(SimpleITK.ReadImage(ct_path), ct_nii, True)
        SimpleITK.WriteImage(SimpleITK.ReadImage(pet_path), pet_nii, True)

        fg, bg = split_clicks(clicks)
        self._write_heatmap(pet_nii, fg,
                            os.path.join(self.nii_path, "TCIA_001_0002.nii.gz"))
        self._write_heatmap(pet_nii, bg,
                            os.path.join(self.nii_path, "TCIA_001_0003.nii.gz"))

        subprocess.run(
            f"nnUNetv2_predict -i {self.nii_path} -o {self.result_path} "
            f"-d 998 -c 3d_fullres -f 0 --disable_tta",
            shell=True, check=True)

        _, pred = self._read_xyz(os.path.join(self.result_path,
                                              self.nii_seg_file))
        return pred > 0

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def process(self):
        ct_mha = os.listdir(os.path.join(self.input_path, "images/ct/"))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, "images/pet/"))[0]
        uuid = os.path.splitext(ct_mha)[0]
        ct_path = os.path.join(self.input_path, "images/ct/", ct_mha)
        pet_path = os.path.join(self.input_path, "images/pet/", pet_mha)

        clicks = self.load_clicks()
        n_points = len(clicks.get("points", []))
        print(f"case={uuid} points={n_points}")

        pet_img, pet = self._read_xyz(pet_path)
        spacing = tuple(pet_img.GetSpacing())  # (x, y, z) mm

        cached_pred, cached_clicks = self.load_cache(uuid)

        if n_points == 0:
            print("iteration 0: nnU-Net inference")
            pred = self.run_nnunet(ct_path, pet_path, clicks)
        elif cached_pred is not None and cached_pred.shape == pet.shape:
            print("warm cache: incremental conformance")
            pred = apply_scribbles(cached_pred, pet, clicks,
                                   prev_clicks=cached_clicks,
                                   spacing_xyz=spacing)
            # On a NEW foreground scribble, also re-run the model with the
            # scribble heatmap channels and union its additions locally: the
            # model recovers the missed lesion's true 3D shape better than
            # thresholding alone. Never lets the model retract elsewhere.
            fg_all, bg_all = split_clicks(clicks)
            fg_prev, _ = split_clicks(cached_clicks or {})
            new_fg = fg_all[len(fg_prev):] if fg_all[:len(fg_prev)] == fg_prev else fg_all
            if new_fg:
                try:
                    nn_pred = self.run_nnunet(ct_path, pet_path, clicks)
                    near = self._near_mask(pet.shape, new_fg, spacing, radius_mm=60.0)
                    pred = pred | (nn_pred & near)
                    for p in bg_all:
                        if all(0 <= c < s for c, s in zip(p, pred.shape)):
                            pred[p] = False
                except Exception:
                    traceback.print_exc()
        else:
            print("cold cache with clicks: nnU-Net + full conformance")
            pred = self.run_nnunet(ct_path, pet_path, clicks)
            pred = apply_scribbles(pred, pet, clicks, prev_clicks=None,
                                   spacing_xyz=spacing)

        self.save_cache(uuid, pred, clicks)

        os.makedirs(self.output_path, exist_ok=True)
        out = os.path.join(self.output_path, uuid + ".mha")
        self._write_mask(pred, pet_img, out)
        print(f"output written: {out} volume_voxels={int(pred.sum())}")


if __name__ == "__main__":
    print("START autoPET V conformance algorithm")
    AutopetInteractive().process()
