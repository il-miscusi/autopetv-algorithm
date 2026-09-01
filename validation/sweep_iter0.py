#!/usr/bin/env python3
"""Colab sweep for autoPET V iteration-0 instance postprocessing.

Run on a T4 after uploading ``apv-src.zip`` to ``/content``. The archive must
contain ``algo-repo/``. Champion inference is executed once per volume; all
dust/SUV policies are then evaluated with the official connectivity-18 and
IoU >= 0.1 detection semantics.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


WORK = Path("/content/autopet_iter0_sweep")
SOURCE_ARCHIVE = Path("/content/apv-src.zip")
RESULTS_PATH = WORK / "sweep_results.json"
CASE_IDS = ("train_0001", "train_0002")
TRACERS = ("PSMA", "FDG")
DUST_THRESHOLDS = (0, 25, 50, 100, 200)
SUV_MAX_THRESHOLDS = (0.0, 2.0, 2.5, 3.0, 4.0)


def run(command: str) -> None:
    print(f"+ {command}", flush=True)
    subprocess.run(command, shell=True, check=True)


def install_dependencies() -> None:
    run(
        "pip install -q SimpleITK connected-components-3d remotezip "
        "acvl-utils==0.2.6 dynamic-network-architectures==0.3.1 "
        "batchgenerators batchgeneratorsv2==0.3.3 blosc2 networkx"
    )


def stage_source() -> Path:
    if not SOURCE_ARCHIVE.is_file():
        raise FileNotFoundError(f"upload {SOURCE_ARCHIVE} before running")
    source = WORK / "source"
    algorithm = WORK / "algo-repo"
    if not algorithm.is_dir():
        source.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(SOURCE_ARCHIVE) as archive:
            archive.extractall(source)
        shutil.copytree(source / "algo-repo", algorithm)
    return algorithm


def stage_weights(algorithm: Path) -> None:
    model = algorithm / "champion" / "model"
    model.mkdir(parents=True, exist_ok=True)
    checkpoints = list(model.rglob("checkpoint_final.pth"))
    if len(checkpoints) == 5:
        return
    archive = WORK / "champion.zip"
    run(
        "wget -q -O "
        f"{archive} "
        "'https://zenodo.org/records/14007247/files/"
        "autoPET-3-LesionTracer.zip?download=1'"
    )
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(model)
    archive.unlink()
    checkpoints = list(model.rglob("checkpoint_final.pth"))
    if len(checkpoints) != 5:
        raise RuntimeError(f"expected 5 checkpoints, found {len(checkpoints)}")


def stage_cases() -> list[tuple[str, str, Path, Path, Path]]:
    import nibabel as nib
    from nibabel.processing import resample_from_to
    from remotezip import RemoteZip

    data = WORK / "cases"
    prepared = WORK / "prepared"
    data.mkdir(parents=True, exist_ok=True)
    prepared.mkdir(parents=True, exist_ok=True)
    url = "https://zenodo.org/records/15281784/files/0001-0020.zip?download=1"
    needed = [
        (case, tracer, filename)
        for case in CASE_IDS
        for tracer in TRACERS
        for filename in ("CT.nii.gz", "PET.nii.gz", "TTB.nii.gz")
    ]
    if not all((data / case / tracer / filename).is_file() for case, tracer, filename in needed):
        with RemoteZip(url) as remote:
            names = remote.namelist()
            for case, tracer, filename in needed:
                destination = data / case / tracer / filename
                if destination.is_file():
                    continue
                pattern = re.compile(rf"(^|/){case}/{tracer}/{filename}$")
                matches = [name for name in names if pattern.search(name)]
                if len(matches) != 1:
                    raise RuntimeError(f"unexpected archive matches: {matches}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with remote.open(matches[0]) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
                print(f"staged {case}/{tracer}/{filename}", flush=True)

    cases = []
    for case in CASE_IDS:
        for tracer in TRACERS:
            tag = f"{tracer}_{case[-4:]}"
            root = data / case / tracer
            ct_path = prepared / f"{tag}_0000.nii.gz"
            pet_path = prepared / f"{tag}_0001.nii.gz"
            gt_path = prepared / f"{tag}_gt.nii.gz"
            if not gt_path.is_file():
                pet_image = nib.load(root / "PET.nii.gz")
                nib.save(resample_from_to(nib.load(root / "CT.nii.gz"), pet_image, order=1), ct_path)
                shutil.copy2(root / "PET.nii.gz", pet_path)
                shutil.copy2(root / "TTB.nii.gz", gt_path)
            cases.append((tag, tracer.lower(), ct_path, pet_path, gt_path))
    return cases


def component_metrics(prediction, ground_truth) -> dict[str, float | int]:
    import cc3d
    import numpy as np

    pred = np.asarray(prediction, dtype=np.uint8)
    gt = np.asarray(ground_truth, dtype=np.uint8)
    pred_labels, num_pred = cc3d.connected_components(pred, connectivity=18, return_N=True)
    gt_labels, num_gt = cc3d.connected_components(gt, connectivity=18, return_N=True)
    pred_volumes = np.bincount(pred_labels.ravel(), minlength=num_pred + 1)
    gt_volumes = np.bincount(gt_labels.ravel(), minlength=num_gt + 1)
    both = (pred_labels > 0) & (gt_labels > 0)
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    if np.any(both):
        stride = int(num_pred) + 1
        pairs, overlap = np.unique(
            gt_labels[both].astype(np.int64) * stride + pred_labels[both],
            return_counts=True,
        )
        for encoded, intersection in zip(pairs, overlap):
            gt_id, pred_id = divmod(int(encoded), stride)
            union = gt_volumes[gt_id] + pred_volumes[pred_id] - int(intersection)
            if union and intersection / union >= 0.1:
                matched_gt.add(gt_id)
                matched_pred.add(pred_id)
    tp = len(matched_gt)
    fp = int(num_pred) - len(matched_pred)
    fn = int(num_gt) - tp
    denominator = int(pred.sum() + gt.sum())
    dice = float(2 * np.count_nonzero(pred & gt) / denominator) if denominator else 1.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0
    return {"dice": dice, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def apply_policy(mask, pet, dust_threshold: int, suv_max_threshold: float):
    import cc3d
    import numpy as np

    filtered = np.asarray(mask, dtype=np.uint8)
    if dust_threshold:
        filtered = (cc3d.dust(filtered, threshold=dust_threshold, connectivity=18) > 0).astype(np.uint8)
    if suv_max_threshold:
        labels, count = cc3d.connected_components(filtered, connectivity=18, return_N=True)
        keep = np.zeros(count + 1, dtype=bool)
        for component_id in range(1, count + 1):
            keep[component_id] = float(np.max(pet[labels == component_id])) >= suv_max_threshold
        filtered = keep[labels].astype(np.uint8)
    return filtered


def summarize(per_case: dict[str, dict[str, float | int]]) -> dict[str, float | int]:
    import numpy as np

    tp = sum(int(metrics["tp"]) for metrics in per_case.values())
    fp = sum(int(metrics["fp"]) for metrics in per_case.values())
    fn = sum(int(metrics["fn"]) for metrics in per_case.values())
    return {
        "mean_dice": float(np.mean([metrics["dice"] for metrics in per_case.values()])),
        "aggregate_f1": float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    start = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    install_dependencies()
    algorithm = stage_source()
    stage_weights(algorithm)
    cases = stage_cases()
    sys.path.insert(0, str(algorithm))

    import nibabel as nib
    import numpy as np
    import process

    policies: dict[str, dict[str, dict[str, float | int]]] = {}
    cache = WORK / "prediction_cache"
    cache.mkdir(exist_ok=True)
    for tag, tracer, ct_path, pet_path, gt_path in cases:
        cached = cache / f"{tag}.npz"
        if cached.is_file():
            with np.load(cached) as archive:
                deployed = archive["deployed"]
                probability = archive["probability"]
                pet = archive["pet"]
                gt = archive["gt"]
        else:
            print(f"CHAMPION_START {tag}", flush=True)
            deployed, probability = process.AutopetInteractive._predict_champion(
                str(ct_path), str(pet_path), tracer, save_probability=True
            )
            pet = np.asarray(nib.load(pet_path).dataobj, dtype=np.float32)
            gt = np.asarray(nib.load(gt_path).dataobj) > 0
            np.savez_compressed(
                cached,
                deployed=deployed.astype(np.uint8),
                probability=probability.astype(np.float16),
                pet=pet.astype(np.float16),
                gt=gt.astype(np.uint8),
            )
            print(f"CHAMPION_DONE {tag}", flush=True)

        for dust in DUST_THRESHOLDS:
            for suv in SUV_MAX_THRESHOLDS:
                name = f"dust={dust}|suvmax={suv:g}"
                candidate = apply_policy(deployed, pet, dust, suv)
                policies.setdefault(name, {})[tag] = component_metrics(candidate, gt)

    results = {
        "cases": [case[0] for case in cases],
        "policies": {
            name: {"summary": summarize(per_case), "per_case": per_case}
            for name, per_case in policies.items()
        },
        "wall_minutes": (time.time() - start) / 60,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    ranking = sorted(
        results["policies"].items(),
        key=lambda item: (item[1]["summary"]["aggregate_f1"], item[1]["summary"]["mean_dice"]),
        reverse=True,
    )
    for name, record in ranking:
        print(name, json.dumps(record["summary"], sort_keys=True), flush=True)
    print(f"RESULTS_PATH={RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
