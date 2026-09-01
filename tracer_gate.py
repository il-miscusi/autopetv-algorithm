"""Classify an unlabeled GC PET/CT input as FDG or PSMA."""

from __future__ import annotations

import json
import os

import numpy as np


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracer_logreg.json")


def feats_from_arrays(ct, pet):
    """Extract the exact ten SITK z-y-x features used to train the gate."""
    ct = np.asarray(ct, dtype=np.float32)
    pet = np.asarray(pet, dtype=np.float32)
    z_size = pet.shape[0]
    top = pet[int(0.85 * z_size):]
    ct_top = ct[int(0.85 * z_size):]
    positive = pet[pet > 0.1]
    if positive.size == 0:
        positive = pet.ravel()
    return np.asarray(
        [
            float(np.percentile(top, 99.5)),
            float(np.percentile(top, 99)),
            float(top[top > 0.1].mean()) if np.any(top > 0.1) else 0.0,
            float(np.percentile(positive, 99.9)),
            float(np.percentile(positive, 99)),
            float(np.percentile(positive, 90)),
            float(positive.mean()),
            float((pet > 2.5).mean()),
            float((pet > 4.0).mean()),
            float((ct_top > 300).mean()),
        ],
        dtype=np.float64,
    )


def predict_tracer(ct, pet, model_path=None, return_prob=False):
    with open(model_path or MODEL_PATH) as handle:
        model = json.load(handle)
    mean = np.asarray(model["mean"])
    std = np.asarray(model["std"])
    weights = np.asarray(model["W"])
    score = float(((feats_from_arrays(ct, pet) - mean) / std) @ weights + float(model["b"]))
    probability_psma = 1.0 / (1.0 + np.exp(-score))
    label = "psma" if probability_psma > 0.5 else "fdg"
    return (label, probability_psma) if return_prob else label
