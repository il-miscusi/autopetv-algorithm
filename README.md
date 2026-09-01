# AutoPET V — Champion M0 + TACE Transactional Scribble Editing

Grand Challenge algorithm container for AutoPET V (MICCAI 2026): interactive
scribble-guided whole-body PET/CT lesion segmentation (FDG + PSMA).

## Method

- **Iteration 0 (M0):** five-fold AutoPET-III champion ensemble
  (LesionTracer, nnU-Net ResEncL, Dataset222 multi-tracer) run in an isolated
  subprocess, followed by a tracer-specific connected-component policy
  (FDG: 18-connectivity dust ≥ 25 voxels; PSMA: ≥ 5 voxels).
- **Tracer routing:** FDG/PSMA is decided directly from PET/CT by a fixed
  three-classifier majority vote over 99 whole-body distribution features.
- **Iterations 1–5:** cumulative scribbles are applied to the previous
  accepted mask (persisted through `/output`, the platform's transaction
  carrier; geometry-checked, failing closed to M0) via TACE — click-local
  Gaussian threshold modulation of the frozen champion probability map —
  adjudicated by a transaction gate: REMOVE proposals that split a previous
  component are rolled back, ADD proposals that merge distinct lesions are
  rejected, and HoleGuard falls back to the previous mask if a proposal
  creates a new enclosed hole. These guards exist because the final ranking
  weights AUC-DMM (lesion-level detection) equally with AUC-Dice.

## Provenance

See [NOTICE](NOTICE). Champion weights are public (Zenodo 14007247,
CC-BY-4.0, MD5-verified at build); scaffolding adapted from public
Apache-2.0 work by YixinChen-AI and MIC-DKFZ; no non-public weights are
used. The Docker build is self-contained: all weights are downloaded and
hash-verified at build time, and the container performs no network access at
inference.

## Build

```
docker build --platform=linux/amd64 -t autopetv-algorithm .
```
