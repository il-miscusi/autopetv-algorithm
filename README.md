# autoPET V — interactive conformance algorithm

Algorithm container for the autoPET V challenge (interactive whole-body
PET/CT lesion segmentation, MICCAI 2026):
https://autopet-v.grand-challenge.org/

Data-centric extension of the organizers' nnU-Net baseline (their pretrained
weights, out-of-competition model, unchanged): predictions are cached across
the interactive loop in `/cache`, and each corrective scribble is enforced by
post-processing — PET-adaptive region growing for foreground (missed-lesion)
scribbles plus a local model re-run union, and uptake-discriminated component
removal or shell chipping for background (false-positive) scribbles. The
model is never retrained; all improvement is pre/post-processing around the
baseline, per the challenge's data-centric award rules.

Build: push a `v*` tag; the workflow bakes the baseline weights into the
image and attaches split `docker save` parts to a GitHub release.

License: Apache-2.0. Baseline weights and simulation code:
https://github.com/lab-midas/autoPETV
