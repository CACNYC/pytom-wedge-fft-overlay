# pytom-wedge-fft-overlay PyTom wedge FFT overlay diagnostic

Diagnostic script to compare PyTom binary missing-wedge models with tomogram FFT XZ views.

This repository contains a diagnostic script for comparing the PyTom binary tomogram missing-wedge model with the missing wedge visible in a tomogram FFT.

## Purpose

The script is intended to test whether the missing-wedge orientation modeled from a supplied `.tlt` or `.rawtlt` file matches the missing-wedge orientation visible in the corresponding tomogram, or whether it appears mirrored.

It performs the comparison in one coordinate system:

1. Reads the tomogram using `pytom_tm.io.read_mrc()`.
2. Builds the PyTom binary tomogram wedge using `pytom_tm.weights.create_wedge(..., per_tilt_weighting=False)`.
3. Computes the tomogram real FFT from the same PyTom-oriented array.
4. Displays both objects in the same reduced real-FFT convention.
5. Extracts the same central XZ slab.
6. Overlays the PyTom wedge contour on the tomogram FFT.

## Important scope

This script visualizes PyTom's binary tomogram missing-wedge model. It does not claim to reproduce the full per-tilt/CTF/dose-weighted template wedge used during template matching.

The tool is intended as an orientation diagnostic, not as proof of biological handedness.

## Example

```bash
python pytom_wedge_fft_overlay.py \
  --tomogram /path/to/tomogram.mrc \
  --tilt-angles /path/to/angles.tlt \
  --voxel-size-angstrom [VOXEL-SIZE-IN-ANGSTROMS] \
  --outdir wedge_fft_check \
  --slab-y 7 \
  --also-negated
