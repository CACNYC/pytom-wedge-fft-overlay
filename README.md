# pytom-wedge-fft-overlay 
## PyTom wedge FFT overlay diagnostic

This repository contains a diagnostic script for comparing the PyTom binary tomogram missing-wedge model with the missing wedge visible in a tomogram FFT XZ view.

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
  --voxel-size-angstrom VOXEL_SIZE_ANGSTROM \
  --outdir wedge_fft_check \
  --slab-y 7 \
  --also-negated
```

## Outputs

The script writes:

tomogram_fft_log_magnitude_xz.png
input_angles_pytom_binary_wedge_xz.png
input_angles_overlay_fft_with_wedge_contour_xz.png
negated_angles_pytom_binary_wedge_xz.png, if --also-negated is used
negated_angles_overlay_fft_with_wedge_contour_xz.png, if --also-negated is used
wedge_fft_overlay_summary.json

## Interpretation

For each angle convention, the overlay shows where PyTom's binary missing-wedge contour falls relative to the tomogram FFT.

The summary JSON also reports a heuristic score:

```sampled FFT log magnitude - missing FFT log magnitude```

A higher score suggests that the modeled sampled region has higher tomogram FFT signal than the modeled missing region. This is a heuristic orientation diagnostic and should be interpreted together with the overlay images.

## Requirements

Run this inside a conda environment where pytom-match-pick is installed.

Additional Python packages:

numpy
scipy
matplotlib
