
#!/usr/bin/env bash

python pytom_wedge_fft_overlay.py \
  --tomogram /path/to/tomogram.mrc \
  --tilt-angles /path/to/angles.tlt \
  --voxel-size-angstrom VOXEL-SIZE-ANGSTROM \
  --outdir wedge_fft_check \
  --slab-y 7 \
  --also-negated
