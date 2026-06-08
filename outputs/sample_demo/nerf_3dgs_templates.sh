#!/usr/bin/env bash
set -euo pipefail

# NeRFStudio path after COLMAP or ns-process-data is available.
ns-process-data images \
  --data outputs/sample_demo/frames \
  --output-dir outputs/sample_demo/nerfstudio

ns-train nerfacto \
  --data outputs/sample_demo/nerfstudio

# 3D Gaussian Splatting-style training placeholder.
# Replace TRAIN_3DGS with your local 3DGS training entry point.
TRAIN_3DGS \
  --source_path outputs/sample_demo/frames \
  --model_path outputs/sample_demo/gaussian_splatting
