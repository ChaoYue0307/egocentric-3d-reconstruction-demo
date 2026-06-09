# Data Notice

This repository does not include raw Xperience-10M files.

Place a local sample episode on your machine and pass it with `--data-root`:

```bash
export DATA_ROOT=/path/to/xperience-10m-sample
python scripts/reconstruction_demo.py --data-root "$DATA_ROOT"
```

The reconstruction preparation script expects:

- `annotation.hdf5`
- one or more camera videos such as `fisheye_cam0.mp4`

Generated frames, calibration exports, SLAM previews, and command templates are
written under `outputs/`.
