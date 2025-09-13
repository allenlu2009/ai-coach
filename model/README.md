# Model Directory

This directory contains RTMPose3D model checkpoints for native 3D pose estimation.

## Required Model

- **rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth** (RTMW3D-L model)
  - Download from: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose3d
  - Used for native 3D pose estimation with real Z coordinates
  - Size: ~230MB

## Usage

The RTMPose analyzer automatically detects and uses models in this directory when `--use-3d` flag is enabled:

```bash
python run_server.py --use-3d
```

If no RTMPose3D checkpoint is found, the system gracefully falls back to 2D pose detection with intelligent depth estimation.