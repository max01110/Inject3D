# Inject3D

A data augmentation pipeline for LiDAR + camera datasets. Injects random 3D objects from Objaverse into scenes in a geometrically consistent way.

Useful for anomaly detection, data augmentation, and 3D vision research in autonomous driving and robotics.

**Note:** Generated results should be manually verified before use in training or evaluation.

![Demo](https://github.com/max01110/Inject3D/blob/main/assets/demo.png)


## How It Works

### Pipeline Overview

1. **Load calibration** — Camera intrinsics (K, distortion) and LiDAR-to-camera extrinsics from YAML
2. **Setup Blender scene** — Virtual camera matching real sensor parameters
3. **Import 3D object** — Random mesh from Objaverse, scaled to realistic size
4. **Place on ground** — RANSAC ground plane fitting, snap object to surface
5. **Collision/occlusion checks** — Reject placements that intersect obstacles or are hidden
6. **Render & composite** — Blender render with distortion correction, alpha composite onto image
7. **Augment point cloud** — Sample points from mesh surface, append to LiDAR with anomaly label
8. **IoU verification** — Align render to projected points, reject if IoU < threshold

### Object Placement

The placement algorithm has two modes:

**Automatic placement:**
1. Fit ground plane to LiDAR points using RANSAC
2. Sample random (x, y) position within configured bounds
3. Query nearby ground points to get local height
4. Place object so its bottom sits on the ground surface
5. Apply random yaw rotation
6. Check for collisions with existing obstacles (KD-tree query)
7. Verify object is visible in camera (not occluded, inside FOV)
8. Retry with different position if checks fail

**Manual placement:**
- Specify exact (x, y) coordinates in LiDAR frame
- Optionally set z manually or auto-adjust to ground height
- Skips collision/occlusion checks (useful for debugging or specific placements)

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--x_range` | Distance range in front of sensor (meters) |
| `--y_range` | Lateral range (meters) |
| `--clearance` | Minimum distance from existing obstacles |
| `--place_tries` | Attempts before giving up on placement |
| `--require_inside_frac` | Fraction of object that must be in camera FOV |
| `--unoccluded_thresh` | Fraction that must be visible (not behind obstacles) |
| `--iou_thresh` | Required IoU between render and projected points |

### Outputs

- `aug_image.png` — Composited image with injected object
- `aug_lidar.bin` — Point cloud with object points appended
- `aug_lidar.label` — Labels with object points marked (default label: 150)


## Installation

### Local Setup

1. **Install Blender** (version 3.6.5 recommended)
   ```bash
   # Download from https://www.blender.org/download/
   # Also install via apt for dependencies:
   apt install blender
   ```

2. **Install Python packages**
   ```bash
   pip install -r requirements.txt
   
   # Inside Blender's Python environment:
   /path/to/blender-3.6.5-linux-x64/3.6/python/bin/python3.10 -m pip install -r requirement_bpy.txt
   ```

### Docker

1. Download Blender .tar and place in `docker/`
2. Update Blender version in `Dockerfile`
3. Build:
   ```bash
   cd docker
   docker build --build-arg USERNAME=$USER -t inject3d:latest .
   ```


## Calibration

Camera intrinsics and extrinsics in a single YAML file:

```yaml
camera_intrinsics:
  width: 1920
  height: 1200
  distortion_model: plumb_bob  # or: equidistant
  K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
  D: [k1, k2, p1, p2, k3]
  P: [fxp, 0, cxp, 0, 0, fyp, cyp, 0, 0, 0, 1, 0]

static_transforms:
  - parent: os_sensor
    child: camera_frame
    translation: {x: tx, y: ty, z: tz}
    rotation: {x: qx, y: qy, z: qz, w: qw}
```

See `input/STU_dataset/calib.yaml` for a complete example.

**Coordinate convention:** LiDAR frame is +X forward, +Y left, +Z up.


## Usage

### Single Frame

```bash
blender -b -P main.py -- \
    --calib input/STU_dataset/calib.yaml \
    --image input/STU_dataset/000000.png \
    --lidar input/STU_dataset/000000.bin \
    --labels input/STU_dataset/000000.label \
    --outdir out/000000
```

### Batch Processing

The `augment_dataset.sh` script processes entire sequences.

#### KITTI Dataset

```bash
# All sequences (00-21)
./augment_dataset.sh

# Specific sequences
./augment_dataset.sh "00,01,02"

# Range of sequences
./augment_dataset.sh "00-05"
```

Default paths:
- Input: `/mnt/data1/datasets/kitti_odom/dataset/sequences/`
- Output: `/mnt/data1/datasets/augmented_kitti_odom/`
- Calibration: `input/KITTI_dataset/KITTI_calibs/calib_XX.yaml`

#### STU Dataset

```bash
./augment_dataset.sh --dataset stu --base_dir /path/to/stu --sequence 201
```

The script auto-detects common STU directory structures. If needed, specify the path explicitly:

```bash
./augment_dataset.sh --dataset stu --base_dir /path/to/stu --sequence 201 \
    --lidar_subpath "train_pointcloud"
```

#### Frame Selection

```bash
# Start from a specific frame
./augment_dataset.sh --dataset stu --base_dir /path --sequence 201 --start_frame 50

# Process only specific frames
./augment_dataset.sh --dataset stu --base_dir /path --sequence 201 --frames "1,3,5,10-15,45"
```

#### Manual Object Placement

Override automatic placement with exact coordinates:

```bash
# Specify position (auto-adjust height to ground)
./augment_dataset.sh --dataset stu --base_dir /path --sequence 201 \
    --manual_x 10.0 --manual_y 1.0 --manual_adjust_to_ground

# Full manual control
./augment_dataset.sh --dataset stu --base_dir /path --sequence 201 \
    --manual_x 10.0 --manual_y 1.0 --manual_z 0.5 --manual_yaw 45
```

### Docker Usage

```bash
docker run --gpus all --rm -it -v $(pwd):/workspace inject3d:latest bash -lc "
    cd /workspace
    blender -b -P main.py -- --calib input/STU_dataset/calib.yaml ...
"
```


## Extra Tools

### Point Cloud Projection

Visualize LiDAR points projected onto the camera image:

```bash
python scripts/project_lidar_on_cam.py \
    --lidar out/000000/aug_lidar.bin \
    --label out/000000/aug_lidar.label \
    --calib input/STU_dataset/calib.yaml \
    --image input/STU_dataset/000000.png \
    --out projection.png
```


## main.py Arguments

Run `blender -b -P main.py -- --help` for the full list. Key parameters:

```
Required:
  --calib       Calibration YAML file
  --image       Input image (distorted)
  --lidar       Input point cloud (.bin)
  --labels      Input labels (.label)
  --outdir      Output directory

Placement:
  --x_range     Forward distance range [min max] (default: 4.0 12.0)
  --y_range     Lateral range [min max] (default: -2.0 2.0)
  --clearance   Min distance from obstacles (default: 0.30)
  --place_tries Placement attempts (default: 150)

Manual placement:
  --manual_x, --manual_y    Exact position in LiDAR frame
  --manual_z                Exact height (optional)
  --manual_yaw              Rotation in degrees
  --manual_adjust_to_ground Auto-adjust Z to ground

Quality:
  --iou_thresh      Required IoU (default: 0.92)
  --iou_max_tries   Retries if IoU fails (default: 10)

Object sizing:
  --target_size     Fixed size in meters (random 0.6-1.5 if not set)
  --size_jitter_frac  Size variation (default: 0.15)
```
