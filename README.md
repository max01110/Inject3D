# Inject3D
This repository provides a data generation pipeline to expand a dataset with lidar and image pairs with random objects in geometrically and spatially consistnet locations

The pipeline is designed for anomaly/data augmentation experiments in autonomous driving, robotics, and 3D vision research.

Note: This data augmentation is not perfect. Please double-check any generated results before using them for training or evaluation.

![alt text](https://github.com/max01110/Inject3D/blob/main/assets/demo.png "Demo images")


# 1. Installation


Clone the repository

```bash
git clone https://github.com/max01110/Inject3D.git
cd Inject3D
```

## 1.1 Local Install

1. Install Blender from [here](https://www.blender.org/download/)

	a) Use Blender version > 2.9


	b) untar the downloaded zip folder and make note of the path to the Blender folder, the folder structure will look like this:
	```
	./blender-3.6.5-linux-x64
	├── blender
	├── 3.6
    ...
	```
	where `blender` is the executable used to run Blender, and `3.6` is the blender version which contains Blender's Python environment

	c) Install Blender using apt to install dependencies
	```
	apt install blender
	```

2. install requirements 

	a) general requirements can be installed with:
	```
	pip install -r requirements.txt
	```
	b). install blender requirements inside the Blender Python environment by running:
    ```
    /blender-3.6.5-linux-x64/3.6/python/bin/python3.10 -m pip install -r requirement_bpy.txt
    ```

## 1.2 Docker

1. Install Blender from [here](https://www.blender.org/download/)

	a) Use Blender version > 2.9 (we recommend 3.6.5)

    b) Place the .tar file in the ```Docker/``` folder

    b) In the ```Dockerfile``` adapt the version of the downloaded blender

2. ```docker build -t inject3d:latest .```


# 2. Setup


## 2.1. Calibration Files

The camera intrinsics and camera-from-LiDAR extrinsics are provided in a single YAML:

```yaml
camera_intrinsics:
  frame_id: port_a_cam_0
  width: 1920
  height: 1200
  distortion_model: plumb_bob      # or: equidistant (fisheye)
  K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]   # 3x3 row-major
  D: [k1, k2, p1, p2, k3]              # radtan (plumb_bob)
  # For fisheye/equidistant use [k1, k2, k3, k4]
  # Optional rectified projection (left 3x3 used if --use_P_for_rect):
  P: [fxp, 0, cxp, 0, 0, fyp, cyp, 0, 0, 0, 1, 0]  # 3x4 row-major

# Extrinsics: camera w.r.t. LiDAR/world ("os_sensor")
static_transforms:
  - parent: os_sensor
    child: port_a_cam_0
    translation: {x: tx, y: ty, z: tz}      # meters
    rotation:    {x: qx, y: qy, z: qz, w: qw} # unit quaternion
```

An example of the calibration file format can be found in ```input/STU_dataset/calib.yaml```

**Frame convention:**
LiDAR/world frame used: +X forward, +Y left, +Z up.

Extrinsics above define T_LiDAR→Cam (camera pose in LiDAR/world).

distortion_model supports plumb_bob (a.k.a. radtan) and equidistant (fisheye).


## 2.2 Point Cloud & Image Pair

**LiDAR point cloud (.bin):** KITTI-style N×3 float32 with columns [x, y, z] in LiDAR frame.

**Labels (.label):** N × uint32 semantic class IDs aligned 1:1 with the .bin.

The pipeline appends new points for the injected mesh and labels them with --anomaly_label (default 150).

**Image:** The raw distorted RGB frame corresponding to the LiDAR sweep (PNG/JPG).

# 3. Usage

## 3.1 Local

```bash
BLENDER=/path/to/blender

$BLENDER -b --python scripts/inject_and_render.py -- \
  --calib input/STU_dataset/calib.yaml \
  --image input/STU_dataset/images/000448.png \
  --lidar input/STU_dataset/lidar/000448.bin \
  --labels input/STU_dataset/labels/000448.label \
```
For a list of possible arguments to customize and adapt your augmented dataset, please see ```main.py```

## 3.2 Docker

```bash
docker run --gpus all --rm -it -v $(pwd):/workspace inject3d:latest bash -lc 

blender-3.6.5-linux-x64/blender -b --python scripts/inject_and_render.py -- \
    --calib input/STU_dataset/calib.yaml \
    --image input/STU_dataset/images/000448.png \
    --lidar input/STU_dataset/lidar/000448.bin \
    --labels input/STU_dataset/labels/000448.label \
    --outdir out/000448 \
```
