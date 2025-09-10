# Inject3D
This repository provides a data generation pipeline to expand a dataset with lidar and image pairs with random objects in geometrically and spatially consistent locations

The pipeline is designed for anomaly/data augmentation experiments in autonomous driving, robotics, and 3D vision research.

Note: This data augmentation is not perfect. Please double-check any generated results before using them for training or evaluation.

![alt text](https://github.com/max01110/Inject3D/blob/main/assets/demo.png "Demo images")


# How Inject3D Works

Inject3D takes a LiDAR point cloud + camera image pair and augments them with a random 3D object in a physically consistent way:

**1. Calibration loading**

Reads a YAML file with camera intrinsics (K, distortion, P) and extrinsics (T<sub>LiDAR→Cam</sub>).

Supports just pinhole (plumb_bob) camera models for now.

**2. Scene preparation (Blender)**

- Starts a clean Blender scene.

- Sets up a virtual camera matching the real sensor intrinsics/extrinsics.

- Imports a random 3D mesh from Objaverse  and scales it to a random defined range.

**3. Ground-aware placement**

- Fits a ground plane to the LiDAR cloud using RANSAC.

- Samples a random (x, y) in front of the ego-vehicle.

- Computes z from the ground plane → snaps the object’s lowest point to the ground.

- Orients the object so its largest face lies flat on the ground, then applies a random yaw.

**4. Collision + occlusion checks**

- Builds a KD-tree of non-ground LiDAR points to reject placements that intersect existing obstacles (--clearance).

- Builds a LiDAR-derived z-buffer in camera space to ensure the object is fully visible (no occlusion).

**5. Rendering & compositing**

- Renders the object with transparent background in Blender.

- Warps the render from rectified → distorted image domain (using OpenCV, this is not perfect and is corrected later).

- Alpha-composites the object over the original camera frame.

**6. Point cloud augmentation**

- Samples points from the object’s mesh surface.

- Appends them to the original .bin LiDAR cloud.

- Writes new .bin + .label files with injected points marked as --anomaly_label.

**7. IoU Check/Alignment**

Given that the blender composite can't fully replicate the camera model, we perform a final alignment/check

- Project augmented point cloud on camera frame
- Check IoU of point cloud object and composite
- Align composite such that it maximizes IoU
- If final IoU < threshold, reject and restart

In practice, we find that using an IoU threshold of 0.92 provides a good balance — it yields results that are nearly perfect while avoiding excessive retries. Achieving a true 100% IoU is unrealistic given the inherent differences between LiDAR point clouds and image-based renderings, so 0.92 serves as a practical compromise. However, feel free to experiment and adapt this as needed.

**Outputs**

- Augmented image (aug_image_distorted.png).

- Augmented point cloud (aug_lidar.bin).

- Augmented labels (aug_lidar.label).


**Disclaimer:** This tool is useful for generating 2D–3D consistent training data (e.g., for voxel-based experiments), but the augmented results are not guaranteed to be perfectly reliable. All generated outputs should be double-checked before use in downstream tasks.

# 1. Installation


Clone the repository

```bash
git clone https://github.com/max01110/Inject3D.git
cd Inject3D
```

## 1.1 Local Install

1. Install Blender from [here](https://www.blender.org/download/)

	a) Use Blender version > 2.9 (we recommend 3.6.5)


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

2. Build docker container 
    ```
    cd docker
    sudo docker build --build-arg USERNAME=$USER -t inject3d:latest .
    ```


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

**LiDAR point cloud (.bin):** KITTI-style N×4 float32 with columns [x, y, z, r] in LiDAR frame (we only use x,y,z)

**Labels (.label):** N × uint32 semantic class IDs aligned 1:1 with the .bin.

The pipeline appends new points for the injected mesh and labels them with --anomaly_label (default 150).

**Image:** The raw distorted RGB frame corresponding to the LiDAR sweep (PNG/JPG).

# 3. Usage
For a list of possible arguments to customize and adapt your augmented dataset, please see ```main.py```

## 3.1 Local

```bash
BLENDER=/path/to/blender

$BLENDER -b --python main.py -- --calib input/STU_dataset/calib.yaml --image input/STU_dataset/000000.png --lidar input/STU_dataset/000000.bin --labels input/STU_dataset/000000.label     --outdir out/000000
```


## 3.2 Docker

```bash
docker run --gpus all --rm -it -v $(pwd):/workspace inject3d:latest bash -lc 

cd /workspace

blender -b --python main.py -- --calib input/STU_dataset/calib.yaml --image input/STU_dataset/000000.png --lidar input/STU_dataset/000000.bin --labels input/STU_dataset/000000.label     --outdir out/000000
```

# 4 Extra Tools

## 4.1 Point Cloud Projection on Camera
We provide a script to project the lidar points onto the camera frame as a check to ensure the augmented object is inserted properly.

To use this script run:

```bash
python scripts/project_lidar_on_cam.py --label out/000000.png/aug_lidar.label --calib input/STU_dataset/calib.yaml --out projection.png --image input/STU_dataset/000000.png 
```

Please refer to the script for more info on specific optional arugments you can pass in


## 4.2 Dataset Augment

We also provid a script to augment a directory of image-lidar pairs.

Configure your settings inside the bash script ```scripts/augment_dataset.sh``

Then, run:

```bash
chmod +x scripts/augment_dataset.sh
./augment_dataset.sh
```
