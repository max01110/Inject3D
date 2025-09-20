#!/usr/bin/env bash
pip install trimesh pygltflib pillow transforms3d
sudo apt install -y libx11-6 libxi6 libxext6 libxxf86vm1 libnvidia-gl-570
pip install pyyaml 


### ------------------- USER CONFIG ------------------- ###
SCRIPT="main.py" 
CALIB="input/STU_dataset/calib.yaml"                # calibration file
IMAGES="../train_images/201/port_a_cam_0"                   # input images dir
LIDAR="../train_pointcloud/nodes/dom/work/nekrasov/data/stu_dataset/train/201/velodyne"  # input lidar dir
LABELS="../train_pointcloud/nodes/dom/work/nekrasov/data/stu_dataset/train/201/labels"   # input labels dir
OUT_IMAGES="out/train/201/images"           # output augmented images
OUT_LIDAR="out/train/201/lidar"          # output augmented lidar
OUT_VOXEL="out/train/201/voxel"          # output augmented voxels (optional)
PER_PAIR=1                                 # number of augmentations per pair
IOU_THRESH=0.90                            # IoU threshold
IOU_MAX_TRIES=10                           # max IoU tries per augmentation
### --------------------------------------------------- ###

mkdir -p "$OUT_IMAGES" "$OUT_LIDAR" "$OUT_VOXEL"

for img_path in "$IMAGES"/*.png; do

  echo $img_path
  stem=$(basename "$img_path" .png)
  lidar_path="$LIDAR/$stem.bin"
  label_path="$LABELS/$stem.label"

  if [[ ! -f "$lidar_path" || ! -f "$label_path" ]]; then
    echo "Skipping $stem (missing lidar or label)"
    continue
  fi

  for ((k=1; k<=PER_PAIR; k++)); do
    tmp_out=$(mktemp -d)

    out_img="$OUT_IMAGES/${stem}_aug${k}.png"
    if [[ -f "$out_img" ]]; then
      echo "Exists, skipping: $out_img"
      continue
    fi
    echo "Generating augmentation $k for $stem..."
    blender -b -P "$SCRIPT" -- \
      --calib "$CALIB" \
      --image "$img_path" \
      --lidar "$lidar_path" \
      --labels "$label_path" \
      --outdir "$tmp_out" \
      --iou_thresh "$IOU_THRESH" \
      --iou_max_tries "$IOU_MAX_TRIES" \
      --fast

    # Example voxel mode usage (commented out by default):
    # blender -b -P "$SCRIPT" -- \
    #   --calib "$CALIB" \
    #   --image "$img_path" \
    #   --lidar "$lidar_path" \
    #   --labels "$label_path" \
    #   --outdir "$tmp_out" \
    #   --iou_thresh "$IOU_THRESH" \
    #   --iou_max_tries "$IOU_MAX_TRIES" \
    #   --fast \
    #   --voxel --voxel_bin "$VOXELS/$stem.bin" --voxel_label "$VOXELS/$stem.label" --voxel_meta "$VOXELS/$stem.meta.json"

    # Move outputs to final destinations with unique names
    if [[ -f "$tmp_out/aug_image.png" ]]; then
      mv "$tmp_out/aug_image.png" "$out_img"
    fi
    if [[ -f "$tmp_out/aug_lidar.bin" ]]; then
      mv "$tmp_out/aug_lidar.bin" "$OUT_LIDAR/${stem}_aug${k}.bin"
    fi
    if [[ -f "$tmp_out/aug_lidar.label" ]]; then
      mv "$tmp_out/aug_lidar.label" "$OUT_LIDAR/${stem}_aug${k}.label"
    fi
    if [[ -f "$tmp_out/aug_voxel.bin" ]]; then
      mv "$tmp_out/aug_voxel.bin" "$OUT_VOXEL/${stem}_aug${k}.bin"
    fi
    if [[ -f "$tmp_out/aug_voxel.label" ]]; then
      mv "$tmp_out/aug_voxel.label" "$OUT_VOXEL/${stem}_aug${k}.label"
    fi

    rm -rf "$tmp_out"
  done
done

echo "All done!"
