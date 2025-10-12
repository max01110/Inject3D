#!/usr/bin/env bash
# pip install trimesh pygltflib pillow transforms3d
# sudo apt install -y libx11-6 libxi6 libxext6 libxxf86vm1 libnvidia-gl-570
# pip install pyyaml 


### ------------------- USER CONFIG ------------------- ###
# Choose dataset: "kitti_sequences" or "stu"
DATASET="kitti_sequences"  # Change to "stu" for STU dataset

# Common settings
SCRIPT="main.py"
PER_PAIR=1                                 # number of augmentations per pair
IOU_THRESH=0.80                          # IoU threshold
IOU_MAX_TRIES=10                           # max IoU tries per augmentation

# KITTI Sequences Configuration (all sequences 00-21)
if [[ "$DATASET" == "kitti_sequences" ]]; then
    echo "Using KITTI sequences configuration (00-21)"
    BASE_PATH="/mnt/data1/datasets/kitti_odom/dataset/sequences"
    OUT_BASE_PATH="/mnt/data1/datasets/augmented_kitti_odom"
    
    # Process all sequences 00-21
    for seq in {00..21}; do
        echo "Processing sequence $seq..."
        
        # Set paths for current sequence
        IMAGES="$BASE_PATH/$seq/image_2"                  # input images dir
        LIDAR="$BASE_PATH/$seq/velodyne"                  # input lidar dir
        LABELS="$BASE_PATH/$seq/labels"                   # input labels dir
        OUT_IMAGES="$OUT_BASE_PATH/$seq/images"          # output augmented images
        OUT_LIDAR="$OUT_BASE_PATH/$seq/lidar"            # output augmented lidar
        
        # Use KITTI calibration (assuming same for all sequences)
        CALIB="input/KITTI_dataset/calib.yaml"
        
        # Create output directories
        mkdir -p "$OUT_IMAGES" "$OUT_LIDAR"
        
        # Process current sequence
        for img_path in "$IMAGES"/*.png; do
            if [[ ! -f "$img_path" ]]; then
                continue
            fi
            
            echo "Processing: $img_path"
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
                
                # Run blender with pointcloud mode
                blender -b -P "$SCRIPT" -- \
                    --calib "$CALIB" \
                    --image "$img_path" \
                    --lidar "$lidar_path" \
                    --labels "$label_path" \
                    --outdir "$tmp_out" \
                    --iou_thresh "$IOU_THRESH" \
                    --iou_max_tries "$IOU_MAX_TRIES" \
                    --x_range 8 40 \
                    --y_range -10 10 \
                    --place_tries 250 \
                    --clearance 0.01 \
                    --ground_ransac_thresh 0.5 \
                    --z_margin 0.2 \
                    --require_inside_frac 0.1 \
                    --unoccluded_thresh 0.1 \
                    --surf_samples 1000 \
                    --fast

                # Move outputs to final destinations
                if [[ -f "$tmp_out/aug_image.png" ]]; then
                    mv "$tmp_out/aug_image.png" "$out_img"
                fi
                if [[ -f "$tmp_out/aug_lidar.bin" ]]; then
                    mv "$tmp_out/aug_lidar.bin" "$OUT_LIDAR/${stem}_aug${k}.bin"
                fi
                if [[ -f "$tmp_out/aug_lidar.label" ]]; then
                    mv "$tmp_out/aug_lidar.label" "$OUT_LIDAR/${stem}_aug${k}.label"
                fi

                rm -rf "$tmp_out"
            done
        done
    done
    
    echo "All sequences processed!"
    exit 0
    
# STU Dataset Configuration
elif [[ "$DATASET" == "stu" ]]; then
    echo "Using STU dataset configuration"
    CALIB="input/STU_dataset/calib.yaml"
    IMAGES="../train_images/201/port_a_cam_0"             # input images dir
    LIDAR="../train_pointcloud/nodes/dom/work/nekrasov/data/stu_dataset/train/201/velodyne"  # input lidar dir
    LABELS="../train_pointcloud/nodes/dom/work/nekrasov/data/stu_dataset/train/201/labels"   # input labels dir
    OUT_IMAGES="out/train/201/images"                     # output augmented images
    OUT_LIDAR="out/train/201/lidar"                       # output augmented lidar
    
    mkdir -p "$OUT_IMAGES" "$OUT_LIDAR"
    
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
                --x_range 25.0 30.0 \
                --y_range -10 10 \
                --place_tries 1000 \
                --clearance 0.01 \
                --ground_ransac_thresh 0.5 \
                --z_margin 0.2 \
                --require_inside_frac 0.1 \
                --unoccluded_thresh 0.1 \
                --surf_samples 1000 \
                --fast

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

            rm -rf "$tmp_out"
        done
    done
    
    echo "All done!"
    
else
    echo "Error: DATASET must be 'kitti_sequences' or 'stu'"
    exit 1
fi
