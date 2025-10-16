#!/usr/bin/env bash
# pip install trimesh pygltflib pillow transforms3d
# sudo apt install -y libx11-6 libxi6 libxext6 libxxf86vm1 libnvidia-gl-570
# pip install pyyaml

# Usage: ./augment_dataset.sh [SEQUENCES]
# Examples:
#   ./augment_dataset.sh                    # Process all sequences (00-21)
#   ./augment_dataset.sh "00,01,02"         # Process specific sequences
#   ./augment_dataset.sh "00-05"            # Process range of sequences
#   ./augment_dataset.sh "all"              # Process all sequences (explicit) 


### ------------------- USER CONFIG ------------------- ###
# Choose dataset: "kitti_sequences" or "stu"
DATASET="kitti_sequences"  # Change to "stu" for STU dataset

# KITTI Sequences to process (comma-separated, e.g., "00,01,02" or "00-05" or "all")
SEQUENCES="${1:-all}"  # Use first argument or default to "all"

# Common settings
SCRIPT="main.py"
PER_PAIR=1                                 # number of augmentations per pair
IOU_THRESH=0.80                          # IoU threshold
IOU_MAX_TRIES=10                           # max IoU tries per augmentation

# Function to parse sequences input
parse_sequences() {
    local input="$1"
    local sequences=()
    
    if [[ "$input" == "all" ]]; then
        # Generate all sequences 00-21
        for i in {00..21}; do
            sequences+=("$i")
        done
    elif [[ "$input" =~ ^[0-9]+-[0-9]+$ ]]; then
        # Handle range format like "00-05"
        local start=$(echo "$input" | cut -d'-' -f1)
        local end=$(echo "$input" | cut -d'-' -f2)
        for ((i=start; i<=end; i++)); do
            sequences+=($(printf "%02d" $i))
        done
    else
        # Handle comma-separated format like "00,01,02"
        IFS=',' read -ra seq_array <<< "$input"
        for seq in "${seq_array[@]}"; do
            # Remove any whitespace and pad with leading zero if needed
            seq=$(echo "$seq" | xargs | printf "%02d" $(cat))
            sequences+=("$seq")
        done
    fi
    
    echo "${sequences[@]}"
}

# KITTI Sequences Configuration
if [[ "$DATASET" == "kitti_sequences" ]]; then
    # Parse sequences input
    SEQUENCE_LIST=($(parse_sequences "$SEQUENCES"))
    echo "Using KITTI sequences configuration: ${SEQUENCE_LIST[*]}"
    BASE_PATH="/mnt/data1/datasets/kitti_odom/dataset/sequences"
    OUT_BASE_PATH="/mnt/data1/datasets/augmented_kitti_odom"
    
    # Process specified sequences
    for seq in "${SEQUENCE_LIST[@]}"; do
        echo "Processing sequence $seq..."
        
        # Set paths for current sequence
        IMAGES="$BASE_PATH/$seq/image_2"                  # input images dir
        LIDAR="$BASE_PATH/$seq/velodyne"                  # input lidar dir
        LABELS="$BASE_PATH/$seq/labels"                   # input labels dir
        OUT_IMAGES="$OUT_BASE_PATH/$seq/images"          # output augmented images
        OUT_LIDAR="$OUT_BASE_PATH/$seq/lidar"            # output augmented lidar
        
        # Use sequence-specific KITTI calibration
        CALIB="input/KITTI_dataset/KITTI_calibs/calib_${seq}.yaml"
        
        # Check if calibration file exists
        if [[ ! -f "$CALIB" ]]; then
            echo "Warning: Calibration file $CALIB not found, skipping sequence $seq"
            continue
        fi
        
        echo "Using calibration file: $CALIB"
        
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
                    --y_range -2 2 \
                    --place_tries 1 \
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
