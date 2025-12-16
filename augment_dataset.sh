#!/usr/bin/env bash
# pip install trimesh pygltflib pillow transforms3d
# sudo apt install -y libx11-6 libxi6 libxext6 libxxf86vm1 libnvidia-gl-570
# pip install pyyaml

# Usage:
#   KITTI: ./augment_dataset.sh [SEQUENCES]
#     Examples:
#       ./augment_dataset.sh                    # Process all sequences (00-21)
#       ./augment_dataset.sh "00,01,02"         # Process specific sequences
#       ./augment_dataset.sh "00-05"            # Process range of sequences
#       ./augment_dataset.sh "all"              # Process all sequences (explicit)
#
#   STU: ./augment_dataset.sh --dataset stu --base_dir BASE_DIR --sequence SEQUENCE [--lidar_subpath SUBPATH]
#     The script will auto-detect common STU path structures. If your structure differs,
#     use --lidar_subpath to specify the subpath to the sequence directory.
#     Examples:
#       ./augment_dataset.sh --dataset stu --base_dir /mnt/data2/datasets/STU --sequence 201
#       ./augment_dataset.sh --dataset stu --base_dir /path/to/stu --sequence 202
#       ./augment_dataset.sh --dataset stu --base_dir /path/to/stu --sequence 201 --lidar_subpath "train"
#
#   Manual Placement (works with both KITTI and STU):
#     Set MANUAL_X and MANUAL_Y in USER CONFIG section, or use command-line:
#       ./augment_dataset.sh --dataset stu --base_dir /path/to/stu --sequence 201 \
#                            --manual_x 10.0 --manual_y 1.0 --manual_z 0.5 --manual_yaw 45.0
#       ./augment_dataset.sh --dataset stu --base_dir /path/to/stu --sequence 201 \
#                            --manual_x 10.0 --manual_y 1.0 --manual_adjust_to_ground

# ./augment_dataset.sh --dataset stu --base_dir /mnt/data2/datasets/STU --sequence 201 --lidar_subpath "train_pointcloud/nodes/dom/work/nekrasov/data/stu_dataset/train"
# ./augment_dataset.sh --dataset stu --base_dir /mnt/data2/datasets/STU --sequence 201 --lidar_subpath "val_pointcloud"

### ------------------- USER CONFIG ------------------- ###
# Choose dataset: "kitti_sequences" or "stu"
# Default to KITTI for backward compatibility (use --dataset stu for STU)
DATASET="kitti_sequences"

# Manual placement settings (set to enable manual placement, leave empty for automatic)
# When manual_x and manual_y are set, object will be placed at exact coordinates
# and all collision/occlusion checks will be skipped
MANUAL_X="6.5"      # X coordinate in LiDAR frame (e.g., "10.0")
MANUAL_Y="0.5"      # Y coordinate in LiDAR frame (e.g., "1.0")
MANUAL_Z="-1.5"      # Z coordinate (optional, leave empty to auto-adjust to ground)
MANUAL_YAW=""    # Yaw rotation in degrees (default: "0.0")
MANUAL_ADJUST_TO_GROUND="1"  # Set to "1" to auto-adjust Z to ground height

# Parse command-line arguments
STU_BASE_DIR=""
STU_SEQUENCE=""
STU_LIDAR_SUBPATH=""
KITTI_SEQUENCES="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --base_dir)
            STU_BASE_DIR="$2"
            shift 2
            ;;
        --sequence)
            STU_SEQUENCE="$2"
            shift 2
            ;;
        --lidar_subpath)
            STU_LIDAR_SUBPATH="$2"
            shift 2
            ;;
        --manual_x)
            MANUAL_X="$2"
            shift 2
            ;;
        --manual_y)
            MANUAL_Y="$2"
            shift 2
            ;;
        --manual_z)
            MANUAL_Z="$2"
            shift 2
            ;;
        --manual_yaw)
            MANUAL_YAW="$2"
            shift 2
            ;;
        --manual_adjust_to_ground)
            MANUAL_ADJUST_TO_GROUND="1"
            shift
            ;;
        *)
            # For backward compatibility, treat first positional arg as KITTI sequences
            # Only if dataset is explicitly kitti_sequences or not set (defaults to kitti)
            if [[ "$DATASET" == "kitti_sequences" ]] || [[ -z "$DATASET" ]] || [[ "$DATASET" == "kitti" ]]; then
                KITTI_SEQUENCES="$1"
            else
                echo "Warning: Unknown argument '$1'. Use --dataset, --base_dir, --sequence, etc."
            fi
            shift
            ;;
    esac
done

# KITTI Sequences to process (comma-separated, e.g., "00,01,02" or "00-05" or "all")
SEQUENCES="${KITTI_SEQUENCES}"

# Build manual placement arguments string if manual placement is enabled
build_manual_placement_args() {
    local args=""
    if [[ -n "$MANUAL_X" && -n "$MANUAL_Y" ]]; then
        args="--manual_x $MANUAL_X --manual_y $MANUAL_Y"
        if [[ -n "$MANUAL_Z" ]]; then
            args="$args --manual_z $MANUAL_Z"
        fi
        if [[ -n "$MANUAL_YAW" ]]; then
            args="$args --manual_yaw $MANUAL_YAW"
        else
            args="$args --manual_yaw 0.0"
        fi
        if [[ -n "$MANUAL_ADJUST_TO_GROUND" ]]; then
            args="$args --manual_adjust_to_ground"
        fi
        echo "$args"
    else
        echo ""
    fi
}

MANUAL_PLACEMENT_ARGS=$(build_manual_placement_args)
if [[ -n "$MANUAL_PLACEMENT_ARGS" ]]; then
    echo "[INFO] Manual placement enabled: $MANUAL_PLACEMENT_ARGS"
else
    echo "[INFO] Automatic placement enabled"
fi

# Common settings
SCRIPT="main.py"
PER_PAIR=1                                 # number of augmentations per pair
IOU_THRESH=0.85                          # IoU threshold
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
                    $MANUAL_PLACEMENT_ARGS \
                    --x_range 8 40 \
                    --y_range -2 2 \
                    --place_tries 150 \
                    --clearance 0.30 \
                    --ground_ransac_thresh 0.15 \
                    --z_margin 0.2 \
                    --require_inside_frac 0.7 \
                    --unoccluded_thresh 0.90 \
                    --surf_samples 3000 \
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
    # Validate required arguments
    if [[ -z "$STU_BASE_DIR" ]]; then
        echo "Error: --base_dir is required for STU dataset"
        echo "Usage: ./augment_dataset.sh --dataset stu --base_dir BASE_DIR --sequence SEQUENCE"
        exit 1
    fi
    
    if [[ -z "$STU_SEQUENCE" ]]; then
        echo "Error: --sequence is required for STU dataset"
        echo "Usage: ./augment_dataset.sh --dataset stu --base_dir BASE_DIR --sequence SEQUENCE"
        exit 1
    fi
    
    echo "Using STU dataset configuration"
    echo "  Base directory: $STU_BASE_DIR"
    echo "  Sequence: $STU_SEQUENCE"
    
    CALIB="input/STU_dataset/calib.yaml"
    
    # Detect dataset split (train/val) and construct paths
    # Determine if this is train or val based on directory structure
    DATASET_SPLIT="train"  # default
    
    # Construct paths based on base directory and sequence
    # Default path structure (can be overridden with --lidar_subpath)
    # If --lidar_subpath is provided, use it; otherwise use common STU structure
    if [[ -z "$STU_LIDAR_SUBPATH" ]]; then
        # Try common STU path structures - check which one exists (both train and val)
        # Option 1: train_pointcloud/.../train/SEQUENCE/velodyne
        LIDAR_PATH1="$STU_BASE_DIR/train_pointcloud/nodes/dom/work/nekrasov/data/stu_dataset/train/$STU_SEQUENCE/velodyne"
        # Option 2: val_pointcloud/.../val/SEQUENCE/velodyne
        LIDAR_PATH1_VAL="$STU_BASE_DIR/val_pointcloud/nodes/dom/work/nekrasov/data/stu_dataset/val/$STU_SEQUENCE/velodyne"
        # Option 3: train_pointcloud/SEQUENCE/velodyne (simpler)
        LIDAR_PATH2_TRAIN="$STU_BASE_DIR/train_pointcloud/$STU_SEQUENCE/velodyne"
        # Option 4: val_pointcloud/SEQUENCE/velodyne (simpler)
        LIDAR_PATH2_VAL="$STU_BASE_DIR/val_pointcloud/$STU_SEQUENCE/velodyne"
        # Option 5: train/SEQUENCE/velodyne
        LIDAR_PATH3_TRAIN="$STU_BASE_DIR/train/$STU_SEQUENCE/velodyne"
        # Option 6: val/SEQUENCE/velodyne
        LIDAR_PATH3_VAL="$STU_BASE_DIR/val/$STU_SEQUENCE/velodyne"
        # Option 7: SEQUENCE/velodyne (direct structure)
        LIDAR_PATH4="$STU_BASE_DIR/$STU_SEQUENCE/velodyne"
        
        if [[ -d "$LIDAR_PATH1" ]]; then
            STU_LIDAR_SUBPATH="train_pointcloud/nodes/dom/work/nekrasov/data/stu_dataset/train"
            DATASET_SPLIT="train"
            echo "  Using detected path structure: train_pointcloud/.../train/..."
        elif [[ -d "$LIDAR_PATH1_VAL" ]]; then
            STU_LIDAR_SUBPATH="val_pointcloud/nodes/dom/work/nekrasov/data/stu_dataset/val"
            DATASET_SPLIT="val"
            echo "  Using detected path structure: val_pointcloud/.../val/..."
        elif [[ -d "$LIDAR_PATH2_TRAIN" ]]; then
            STU_LIDAR_SUBPATH="train_pointcloud"
            DATASET_SPLIT="train"
            echo "  Using detected path structure: train_pointcloud/SEQUENCE/..."
        elif [[ -d "$LIDAR_PATH2_VAL" ]]; then
            STU_LIDAR_SUBPATH="val_pointcloud"
            DATASET_SPLIT="val"
            echo "  Using detected path structure: val_pointcloud/SEQUENCE/..."
        elif [[ -d "$LIDAR_PATH3_TRAIN" ]]; then
            STU_LIDAR_SUBPATH="train"
            DATASET_SPLIT="train"
            echo "  Using detected path structure: train/SEQUENCE/..."
        elif [[ -d "$LIDAR_PATH3_VAL" ]]; then
            STU_LIDAR_SUBPATH="val"
            DATASET_SPLIT="val"
            echo "  Using detected path structure: val/SEQUENCE/..."
        elif [[ -d "$LIDAR_PATH4" ]]; then
            STU_LIDAR_SUBPATH=""
            echo "  Using detected path structure: SEQUENCE/..."
        else
            # Default fallback
            STU_LIDAR_SUBPATH="train"
            echo "  Warning: Could not auto-detect LiDAR path structure, using default: train/"
        fi
    else
        echo "  Using custom LiDAR subpath: $STU_LIDAR_SUBPATH"
        # Try to detect split from subpath
        if [[ "$STU_LIDAR_SUBPATH" == *"val"* ]]; then
            DATASET_SPLIT="val"
        fi
    fi
    
    # Construct paths
    if [[ -z "$STU_LIDAR_SUBPATH" ]]; then
        LIDAR="$STU_BASE_DIR/$STU_SEQUENCE/velodyne"
        LABELS="$STU_BASE_DIR/$STU_SEQUENCE/labels"
    else
        LIDAR="$STU_BASE_DIR/$STU_LIDAR_SUBPATH/$STU_SEQUENCE/velodyne"
        LABELS="$STU_BASE_DIR/$STU_LIDAR_SUBPATH/$STU_SEQUENCE/labels"
    fi
    
    # Try different image path structures (both train and val)
    IMAGES_PATH1_TRAIN="$STU_BASE_DIR/train_images/$STU_SEQUENCE/port_a_cam_0"
    IMAGES_PATH1_VAL="$STU_BASE_DIR/val_images/$STU_SEQUENCE/port_a_cam_0"
    IMAGES_PATH2="$STU_BASE_DIR/images/$STU_SEQUENCE/port_a_cam_0"
    IMAGES_PATH3="$STU_BASE_DIR/$STU_SEQUENCE/images/port_a_cam_0"
    IMAGES_PATH4="$STU_BASE_DIR/$STU_SEQUENCE/port_a_cam_0"
    
    if [[ -d "$IMAGES_PATH1_TRAIN" ]]; then
        IMAGES="$IMAGES_PATH1_TRAIN"
        DATASET_SPLIT="train"
    elif [[ -d "$IMAGES_PATH1_VAL" ]]; then
        IMAGES="$IMAGES_PATH1_VAL"
        DATASET_SPLIT="val"
    elif [[ -d "$IMAGES_PATH2" ]]; then
        IMAGES="$IMAGES_PATH2"
    elif [[ -d "$IMAGES_PATH3" ]]; then
        IMAGES="$IMAGES_PATH3"
    elif [[ -d "$IMAGES_PATH4" ]]; then
        IMAGES="$IMAGES_PATH4"
    else
        # Default fallback
        IMAGES="$STU_BASE_DIR/train_images/$STU_SEQUENCE/port_a_cam_0"
        echo "  Warning: Could not auto-detect images path, using default: train_images/SEQUENCE/port_a_cam_0"
    fi

    
    
    OUT_IMAGES="out/stu/$DATASET_SPLIT/$STU_SEQUENCE/images"
    OUT_LIDAR="out/stu/$DATASET_SPLIT/$STU_SEQUENCE/lidar"
    
    echo "  Images: $IMAGES"
    echo "  LiDAR: $LIDAR"
    echo "  Labels: $LABELS"
    
    # Check if input directories exist
    if [[ ! -d "$IMAGES" ]]; then
        echo "Error: Images directory not found: $IMAGES"
        exit 1
    fi
    if [[ ! -d "$LIDAR" ]]; then
        echo "Error: LiDAR directory not found: $LIDAR"
        exit 1
    fi
    if [[ ! -d "$LABELS" ]]; then
        echo "Error: Labels directory not found: $LABELS"
        exit 1
    fi
    
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
                $MANUAL_PLACEMENT_ARGS \
                --x_range 8.0 10.5 \
                --y_range -2 2 \
                --place_tries 150 \
                --clearance 0.25 \
                --ground_ransac_thresh 0.15 \
                --z_margin 0.2 \
                --require_inside_frac 0.7 \
                --unoccluded_thresh 0.90 \
                --surf_samples 10000 \
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
