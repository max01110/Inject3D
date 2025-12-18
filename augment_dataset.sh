#!/usr/bin/env bash
#
# augment_dataset.sh - Batch augmentation for KITTI and STU datasets
#
# KITTI usage:
#   ./augment_dataset.sh [SEQUENCES]
#   ./augment_dataset.sh "00,01,02"    # specific sequences
#   ./augment_dataset.sh "00-05"       # range
#   ./augment_dataset.sh "all"         # all (00-21)
#
# STU usage:
#   ./augment_dataset.sh --dataset stu --base_dir /path/to/stu --sequence 201
#   ./augment_dataset.sh --dataset stu --base_dir /path/to/stu --sequence 201 --start_frame 50
#   ./augment_dataset.sh --dataset stu --base_dir /path/to/stu --sequence 201 --frames "1,3,5,10-15"
#
# Manual placement (both datasets):
#   ./augment_dataset.sh --manual_x 10.0 --manual_y 1.0 [--manual_z 0.5] [--manual_yaw 45]
#   ./augment_dataset.sh --manual_x 10.0 --manual_y 1.0 --manual_adjust_to_ground
#
# See README.md for full documentation.

set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT="main.py"
PER_PAIR=1
IOU_THRESH=0.85
IOU_MAX_TRIES=10

# Dataset defaults
DATASET="kitti_sequences"
KITTI_SEQUENCES="all"
KITTI_BASE="/mnt/data1/datasets/kitti_odom/dataset/sequences"
KITTI_OUT="/mnt/data1/datasets/augmented_kitti_odom"

# STU-specific
STU_BASE_DIR=""
STU_SEQUENCE=""
STU_LIDAR_SUBPATH=""

# Frame selection
START_FRAME=0
SPECIFIC_FRAMES=""

# Manual placement (leave empty for automatic)
MANUAL_X=""
MANUAL_Y=""
MANUAL_Z=""
MANUAL_YAW=""
MANUAL_ADJUST_TO_GROUND="1"

# ==============================================================================
# Argument parsing
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)       DATASET="$2"; shift 2 ;;
        --base_dir)      STU_BASE_DIR="$2"; shift 2 ;;
        --sequence)      STU_SEQUENCE="$2"; shift 2 ;;
        --lidar_subpath) STU_LIDAR_SUBPATH="$2"; shift 2 ;;
        --manual_x)      MANUAL_X="$2"; shift 2 ;;
        --manual_y)      MANUAL_Y="$2"; shift 2 ;;
        --manual_z)      MANUAL_Z="$2"; shift 2 ;;
        --manual_yaw)    MANUAL_YAW="$2"; shift 2 ;;
        --manual_adjust_to_ground) MANUAL_ADJUST_TO_GROUND="1"; shift ;;
        --start_frame)   START_FRAME="$2"; shift 2 ;;
        --frames)        SPECIFIC_FRAMES="$2"; shift 2 ;;
        *)
            if [[ "$DATASET" == "kitti_sequences" || "$DATASET" == "kitti" ]]; then
                KITTI_SEQUENCES="$1"
            fi
            shift
            ;;
    esac
done

# ==============================================================================
# Helper functions
# ==============================================================================

# Parse sequence input: "all", "00-05", or "00,01,02"
parse_sequences() {
    local input="$1"
    local sequences=()
    
    if [[ "$input" == "all" ]]; then
        for i in {00..21}; do sequences+=("$i"); done
    elif [[ "$input" =~ ^[0-9]+-[0-9]+$ ]]; then
        local start end
        start=$(echo "$input" | cut -d'-' -f1)
        end=$(echo "$input" | cut -d'-' -f2)
        for ((i=start; i<=end; i++)); do
            sequences+=("$(printf "%02d" "$i")")
        done
    else
        IFS=',' read -ra arr <<< "$input"
        for seq in "${arr[@]}"; do
            seq=$(echo "$seq" | xargs | printf "%02d" "$(cat)")
            sequences+=("$seq")
        done
    fi
    echo "${sequences[@]}"
}

# Parse frame list: "1,3,5,10-15" -> space-separated numbers
parse_frames() {
    local input="$1"
    local frames=()
    input=$(echo "$input" | tr -d ' ')
    
    IFS=',' read -ra parts <<< "$input"
    for part in "${parts[@]}"; do
        if [[ "$part" =~ ^[0-9]+-[0-9]+$ ]]; then
            local start end
            start=$(echo "$part" | cut -d'-' -f1)
            end=$(echo "$part" | cut -d'-' -f2)
            for ((i=start; i<=end; i++)); do frames+=("$i"); done
        else
            frames+=("$part")
        fi
    done
    echo "${frames[@]}"
}

# Build manual placement args if configured
build_manual_args() {
    if [[ -n "$MANUAL_X" && -n "$MANUAL_Y" ]]; then
        local args="--manual_x $MANUAL_X --manual_y $MANUAL_Y"
        [[ -n "$MANUAL_Z" ]] && args="$args --manual_z $MANUAL_Z"
        args="$args --manual_yaw ${MANUAL_YAW:-0.0}"
        [[ -n "$MANUAL_ADJUST_TO_GROUND" ]] && args="$args --manual_adjust_to_ground"
        echo "$args"
    fi
}

# Check if frame should be processed
declare -A FRAMES_TO_PROCESS
should_process_frame() {
    local frame_num="$1"
    [[ -z "$SPECIFIC_FRAMES" ]] && return 0
    [[ -v FRAMES_TO_PROCESS[$frame_num] ]] && return 0
    return 1
}

# Process a single image-lidar pair
process_frame() {
    local img_path="$1"
    local lidar_dir="$2"
    local label_dir="$3"
    local out_images="$4"
    local out_lidar="$5"
    local calib="$6"
    local extra_args="$7"
    
    local stem
    stem=$(basename "$img_path" .png)
    local frame_num=$((10#$stem))
    
    # Skip if before start frame or not in specific frames list
    [[ "$frame_num" -lt "$START_FRAME" ]] && return
    should_process_frame "$frame_num" || return
    
    local lidar_path="$lidar_dir/$stem.bin"
    local label_path="$label_dir/$stem.label"
    
    if [[ ! -f "$lidar_path" || ! -f "$label_path" ]]; then
        echo "Skipping $stem (missing lidar or label)"
        return
    fi
    
    echo "Processing: $img_path"
    
    for ((k=1; k<=PER_PAIR; k++)); do
        local out_img="$out_images/${stem}_aug${k}.png"
        if [[ -f "$out_img" ]]; then
            echo "  Exists, skipping: $out_img"
            continue
        fi
        
        local tmp_out
        tmp_out=$(mktemp -d)
        
        echo "  Generating augmentation $k..."
        blender -b -P "$SCRIPT" -- \
            --calib "$calib" \
            --image "$img_path" \
            --lidar "$lidar_path" \
            --labels "$label_path" \
            --outdir "$tmp_out" \
            --iou_thresh "$IOU_THRESH" \
            --iou_max_tries "$IOU_MAX_TRIES" \
            $extra_args
        
        # Move outputs
        [[ -f "$tmp_out/aug_image.png" ]] && mv "$tmp_out/aug_image.png" "$out_img"
        [[ -f "$tmp_out/aug_lidar.bin" ]] && mv "$tmp_out/aug_lidar.bin" "$out_lidar/${stem}_aug${k}.bin"
        [[ -f "$tmp_out/aug_lidar.label" ]] && mv "$tmp_out/aug_lidar.label" "$out_lidar/${stem}_aug${k}.label"
        
        rm -rf "$tmp_out"
    done
}

# Auto-detect STU directory structure
detect_stu_paths() {
    local base="$1"
    local seq="$2"
    
    # LiDAR paths to try (in order of priority)
    local lidar_paths=(
        "train_pointcloud/nodes/dom/work/nekrasov/data/stu_dataset/train/$seq/velodyne"
        "val_pointcloud/nodes/dom/work/nekrasov/data/stu_dataset/val/$seq/velodyne"
        "train_pointcloud/$seq/velodyne"
        "val_pointcloud/$seq/velodyne"
        "train/$seq/velodyne"
        "val/$seq/velodyne"
        "$seq/velodyne"
    )
    
    # Image paths to try
    local image_paths=(
        "train_images/$seq/port_a_cam_0"
        "val_images/$seq/port_a_cam_0"
        "images/$seq/port_a_cam_0"
        "$seq/images/port_a_cam_0"
        "$seq/port_a_cam_0"
    )
    
    # Find LiDAR path
    DETECTED_LIDAR=""
    DETECTED_LABELS=""
    for p in "${lidar_paths[@]}"; do
        if [[ -d "$base/$p" ]]; then
            DETECTED_LIDAR="$base/$p"
            DETECTED_LABELS="${DETECTED_LIDAR%/velodyne}/labels"
            break
        fi
    done
    
    # Find image path
    DETECTED_IMAGES=""
    for p in "${image_paths[@]}"; do
        if [[ -d "$base/$p" ]]; then
            DETECTED_IMAGES="$base/$p"
            break
        fi
    done
    
    # Detect split from path
    if [[ "$DETECTED_LIDAR" == *"val"* || "$DETECTED_IMAGES" == *"val"* ]]; then
        DETECTED_SPLIT="val"
    else
        DETECTED_SPLIT="train"
    fi
}

# ==============================================================================
# Main
# ==============================================================================

MANUAL_ARGS=$(build_manual_args)
if [[ -n "$MANUAL_ARGS" ]]; then
    echo "[INFO] Manual placement: $MANUAL_ARGS"
else
    echo "[INFO] Automatic placement"
fi

[[ "$START_FRAME" -gt 0 ]] && echo "[INFO] Starting from frame $START_FRAME"

# Populate frame filter if specified
if [[ -n "$SPECIFIC_FRAMES" ]]; then
    echo "[INFO] Processing specific frames: $SPECIFIC_FRAMES"
    FRAME_LIST=($(parse_frames "$SPECIFIC_FRAMES"))
    for f in "${FRAME_LIST[@]}"; do FRAMES_TO_PROCESS[$f]=1; done
    echo "[INFO] Total frames to process: ${#FRAME_LIST[@]}"
fi

# ------------------------------------------------------------------------------
# KITTI dataset
# ------------------------------------------------------------------------------
if [[ "$DATASET" == "kitti_sequences" || "$DATASET" == "kitti" ]]; then
    SEQUENCE_LIST=($(parse_sequences "$KITTI_SEQUENCES"))
    echo "Processing KITTI sequences: ${SEQUENCE_LIST[*]}"
    
    for seq in "${SEQUENCE_LIST[@]}"; do
        echo "=== Sequence $seq ==="
        
        CALIB="input/KITTI_dataset/KITTI_calibs/calib_${seq}.yaml"
        if [[ ! -f "$CALIB" ]]; then
            echo "Warning: Calibration not found ($CALIB), skipping"
            continue
        fi
        
        IMAGES="$KITTI_BASE/$seq/image_2"
        LIDAR="$KITTI_BASE/$seq/velodyne"
        LABELS="$KITTI_BASE/$seq/labels"
        OUT_IMAGES="$KITTI_OUT/$seq/images"
        OUT_LIDAR="$KITTI_OUT/$seq/lidar"
        
        mkdir -p "$OUT_IMAGES" "$OUT_LIDAR"
        
        # KITTI-specific Blender args
        EXTRA_ARGS="$MANUAL_ARGS --x_range 8 40 --y_range -2 2 --place_tries 150"
        EXTRA_ARGS="$EXTRA_ARGS --clearance 0.30 --ground_ransac_thresh 0.15 --z_margin 0.2"
        EXTRA_ARGS="$EXTRA_ARGS --require_inside_frac 0.7 --unoccluded_thresh 0.90"
        EXTRA_ARGS="$EXTRA_ARGS --surf_samples 3000 --fast"
        
        for img in "$IMAGES"/*.png; do
            [[ -f "$img" ]] || continue
            process_frame "$img" "$LIDAR" "$LABELS" "$OUT_IMAGES" "$OUT_LIDAR" "$CALIB" "$EXTRA_ARGS"
        done
    done
    
    echo "Done!"
    exit 0
fi

# ------------------------------------------------------------------------------
# STU dataset
# ------------------------------------------------------------------------------
if [[ "$DATASET" == "stu" ]]; then
    if [[ -z "$STU_BASE_DIR" || -z "$STU_SEQUENCE" ]]; then
        echo "Error: --base_dir and --sequence required for STU"
        echo "Usage: ./augment_dataset.sh --dataset stu --base_dir BASE --sequence SEQ"
        exit 1
    fi
    
    echo "Processing STU sequence $STU_SEQUENCE"
    echo "  Base: $STU_BASE_DIR"
    
    CALIB="input/STU_dataset/calib.yaml"
    
    # Use custom subpath or auto-detect
    if [[ -n "$STU_LIDAR_SUBPATH" ]]; then
        LIDAR="$STU_BASE_DIR/$STU_LIDAR_SUBPATH/$STU_SEQUENCE/velodyne"
        LABELS="$STU_BASE_DIR/$STU_LIDAR_SUBPATH/$STU_SEQUENCE/labels"
        detect_stu_paths "$STU_BASE_DIR" "$STU_SEQUENCE"
        IMAGES="$DETECTED_IMAGES"
    else
        detect_stu_paths "$STU_BASE_DIR" "$STU_SEQUENCE"
        LIDAR="$DETECTED_LIDAR"
        LABELS="$DETECTED_LABELS"
        IMAGES="$DETECTED_IMAGES"
    fi
    
    echo "  Images: $IMAGES"
    echo "  LiDAR:  $LIDAR"
    echo "  Labels: $LABELS"
    
    # Validate paths
    for path in "$IMAGES" "$LIDAR" "$LABELS"; do
        if [[ ! -d "$path" ]]; then
            echo "Error: Directory not found: $path"
            exit 1
        fi
    done
    
    OUT_IMAGES="out/stu/$DETECTED_SPLIT/$STU_SEQUENCE/images"
    OUT_LIDAR="out/stu/$DETECTED_SPLIT/$STU_SEQUENCE/lidar"
    mkdir -p "$OUT_IMAGES" "$OUT_LIDAR"
    
    # STU-specific Blender args
    EXTRA_ARGS="$MANUAL_ARGS --x_range 5.5 12.0 --y_range -2 2 --place_tries 100"
    EXTRA_ARGS="$EXTRA_ARGS --clearance 0.0 --ground_ransac_thresh 0.15 --z_margin 0.2"
    EXTRA_ARGS="$EXTRA_ARGS --require_inside_frac 0.5 --unoccluded_thresh 0.50"
    EXTRA_ARGS="$EXTRA_ARGS --surf_samples 5000 --fast"
    
    for img in "$IMAGES"/*.png; do
        [[ -f "$img" ]] || continue
        process_frame "$img" "$LIDAR" "$LABELS" "$OUT_IMAGES" "$OUT_LIDAR" "$CALIB" "$EXTRA_ARGS"
    done
    
    echo "Done!"
    exit 0
fi

echo "Error: Unknown dataset '$DATASET' (use 'kitti_sequences' or 'stu')"
exit 1
 