"""
Anomaly injection module for LiDAR + camera data augmentation.
"""
from .calib import load_calib
from .blender_utils import (
    to_blender_matrix, ensure_collection, clear_scene, import_mesh,
    triangulate_and_smooth, set_camera_from_extrinsics,
    print_pose, print_camera_intrinsics, print_relative
)
from .mesh_ops import (
    _gather_world_vertices, measure_object_dims_world, fit_object_longest_to,
    _collect_world_triangles, _sample_points_on_triangles_world
)
from .image_ops import (
    build_rectified_to_distorted_map, rgba_rectified_to_distorted,
    alpha_composite_rgba_over_bgr, composite_over_background
)
from .proj_ops import (
    invert_lidar_cam, project_points_distorted, mask_from_points,
    estimate_affine_scale_translate, warp_rgba_affine_scale_translate, compute_iou
)
from .objaverse_io import get_random_objaverse
from .io_utils import write_augmented_pointcloud
from .placement import place_random_on_lidar_ground, place_manual
