"""
Occlusion and visibility checking using LiDAR-derived z-buffer.
"""
import numpy as np
import cv2

from .proj_ops import invert_lidar_cam
from .mesh_ops import _collect_world_triangles, _sample_points_on_triangles_world


def build_scene_zbuffer_from_lidar(lidar_bin_path, T_lidar_cam, model, K, D,
                                    width, height, dilate_px=1):
    """
    Build per-pixel depth buffer from LiDAR points.
    Returns (H,W) array where np.inf = no depth.
    """
    pts = np.fromfile(lidar_bin_path, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)

    R_cl, t_cl = invert_lidar_cam(T_lidar_cam)
    pts_c = (R_cl @ pts.T).T + t_cl
    z = pts_c[:, 2]

    front = z > 1e-6
    if not np.any(front):
        return np.full((height, width), np.inf, np.float32)

    pts_c, z = pts_c[front], z[front]

    # Project to image
    obj = pts_c.reshape(-1, 1, 3).astype(np.float64)
    rvec, tvec = np.zeros((3, 1)), np.zeros((3, 1))
    uv, _ = cv2.projectPoints(obj, rvec, tvec, K.astype(np.float64), D.reshape(-1, 1))
    uv = uv.reshape(-1, 2)

    u = np.round(uv[:, 0]).astype(np.int32)
    v = np.round(uv[:, 1]).astype(np.int32)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[valid], v[valid], z[valid]

    zbuf = np.full((height, width), np.inf, np.float32)
    if len(z):
        np.minimum.at(zbuf, (v, u), z.astype(np.float32))

    # Dilate occupied pixels
    if dilate_px > 0 and np.isfinite(zbuf).any():
        occ = np.isfinite(zbuf).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px+1, 2*dilate_px+1))
        occ_d = cv2.dilate(occ, kernel)
        zmin = cv2.erode(np.where(np.isfinite(zbuf), zbuf, 1e9).astype(np.float32), kernel)
        zbuf = np.where((occ_d == 1) & ~np.isfinite(zbuf), zmin, zbuf)

    return zbuf


def project_world_points_to_cam_uv(points, T_lidar_cam, model, K, D):
    """Project world points to image coordinates. Returns (uv, z)."""
    R_cl, t_cl = invert_lidar_cam(T_lidar_cam)
    pts_c = (R_cl @ points.T).T + t_cl
    z = pts_c[:, 2]

    obj = pts_c.reshape(-1, 1, 3).astype(np.float64)
    rvec, tvec = np.zeros((3, 1)), np.zeros((3, 1))
    uv, _ = cv2.projectPoints(obj, rvec, tvec, K.astype(np.float64), D.reshape(-1, 1))
    return uv.reshape(-1, 2).astype(np.float32), z.astype(np.float32)


def fraction_unoccluded(obj_parent, zbuf, T_lidar_cam, model, K, D, width, height,
                        n_surface_samples=3000, z_margin=0.05, require_inside_frac=0.85):
    """
    Check what fraction of object surface is visible.
    Returns (inside_frac, unoccluded_frac).
    """
    tris = _collect_world_triangles(obj_parent)
    surf = _sample_points_on_triangles_world(tris, n_surface_samples)

    uv, z = project_world_points_to_cam_uv(surf, T_lidar_cam, model, K, D)

    front = z > 1e-6
    if not np.any(front):
        return 0.0, 0.0

    uv, z = uv[front], z[front]
    u = np.round(uv[:, 0]).astype(np.int32)
    v = np.round(uv[:, 1]).astype(np.int32)

    in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(in_img):
        return 0.0, 0.0

    u, v, z = u[in_img], v[in_img], z[in_img]
    inside_frac = len(z) / n_surface_samples

    scene_z = zbuf[v, u]
    unoccluded = np.isinf(scene_z) | (z <= scene_z - z_margin)
    unoccluded_frac = np.count_nonzero(unoccluded) / len(z)

    return inside_frac, unoccluded_frac
