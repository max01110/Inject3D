import numpy as np
import cv2

from .proj_ops import invert_lidar_cam
from .mesh_ops import  _collect_world_triangles, _sample_points_on_triangles_world

def build_scene_zbuffer_from_lidar(lidar_bin_path, T_lidar_cam, model, K, D, width, height, dilate_px=1):
    """
    Create a per-pixel nearest-depth Z-buffer (in camera meters) from LiDAR.
    Returns zbuf with shape (H,W), where np.inf means 'no LiDAR point'.
    """
    pts = np.fromfile(lidar_bin_path, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
    R_cl, t_cl = invert_lidar_cam(T_lidar_cam)  # camera-from-lidar
    pts_c = (R_cl @ pts.T).T + t_cl
    z = pts_c[:, 2]
    m_front = z > 1e-6
    if not np.any(m_front):
        return np.full((height, width), np.inf, np.float32)

    pts_c = pts_c[m_front]
    z = z[m_front]

    # Project to pixels 
    obj = pts_c.reshape(-1,1,3).astype(np.float64)
    rvec = np.zeros((3,1)); tvec = np.zeros((3,1))
    if model.lower() in ["equidistant", "fisheye"]:
        uv, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K.astype(np.float64), D.astype(np.float64))
    else:
        uv, _ = cv2.projectPoints(obj, rvec, tvec, K.astype(np.float64), D.reshape(-1,1))
    uv = uv.reshape(-1, 2)

    u = np.round(uv[:,0]).astype(np.int32)
    v = np.round(uv[:,1]).astype(np.int32)
    in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[in_img]; v = v[in_img]; z = z[in_img]

    zbuf = np.full((height, width), np.inf, np.float32)
    if len(z):
        #for repeated pixels, keep the nearest depth
        np.minimum.at(zbuf, (v, u), z.astype(np.float32))

    # Conservative dilation (optional): expand occupied pixels slightly
    if dilate_px and dilate_px > 0 and np.isfinite(zbuf).any():
        occ = (np.isfinite(zbuf)).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px+1, 2*dilate_px+1))
        occ_d = cv2.dilate(occ, kernel)
        # Where dilation created new occupancy, copy nearest depths from original (approx by local min filter)
        # Simple approach: blur the original zbuf where occ_d==1 and zbuf==inf → fill with min of neighbors
        zmin = cv2.erode(np.where(np.isfinite(zbuf), zbuf, 1e9).astype(np.float32), kernel)
        zbuf = np.where((occ_d==1) & ~np.isfinite(zbuf), zmin, zbuf)

    return zbuf

def project_world_points_to_cam_uv(points_world, T_lidar_cam, model, K, D):
    """
    points_world: (N,3) in LiDAR/world
    Returns uv (N,2 float32) and z (N,)
    """
    R_cl, t_cl = invert_lidar_cam(T_lidar_cam)
    pts_c = (R_cl @ points_world.T).T + t_cl
    z = pts_c[:,2]
    obj = pts_c.reshape(-1,1,3).astype(np.float64)
    rvec = np.zeros((3,1)); tvec = np.zeros((3,1))
    if model.lower() in ["equidistant", "fisheye"]:
        uv, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K.astype(np.float64), D.astype(np.float64))
    else:
        uv, _ = cv2.projectPoints(obj, rvec, tvec, K.astype(np.float64), D.reshape(-1,1))
    return uv.reshape(-1,2).astype(np.float32), z.astype(np.float32)

def fraction_unoccluded(obj_parent, zbuf, T_lidar_cam, model, K, D, width, height,
                        n_surface_samples=3000, z_margin=0.05, require_inside_frac=0.85):
    """
    Returns (inside_frac, unoccluded_frac).
    inside_frac: fraction of sampled surface points that are in front of the camera and inside the image.
    unoccluded_frac: within those, fraction whose depth is strictly less than scene depth - margin,
                     or where the scene has no depth (np.inf).
    """
    # Sample object surface in world (LiDAR) coords
    tris_world = _collect_world_triangles(obj_parent)
    surf = _sample_points_on_triangles_world(tris_world, n_surface_samples)

    uv, z = project_world_points_to_cam_uv(surf, T_lidar_cam, model, K, D)
    in_front = z > 1e-6
    if not np.any(in_front):
        return 0.0, 0.0

    uv = uv[in_front]; z = z[in_front]
    u = np.round(uv[:,0]).astype(np.int32)
    v = np.round(uv[:,1]).astype(np.int32)
    in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(in_img):
        return 0.0, 0.0

    u = u[in_img]; v = v[in_img]; z = z[in_img]
    inside_frac = float(len(z)) / float(n_surface_samples)

    # Gather scene depths at those pixels
    scene_z = zbuf[v, u]
    # Unoccluded if either no scene depth there, or object is closer by at least margin
    unocc = (np.isinf(scene_z)) | (z <= (scene_z - z_margin))
    unoccluded_frac = float(np.count_nonzero(unocc)) / float(len(z))
    return inside_frac, unoccluded_frac
