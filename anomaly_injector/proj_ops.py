"""
Point projection and mask operations.
"""
import numpy as np
import cv2


def invert_lidar_cam(T_lidar_cam):
    """Invert LiDAR-to-camera transform. Returns (R_cl, t_cl)."""
    R_lc = T_lidar_cam[:3, :3]
    t_lc = T_lidar_cam[:3, 3]
    R_cl = R_lc.T
    t_cl = -R_cl @ t_lc
    return R_cl, t_cl


def project_points_distorted(pts_lidar, T_lidar_cam, model, K, D, min_z=1e-6):
    """Project LiDAR points to distorted image coordinates."""
    R_cl, t_cl = invert_lidar_cam(T_lidar_cam)
    pts_c = (R_cl @ pts_lidar.T).T + t_cl
    z = pts_c[:, 2]

    front = z > min_z
    pts_c = pts_c[front]
    if len(pts_c) == 0:
        return np.empty((0, 2), np.float32)

    obj = pts_c.reshape(-1, 1, 3).astype(np.float64)
    rvec, tvec = np.zeros((3, 1)), np.zeros((3, 1))

    if model in ["equidistant", "fisheye"]:
        uv, _ = cv2.fisheye.projectPoints(obj, rvec, tvec,
                                          K.astype(np.float64), D.astype(np.float64))
    else:
        uv, _ = cv2.projectPoints(obj, rvec, tvec,
                                  K.astype(np.float64), D.reshape(-1, 1))

    return uv.reshape(-1, 2).astype(np.float32)


def mask_from_points(uv, width, height, radius=2):
    """Create binary mask from projected points."""
    mask = np.zeros((height, width), np.uint8)
    if len(uv) == 0:
        return mask

    u = np.round(uv[:, 0]).astype(np.int32)
    v = np.round(uv[:, 1]).astype(np.int32)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v = u[valid], v[valid]

    if radius <= 1:
        mask[v, u] = 255
    else:
        for x, y in zip(u, v):
            cv2.circle(mask, (int(x), int(y)), radius, 255, -1, lineType=cv2.LINE_AA)

    return mask


def _bbox(mask):
    """Get bounding box of mask. Returns (x0, y0, x1, y1) or None."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _center_and_size(mask):
    """Get center and size of mask bounding box."""
    bb = _bbox(mask)
    if bb is None:
        return None, None
    x0, y0, x1, y1 = bb
    w, h = max(1, x1 - x0 + 1), max(1, y1 - y0 + 1)
    cx, cy = x0 + w * 0.5, y0 + h * 0.5
    return (cx, cy), (w, h)


def estimate_affine_scale_translate(src_mask, dst_mask):
    """Estimate scale and translation to align src_mask to dst_mask."""
    cs, ss = _center_and_size(src_mask)
    cd, sd = _center_and_size(dst_mask)
    if cs is None or cd is None:
        return 1.0, 1.0, 0.0, 0.0

    sx = sd[0] / max(1e-6, ss[0])
    sy = sd[1] / max(1e-6, ss[1])
    dx = cd[0] - sx * cs[0]
    dy = cd[1] - sy * cs[1]
    return float(sx), float(sy), float(dx), float(dy)


def warp_rgba_affine_scale_translate(rgba, sx, sy, dx, dy):
    """Apply scale and translation to RGBA image."""
    h, w = rgba.shape[:2]
    A = np.array([[sx, 0.0, dx], [0.0, sy, dy]], dtype=np.float32)
    return cv2.warpAffine(rgba, A, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)


def compute_iou(mask_a, mask_b):
    """Compute intersection over union of two masks."""
    a = (mask_a > 0).astype(np.uint8)
    b = (mask_b > 0).astype(np.uint8)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0
