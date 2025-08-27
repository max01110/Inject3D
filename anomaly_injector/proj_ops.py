
import numpy as np
import cv2



def invert_lidar_cam(T_lidar_cam):
    R_lc = T_lidar_cam[:3,:3]; t_lc = T_lidar_cam[:3,3]
    R_cl = R_lc.T
    t_cl = -R_cl @ t_lc         # camera origin in LiDAR/world frame
    return R_cl, t_cl

def project_points_distorted(pts_lidar, T_lidar_cam, model, K, D, min_z=1e-6):
    R_cl, t_cl = invert_lidar_cam(T_lidar_cam)
    pts_c = (R_cl @ pts_lidar.T).T + t_cl
    z = pts_c[:,2]
    m = z > min_z
    pts_c = pts_c[m]
    if len(pts_c)==0:
        return np.empty((0,2), np.float32)

    obj = pts_c.reshape(-1,1,3).astype(np.float64)
    rvec = np.zeros((3,1)); tvec = np.zeros((3,1))
    if model in ["equidistant","fisheye"]:
        uv, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K.astype(np.float64), D.astype(np.float64))
    else:
        uv, _ = cv2.projectPoints(obj, rvec, tvec, K.astype(np.float64), D.reshape(-1,1))
    return uv.reshape(-1,2).astype(np.float32)

def _bbox(mask_u8):
    ys, xs = np.where(mask_u8>0)
    if len(xs)==0: return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return (int(x0), int(y0), int(x1), int(y1))

def _center_and_size(mask_u8):
    bb = _bbox(mask_u8)
    if bb is None: return None, None
    x0,y0,x1,y1 = bb
    w = max(1, x1 - x0 + 1)
    h = max(1, y1 - y0 + 1)
    cx = x0 + w*0.5
    cy = y0 + h*0.5
    return (cx,cy), (w,h)


def mask_from_points(uv, width, height, radius=2):
    M = np.zeros((height, width), np.uint8)
    if len(uv)==0: return M
    u = np.round(uv[:,0]).astype(np.int32)
    v = np.round(uv[:,1]).astype(np.int32)
    m = (u>=0)&(u<width)&(v>=0)&(v<height)
    u=u[m]; v=v[m]
    if radius <= 1:
        M[v, u] = 255
    else:
        for x,y in zip(u,v):
            cv2.circle(M, (int(x),int(y)), radius, 255, -1, lineType=cv2.LINE_AA)
    return M



def estimate_affine_scale_translate(src_mask, dst_mask):
    cs, ss = _center_and_size(src_mask)
    cd, sd = _center_and_size(dst_mask)
    if cs is None or cd is None:
        return 1.0, 1.0, 0.0, 0.0
    sx = sd[0] / max(1e-6, ss[0])
    sy = sd[1] / max(1e-6, ss[1])
    dx = cd[0] - sx*cs[0]
    dy = cd[1] - sy*cs[1]
    return float(sx), float(sy), float(dx), float(dy)

def warp_rgba_affine_scale_translate(rgba, sx, sy, dx, dy):
    h, w = rgba.shape[:2]
    A = np.array([[sx, 0.0, dx],
                  [0.0, sy, dy]], dtype=np.float32)
    return cv2.warpAffine(rgba, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def compute_iou(maskA, maskB):
    A = (maskA>0).astype(np.uint8); B=(maskB>0).astype(np.uint8)
    inter = int(np.logical_and(A,B).sum())
    union = int(np.logical_or(A,B).sum())
    return 0.0 if union == 0 else inter/union
