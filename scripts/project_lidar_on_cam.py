#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# Import calib and proj_ops directly (avoid __init__.py which imports Blender modules)
import importlib.util
_pkg_root = Path(__file__).resolve().parents[1] / "anomaly_injector"

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_calib_mod = _load_module("calib", _pkg_root / "calib.py")
_proj_mod = _load_module("proj_ops", _pkg_root / "proj_ops.py")
load_calib = _calib_mod.load_calib
invert_lidar_cam = _proj_mod.invert_lidar_cam


def euler_cam_frame(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    rz, ry, rx = map(np.deg2rad, (yaw_deg, pitch_deg, roll_deg))
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    return Rz @ Ry @ Rx


def load_yaml_and_extrinsics(calib_path):
    """
    Wrapper that reuses the same calibration + extrinsics logic as the main
    injection pipeline (`anomaly_injector.calib.load_calib` and
    `anomaly_injector.proj_ops.invert_lidar_cam`), so projections here match
    the IoU check in `main.py`.
    """
    model, K, D, width, height, T_lidar_cam, P = load_calib(calib_path)
    R_cl, t_cl = invert_lidar_cam(T_lidar_cam)
    R_rect = np.eye(3, dtype=np.float64)

    return {
        "model": model,
        "K": K,
        "D": D,
        "P": P,
        "R_rect": R_rect,
        "width": width,
        "height": height,
        "R_cl": R_cl,
        "t_cl": t_cl,
    }


#Point loading

def _sibling_bin_path(label_path: Path):
    cand = label_path.with_suffix(".bin")
    return cand if cand.exists() else None


def load_points_and_labels(label_path: Path):
    """
    Try paired .bin first (float32 xyzI). Otherwise parse packed .label:
      - 20 bytes/pt: 16 (xyzI float32) + 4 (label uint32)
      - 16 bytes/pt: 12 (xyz float32)   + 4 (label uint32)
    """
    pair_bin = _sibling_bin_path(label_path)
    if pair_bin is not None:
        pts_all = np.fromfile(pair_bin, dtype=np.float32).reshape(-1, 4)
        pts = pts_all[:, :3].copy()
        labels = np.fromfile(label_path, dtype=np.uint32)
        if len(labels) != len(pts):
            raise ValueError(f"Mismatch: {pair_bin.name} has {len(pts)}, {label_path.name} has {len(labels)}.")
        return pts, labels

    file_size = label_path.stat().st_size
    if file_size % 20 == 0:
        n = file_size // 20
        raw = np.fromfile(label_path, dtype=np.uint8).reshape(n, 20)
        xyz_i = raw[:, :16].copy().view(np.float32).reshape(n, 4)
        lbl = raw[:, 16:].copy().view(np.uint32).reshape(n)
        return xyz_i[:, :3].astype(np.float32), lbl
    if file_size % 16 == 0:
        n = file_size // 16
        raw = np.fromfile(label_path, dtype=np.uint8).reshape(n, 16)
        xyz = raw[:, :12].copy().view(np.float32).reshape(n, 3)
        lbl = raw[:, 12:].copy().view(np.uint32).reshape(n)
        return xyz.astype(np.float32), lbl

    raise ValueError("Cannot get xyz from .label. Provide matching .bin, or 16/20 bytes/pt packing.")


# ============================ Projection models ============================



def project_plumb_bob(points_cam, K, D):
    """Pinhole with radial-tangential distortion (ROS 'plumb_bob' D=[k1,k2,p1,p2,k3])."""
    obj = points_cam.astype(np.float64).reshape(-1, 1, 3)
    rvec = np.zeros((3, 1)); tvec = np.zeros((3, 1))
    uv, _ = cv2.projectPoints(obj, rvec, tvec, K.astype(np.float64), D.reshape(-1, 1))
    return uv.reshape(-1, 2)



#Colors & Rendering 

def hsv_to_rgb(h, s, v):
    i = int(h * 6)
    f = h * 6 - i
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    i %= 6
    if   i == 0: r, g, b = v, t, p
    elif i == 1: r, g, b = q, v, p
    elif i == 2: r, g, b = p, v, t
    elif i == 3: r, g, b = p, q, v
    elif i == 4: r, g, b = t, p, v
    else:        r, g, b = v, p, q
    return int(r * 255), int(g * 255), int(b * 255)


def parse_color_arg(s):
    if s.startswith("#") and len(s) == 7:
        return int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)
    parts = s.split(",")
    if len(parts) == 3:
        return tuple(int(p) for p in parts)
    raise ValueError("Use '#RRGGBB' or 'R,G,B' for colors.")


def muted_color_map(labels, anomaly_label=None, anomaly_rgb=(255, 64, 160), sat=0.35, val=0.75):
    uniq = np.unique(labels.astype(np.uint64))
    lut = {}
    for lb in uniq:
        lb_i = int(lb)
        if anomaly_label is not None and lb_i == int(anomaly_label):
            lut[lb_i] = anomaly_rgb
            continue
        hue = (lb_i * 0.6180339887498949) % 1.0
        v = val * (0.92 + 0.08 * ((lb_i * 37) % 5) / 4.0)
        s = sat * (0.95 + 0.10 * ((lb_i * 53) % 3) / 2.0)
        lut[lb_i] = hsv_to_rgb(hue, s, v)
    return lut


def render_overlay(width, height, uv, labels, bg_img=None, dot_size=1,
                   anomaly_label=None, anomaly_rgb=(255, 64, 160),
                   palette_s=0.35, palette_v=0.75):
    lut = muted_color_map(labels, anomaly_label=anomaly_label,
                          anomaly_rgb=anomaly_rgb, sat=palette_s, val=palette_v)

    u = np.round(uv[:, 0]).astype(np.int32)
    v = np.round(uv[:, 1]).astype(np.int32)
    m = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, labs = u[m], v[m], labels[m].astype(np.uint64)

    if bg_img is None:
        color = np.zeros((height, width, 3), dtype=np.uint8)
        alpha = np.zeros((height, width), dtype=np.uint8)
    else:
        img = bg_img if (bg_img.shape[1] == width and bg_img.shape[0] == height) \
            else cv2.resize(bg_img, (width, height), interpolation=cv2.INTER_AREA)
        color = img.astype(np.uint8)
        alpha = np.full((height, width), 255, dtype=np.uint8)

    if dot_size <= 1:
        cols = np.array([lut[int(lb)] for lb in labs], dtype=np.uint8)  # RGB
        color[v, u] = cols[:, ::-1]  # to BGR
        alpha[v, u] = 255
    else:
        for x, y, lb in zip(u, v, labs):
            r, g, b = lut[int(lb)]
            cv2.circle(color, (int(x), int(y)), dot_size, (b, g, r), -1, lineType=cv2.LINE_AA)
            cv2.circle(alpha, (int(x), int(y)), dot_size, 255, -1, lineType=cv2.LINE_AA)

    return np.dstack([color, alpha])


# Main ============================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--max_points", type=int, default=40_000_000)
    ap.add_argument("--min_depth", type=float, default=0.1)
    ap.add_argument("--max_depth", type=float, default=200.0)
    ap.add_argument("--dot_size", type=int, default=3)

    ap.add_argument("--yaw_deg", type=float, default=0.0)
    ap.add_argument("--pitch_deg", type=float, default=0.0)
    ap.add_argument("--roll_deg", type=float, default=0.0)

    # anomaly + palette controls
    ap.add_argument("--anomaly_label", type=int, default=150, help="Label id to highlight distinctly (default: 150).")
    ap.add_argument("--anomaly_color", type=str, default="#ff40a0", help="RGB for anomaly as '#RRGGBB' or 'R,G,B'.")
    ap.add_argument("--palette_s", type=float, default=0.35)
    ap.add_argument("--palette_v", type=float, default=0.75)

    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml_and_extrinsics(args.calib)

    K, D, P = cfg["K"], cfg["D"], cfg["P"]
    W, H = cfg["width"], cfg["height"]
    R_cl, t_cl = cfg["R_cl"], cfg["t_cl"]
    model = cfg["model"]

    pts_l, labels = load_points_and_labels(Path(args.label))
    if args.max_points and len(pts_l) > args.max_points:
        idx = np.random.RandomState(0).choice(len(pts_l), size=args.max_points, replace=False)
        pts_l, labels = pts_l[idx], labels[idx]

    # LiDAR -> Camera
    pts_c = (R_cl @ pts_l.T).T + t_cl
    R_fix = euler_cam_frame(args.yaw_deg, args.pitch_deg, args.roll_deg)
    pts_c = (R_fix @ pts_c.T).T

    # Keep points in front of camera and within depth bounds
    z = pts_c[:, 2]
    m = (z > max(args.min_depth, 1e-6)) & (z < args.max_depth)
    pts_c, labels = pts_c[m], labels[m]

    if len(pts_c) == 0:
        sys.exit("No points after depth filtering; try different transform/yaw/pitch/roll.")


    uv, used = project_plumb_bob(pts_c, K, D), "plumb_bob (radtan)"

    # Filter out invalid projections (NaN, inf, or out of bounds)
    valid_uv = np.isfinite(uv).all(axis=1)
    uv, labels = uv[valid_uv], labels[valid_uv]

    #optional background
    bg = cv2.imread(args.image, cv2.IMREAD_COLOR) if args.image else None
    if args.image and bg is None:
        print(f"Warning: could not read background image: {args.image}")

    anomaly_rgb = parse_color_arg(args.anomaly_color)
    out = render_overlay(
        W, H, uv, labels, bg_img=bg, dot_size=args.dot_size,
        anomaly_label=args.anomaly_label, anomaly_rgb=anomaly_rgb,
        palette_s=args.palette_s, palette_v=args.palette_v,
    )

    # Save (PNG/WEBP preserve alpha)
    if out.shape[2] == 4 and not (args.out.lower().endswith(".png") or args.out.lower().endswith(".webp")):
        print("Warning: Alpha channel present. Use a PNG/WEBP filename to preserve transparency.")

    cv2.imwrite(args.out, out)
    print(
        f"Saved {args.out}\n"
        f"  model={model}, used_projection={used}\n"
        f"  anomaly_label={args.anomaly_label}, anomaly_color={args.anomaly_color}\n"
        f"  palette_s={args.palette_s}, palette_v={args.palette_v}\n"
        f"  yaw/pitch/roll=({args.yaw_deg}, {args.pitch_deg}, {args.roll_deg})"
    )


if __name__ == "__main__":
    main()
