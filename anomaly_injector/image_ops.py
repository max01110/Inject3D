"""
Image distortion and compositing operations.
"""
import numpy as np
import cv2
from PIL import Image


def build_rectified_to_distorted_map(model, K, D, width, height, K_rect=None):
    """Build pixel remap from rectified to distorted coordinates."""
    K = np.asarray(K, np.float64).reshape(3, 3)
    K_rect = np.asarray(K_rect if K_rect is not None else K, np.float64).reshape(3, 3)
    D = np.asarray(D, np.float64).reshape(-1, 1)

    W, H = int(width), int(height)
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    pts_d = np.stack([uu, vv], axis=-1).reshape(-1, 1, 2)

    rect_pix = cv2.undistortPoints(pts_d, K, D, R=np.eye(3), P=K_rect)
    rect_pix = rect_pix.reshape(H, W, 2).astype(np.float32)
    return rect_pix[..., 0], rect_pix[..., 1]


def rgba_rectified_to_distorted(rgba_rect, mapx, mapy):
    """Remap RGBA image from rectified to distorted domain."""
    b, g, r, a = cv2.split(rgba_rect)
    channels = [cv2.remap(ch, mapx, mapy, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
                for ch in [b, g, r, a]]
    return cv2.merge(channels)


def alpha_composite_rgba_over_bgr(rgba, bg_bgr):
    """Composite RGBA foreground over BGR background."""
    if bg_bgr.shape[:2] != rgba.shape[:2]:
        bg_bgr = cv2.resize(bg_bgr, (rgba.shape[1], rgba.shape[0]),
                            interpolation=cv2.INTER_AREA)

    b, g, r, a = cv2.split(rgba)
    alpha = (a.astype(np.float32) / 255.0)[..., None]
    fg = cv2.merge([b, g, r]).astype(np.float32)
    bg = bg_bgr.astype(np.float32)
    out = (fg * alpha + bg * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
    return out


def composite_over_background(render_path, bg_path, out_path):
    """Composite rendered RGBA over background and save."""
    fg = Image.open(render_path).convert("RGBA")
    bg = Image.open(bg_path).convert("RGBA")
    if bg.size != fg.size:
        bg = bg.resize(fg.size, Image.BICUBIC)
    out = Image.alpha_composite(bg, fg)
    out.convert("RGB").save(out_path)
