import numpy as np
from PIL import Image
import cv2



def build_rectified_to_distorted_map(model, K, D, width, height, K_rect=None):
    K = np.asarray(K, np.float64).reshape(3,3)
    if K_rect is None: K_rect = K
    K_rect = np.asarray(K_rect, np.float64).reshape(3,3)
    D = np.asarray(D, np.float64).reshape(-1,1)

    W, H = int(width), int(height)
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    pts_d = np.stack([uu, vv], axis=-1).reshape(-1,1,2)

    rect_pix = cv2.undistortPoints(pts_d, K, D, R=np.eye(3), P=K_rect)

    rect_pix = rect_pix.reshape(H, W, 2).astype(np.float32)
    return rect_pix[...,0], rect_pix[...,1]

def rgba_rectified_to_distorted(rgba_rectified, mapx, mapy):
    b,g,r,a = cv2.split(rgba_rectified)
    rD = cv2.remap(r, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    gD = cv2.remap(g, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    bD = cv2.remap(b, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    aD = cv2.remap(a, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return cv2.merge([bD,gD,rD,aD])


def alpha_composite_rgba_over_bgr(rgba, bg_bgr):
    if (bg_bgr.shape[1], bg_bgr.shape[0]) != (rgba.shape[1], rgba.shape[0]):
        bg_bgr = cv2.resize(bg_bgr, (rgba.shape[1], rgba.shape[0]), interpolation=cv2.INTER_AREA)
    b,g,r,a = cv2.split(rgba)
    aF = (a.astype(np.float32)/255.0)[...,None]
    fgBGR = cv2.merge([b,g,r]).astype(np.float32)
    out = (fgBGR*aF + bg_bgr.astype(np.float32)*(1.0 - aF)).clip(0,255).astype(np.uint8)
    return out


def composite_over_background(render_path, bg_image_path, out_path):
    fg = Image.open(render_path).convert("RGBA")
    bg = Image.open(bg_image_path).convert("RGBA")
    if bg.size != fg.size:
        bg = bg.resize(fg.size, Image.BICUBIC)
    out = Image.alpha_composite(bg, fg)
    # out.convert("RGB").save(out_path)


