
import numpy as np



def write_augmented_pointcloud(lidar_bin, lidar_label, pts_new, anomaly_label, out_bin, out_label):
    pts = np.fromfile(lidar_bin, dtype=np.float32).reshape(-1,4)
    labels = np.fromfile(lidar_label, dtype=np.uint32)
    if labels.shape[0] != pts.shape[0]:
        raise RuntimeError("lidar.bin and lidar.label count mismatch")
    add = np.zeros((pts_new.shape[0], 4), dtype=np.float32)
    add[:,:3] = pts_new; add[:,3]=0.0
    pts_aug = np.vstack([pts, add])
    labels_aug = np.concatenate([labels, np.full((pts_new.shape[0],), anomaly_label, dtype=np.uint32)])
    pts_aug.astype(np.float32).tofile(out_bin)
    labels_aug.astype(np.uint32).tofile(out_label)
