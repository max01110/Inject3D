"""
Object placement on LiDAR ground plane.
"""
import math
import random
import numpy as np
import numpy.linalg as npl
from mathutils import Vector, Quaternion

from .mesh_ops import _gather_world_vertices
from .collision import build_non_ground_kdtree, has_collision_with_scene
from .occlusion import fraction_unoccluded
from .proj_ops import invert_lidar_cam

# SemanticKITTI road label
ROAD_LABELS = {40}


# -----------------------------------------------------------------------------
# LiDAR I/O
# -----------------------------------------------------------------------------

def load_lidar_xyz(lidar_bin_path):
    """Load (N,3) XYZ from KITTI-style .bin file."""
    pts = np.fromfile(lidar_bin_path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3].astype(np.float64)


def load_lidar_labels(lidar_label_path):
    """Load (N,) labels from .label file."""
    return np.fromfile(lidar_label_path, dtype=np.uint32)


# -----------------------------------------------------------------------------
# Ground plane fitting
# -----------------------------------------------------------------------------

def fit_ground_plane_ransac(xyz, dist_thresh=0.15, max_iters=300, seed=None):
    """
    RANSAC plane fitting. Returns (normal, d) where n·p + d = 0.
    Normal points up (n_z >= 0).
    """
    rng = np.random.default_rng(seed)
    N = xyz.shape[0]
    if N < 3:
        raise RuntimeError("Not enough points for plane fitting")

    best_inliers, best_n, best_d = -1, None, None

    for _ in range(max_iters):
        idx = rng.choice(N, size=3, replace=False)
        p0, p1, p2 = xyz[idx]
        n = np.cross(p1 - p0, p2 - p0)
        nn = npl.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = -np.dot(n, p0)

        inliers = int((np.abs(xyz @ n + d) < dist_thresh).sum())
        if inliers > best_inliers:
            best_inliers, best_n, best_d = inliers, n.copy(), float(d)

    if best_n is None:
        raise RuntimeError("RANSAC failed")

    # Ensure normal points up
    if best_n[2] < 0:
        best_n, best_d = -best_n, -best_d

    return best_n, best_d


def z_on_plane_at_xy(n, d, x, y, fallback_z=-1.5):
    """Solve for z on plane at (x, y)."""
    if abs(n[2]) < 1e-6:
        return float(fallback_z)
    return float((-d - n[0]*x - n[1]*y) / n[2])


def _reorient_plane_to_camera(n, d, T_lidar_cam):
    """Ensure ground plane is below the camera."""
    try:
        n = np.asarray(n, dtype=np.float64)
        R_lc = T_lidar_cam[:3, :3]
        t_lc = T_lidar_cam[:3, 3]
        R_fix = np.diag([1, -1, -1]).astype(np.float64)
        R_blcam_w = (R_lc @ R_fix).T
        n_cam_bl = R_blcam_w @ n
        s_cam = float(n @ t_lc + d)
        if n_cam_bl[1] <= 0 or s_cam <= 0:
            n, d = -n, -float(d)
        return n, float(d)
    except Exception:
        return n, d


# -----------------------------------------------------------------------------
# Ground KD-tree helpers
# -----------------------------------------------------------------------------

def _build_ground_kdtree(xyz, n, d, ground_thresh=0.15, labels=None, prefer_road=True):
    """Build KD-tree of points near ground plane."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None, None

    near_mask = np.abs(xyz @ n + d) <= ground_thresh

    if labels is not None and prefer_road:
        road_mask = np.isin(labels, list(ROAD_LABELS))
        road_ground = near_mask & road_mask
        ground_pts = xyz[road_ground] if road_ground.sum() >= 100 else xyz[near_mask]
    else:
        ground_pts = xyz[near_mask]

    if ground_pts.size == 0:
        return None, ground_pts
    return cKDTree(ground_pts), ground_pts


def _build_road_only_kdtree(xyz, labels):
    """Build KD-tree of road-labeled points only."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None, None

    if labels is None:
        return None, None

    road_pts = xyz[np.isin(labels, list(ROAD_LABELS))]
    if road_pts.size == 0:
        return None, None
    return cKDTree(road_pts), road_pts


# -----------------------------------------------------------------------------
# Object orientation
# -----------------------------------------------------------------------------

def _pca_axes(V):
    """PCA on vertices, returns (eigenvectors, eigenvalues) sorted by descending variance."""
    if V.ndim != 2 or V.shape[1] != 3 or V.shape[0] < 3:
        raise RuntimeError("Not enough vertices for PCA")
    X = V - V.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / max(1, X.shape[0] - 1)
    vals, vecs = npl.eigh(cov)
    order = np.argsort(vals)[::-1]
    return vecs[:, order], vals[order]


def _rotation_between(v_from, v_to):
    """Quaternion rotating v_from to v_to."""
    a = Vector(v_from).normalized()
    b = Vector(v_to).normalized()
    if (a + b).length < 1e-8:
        ortho = Vector((1, 0, 0)) if abs(a.x) < 0.9 else Vector((0, 1, 0))
        axis = a.cross(ortho).normalized()
        return Quaternion(axis, math.pi)
    return a.rotation_difference(b)


def orient_largest_face_to_ground(obj_parent, ground_n=(0, 0, 1), yaw_range_deg=(0, 360), seed=None):
    """
    Align object's thickness axis (smallest PCA variance) with ground normal,
    then apply random yaw.
    """
    V = _gather_world_vertices(obj_parent)
    if V.size == 0:
        return

    axes, _ = _pca_axes(V)
    thickness = axes[:, 2]  # smallest variance = face normal

    gn = np.array(ground_n, dtype=np.float64)
    gn = gn / max(npl.norm(gn), 1e-9)

    q_align = _rotation_between(thickness, gn)
    R_align = q_align.to_matrix().to_4x4()

    rng = np.random.default_rng(seed) if seed else None
    yaw = rng.uniform(*yaw_range_deg) if rng else random.uniform(*yaw_range_deg)
    q_yaw = Quaternion(Vector(gn.tolist()), math.radians(yaw))
    R_yaw = q_yaw.to_matrix().to_4x4()

    obj_parent.matrix_world = obj_parent.matrix_world @ R_align @ R_yaw


def snap_object_bottom_to_plane(obj_parent, n, d, contact_offset=0.02):
    """Translate object so its lowest point sits on the plane."""
    V = _gather_world_vertices(obj_parent)
    if V.size == 0:
        return
    sd = V @ n + d
    shift = (-(sd.min()) + contact_offset) * n
    mw = obj_parent.matrix_world.copy()
    mw.translation = mw.translation + Vector(shift.tolist())
    obj_parent.matrix_world = mw


def _adjust_height_to_ground_points(obj_parent, ground_kdt, ground_pts,
                                    contact_offset=0.02, height_margin=0.10,
                                    k_neighbors=20, max_radius=1.5):
    """Fine-tune height using nearby ground points."""
    if ground_kdt is None or ground_pts is None or ground_pts.size == 0:
        return

    V = _gather_world_vertices(obj_parent)
    if V.size == 0:
        return

    z_min = V[:, 2].min()
    bottom_pts = V[V[:, 2] <= z_min + 0.03]
    if bottom_pts.size == 0:
        return

    xy_center = bottom_pts[:, :2].mean(axis=0)
    query = np.array([xy_center[0], xy_center[1], z_min], dtype=np.float64)
    dists, idxs = ground_kdt.query(query, k=k_neighbors,
                                   distance_upper_bound=float(max_radius), workers=-1)
    dists, idxs = np.atleast_1d(dists), np.atleast_1d(idxs)
    valid = idxs < ground_pts.shape[0]
    if not valid.any():
        return

    ground_z = np.median(ground_pts[idxs[valid], 2])
    delta = (ground_z + contact_offset) - z_min
    if abs(delta) < height_margin:
        return

    mw = obj_parent.matrix_world.copy()
    mw.translation.z += float(delta)
    obj_parent.matrix_world = mw

    # Safety check
    V_new = _gather_world_vertices(obj_parent)
    if V_new.size > 0 and V_new[:, 2].min() < ground_z:
        correction = ground_z + contact_offset - V_new[:, 2].min()
        mw = obj_parent.matrix_world.copy()
        mw.translation.z += correction
        obj_parent.matrix_world = mw


# -----------------------------------------------------------------------------
# Placement functions
# -----------------------------------------------------------------------------

def place_random_on_lidar_ground(obj_parent, lidar_bin_path,
                                 x_range=(4.0, 12.0), y_range=(-2.0, 2.0),
                                 seed=None, ransac_thresh=0.15, contact_offset=0.02,
                                 yaw_range_deg=(0.0, 360.0), clearance=0.20, tries=120,
                                 n_surface_samples=2500, avoid_occlusion=False,
                                 zbuf=None, T_lidar_cam=None, model=None, K=None, D=None,
                                 width=None, height=None, z_margin=0.05,
                                 require_inside_frac=0.85, unoccluded_thresh=1.0,
                                 height_margin=0.10, lidar_label_path=None,
                                 allow_expand_bounds=True, max_y_expand=2.0, max_x_expand=4.0,
                                 require_on_road=True, road_max_distance=1.0):
    """
    Place object randomly on ground within bounds.
    Returns True if placement succeeded.
    """
    import bpy
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed) if seed else None
    rand = lambda a, b: float(rng.uniform(a, b) if rng else random.uniform(a, b))

    # Load LiDAR
    xyz = load_lidar_xyz(lidar_bin_path)
    labels = None
    if lidar_label_path:
        try:
            labels = load_lidar_labels(lidar_label_path)
        except Exception:
            pass

    # Build ground point set
    if labels is not None:
        road_mask = np.isin(labels, list(ROAD_LABELS))
        if road_mask.sum() >= 100:
            ground_pts = xyz[road_mask]
        else:
            z_thresh = np.percentile(xyz[:, 2], 20)
            ground_pts = xyz[xyz[:, 2] <= z_thresh]
    else:
        z_thresh = np.percentile(xyz[:, 2], 20)
        ground_pts = xyz[xyz[:, 2] <= z_thresh]

    if ground_pts.shape[0] < 10:
        ground_pts = xyz
    ground_kdt = cKDTree(ground_pts)

    # Obstacle points (above 30th percentile Z)
    z_obstacle = np.percentile(xyz[:, 2], 30)
    obstacle_pts = xyz[xyz[:, 2] > z_obstacle]
    obstacle_kdt = cKDTree(obstacle_pts) if obstacle_pts.shape[0] > 0 else None

    stats = {'collision': 0, 'outside_fov': 0, 'occluded': 0, 'no_ground': 0}
    print(f"[PLACEMENT] Searching (tries={tries}, x={x_range}, y={y_range})")

    for attempt in range(tries):
        x, y = rand(*x_range), rand(*y_range)

        # Find ground height at (x, y)
        query = np.array([x, y, 0.0], dtype=np.float64)
        dists, indices = ground_kdt.query(query, k=min(10, ground_pts.shape[0]))
        dists, indices = np.atleast_1d(dists), np.atleast_1d(indices)

        valid = dists < 5.0
        if not valid.any():
            stats['no_ground'] += 1
            continue

        ground_z = float(np.median(ground_pts[indices[valid], 2]))

        # Place object temporarily to measure bounds
        yaw = rand(math.radians(yaw_range_deg[0]), math.radians(yaw_range_deg[1]))
        obj_parent.location = (x, y, 0.0)
        obj_parent.rotation_euler = (0.0, 0.0, yaw)
        bpy.context.view_layer.update()

        V = _gather_world_vertices(obj_parent)
        if V.size == 0:
            continue

        obj_bottom = float(V[:, 2].min())
        final_z = ground_z + contact_offset - obj_bottom
        obj_parent.location = (x, y, final_z)
        bpy.context.view_layer.update()

        # Ensure object is above ground
        V_final = _gather_world_vertices(obj_parent)
        if V_final.size > 0:
            final_bottom = float(V_final[:, 2].min())
            if final_bottom < ground_z:
                obj_parent.location = (x, y, final_z + ground_z + contact_offset - final_bottom)
                bpy.context.view_layer.update()

        # Collision check
        if obstacle_kdt is not None and clearance > 0:
            center = np.array([x, y, ground_z + 0.5], dtype=np.float64)
            nearby = obstacle_kdt.query_ball_point(center, r=clearance)
            if len(nearby) > 200:
                stats['collision'] += 1
                continue

        # Visibility check
        if avoid_occlusion:
            inside_frac, unocc_frac = fraction_unoccluded(
                obj_parent, zbuf, T_lidar_cam, model, K, D, width, height,
                n_surface_samples=min(1500, n_surface_samples),
                z_margin=z_margin, require_inside_frac=require_inside_frac
            )
            if inside_frac < require_inside_frac:
                stats['outside_fov'] += 1
                continue
            if unocc_frac < unoccluded_thresh:
                stats['occluded'] += 1
                continue

        # Success
        print(f"[PLACEMENT] Found at x={x:.2f}, y={y:.2f}, z={ground_z:.2f} (attempt {attempt+1})")
        print(f"[PLACEMENT] Stats: {stats}")
        return True

    print(f"[PLACEMENT] Failed after {tries} tries. Stats: {stats}")
    return False


def place_manual(obj_parent, x, y, z=None, yaw_deg=0.0,
                 lidar_bin_path=None, adjust_to_ground=False,
                 ransac_thresh=0.15, contact_offset=0.02,
                 height_margin=0.10, lidar_label_path=None):
    """
    Place object at exact coordinates, skipping placement checks.
    If z is None and adjust_to_ground=True, auto-adjust to ground height.
    """
    import bpy
    from mathutils import Euler

    obj_parent.rotation_euler = Euler((0.0, 0.0, math.radians(yaw_deg)), 'XYZ')

    if z is None and adjust_to_ground:
        if lidar_bin_path is None:
            print("[ERROR] lidar_bin_path required for ground adjustment")
            return False

        xyz = load_lidar_xyz(lidar_bin_path)
        n, d = fit_ground_plane_ransac(xyz, dist_thresh=ransac_thresh)

        labels = None
        if lidar_label_path:
            try:
                labels = load_lidar_labels(lidar_label_path)
            except Exception:
                pass

        ground_kdt, ground_pts = _build_ground_kdtree(xyz, n, d,
                                                      ground_thresh=ransac_thresh,
                                                      labels=labels, prefer_road=True)

        # Find ground Z at (x, y)
        if ground_kdt is not None and ground_pts.shape[0] > 0:
            query = np.array([[x, y]], dtype=np.float64)
            dists, indices = ground_kdt.query(query, k=min(5, len(ground_pts)), workers=-1)
            if isinstance(dists, np.ndarray) and dists.size > 0:
                idx = indices.flat[0] if indices.size > 0 else 0
                ground_z = float(ground_pts[idx, 2])
            else:
                ground_z = z_on_plane_at_xy(n, d, x, y)
        else:
            ground_z = z_on_plane_at_xy(n, d, x, y)

        # Get object bottom
        obj_parent.location = (x, y, 0.0)
        bpy.context.view_layer.update()
        V = _gather_world_vertices(obj_parent)
        if V.size > 0:
            obj_bottom = float(V[:, 2].min())
            z = ground_z + contact_offset - obj_bottom
        else:
            z = ground_z + contact_offset

    elif z is None:
        z = 0.0

    obj_parent.location = (x, y, z)
    bpy.context.view_layer.update()

    print(f"[MANUAL] Placed at ({x:.3f}, {y:.3f}, {z:.3f}), yaw={yaw_deg:.1f}°")
    return True
