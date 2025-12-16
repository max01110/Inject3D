
import math, random, numpy as np, numpy.linalg as npl
from mathutils import Vector, Quaternion
from .mesh_ops import _gather_world_vertices
from .collision import build_non_ground_kdtree, has_collision_with_scene
from .occlusion import fraction_unoccluded
from .proj_ops import invert_lidar_cam


def _world_to_blender_cam(p_world, T_lidar_cam):
    """
    Convert world/LiDAR point to Blender camera frame used by print_relative:
    - Camera looks along -Z, +Y is up, +X is right.
    The Blender camera rotation is R_world_cam_bl = R_lidar_cam @ R_fix with
    R_fix = diag(1,-1,-1).
    """
    R_lc = T_lidar_cam[:3, :3]
    t_lc = T_lidar_cam[:3, 3]  # Camera position in LiDAR/world frame
    R_fix = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
    R_wcam_bl = R_lc @ R_fix                   # world->camera (blender) rotation
    R_blcam_w = R_wcam_bl.T                    # inverse rotation
    return R_blcam_w @ (p_world - t_lc)


def _blender_cam_to_world(p_cam_bl, T_lidar_cam):
    """Convert Blender camera-frame point to world/LiDAR coordinates."""
    R_lc = T_lidar_cam[:3, :3]
    t_lc = T_lidar_cam[:3, 3]  # Camera position in LiDAR/world frame
    R_fix = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
    R_wcam_bl = R_lc @ R_fix
    return R_wcam_bl @ p_cam_bl + t_lc

def load_lidar_xyz(lidar_bin_path):
    """Return (N,3) XYZ from a KITTI-style lidar.bin (Nx4 float32)."""
    pts = np.fromfile(lidar_bin_path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3].astype(np.float64)

def load_lidar_labels(lidar_label_path):
    """Return (N,) labels from a KITTI-style .label file (N uint32)."""
    return np.fromfile(lidar_label_path, dtype=np.uint32)

# SemanticKITTI-style road/ground labels
ROAD_LABELS = {40, 44, 48, 49, 60, 72}  # road, parking, sidewalk, other-ground, lane-marking, terrain

def fit_ground_plane_ransac(xyz, dist_thresh=0.15, max_iters=300, seed=None):
    """
    Fit a plane n·p + d = 0 to LiDAR points with RANSAC.
    Returns (n, d) with ||n||=1 and n pointing 'up' (n_z >= 0).
    """
    rng = np.random.default_rng(seed)
    N = xyz.shape[0]
    if N < 3:
        raise RuntimeError("Not enough LiDAR points to fit a plane")

    best_inliers = -1
    best_n, best_d = None, None

    for _ in range(max_iters):
        idx = rng.choice(N, size=3, replace=False)
        p0, p1, p2 = xyz[idx]
        # Normal from cross product
        n = np.cross(p1 - p0, p2 - p0)
        nn = npl.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        # Plane offset d
        d = -np.dot(n, p0)

        # Count inliers
        dist = np.abs(xyz @ n + d)
        inliers = int((dist < dist_thresh).sum())
        if inliers > best_inliers:
            best_inliers = inliers
            best_n, best_d = n.copy(), float(d)

    if best_n is None:
        raise RuntimeError("RANSAC failed to find a ground plane")

    # Make the normal point 'up' in LiDAR frame (z-up)
    if best_n[2] < 0:
        best_n = -best_n
        best_d = -best_d

    return best_n, best_d

def z_on_plane_at_xy(n, d, x, y, fallback_z=-1.5):
    """
    Solve for z on plane n·[x,y,z] + d = 0  ->  z = (-d - n_x x - n_y y) / n_z
    If the plane is near vertical (n_z≈0), fall back to a reasonable z.
    """
    nz = n[2]
    if abs(nz) < 1e-6:
        return float(fallback_z)
    return float((-d - n[0]*x - n[1]*y) / nz)



def _pca_axes_from_vertices(V_world):
    """
    Compute principal axes from world-space vertices (M,3).
    Returns eigenvectors as columns in a (3,3) matrix, sorted by descending variance,
    and the corresponding variances (length 3).
    """
    if V_world.ndim != 2 or V_world.shape[1] != 3 or V_world.shape[0] < 3:
        raise RuntimeError("Not enough vertices for PCA.")
    C = V_world.mean(axis=0, keepdims=True)
    X = V_world - C
    # Covariance
    cov = (X.T @ X) / max(1, X.shape[0]-1)
    vals, vecs = npl.eigh(cov)  # ascending order
    order = np.argsort(vals)[::-1]  # descending variance
    vals = vals[order]
    vecs = vecs[:, order]  # columns = principal directions
    return vecs, vals

def _rotation_between(v_from, v_to):
    """
    Quaternion that rotates unit v_from to unit v_to.
    """
    a = Vector(v_from).normalized()
    b = Vector(v_to).normalized()
    # Handle opposite vectors robustly
    if (a + b).length < 1e-8:
        # 180-degree turn around any axis orthogonal to a
        ortho = Vector((1,0,0)) if abs(a.x) < 0.9 else Vector((0,1,0))
        axis = a.cross(ortho).normalized()
        return Quaternion(axis, math.pi)
    return a.rotation_difference(b)

def orient_largest_face_to_ground(obj_parent, ground_n_world=(0,0,1), yaw_range_deg=(0.0, 360.0), seed=None):
    """
    Heuristic: align the object's *thickness* axis (smallest-variance PCA axis)
    with the ground normal so the largest face lies on the ground. Then apply
    a random yaw about the ground normal.
    """
    # Get vertices in world space
    V = _gather_world_vertices(obj_parent)  # returns (M,3)
    if V.size == 0:
        return

    # PCA principal directions (world)
    axes, vars_desc = _pca_axes_from_vertices(V)  # axes[:,0]=largest var, axes[:,2]=smallest var
    thickness_axis = axes[:, 2]  # smallest variance direction → face normal

    # Make sure ground normal is a unit vector
    gn = np.array(ground_n_world, dtype=np.float64)
    if npl.norm(gn) < 1e-9:
        gn = np.array([0.0,0.0,1.0], dtype=np.float64)
    gn = gn / npl.norm(gn)

    # Rotate thickness axis --> ground normal
    q_align = _rotation_between(thickness_axis, gn)
    R_align = q_align.to_matrix().to_4x4()

    # Optional random yaw about ground normal (keep it physically plausible but random)
    if seed is not None:
        rng = np.random.default_rng(seed)
        yaw_deg = float(rng.uniform(yaw_range_deg[0], yaw_range_deg[1]))
    else:
        yaw_deg = random.uniform(yaw_range_deg[0], yaw_range_deg[1])

    q_yaw = Quaternion(Vector(gn.tolist()), math.radians(yaw_deg))
    R_yaw = q_yaw.to_matrix().to_4x4()

    # Apply rotations around the object origin in world space
    mw = obj_parent.matrix_world.copy()
    # first align thickness to up, then add a random yaw around up
    mw = mw @ R_align @ R_yaw
    obj_parent.matrix_world = mw

def snap_object_bottom_to_plane(obj_parent, n, d, contact_offset=0.02):
    """
    Translate the object along the plane normal so its lowest point
    lies on the plane (with a tiny positive offset).
    """
    V = _gather_world_vertices(obj_parent)  # (M,3)
    if V.size == 0:
        return
    # Signed distance of each vertex to plane
    sd = V @ n + d
    s_min = sd.min()
    # Shift so the closest vertex touches plane, then add a small offset up
    shift = (-(s_min) + contact_offset) * n
    mw = obj_parent.matrix_world.copy()
    mw.translation = mw.translation + Vector(shift.tolist())
    obj_parent.matrix_world = mw


def _build_ground_kdtree(xyz, n, d, ground_thresh=0.15, labels=None, prefer_road=True):
    """
    KD-tree of points close to the ground plane.
    If labels provided and prefer_road=True, prioritize road-labeled points.
    """
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None, None

    dist = np.abs(xyz @ n + d)
    near_plane_mask = dist <= float(ground_thresh)
    
    # If we have labels, prefer road-labeled points
    if labels is not None and prefer_road:
        road_mask = np.isin(labels, list(ROAD_LABELS))
        road_ground_mask = near_plane_mask & road_mask
        
        # Use road points if we have enough, otherwise fall back to all ground points
        if road_ground_mask.sum() >= 100:
            ground_pts = xyz[road_ground_mask]
        else:
            ground_pts = xyz[near_plane_mask]
    else:
        ground_pts = xyz[near_plane_mask]
    
    if ground_pts.size == 0:
        return None, ground_pts
    return cKDTree(ground_pts), ground_pts


def _adjust_height_to_ground_points(obj_parent,
                                    ground_kdtree,
                                    ground_pts,
                                    contact_offset=0.02,
                                    height_margin=0.10,
                                    k_neighbors=20,
                                    max_radius=1.5):
    """
    Small corrective translation so the object's lowest point touches nearby
    ground points instead of relying solely on a plane fit.
    """
    if ground_kdtree is None or ground_pts is None or ground_pts.size == 0:
        return

    V = _gather_world_vertices(obj_parent)
    if V.size == 0:
        return

    # Use vertices near the bottom face to estimate contact XY
    z_min = V[:, 2].min()
    bottom_mask = (V[:, 2] <= z_min + 0.03)  # within 3 cm of the lowest point
    bottom_pts = V[bottom_mask]
    if bottom_pts.size == 0:
        return
    xy_center = bottom_pts[:, :2].mean(axis=0)

    # Query nearest ground points around that XY
    query_pt = np.array([xy_center[0], xy_center[1], z_min], dtype=np.float64)
    dists, idxs = ground_kdtree.query(query_pt, k=k_neighbors, distance_upper_bound=float(max_radius), workers=-1)

    # Handle scipy returning scalars when k=1
    dists = np.atleast_1d(dists)
    idxs = np.atleast_1d(idxs)

    valid = idxs < ground_pts.shape[0]
    if not valid.any():
        return

    ground_z = np.median(ground_pts[idxs[valid], 2])
    current_bottom_z = z_min
    desired_bottom_z = ground_z + float(contact_offset)
    delta = desired_bottom_z - current_bottom_z

    if abs(delta) < float(height_margin):
        return  # Already close enough

    mw = obj_parent.matrix_world.copy()
    mw.translation.z += float(delta)
    obj_parent.matrix_world = mw
    
    # Safety check: ensure object is not below ground after adjustment
    V_new = _gather_world_vertices(obj_parent)
    if V_new.size > 0:
        new_bottom_z = V_new[:, 2].min()
        if new_bottom_z < ground_z:
            # Push object up so it sits on ground
            correction = ground_z + float(contact_offset) - new_bottom_z
            mw = obj_parent.matrix_world.copy()
            mw.translation.z += float(correction)
            obj_parent.matrix_world = mw


def _reorient_plane_to_camera(n, d, T_lidar_cam):
    """
    Ensure ground plane is below the camera:
    - In Blender camera frame, ground normal should have positive Y (points 'up').
    - Camera should lie above the plane along the ground normal (signed distance > 0).
    Returns possibly flipped (n, d).
    """
    try:
        n = np.asarray(n, dtype=np.float64)
        R_lc = T_lidar_cam[:3, :3]
        t_lc = T_lidar_cam[:3, 3]
        R_fix = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
        R_wcam_bl = R_lc @ R_fix
        R_blcam_w = R_wcam_bl.T
        n_cam_bl = R_blcam_w @ n
        s_cam = float(n @ t_lc + d)
        if (n_cam_bl[1] <= 0.0) or (s_cam <= 0.0):
            n = -n
            d = -float(d)
        return n, float(d)
    except Exception:
        return n, d

def place_random_on_lidar_ground(obj_parent,
                                 lidar_bin_path,
                                 x_range=(4.0, 12.0),
                                 y_range=(-2.0, 2.0),
                                 seed=None,
                                 ransac_thresh=0.15,
                                 contact_offset=0.02,
                                 yaw_range_deg=(0.0, 360.0),
                                 clearance=0.20,
                                 tries=120,
                                 n_surface_samples=2500,
                                 avoid_occlusion=False,
                                 zbuf=None,
                                 T_lidar_cam=None,
                                 model=None, K=None, D=None, width=None, height=None,
                                 z_margin=0.05,
                                 require_inside_frac=0.85,
                                 unoccluded_thresh=1.0,
                                 height_margin=0.10,
                                 lidar_label_path=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
        rand = lambda a, b: float(rng.uniform(a, b))
    else:
        rand = lambda a, b: float(random.uniform(a, b))

    # Fit ground and build KD-tree obstacles
    xyz = load_lidar_xyz(lidar_bin_path)
    labels = None
    if lidar_label_path is not None:
        try:
            labels = load_lidar_labels(lidar_label_path)
        except Exception:
            labels = None
    
    n, d = fit_ground_plane_ransac(xyz, dist_thresh=ransac_thresh, seed=seed)
    if T_lidar_cam is not None:
        n, d = _reorient_plane_to_camera(n, d, T_lidar_cam)
    kdt, _ = build_non_ground_kdtree(xyz, n, d, ground_thresh=ransac_thresh)
    # Use road-labeled points for ground height adjustment when labels available
    ground_kdt, ground_pts = _build_ground_kdtree(xyz, n, d, ground_thresh=ransac_thresh, 
                                                   labels=labels, prefer_road=True)


    best_pose = None
    for _ in range(int(tries)):
        # Propose XY on ground
        x = rand(*x_range); y = rand(*y_range)
        z = z_on_plane_at_xy(n, d, x, y, fallback_z=np.percentile(xyz[:,2], 5))

        # Place & orient
        mw = obj_parent.matrix_world.copy()
        mw.translation = Vector((x, y, z))
        obj_parent.matrix_world = mw

        orient_largest_face_to_ground(obj_parent, ground_n_world=n.tolist(),
                                      yaw_range_deg=yaw_range_deg, seed=seed)
        snap_object_bottom_to_plane(obj_parent, n, d, contact_offset=contact_offset)
        _adjust_height_to_ground_points(
            obj_parent,
            ground_kdt,
            ground_pts,
            contact_offset=contact_offset,
            height_margin=height_margin
        )

        # Enforce: object must be below the camera in Blender camera Y (y_cam < 0)
        if T_lidar_cam is not None:
            try:
                p_world = np.array(list(obj_parent.matrix_world.translation), dtype=np.float64)
                p_cam_bl = _world_to_blender_cam(p_world, T_lidar_cam)
                if not (p_cam_bl[1] < 0.0):
                    # Reject this sample if it's not below the camera
                    continue
            except Exception:
                pass

        # Collision check vs scene
        if has_collision_with_scene(obj_parent, kdt, clearance=clearance, n_samples=n_surface_samples):
            continue

        #Occlusion rejection
        if avoid_occlusion:
            # Use fewer samples for a faster visibility test
            fast_samples = int(min(max(800, 0.25 * n_surface_samples), 3000))
            inside_frac, unocc_frac = fraction_unoccluded(
                obj_parent, zbuf, T_lidar_cam, model, K, D, width, height,
                n_surface_samples=fast_samples,
                z_margin=z_margin,
                require_inside_frac=require_inside_frac
            )
            # Optional refine if borderline (disabled by default for speed)
            # if inside_frac >= require_inside_frac and (unocc_frac < unoccluded_thresh + 0.05):
            #     inside_frac, unocc_frac = fraction_unoccluded(
            #         obj_parent, zbuf, T_lidar_cam, model, K, D, width, height,
            #         n_surface_samples=min(n_surface_samples, 6000),
            #         z_margin=z_margin,
            #         require_inside_frac=require_inside_frac
            #     )
            # Require enough of the object to be actually visible,
            # and all visible samples to be unoccluded (or above threshold)
            if inside_frac < require_inside_frac:
                continue
            if unocc_frac < unoccluded_thresh:
                continue

        best_pose = obj_parent.matrix_world.copy()
        break

    if best_pose is None:
        print(f"[WARN] Could not find a valid placement in {tries} tries. Applying manual fallback placement in front of and below the camera.")

        # Manual fallback: place at a fixed Blender-camera-relative location (no snap/orient)
        # Blender camera convention: looking along -Z, X right, Y up. We want Y<0 and Z<0.
        if T_lidar_cam is not None:
            try:
                # Desired point in camera frame (meters) - random distance within x_range
                random_z = random.uniform(x_range[0], x_range[1])
                p_c_bl = np.array([0.0, -2.0, -random_z], dtype=np.float64)
                # Transform to LiDAR/world frame using Blender camera convention
                p_l = _blender_cam_to_world(p_c_bl, T_lidar_cam)
                # Directly set world translation
                mw_fb = obj_parent.matrix_world.copy()
                mw_fb.translation = Vector((float(p_l[0]), float(p_l[1]), float(p_l[2])))
                obj_parent.matrix_world = mw_fb

                best_pose = obj_parent.matrix_world.copy()
            except Exception as e:
                print(f"[WARN] Manual fallback placement failed: {e}. Keeping last pose.")
                best_pose = None
        else:
            print("[WARN] No T_lidar_cam provided; cannot compute camera-relative fallback.")
            best_pose = None
    else:
        obj_parent.matrix_world = best_pose

        # Final validation: ensure below camera (y_cam < 0). If not, override with manual fallback.
        if T_lidar_cam is not None:
            try:
                p_world = np.array(list(obj_parent.matrix_world.translation), dtype=np.float64)
                p_cam_bl = _world_to_blender_cam(p_world, T_lidar_cam)
                if not (p_cam_bl[1] < 0.0):
                    # Directly set a random camera-relative position within x_range
                    random_z = random.uniform(x_range[0], x_range[1])
                    p_c_bl = np.array([0.0, -2.0, -random_z], dtype=np.float64)
                    p_l = _blender_cam_to_world(p_c_bl, T_lidar_cam)
                    mw_fb = obj_parent.matrix_world.copy()
                    mw_fb.translation = Vector((float(p_l[0]), float(p_l[1]), float(p_l[2])))
                    obj_parent.matrix_world = mw_fb
            except Exception:
                pass



