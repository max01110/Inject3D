
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
    t_lc = T_lidar_cam[:3, 3]
    R_fix = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
    R_wcam_bl = R_lc @ R_fix                   # world->camera (blender) rotation
    R_blcam_w = R_wcam_bl.T                    # inverse rotation
    return R_blcam_w @ (p_world - t_lc)


def _blender_cam_to_world(p_cam_bl, T_lidar_cam):
    """Convert Blender camera-frame point to world/LiDAR coordinates."""
    R_lc = T_lidar_cam[:3, :3]
    t_lc = T_lidar_cam[:3, 3]
    R_fix = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
    R_wcam_bl = R_lc @ R_fix
    return R_wcam_bl @ p_cam_bl + t_lc

def load_lidar_xyz(lidar_bin_path):
    """Return (N,3) XYZ from a KITTI-style lidar.bin (Nx4 float32)."""
    pts = np.fromfile(lidar_bin_path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3].astype(np.float64)

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
                                 unoccluded_thresh=1.0):
    if seed is not None:
        rng = np.random.default_rng(seed)
        rand = lambda a, b: float(rng.uniform(a, b))
    else:
        rand = lambda a, b: float(random.uniform(a, b))

    # Fit ground and build KD-tree obstacles
    xyz = load_lidar_xyz(lidar_bin_path)
    n, d = fit_ground_plane_ransac(xyz, dist_thresh=ransac_thresh, seed=seed)
    if T_lidar_cam is not None:
        n, d = _reorient_plane_to_camera(n, d, T_lidar_cam)
    kdt, _ = build_non_ground_kdtree(xyz, n, d, ground_thresh=ransac_thresh)


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
                # Desired point in camera frame (meters)
                p_c_bl = np.array([0.0, -2.0, -8.0], dtype=np.float64)
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
                    # Directly set a fixed camera-relative position
                    p_c_bl = np.array([0.0, -2.0, -8.0], dtype=np.float64)
                    p_l = _blender_cam_to_world(p_c_bl, T_lidar_cam)
                    mw_fb = obj_parent.matrix_world.copy()
                    mw_fb.translation = Vector((float(p_l[0]), float(p_l[1]), float(p_l[2])))
                    obj_parent.matrix_world = mw_fb
            except Exception:
                pass



