
import math, random, numpy as np, numpy.linalg as npl
from mathutils import Vector, Quaternion
from .mesh_ops import _gather_world_vertices
from .collision import build_non_ground_kdtree, has_collision_with_scene
from .occlusion import fraction_unoccluded

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


def place_random_on_lidar_ground(obj_parent,
                                 lidar_bin_path,
                                 x_range=(4.0, 12.0),
                                 y_range=(-2.0, 2.0),
                                 seed=None,
                                 ransac_thresh=0.15,
                                 contact_offset=0.02,
                                 yaw_range_deg=(0.0, 360.0),
                                 clearance=0.20,
                                 tries=200,
                                 n_surface_samples=4000,
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

        # Collision check vs scene
        if has_collision_with_scene(obj_parent, kdt, clearance=clearance, n_samples=n_surface_samples):
            continue

        #Occlusion rejection
        if avoid_occlusion:
            inside_frac, unocc_frac = fraction_unoccluded(
                obj_parent, zbuf, T_lidar_cam, model, K, D, width, height,
                n_surface_samples=min(n_surface_samples, 6000),  # cap for speed
                z_margin=z_margin,
                require_inside_frac=require_inside_frac
            )
            # Require enough of the object to be actually visible,
            # and all visible samples to be unoccluded (or above threshold)
            if inside_frac < require_inside_frac:
                continue
            if unocc_frac < unoccluded_thresh:
                continue

        best_pose = obj_parent.matrix_world.copy()
        break

    if best_pose is None:
        print(f"[WARN] Could not find a collision- and occlusion-free placement in {tries} tries; keeping last pose.")
    else:
        obj_parent.matrix_world = best_pose



