
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

# SemanticKITTI-style road label (strict - only actual road surface)
ROAD_LABELS = {40}  # road only

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


def _build_road_only_kdtree(xyz, labels):
    """
    Build a KD-tree containing ONLY road-labeled points (no fallback).
    Returns (kdtree, road_pts) or (None, None) if no road points.
    """
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None, None
    
    if labels is None:
        return None, None
    
    road_mask = np.isin(labels, list(ROAD_LABELS))
    road_pts = xyz[road_mask]
    
    if road_pts.size == 0:
        return None, None
    
    return cKDTree(road_pts), road_pts


def _is_on_road(x, y, road_kdtree, road_pts, max_distance=1.5):
    """
    Check if position (x, y) is within max_distance of a road point.
    Returns True if on or near road, False otherwise.
    """
    if road_kdtree is None or road_pts is None:
        return True  # If no road data, allow placement (fallback behavior)
    
    # Query using XY only (ignore Z for this check)
    # Find nearest road point in 2D
    query_pt = np.array([x, y, 0.0], dtype=np.float64)
    
    # Get nearby road points
    dist, idx = road_kdtree.query(query_pt, k=1)
    
    if idx >= road_pts.shape[0]:
        return False
    
    # Check 2D distance (XY only)
    road_pt = road_pts[idx]
    dist_2d = np.sqrt((x - road_pt[0])**2 + (y - road_pt[1])**2)
    
    return dist_2d <= max_distance


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
                                 lidar_label_path=None,
                                 allow_expand_bounds=True,
                                 max_y_expand=2.0,
                                 max_x_expand=4.0,
                                 require_on_road=True,
                                 road_max_distance=1.0):
    """
    Place object randomly on LiDAR ground within bounds.
    
    Simple placement logic (like manual placement but with randomness):
    1. Pick random x, y within bounds
    2. Find ground height from nearest LiDAR points (same as MANUAL_ADJUST_TO_GROUND)
    3. Place object so its bottom sits on top of ground
    4. Check for collision with scene points
    5. Check object is visible in camera (not occluded)
    
    Returns True if placement succeeded, False if no valid placement found.
    """
    import bpy
    from scipy.spatial import cKDTree
    
    if seed is not None:
        rng = np.random.default_rng(seed)
        rand = lambda a, b: float(rng.uniform(a, b))
    else:
        rand = lambda a, b: float(random.uniform(a, b))

    # Load LiDAR points
    xyz = load_lidar_xyz(lidar_bin_path)
    labels = None
    if lidar_label_path is not None:
        try:
            labels = load_lidar_labels(lidar_label_path)
        except Exception:
            labels = None
    
    # Build KD-tree of ALL points for ground height queries (same as manual placement)
    # Prefer road-labeled points if available, otherwise use lowest points
    if labels is not None:
        road_mask = np.isin(labels, list(ROAD_LABELS))
        if road_mask.sum() >= 100:
            ground_pts = xyz[road_mask]
        else:
            # Use lowest 20% of points as ground
            z_thresh = np.percentile(xyz[:, 2], 20)
            ground_pts = xyz[xyz[:, 2] <= z_thresh]
    else:
        # Use lowest 20% of points as ground
        z_thresh = np.percentile(xyz[:, 2], 20)
        ground_pts = xyz[xyz[:, 2] <= z_thresh]
    
    if ground_pts.shape[0] < 10:
        ground_pts = xyz  # Fallback to all points
    
    ground_kdt = cKDTree(ground_pts)
    
    # Build KD-tree for collision detection (non-ground points)
    # Simple approach: points above the 30th percentile Z are obstacles
    z_obstacle_thresh = np.percentile(xyz[:, 2], 30)
    obstacle_pts = xyz[xyz[:, 2] > z_obstacle_thresh]
    obstacle_kdt = cKDTree(obstacle_pts) if obstacle_pts.shape[0] > 0 else None

    best_pose = None
    rejection_stats = {'collision': 0, 'outside_fov': 0, 'occluded': 0, 'no_ground': 0}
    
    print(f"[PLACEMENT] Starting random placement search (tries={tries}, x_range={x_range}, y_range={y_range})")
    print(f"[PLACEMENT] Ground points: {ground_pts.shape[0]}, Obstacle points: {obstacle_pts.shape[0] if obstacle_pts is not None else 0}")
    
    for attempt in range(tries):
        # Pick random x, y within bounds
        x = rand(*x_range)
        y = rand(*y_range)
        
        # Find ground height at (x, y) by querying nearest ground points (same as manual placement)
        query_pt = np.array([x, y, 0.0], dtype=np.float64)
        dists, indices = ground_kdt.query(query_pt, k=min(10, ground_pts.shape[0]))
        
        # Handle scipy returning scalars
        dists = np.atleast_1d(dists)
        indices = np.atleast_1d(indices)
        
        # Filter to nearby points (within 5m horizontal distance - more lenient)
        valid_mask = dists < 5.0
        if not valid_mask.any():
            rejection_stats['no_ground'] += 1
            continue
        
        # Get ground Z from nearest points (use median for robustness)
        nearby_z = ground_pts[indices[valid_mask], 2]
        ground_z = float(np.median(nearby_z))
        
        # Place object temporarily at (x, y, 0) to measure its size
        obj_parent.location = (x, y, 0.0)
        obj_parent.rotation_euler = (0.0, 0.0, rand(math.radians(yaw_range_deg[0]), math.radians(yaw_range_deg[1])))
        bpy.context.view_layer.update()
        
        # Get object's bottom Z
        V = _gather_world_vertices(obj_parent)
        if V.size == 0:
            continue
        obj_bottom_z = float(V[:, 2].min())
        
        # Calculate final Z so object bottom sits ON TOP of ground (with small offset)
        final_z = ground_z + contact_offset - obj_bottom_z
        
        # Set final position
        obj_parent.location = (x, y, final_z)
        bpy.context.view_layer.update()
        
        # Verify object is above ground (safety check)
        V_final = _gather_world_vertices(obj_parent)
        if V_final.size > 0:
            final_bottom = float(V_final[:, 2].min())
            if final_bottom < ground_z:
                # Push up to ensure on top
                correction = ground_z + contact_offset - final_bottom
                obj_parent.location = (x, y, final_z + correction)
                bpy.context.view_layer.update()

        # Check 1: No collision with obstacle points (relaxed check)
        # Only reject if there are MANY points very close to the object center
        if obstacle_kdt is not None and clearance > 0:
            obj_center = np.array([x, y, ground_z + 0.5], dtype=np.float64)
            # Use a smaller radius (just the clearance, not clearance + 1.0)
            nearby_obstacles = obstacle_kdt.query_ball_point(obj_center, r=clearance)
            # Only reject if there are obstacle points VERY close (within clearance)
            if len(nearby_obstacles) > 200:  # Much more lenient threshold
                rejection_stats['collision'] += 1
                continue

        # Check 2: Object is visible (not occluded, inside FOV)
        if avoid_occlusion:
            inside_frac, unocc_frac = fraction_unoccluded(
                obj_parent, zbuf, T_lidar_cam, model, K, D, width, height,
                n_surface_samples=min(1500, n_surface_samples),
                z_margin=z_margin,
                require_inside_frac=require_inside_frac
            )
            
            if inside_frac < require_inside_frac:
                rejection_stats['outside_fov'] += 1
                if (attempt + 1) % 25 == 0:
                    print(f"[PLACEMENT] Attempt {attempt+1}/{tries} - Stats: collision={rejection_stats['collision']}, "
                          f"fov={rejection_stats['outside_fov']}, occluded={rejection_stats['occluded']}, "
                          f"no_ground={rejection_stats['no_ground']}")
                continue
            if unocc_frac < unoccluded_thresh:
                rejection_stats['occluded'] += 1
                if (attempt + 1) % 25 == 0:
                    print(f"[PLACEMENT] Attempt {attempt+1}/{tries} - Stats: collision={rejection_stats['collision']}, "
                          f"fov={rejection_stats['outside_fov']}, occluded={rejection_stats['occluded']}, "
                          f"no_ground={rejection_stats['no_ground']}")
                continue

        # Valid placement found!
        best_pose = obj_parent.matrix_world.copy()
        print(f"[PLACEMENT] Success at x={x:.2f}, y={y:.2f}, ground_z={ground_z:.2f} (attempt {attempt+1}/{tries})")
        print(f"[PLACEMENT] Rejection breakdown: collision={rejection_stats['collision']}, "
              f"outside_fov={rejection_stats['outside_fov']}, occluded={rejection_stats['occluded']}, "
              f"no_ground={rejection_stats['no_ground']}")
        break
    
    if best_pose is None:
        print(f"[REJECT] Could not find valid placement after {tries} tries.")
        print(f"  Rejection breakdown: collision={rejection_stats['collision']}, "
              f"outside_fov={rejection_stats['outside_fov']}, occluded={rejection_stats['occluded']}, "
              f"no_ground={rejection_stats['no_ground']}")
        return False
    
    obj_parent.matrix_world = best_pose
    return True


def place_manual(obj_parent,
                x, y, z=None,
                yaw_deg=0.0,
                lidar_bin_path=None,
                adjust_to_ground=False,
                ransac_thresh=0.15,
                contact_offset=0.02,
                height_margin=0.10,
                lidar_label_path=None):
    """
    Manually place object at exact coordinates, skipping all placement checks.
    
    Args:
        obj_parent: Blender object to place
        x, y: X and Y coordinates in LiDAR/world frame (required)
        z: Z coordinate in LiDAR/world frame. If None and adjust_to_ground=True,
           will be automatically adjusted to ground height
        yaw_deg: Yaw rotation in degrees (default: 0.0)
        lidar_bin_path: Path to LiDAR .bin file (required if adjust_to_ground=True)
        adjust_to_ground: If True, adjust Z to ground height (default: False)
        ransac_thresh: RANSAC threshold for ground fitting (if adjusting to ground)
        contact_offset: Offset from ground for object contact point
        height_margin: Margin for height adjustment
        lidar_label_path: Path to label file (optional, for road-aware ground adjustment)
    
    Returns:
        True if placement succeeded, False otherwise
    """
    import bpy
    from mathutils import Euler
    
    # Set rotation (yaw around Z axis)
    yaw_rad = math.radians(yaw_deg)
    obj_parent.rotation_euler = Euler((0.0, 0.0, yaw_rad), 'XYZ')
    
    # Set position
    if z is None and adjust_to_ground:
        if lidar_bin_path is None:
            print("[ERROR] lidar_bin_path required when adjust_to_ground=True and z=None")
            return False
        
        # Load LiDAR and fit ground
        xyz = load_lidar_xyz(lidar_bin_path)
        
        # Fit ground plane
        n, d = fit_ground_plane_ransac(xyz, dist_thresh=ransac_thresh, seed=None)
        
        # Adjust height to ground at (x, y)
        labels = None
        if lidar_label_path:
            try:
                labels = load_lidar_labels(lidar_label_path)
            except Exception:
                pass
        
        # Build ground KD-tree for height adjustment
        ground_kdt, ground_pts = _build_ground_kdtree(xyz, n, d, ground_thresh=ransac_thresh,
                                                      labels=labels, prefer_road=True)
        
        if ground_kdt is not None and ground_pts.shape[0] > 0:
            # Find nearest ground point to (x, y)
            query_xy = np.array([[x, y]], dtype=np.float64)
            dists, indices = ground_kdt.query(query_xy, k=min(5, len(ground_pts)), workers=-1)
            if isinstance(dists, np.ndarray) and dists.size > 0:
                if dists.ndim == 1:
                    nearest_idx = indices[0] if isinstance(indices, np.ndarray) else indices
                    ground_z = float(ground_pts[nearest_idx, 2])
                else:
                    nearest_idx = indices[0, 0] if isinstance(indices, np.ndarray) else indices[0]
                    ground_z = float(ground_pts[nearest_idx, 2])
            else:
                # Fallback: use plane equation
                ground_z = float(-(n[0] * x + n[1] * y + d) / n[2]) if abs(n[2]) > 1e-6 else 0.0
        else:
            # Fallback: use plane equation
            ground_z = float(-(n[0] * x + n[1] * y + d) / n[2]) if abs(n[2]) > 1e-6 else 0.0
        
        # Get object's lowest point
        V = _gather_world_vertices(obj_parent)
        if V.size > 0:
            # Temporarily place at (x, y, 0) to get object bounds
            obj_parent.location = (x, y, 0.0)
            bpy.context.view_layer.update()
            V_temp = _gather_world_vertices(obj_parent)
            obj_bottom_z = float(V_temp[:, 2].min())
            
            # Calculate z so object bottom touches ground
            z = ground_z + float(contact_offset) - obj_bottom_z
        else:
            z = ground_z + float(contact_offset)
    
    elif z is None:
        z = 0.0  # Default to z=0 if not specified and not adjusting to ground
    
    # Set final position
    obj_parent.location = (x, y, z)
    bpy.context.view_layer.update()
    
    print(f"[MANUAL PLACEMENT] Object placed at ({x:.3f}, {y:.3f}, {z:.3f}), yaw={yaw_deg:.1f}°")
    return True


