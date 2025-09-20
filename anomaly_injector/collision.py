import numpy as np
from scipy.spatial import cKDTree
from .mesh_ops import _collect_world_triangles, _sample_points_on_triangles_world


def build_non_ground_kdtree(xyz, n, d, ground_thresh=0.15):
    """
    Keep points sufficiently far from the ground plane → potential obstacles.
    Returns (kdtree, pts_non_ground).
    """
    dist = np.abs(xyz @ n + d)
    mask = dist > ground_thresh
    pts_ng = xyz[mask]
    if pts_ng.size == 0:
        return None, pts_ng
    return cKDTree(pts_ng), pts_ng

def sample_mesh_surface_world(obj_parent, n_points=4000):
    """
    Importance-sample surface points in *world* space (area-weighted).
    """
    tris_world = _collect_world_triangles(obj_parent)
    return _sample_points_on_triangles_world(tris_world, n_points)

def has_collision_with_scene(obj_parent, kdtree, clearance=0.15, n_samples=4000,
                             coarse_samples=1024, early_margin=0.05):
    """
    True if any sampled surface point of the object is closer than `clearance`
    to a non-ground LiDAR point.

    Speed: uses a coarse stage to early-exit when obviously clear.
    """
    if kdtree is None:
        return False

    # Coarse stage
    s_coarse = int(min(max(1, coarse_samples), max(1, n_samples)))
    pts_coarse = sample_mesh_surface_world(obj_parent, n_points=s_coarse)
    if pts_coarse.shape[0] == 0:
        return False
    dists_coarse, _ = kdtree.query(pts_coarse, k=1, workers=-1)
    if np.all(dists_coarse >= float(clearance) + float(early_margin)):
        return False

    # Fine stage (only if needed)
    if s_coarse >= n_samples:
        return bool((dists_coarse < float(clearance)).any())

    pts_fine = sample_mesh_surface_world(obj_parent, n_points=int(min(n_samples, 6000)))
    if pts_fine.shape[0] == 0:
        return False
    dists_fine, _ = kdtree.query(pts_fine, k=1, workers=-1)
    return bool((dists_fine < float(clearance)).any())
