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

def has_collision_with_scene(obj_parent, kdtree, clearance=0.15, n_samples=4000):
    """
    True if any sampled surface point of the object is closer than `clearance`
    to a non-ground LiDAR point.
    """
    if kdtree is None:
        return False
    pts_surf = sample_mesh_surface_world(obj_parent, n_points=n_samples)
    #early out for very small meshes
    if pts_surf.shape[0] == 0:
        return False
    dists, _ = kdtree.query(pts_surf, k=1, workers=-1)
    return bool((dists < float(clearance)).any())
