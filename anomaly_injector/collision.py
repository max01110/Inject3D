"""
Collision detection with scene geometry.
"""
import numpy as np
from scipy.spatial import cKDTree

from .mesh_ops import _collect_world_triangles, _sample_points_on_triangles_world


def build_non_ground_kdtree(xyz, n, d, ground_thresh=0.15):
    """Build KD-tree of non-ground points (potential obstacles)."""
    dist = np.abs(xyz @ n + d)
    pts_ng = xyz[dist > ground_thresh]
    if pts_ng.size == 0:
        return None, pts_ng
    return cKDTree(pts_ng), pts_ng


def sample_mesh_surface_world(obj_parent, n_points=4000):
    """Sample points from mesh surface in world coordinates."""
    tris = _collect_world_triangles(obj_parent)
    return _sample_points_on_triangles_world(tris, n_points)


def has_collision_with_scene(obj_parent, kdtree, clearance=0.15, n_samples=4000,
                             coarse_samples=1024, early_margin=0.05):
    """
    Check if object collides with scene points.
    Uses coarse-to-fine sampling for efficiency.
    """
    if kdtree is None:
        return False

    # Coarse check
    n_coarse = min(max(1, coarse_samples), max(1, n_samples))
    pts_coarse = sample_mesh_surface_world(obj_parent, n_points=n_coarse)
    if pts_coarse.shape[0] == 0:
        return False

    dists, _ = kdtree.query(pts_coarse, k=1, workers=-1)
    if np.all(dists >= clearance + early_margin):
        return False

    if n_coarse >= n_samples:
        return bool((dists < clearance).any())

    # Fine check
    pts_fine = sample_mesh_surface_world(obj_parent, n_points=min(n_samples, 6000))
    if pts_fine.shape[0] == 0:
        return False

    dists_fine, _ = kdtree.query(pts_fine, k=1, workers=-1)
    return bool((dists_fine < clearance).any())
