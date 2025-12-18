"""
Blender mesh operations.
"""
import sys
import random
import numpy as np

try:
    import bpy
    from mathutils import Matrix
except ImportError:
    print("This script must be run inside Blender.")
    sys.exit(1)


def _gather_world_vertices(obj_parent):
    """Collect all vertices from object hierarchy in world coordinates."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    verts = []

    stack = [obj_parent]
    while stack:
        obj = stack.pop()
        if obj.type == 'MESH':
            eval_obj = obj.evaluated_get(depsgraph)
            mesh = eval_obj.to_mesh()
            try:
                mw = eval_obj.matrix_world
                for v in mesh.vertices:
                    co = mw @ v.co
                    verts.append([co.x, co.y, co.z])
            finally:
                eval_obj.to_mesh_clear()
        stack.extend(list(obj.children))

    if not verts:
        raise RuntimeError("No mesh vertices found")
    return np.asarray(verts, dtype=np.float64)


def measure_object_dims_world(obj_parent):
    """Get object dimensions in world space. Returns (sizes, longest)."""
    V = _gather_world_vertices(obj_parent)
    sizes = V.max(axis=0) - V.min(axis=0)
    return sizes, float(sizes.max())


def fit_object_longest_to(obj_parent, target_size=1.0, jitter_frac=0.15):
    """Scale object so longest dimension equals target_size (with jitter)."""
    _, longest = measure_object_dims_world(obj_parent)
    if longest < 1e-6:
        raise RuntimeError("Object has zero size")

    base_scale = target_size / longest
    jitter = 1.0 + (random.uniform(-jitter_frac, jitter_frac) if jitter_frac > 0 else 0.0)
    scale = base_scale * jitter

    S = Matrix.Diagonal((scale, scale, scale, 1.0))
    obj_parent.matrix_world = obj_parent.matrix_world @ S


def _collect_world_triangles(obj_parent):
    """Collect all triangles from object hierarchy in world coordinates."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    tris = []

    def process_mesh(obj):
        if obj.type != 'MESH':
            return
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        try:
            mesh.calc_loop_triangles()
            mw = eval_obj.matrix_world
            coords = np.array([mw @ v.co for v in mesh.vertices], dtype=np.float64)
            for lt in mesh.loop_triangles:
                i0, i1, i2 = lt.vertices
                tris.append(np.stack([coords[i0], coords[i1], coords[i2]]))
        finally:
            eval_obj.to_mesh_clear()

    stack = [obj_parent]
    while stack:
        obj = stack.pop()
        process_mesh(obj)
        stack.extend(list(obj.children))

    if not tris:
        raise RuntimeError("No triangles found")
    return np.stack(tris, axis=0)


def _sample_points_on_triangles_world(tris, n_points):
    """Sample points uniformly on triangle mesh (area-weighted)."""
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    areas = np.where(areas > 1e-12, areas, 0.0)
    probs = areas / (areas.sum() + 1e-18)

    idx = np.random.choice(len(tris), size=n_points, p=probs)
    u, v = np.random.rand(n_points), np.random.rand(n_points)
    su = np.sqrt(u)
    b0, b1, b2 = 1.0 - su, su * (1.0 - v), su * v

    pts = b0[:, None] * v0[idx] + b1[:, None] * v1[idx] + b2[:, None] * v2[idx]
    return pts
