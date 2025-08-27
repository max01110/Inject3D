import sys, random
import numpy as np

try:
    import bpy
except Exception:
        print("This script must be run from inside Blender (has to import bpy).")
        sys.exit(1)

from mathutils import Matrix



def _gather_world_vertices(obj_parent):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    verts = []
    stack = [obj_parent]
    while stack:
        o = stack.pop()
        if o.type == 'MESH':
            eval_o = o.evaluated_get(depsgraph)
            me = eval_o.to_mesh()
            try:
                mw = eval_o.matrix_world
                for v in me.vertices:
                    co = mw @ v.co
                    verts.append([co.x, co.y, co.z])
            finally:
                eval_o.to_mesh_clear()
        stack.extend(list(o.children))
    if not verts: raise RuntimeError("No mesh vertices found to measure object size.")
    return np.asarray(verts, dtype=np.float64)

def measure_object_dims_world(obj_parent):
    V = _gather_world_vertices(obj_parent)
    sizes = V.max(axis=0) - V.min(axis=0)
    return sizes, float(sizes.max())

def fit_object_longest_to(obj_parent, target_size=1.0, jitter_frac=0.15):
    _, longest = measure_object_dims_world(obj_parent)
    if longest < 1e-6: raise RuntimeError("Object appears degenerate (zero size)")
    base = target_size / longest
    j = 1.0 + (random.uniform(-jitter_frac, jitter_frac) if jitter_frac>0 else 0.0)
    s = base * j
    S = Matrix.Diagonal((s, s, s, 1.0))
    obj_parent.matrix_world = obj_parent.matrix_world @ S


def _collect_world_triangles(obj_parent):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    tris = []
    def add_obj_mesh(obj):
        if obj.type != 'MESH': return
        eval_obj = obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()
        try:
            eval_mesh.calc_loop_triangles()
            mw = eval_obj.matrix_world
            co = np.array([mw @ v.co for v in eval_mesh.vertices], dtype=np.float64)
            for lt in eval_mesh.loop_triangles:
                i0,i1,i2 = lt.vertices
                tris.append(np.stack([co[i0],co[i1],co[i2]], axis=0))
        finally:
            eval_obj.to_mesh_clear()
    stack=[obj_parent]
    while stack:
        o=stack.pop()
        if o.type=='MESH': add_obj_mesh(o)
        stack.extend(list(o.children))
    if not tris: raise RuntimeError("No triangles found on imported object")
    return np.stack(tris, axis=0)

def _sample_points_on_triangles_world(tris_world, n_points):
    v0 = tris_world[:,0,:]; v1 = tris_world[:,1,:]; v2 = tris_world[:,2,:]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    areas = np.where(areas > 1e-12, areas, 0.0)
    probs = areas / (areas.sum() + 1e-18)
    idx = np.random.choice(len(tris_world), size=n_points, p=probs)
    u = np.random.rand(n_points); v = np.random.rand(n_points)
    su = np.sqrt(u); b0 = 1.0 - su; b1 = su*(1.0 - v); b2 = su*v
    p = (b0[:,None]*v0[idx] + b1[:,None]*v1[idx] + b2[:,None]*v2[idx])
    return p
