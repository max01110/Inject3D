"""
Blender scene setup and utilities.
"""
import sys
import math
import numpy as np
from pathlib import Path

try:
    import bpy
    from mathutils import Matrix, Vector
except ImportError:
    print("This script must be run inside Blender.")
    sys.exit(1)


def to_blender_matrix(T):
    """Convert 4x4 numpy array to Blender Matrix."""
    return Matrix([list(T[i]) for i in range(4)])


def ensure_collection(name):
    """Get or create a Blender collection."""
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def clear_scene():
    """Reset Blender to empty scene."""
    bpy.ops.wm.read_homefile(use_empty=True)
    for obj in list(bpy.data.objects):
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass


def import_mesh(filepath):
    """Import mesh file and return parent empty."""
    ext = Path(filepath).suffix.lower()
    importers = {
        ".glb": lambda: bpy.ops.import_scene.gltf(filepath=filepath),
        ".gltf": lambda: bpy.ops.import_scene.gltf(filepath=filepath),
        ".obj": lambda: bpy.ops.import_scene.obj(filepath=filepath),
        ".fbx": lambda: bpy.ops.import_scene.fbx(filepath=filepath),
    }
    if ext not in importers:
        raise ValueError(f"Unsupported mesh format: {ext}")

    importers[ext]()

    objs = list(bpy.context.selected_objects)
    parent = bpy.data.objects.new("ObjaverseAsset", None)
    bpy.context.scene.collection.objects.link(parent)
    for o in objs:
        o.parent = parent
    return parent


def triangulate_and_smooth(obj_parent):
    """Apply scale and smooth shading to mesh children."""
    bpy.ops.object.select_all(action='DESELECT')
    for child in obj_parent.children:
        if child.type != 'MESH':
            continue
        bpy.context.view_layer.objects.active = child
        child.select_set(True)
        try:
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            bpy.ops.object.shade_smooth()
        except Exception:
            pass
        child.select_set(False)


def set_camera_from_extrinsics(K, width, height, T_lidar_cam):
    """
    Create camera matching given intrinsics and extrinsics.
    K: 3x3 intrinsic matrix
    T_lidar_cam: 4x4 camera pose in LiDAR frame
    """
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Set intrinsics
    cam_data.sensor_width = float(width)
    cam_data.sensor_height = float(height)
    cam_data.lens = float(K[0, 0])
    cam_data.shift_x = (-(K[0, 2]) + width / 2.0) / width
    cam_data.shift_y = ((K[1, 2]) - height / 2.0) / height

    # Set extrinsics (convert to Blender convention)
    R_lidar_cam = T_lidar_cam[:3, :3]
    t_lidar_cam = T_lidar_cam[:3, 3]
    R_fix = np.diag([1, -1, -1]).astype(np.float64)
    R_world_cam = R_lidar_cam @ R_fix

    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = R_world_cam
    T_world_cam[:3, 3] = t_lidar_cam
    cam_obj.matrix_world = to_blender_matrix(T_world_cam)

    # Render settings
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.film_transparent = True
    try:
        scene.cycles.device = 'GPU'
    except Exception:
        pass

    return cam_obj


# -----------------------------------------------------------------------------
# Debug utilities
# -----------------------------------------------------------------------------

def print_pose(obj, name=""):
    """Print object pose in world frame."""
    M = obj.matrix_world
    loc = M.to_translation()
    R = M.to_3x3()
    quat = R.to_quaternion()
    eul = R.to_euler('XYZ')
    fwd = R @ Vector((0, 0, -1))
    up = R @ Vector((0, 1, 0))
    right = R @ Vector((1, 0, 0))

    print(f"\n[{name}]")
    print(f"  loc: {loc.x:.3f}, {loc.y:.3f}, {loc.z:.3f}")
    print(f"  euler: {math.degrees(eul.x):.1f}, {math.degrees(eul.y):.1f}, {math.degrees(eul.z):.1f}")
    print(f"  quat: {quat.x:.4f}, {quat.y:.4f}, {quat.z:.4f}, {quat.w:.4f}")


def print_camera_intrinsics(cam_obj):
    """Print camera intrinsic parameters."""
    c = cam_obj.data
    hfov = 2 * math.degrees(math.atan(c.sensor_width / (2 * c.lens)))
    vfov = 2 * math.degrees(math.atan(c.sensor_height / (2 * c.lens)))
    print(f"\n[Camera]")
    print(f"  f={c.lens:.1f}px  sensor={c.sensor_width:.0f}x{c.sensor_height:.0f}")
    print(f"  shift=({c.shift_x:.4f}, {c.shift_y:.4f})  fov={hfov:.1f}°x{vfov:.1f}°")


def print_relative(cam_obj, obj):
    """Print object position relative to camera."""
    M_cam_inv = cam_obj.matrix_world.inverted()
    t_cam_obj = (M_cam_inv @ obj.matrix_world).to_translation()
    dist = (obj.matrix_world.translation - cam_obj.matrix_world.translation).length
    print(f"\n[Object→Camera]")
    print(f"  cam frame: {t_cam_obj.x:.3f}, {t_cam_obj.y:.3f}, {t_cam_obj.z:.3f}")
    print(f"  range: {dist:.2f}m")
