import sys, math
import numpy as np
from math import radians
from pathlib import Path

# --- Blender ---
try:
    import bpy
except Exception:
        print("This script must be run from inside Blender (has to import bpy).")
        sys.exit(1)

from mathutils import Matrix, Vector
import cv2


def to_blender_matrix(T):
    return Matrix([[T[0,0], T[0,1], T[0,2], T[0,3]],
                   [T[1,0], T[1,1], T[1,2], T[1,3]],
                   [T[2,0], T[2,1], T[2,2], T[2,3]],
                   [T[3,0], T[3,1], T[3,2], T[3,3]]])


def ensure_collection(name):
    if name in bpy.data.collections: return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col

def clear_scene():
    bpy.ops.wm.read_homefile(use_empty=True)
    for obj in list(bpy.data.objects):
        try: bpy.data.objects.remove(obj, do_unlink=True)
        except: pass

def import_mesh(filepath):
    ext = Path(filepath).suffix.lower()
    if ext in [".glb", ".gltf"]:
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == ".obj":
        bpy.ops.import_scene.obj(filepath=filepath)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=filepath)
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")
    objs = [o for o in bpy.context.selected_objects]
    parent = bpy.data.objects.new("ObjaverseAsset", None)
    bpy.context.scene.collection.objects.link(parent)
    for o in objs: o.parent = parent
    return parent


def triangulate_and_smooth(obj_parent):
    bpy.ops.object.select_all(action='DESELECT')
    for c in obj_parent.children:
        if c.type == 'MESH':
            bpy.context.view_layer.objects.active = c
            c.select_set(True)
            try:
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                bpy.ops.object.shade_smooth()
            except: pass
            c.select_set(False)


def set_camera_from_extrinsics(K, width, height, T_lidar_cam):
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    cam_data.sensor_width  = float(width)
    cam_data.sensor_height = float(height)
    cam_data.lens = float(K[0,0])
    cam_data.shift_x = (-(K[0,2]) + width/2.0)/width
    cam_data.shift_y = ((K[1,2]) - height/2.0)/height

    R_lidar_cam = T_lidar_cam[:3,:3]
    t_lidar_cam = T_lidar_cam[:3, 3]
    R_fix = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
    R_world_cam_blender = R_lidar_cam @ R_fix

    # t_lidar_cam IS the camera position in LiDAR/world frame (from ROS static_transform convention)
    T_world_cam = np.eye(4)
    T_world_cam[:3,:3] = R_world_cam_blender
    T_world_cam[:3, 3] = t_lidar_cam
    cam_obj.matrix_world = to_blender_matrix(T_world_cam)

    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.film_transparent = True
    try:
        scene.cycles.device = 'GPU'
    except: pass
    return cam_obj

#====================DEBUG PRINTERS==========================


def print_pose(o, name=""):
    M = o.matrix_world
    loc = M.to_translation()
    R  = M.to_3x3()
    quat = R.to_quaternion()
    eul  = R.to_euler('XYZ')
    fwd  = R @ Vector((0, 0, -1))
    up   = R @ Vector((0, 1,  0))
    right= R @ Vector((1, 0,  0))
    print(f"\n[{name}]")
    print(f"location (world/LiDAR): {loc.x:.3f}, {loc.y:.3f}, {loc.z:.3f}  [m]")
    print(f"euler XYZ (deg):        {math.degrees(eul.x):.2f}, {math.degrees(eul.y):.2f}, {math.degrees(eul.z):.2f}")
    print(f"quaternion (x,y,z,w):   {quat.x:.6f}, {quat.y:.6f}, {quat.z:.6f}, {quat.w:.6f}")
    print(f"axes (world): right={tuple(round(v,3) for v in right)}, up={tuple(round(v,3) for v in up)}, forward={tuple(round(v,3) for v in fwd)}")

def print_camera_intrinsics(cam_obj):
    c = cam_obj.data
    hfov = 2*math.degrees(math.atan(c.sensor_width /(2*c.lens)))
    vfov = 2*math.degrees(math.atan(c.sensor_height/(2*c.lens)))
    print(f"\n[Camera intrinsics]")
    print(f" lens(f)= {c.lens:.3f} px   sensor(WxH)= {c.sensor_width:.3f} x {c.sensor_height:.3f}")
    print(f" shift_x= {c.shift_x:.6f}   shift_y= {c.shift_y:.6f}")
    print(f" FOV(h/v)= {hfov:.2f}° / {vfov:.2f}°")

def print_relative(cam_obj, obj):
    M_cam_inv = cam_obj.matrix_world.inverted()
    t_cam_obj = (M_cam_inv @ obj.matrix_world).to_translation()
    dist = (obj.matrix_world.translation - cam_obj.matrix_world.translation).length
    print(f"\n[Object relative to camera]")
    print(f" translation in camera frame: {t_cam_obj.x:.3f}, {t_cam_obj.y:.3f}, {t_cam_obj.z:.3f}  (in front if z<0)")
    print(f" range: {dist:.3f} m")
