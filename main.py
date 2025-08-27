import os, sys, argparse, tempfile, cv2, random
import numpy as np
from pathlib import Path
from math import radians

sys.path.append(os.path.dirname(__file__))

try:
    import bpy
except Exception:
        print("This script must be run from inside Blender (has to import bpy).")
        sys.exit(1)

# Make package importable when run from Blender
sys.path.append(str(Path(__file__).resolve().parents[1]))

from anomaly_injector import (
    load_calib,
    clear_scene,
    import_mesh,
    triangulate_and_smooth,
    build_rectified_to_distorted_map,
    rgba_rectified_to_distorted,
    alpha_composite_rgba_over_bgr,
    composite_over_background
)

from anomaly_injector.proj_ops import invert_lidar_cam, project_points_distorted, mask_from_points, estimate_affine_scale_translate, warp_rgba_affine_scale_translate, compute_iou
from anomaly_injector.objaverse_io import get_random_objaverse
from anomaly_injector.io_utils import write_augmented_pointcloud
from anomaly_injector.occlusion import build_scene_zbuffer_from_lidar
from anomaly_injector.placement import place_random_on_lidar_ground
from anomaly_injector.blender_utils import print_pose, print_camera_intrinsics, print_relative
from anomaly_injector.image_ops import composite_over_background
from anomaly_injector.mesh_ops import fit_object_longest_to, _collect_world_triangles, _sample_points_on_triangles_world
from anomaly_injector.blender_utils import set_camera_from_extrinsics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib", required=True)
    parser.add_argument("--image", required=True, help="raw distorted frame (e.g., image2.png)")
    parser.add_argument("--lidar", required=True, help="lidar.bin (Nx4 float32)")
    parser.add_argument("--labels", required=True, help="lidar.label (N uint32)")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_mesh_points", type=int, default=20000)
    parser.add_argument("--anomaly_label", type=int, default=150)
    parser.add_argument("--use_P_for_rect", action="store_true",
                        help="Use intrinsics from 3x4 P (left 3x3) as the rectified K used for the Blender render.")
    # Placement tweak: choose which local axis is the 'side' (default X)
    parser.add_argument("--side_axis", choices=['X','Y','Z'], default='X')
    # Optional: pixel to target on plane (defaults to center-ish)
    parser.add_argument("--place_u", type=float, default=None)
    parser.add_argument("--place_v", type=float, default=None)

    args, _ = parser.parse_known_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])

    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    #loading calibration
    model, K, D, width, height, T_lidar_cam, P = load_calib(args.calib)
    print(f"distortion_model={model}")

    #clearing the scene
    clear_scene()

    #setting camera
    cam_obj = set_camera_from_extrinsics(K, width, height, T_lidar_cam)

    #lighting
    world = bpy.data.worlds.new("World") if not bpy.data.worlds else bpy.data.worlds[0]
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs[1].default_value = 1.0
    light_data = bpy.data.lights.new(name="area_key", type='AREA')
    light_data.energy = 3000
    light_obj = bpy.data.objects.new(name="area_key", object_data=light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.location = (2, -2, 3)
    light_obj.rotation_euler = (radians(60), 0, radians(30))

    print("Fetching random Objaverse object…")
    with tempfile.TemporaryDirectory() as td:
        uid, mesh_path = get_random_objaverse(td)
        print(f"Picked UID: {uid} at {mesh_path}")

        print("Import mesh into Blender…")
        obj_parent = import_mesh(mesh_path)
        triangulate_and_smooth(obj_parent)

        print("Sizing object") #######NEED TO SET AS AN ARGUMENT **********
        fit_object_longest_to(obj_parent, target_size=random.uniform(0.5, 2.2), jitter_frac=0.15)

        zbuf = build_scene_zbuffer_from_lidar(args.lidar, T_lidar_cam, model, K, D, width, height, dilate_px=1)


        print("Placing object on LiDAR ground")
        place_random_on_lidar_ground(
            obj_parent,
            lidar_bin_path=args.lidar,
            x_range=tuple(args.x_range) if hasattr(args, "x_range") else (4.0, 12.0),
            y_range=tuple(args.y_range) if hasattr(args, "y_range") else (-2.0, 2.0),
            seed=args.seed,
            ransac_thresh=args.ground_ransac_thresh if hasattr(args, "ground_ransac_thresh") else 0.15,
            contact_offset=args.ground_contact_offset if hasattr(args, "ground_contact_offset") else 0.02,
            yaw_range_deg=(getattr(args, "yaw_min", 0.0), getattr(args, "yaw_max", 360.0)),
            clearance=getattr(args, "clearance", 0.20),
            tries=getattr(args, "place_tries", 200),
            n_surface_samples=getattr(args, "surf_samples", 4000),
            avoid_occlusion=True,
            zbuf=zbuf,
            T_lidar_cam=T_lidar_cam, model=model, K=K, D=D, width=width, height=height,
            z_margin=getattr(args, "z_margin", 0.05),
            require_inside_frac=getattr(args, "require_inside_frac", 0.85),
            unoccluded_thresh=getattr(args, "unoccluded_thresh", 1.0)
        )

 
        #Render only the object with transparent background
        scene = bpy.context.scene
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.film_transparent = True

        with tempfile.NamedTemporaryFile(suffix=".png", dir=args.outdir, delete=False) as tmp:
            tmp_rgba_path = tmp.name

        scene.render.filepath = tmp_rgba_path

        cam = bpy.context.scene.camera
        obj = bpy.data.objects.get("ObjaverseAsset") or obj_parent
        print_pose(cam, "Camera"); print_camera_intrinsics(cam)
        print_pose(obj, "Object"); print_relative(cam, obj)

        try:
            print("Rendering object layer")
            bpy.ops.render.render(write_still=True)

            preview_path = os.path.join(args.outdir, "aug_image_rectified_preview.png")
            composite_over_background(tmp_rgba_path, args.image, preview_path)

            print("Warping RGBA render to raw distorted image + compositing")
            K_rect = K
            if args.use_P_for_rect and P is not None:
                K_rect = P[:, :3].copy()
                if abs(K_rect[2,2]) < 1e-12: K_rect[2,2] = 1.0

            mapx, mapy = build_rectified_to_distorted_map(model, K, D, width, height, K_rect=K_rect)
            rgba_rect = cv2.imread(tmp_rgba_path, cv2.IMREAD_UNCHANGED)
            bg_bgr    = cv2.imread(args.image, cv2.IMREAD_COLOR)
            if rgba_rect is None or bg_bgr is None:
                raise RuntimeError("Could not read RGBA or background image.")
            rgba_raw = rgba_rectified_to_distorted(rgba_rect, mapx, mapy)


            # tmp_rgba_raw_path = os.path.join(args.outdir, "object_rgba_distorted.png")
            # cv2.imwrite(tmp_rgba_raw_path, rgba_raw)

            # out_img_path = os.path.join(args.outdir, "aug_image_distorted.png")
            # cv2.imwrite(out_img_path, alpha_composite_rgba_over_bgr(rgba_raw, bg_bgr))
            # print(f"   Saved augmented image (distorted domain): {out_img_path}")

            print("sampling mesh surface points for LiDAR fusion")
            tris_world = _collect_world_triangles(obj_parent)
            pts_mesh = _sample_points_on_triangles_world(tris_world, args.n_mesh_points)
            out_bin = os.path.join(args.outdir, "aug_lidar.bin")
            out_label = os.path.join(args.outdir, "aug_lidar.label")
            print("Writing augmented point cloud and labels")
            write_augmented_pointcloud(args.lidar, args.labels, pts_mesh, args.anomaly_label, out_bin, out_label)
            print(f"   Saved: {out_bin} and {out_label}")

            print("align distorted RGBA to projected mesh points…")
            uv_mesh = project_points_distorted(pts_mesh, T_lidar_cam, model, K, D)
            render_mask = (rgba_raw[...,3] > 0).astype(np.uint8)*255
            pts_mask    = mask_from_points(uv_mesh, width, height, radius=5)
            iou_before = compute_iou(render_mask, pts_mask)
            print(f"IoU BEFORE: {iou_before:.3f}")

            sx, sy, dx, dy = estimate_affine_scale_translate(render_mask, pts_mask)
            rgba_fit = warp_rgba_affine_scale_translate(rgba_raw, sx, sy, dx, dy)
            mask_fit = (rgba_fit[...,3] > 0).astype(np.uint8)*255
            iou_after = compute_iou(mask_fit, pts_mask)
            print(f"IoU AFTER:  {iou_after:.3f}")
            print(f"(sx, sy, dx, dy) = ({sx:.5f}, {sy:.5f}, {dx:.2f}, {dy:.2f})")

            out_fit_path = os.path.join(args.outdir, "aug_image.png")
            cv2.imwrite(out_fit_path, alpha_composite_rgba_over_bgr(rgba_fit, bg_bgr))

        finally:
            try:
                os.remove(tmp_rgba_path)
            except FileNotFoundError:
                pass
    print("Done")

if __name__ == "__main__":
    main()

