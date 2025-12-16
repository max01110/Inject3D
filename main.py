import os, sys, argparse, tempfile, cv2, random
import numpy as np
from pathlib import Path
from math import radians

sys.path.append(os.path.dirname(__file__))

try:
    import bpy
    from mathutils import Vector
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
from anomaly_injector.occlusion import build_scene_zbuffer_from_lidar, fraction_unoccluded
from anomaly_injector.collision import has_collision_with_scene, build_non_ground_kdtree
from anomaly_injector.placement import (
    place_random_on_lidar_ground,
    load_lidar_xyz,
    fit_ground_plane_ransac,
    z_on_plane_at_xy,
    orient_largest_face_to_ground,
    snap_object_bottom_to_plane,
    _reorient_plane_to_camera,
)
from anomaly_injector.blender_utils import print_pose, print_camera_intrinsics, print_relative
from anomaly_injector.image_ops import composite_over_background
from anomaly_injector.mesh_ops import fit_object_longest_to, _collect_world_triangles, _sample_points_on_triangles_world
from anomaly_injector.blender_utils import set_camera_from_extrinsics


def _delete_object_hierarchy(obj):
    """Remove obj and all its children, then purge orphan meshes/materials."""
    import bpy
    # Deselect all, select hierarchy
    bpy.ops.object.select_all(action='DESELECT')
    def _select_tree(o):
        o.select_set(True)
        for ch in o.children:
            _select_tree(ch)
    _select_tree(obj)
    bpy.ops.object.delete()
    # Purge orphan data blocks to free memory
    for datablock in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.textures):
        for x in list(datablock):
            if x.users == 0:
                datablock.remove(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib", required=True) #calibration file for camera and lidar extrinsics/instrincs  
    parser.add_argument("--image", required=True, help="raw distorted frame (e.g., image2.png)") #original image
    parser.add_argument("--lidar", required=True, help="lidar.bin (Nx4 float32)") #original lidar point cloud (.bin file)
    parser.add_argument("--labels", required=True, help="lidar.label (N uint32)") #original lidar labels (.label file)
    parser.add_argument("--outdir", required=True)  #output directory
    parser.add_argument("--seed", type=int, default=None) #random seed
    parser.add_argument("--n_mesh_points", type=int, default=20000) #number of mesh points to sample to incporporate object into lidar
    parser.add_argument("--anomaly_label", type=int, default=150) #label to assign to the new object points in the lidar label file

    # Placement
    parser.add_argument("--x_range", nargs=2, type=float, metavar=("XMIN","XMAX"), default=(4.0, 12.0)) #distance range in front of the lidar to place the object
    parser.add_argument("--y_range", nargs=2, type=float, metavar=("YMIN","YMAX"), default=(-2.0, 2.0)) #lateral range to place the object
    parser.add_argument("--ground_ransac_thresh", type=float, default=0.15) #RANSAC threshold for ground plane fitting -> threshold distance from randomly selected plane (for) to other lidar points (larger num means more robust, but less accurate)
    parser.add_argument("--ground_contact_offset", type=float, default=0.02) #offset above the ground plane to place the object
    parser.add_argument("--yaw_min", type=float, default=0.0) #minimum yaw angle for object placement
    parser.add_argument("--yaw_max", type=float, default=0.0) #maximum yaw angle for object placement
    parser.add_argument("--clearance", type=float, default=0.20) #minimum clearance from other points in the lidar point cloud to insert the object
    parser.add_argument("--place_tries", type=int, default=1) #number of attempts to place the object without collisions 
    parser.add_argument("--surf_samples", type=int, default=40000) #number of surface points (from objaverse object) to sample for collision & occlusion checking
    parser.add_argument("--z_margin", type=float, default=0.05) #margin for z-buffer occlusion checking (the object’s depth being at least z_margin closer than the scene depth)
    parser.add_argument("--require_inside_frac", type=float, default=0.85) #fraction of object points that must be inside the image
    parser.add_argument("--unoccluded_thresh", type=float, default=1.0) #fraction of object points that must be unoccluded in the z-buffer


    # Sizing
    parser.add_argument("--target_size", type=float, default=None,
                        help="If set, use this longest-side size (m) instead of random [0.5,2.2]")
    parser.add_argument("--size_jitter_frac", type=float, default=0.15)

    # IoU control
    parser.add_argument("--iou_thresh", type=float, default=0.92, #minimum IoU between composite image and point cloud projection
                        help="Post-alignment IoU required to accept the object (default: 0.60)") 
    parser.add_argument("--iou_max_tries", type=int, default=10,  #max number of retries before script stops if IoU is below threshold
                        help="Max fresh-object attempts if IoU is below threshold (default: 10)")

    args, _ = parser.parse_known_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])

    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # Load calib once
    model, K, D, width, height, T_lidar_cam, P = load_calib(args.calib)

    # Clear scene and set camera/lights once
    clear_scene()
    cam_obj = set_camera_from_extrinsics(K, width, height, T_lidar_cam)

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

    # Build a scene z-buffer from LiDAR once
    zbuf = build_scene_zbuffer_from_lidar(args.lidar, T_lidar_cam, model, K, D, width, height, dilate_px=1)

    # Prepare background image once
    bg_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bg_bgr is None:
        raise RuntimeError(f"Could not read background image: {args.image}")

    # Static rect<->raw mapping once
    K_rect = K
    mapx, mapy = build_rectified_to_distorted_map(model, K, D, width, height, K_rect=K_rect)

    accepted = False
    best_report = None

    with tempfile.TemporaryDirectory() as td:
        for attempt in range(1, args.iou_max_tries + 1):
            print(f"\n=== IoU attempt {attempt}/{args.iou_max_tries} ===")
            # Fetch a fresh Objaverse asset each try
            uid, mesh_path = get_random_objaverse(td)
            print(f"Picked UID: {uid} at {mesh_path}")

            # Import & size
            obj_parent = import_mesh(mesh_path)
            triangulate_and_smooth(obj_parent)
            target_size = args.target_size if args.target_size is not None else random.uniform(0.5, 2.2)
            fit_object_longest_to(obj_parent, target_size=target_size, jitter_frac=args.size_jitter_frac)
      
            # Placement on LiDAR ground
            assert args.x_range[0] < args.x_range[1]
            assert args.y_range[0] < args.y_range[1]
            assert args.yaw_min <= args.yaw_max
            place_random_on_lidar_ground(
                obj_parent,
                lidar_bin_path=args.lidar,
                x_range=tuple(args.x_range),
                y_range=tuple(args.y_range),
                seed=args.seed,
                ransac_thresh=args.ground_ransac_thresh,
                contact_offset=args.ground_contact_offset,
                yaw_range_deg=(args.yaw_min, args.yaw_max),
                clearance=args.clearance,
                tries=args.place_tries,
                n_surface_samples=args.surf_samples,
                avoid_occlusion=True,
                zbuf=zbuf,
                T_lidar_cam=T_lidar_cam, model=model, K=K, D=D, width=width, height=height,
                z_margin=args.z_margin,
                require_inside_frac=args.require_inside_frac,
                unoccluded_thresh=args.unoccluded_thresh,
                lidar_label_path=args.labels,
            )
            # Print camera-frame translation (pre-safety) using the same method as print_relative
            cam = bpy.context.scene.camera
            obj = bpy.data.objects.get("ObjaverseAsset") or obj_parent
            M_cam_inv = cam.matrix_world.inverted()
            t_cam_obj = (M_cam_inv @ obj.matrix_world).to_translation()
            print(f"[DEBUG] Before safety: cam-frame translation = ({t_cam_obj.x:.3f}, {t_cam_obj.y:.3f}, {t_cam_obj.z:.3f})")
            # Final safety: if object ends up above the camera (y_cam >= 0), try multiple random placements
            if t_cam_obj.y >= -0.4:
                print("[WARN] Object above camera after placement; trying random camera-relative locations.")
                fallback_tries = 50
                for fallback_attempt in range(fallback_tries):
                    try:
                        # Set directly to a random Blender camera-frame location within x_range (no snap/orient)
                        random_z = random.uniform(args.x_range[0], args.x_range[1])
                        random_y = random.uniform(args.y_range[0], args.y_range[1])
                        p_c = Vector((random_y, -1.65, -random_z))
                        p_w = cam.matrix_world @ p_c
                        mw_fb = obj.matrix_world.copy()
                        mw_fb.translation = Vector((float(p_w.x), float(p_w.y), float(p_w.z)))
                        obj.matrix_world = mw_fb
                        
                        # Check collision with scene
                        if has_collision_with_scene(obj, kdt, clearance=args.clearance, n_samples=args.surf_samples):
                            continue  # Try another random position
                        
                        # Check occlusion if zbuf is available
                        if zbuf is not None:
                            inside_frac, unoccluded_frac = fraction_unoccluded(
                                obj, zbuf, T_lidar_cam, model, K, D, width, height,
                                n_surface_samples=args.surf_samples, z_margin=args.z_margin,
                                require_inside_frac=args.require_inside_frac
                            )
                            if unoccluded_frac < args.unoccluded_thresh:
                                print("==================OCCLUSION==================")
                                continue  # Try another random position
                        
                        # Found a valid placement!
                        print(f"[INFO] Main.py fallback placement succeeded after {fallback_attempt + 1} attempts")
                        break
                        
                    except Exception as e:
                        if fallback_attempt == fallback_tries - 1:  # Last attempt
                            print(f"[WARN] Main.py fallback placement failed after {fallback_tries} attempts: {e}")
                        continue
                # Recompute and print post-safety camera-frame translation
                M_cam_inv = cam.matrix_world.inverted()
                t_cam_obj = (M_cam_inv @ obj.matrix_world).to_translation()
                print(f"[DEBUG] After safety: cam-frame translation = ({t_cam_obj.x:.3f}, {t_cam_obj.y:.3f}, {t_cam_obj.z:.3f})")

            # Optional: verify ground plane orientation — ensure it's below camera
            xyz_lidar = load_lidar_xyz(args.lidar)
            n_raw, d_raw = fit_ground_plane_ransac(xyz_lidar, dist_thresh=args.ground_ransac_thresh)
            n_fix, d_fix = _reorient_plane_to_camera(n_raw, d_raw, T_lidar_cam)
            if (n_fix is not None) and (d_fix is not None):
                cam_pt = cam.matrix_world.translation
                s = float(n_fix[0]*cam_pt.x + n_fix[1]*cam_pt.y + n_fix[2]*cam_pt.z + d_fix)
                print(f"[DEBUG] Ground plane signed dist at camera (should be >0): {s:.4f}")
            # Configure render (transparent)
            scene = bpy.context.scene
            scene.render.image_settings.file_format = 'PNG'
            scene.render.image_settings.color_mode = 'RGBA'
            scene.render.film_transparent = True

            # Render to temp file (rectified view)
            with tempfile.NamedTemporaryFile(suffix=".png", dir=args.outdir, delete=False) as tmp:
                tmp_rgba_path = tmp.name
            try:
                scene.render.filepath = tmp_rgba_path

                cam = bpy.context.scene.camera
                obj = bpy.data.objects.get("ObjaverseAsset") or obj_parent
                print_pose(cam, "Camera"); print_camera_intrinsics(cam)
                print_pose(obj, "Object"); print_relative(cam, obj)

                print("Rendering object layer (rectified)")
                bpy.ops.render.render(write_still=True)

                # Distort render into raw domain
                rgba_rect = cv2.imread(tmp_rgba_path, cv2.IMREAD_UNCHANGED)
                if rgba_rect is None:
                    raise RuntimeError("Could not read rectified RGBA render.")
                rgba_raw = rgba_rectified_to_distorted(rgba_rect, mapx, mapy)

                print("Sampling mesh surface points for IoU check…")
                tris_world = _collect_world_triangles(obj_parent)
                
                # Adaptive sampling based on object size
                object_scale = np.linalg.norm(obj_parent.matrix_world.to_scale())
                adaptive_n_points = int(args.n_mesh_points * object_scale)
                # Ensure minimum and maximum bounds for sampling
                adaptive_n_points = max(1000, min(adaptive_n_points, 50000))
                
                print(f"[DEBUG] Object scale: {object_scale:.3f}")
                print(f"[DEBUG] Adaptive sampling points: {adaptive_n_points} (base: {args.n_mesh_points})")
                
                pts_mesh = _sample_points_on_triangles_world(tris_world, adaptive_n_points)

                # Build masks and compute IoU BEFORE/AFTER alignment
                uv_mesh = project_points_distorted(pts_mesh, T_lidar_cam, model, K, D)
                render_mask = (rgba_raw[..., 3] > 0).astype(np.uint8) * 255
                
                # Adaptive radius based on object distance and scale
                object_distance = np.linalg.norm(obj_parent.matrix_world.translation - cam.matrix_world.translation)
                base_radius = 5
                distance_factor = max(0.5, min(2.0, object_distance / 10.0))  # Scale with distance
                scale_factor = max(0.5, min(2.0, object_scale))  # Scale with object size
                adaptive_radius = int(base_radius * distance_factor * scale_factor)
                adaptive_radius = max(1, min(adaptive_radius, 20))  # Reasonable bounds
                
                print(f"[DEBUG] Object distance: {object_distance:.3f}m")
                print(f"[DEBUG] Adaptive mask radius: {adaptive_radius} (base: {base_radius})")
                
                pts_mask = mask_from_points(uv_mesh, width, height, radius=adaptive_radius)
                
                # Debug projection quality
                valid_points = np.sum((uv_mesh[:,0] >= 0) & (uv_mesh[:,0] < width) & 
                                    (uv_mesh[:,1] >= 0) & (uv_mesh[:,1] < height))
                print(f"[DEBUG] Valid projected points: {valid_points}/{len(uv_mesh)} ({100*valid_points/len(uv_mesh):.1f}%)")
                print(f"[DEBUG] UV range - U: [{uv_mesh[:,0].min():.1f}, {uv_mesh[:,0].max():.1f}]")
                print(f"[DEBUG] UV range - V: [{uv_mesh[:,1].min():.1f}, {uv_mesh[:,1].max():.1f}]")
                
                iou_before  = compute_iou(render_mask, pts_mask)

                sx, sy, dx, dy = estimate_affine_scale_translate(render_mask, pts_mask)
                rgba_fit  = warp_rgba_affine_scale_translate(rgba_raw, sx, sy, dx, dy)
                mask_fit  = (rgba_fit[..., 3] > 0).astype(np.uint8) * 255
                iou_after = compute_iou(mask_fit, pts_mask)

                print(f"IoU BEFORE: {iou_before:.3f}")
                print(f"IoU AFTER:  {iou_after:.3f}")
                print(f"(sx, sy, dx, dy) = ({sx:.5f}, {sy:.5f}, {dx:.2f}, {dy:.2f})")

                # Decide: accept or retry with a new object
                if iou_after >= args.iou_thresh:
                    print(f"IoU {iou_after:.3f} ≥ threshold {args.iou_thresh:.3f} — accepting object.")
                    # Write final composites
                    out_fit_path = os.path.join(args.outdir, "aug_image.png")
                    cv2.imwrite(out_fit_path, alpha_composite_rgba_over_bgr(rgba_fit, bg_bgr))
                    # Optional preview in rectified domain
                    preview_path = os.path.join(args.outdir, "aug_image_rectified_preview.png")
                    composite_over_background(tmp_rgba_path, args.image, preview_path)

                    # Now write augmented point cloud + labels
                    out_bin   = os.path.join(args.outdir, "aug_lidar.bin")
                    out_label = os.path.join(args.outdir, "aug_lidar.label")
                    write_augmented_pointcloud(args.lidar, args.labels, pts_mesh,
                                               args.anomaly_label, out_bin, out_label)
                    print(f"Saved: {out_fit_path}")
                    print(f"Saved: {out_bin} and {out_label}")

                    best_report = (iou_before, iou_after, sx, sy, dx, dy)
                    accepted = True
                    break
                else:
                    print(f"IoU {iou_after:.3f} < threshold {args.iou_thresh:.3f} — retrying with a new object.")
            finally:
                # Remove the temp RGBA file for this attempt
                try:
                    os.remove(tmp_rgba_path)
                except FileNotFoundError:
                    pass
                # Delete the object we just used (whether accepted or not).
                # If accepted, we already broke out; this cleanup only runs on failure.
                if not accepted:
                    _delete_object_hierarchy(obj_parent)

    if not accepted:
        raise RuntimeError(f"Failed to achieve IoU ≥ {args.iou_thresh} after {args.iou_max_tries} attempts.")

    print("Done.")


if __name__ == "__main__":
    main()

