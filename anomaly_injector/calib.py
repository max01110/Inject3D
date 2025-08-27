import numpy as np, yaml
from transforms3d.quaternions import quat2mat


def load_calib(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    cam = data["camera_intrinsics"]
    model = cam["distortion_model"].lower()
    K = np.array(cam["K"], dtype=np.float64).reshape(3,3)
    D = np.array(cam["D"], dtype=np.float64).reshape(-1, 1)
    width  = int(cam["width"]); height = int(cam["height"])
    P = None
    if "P" in cam and cam["P"] is not None:
        P = np.array(cam["P"], dtype=np.float64).reshape(3,4)

    frame_id = cam.get("frame_id", "")
    st = _find_static_transform(data, frame_id)
    if st is None:
        raise RuntimeError("No static transform found with parent='os_sensor'.")
    if st.get("child") != frame_id:
        print(f"[WARN] static_transforms child '{st.get('child')}' != intrinsics frame_id '{frame_id}'. Using '{st.get('child')}'.")

    t = st["translation"]; q = st["rotation"]
    tvec = np.array([t["x"], t["y"], t["z"]], dtype=np.float64)
    qxyzw = np.array([q["x"], q["y"], q["z"], q["w"]], dtype=np.float64)
    R = quat2mat([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]])

    T_lidar_cam = np.eye(4)
    T_lidar_cam[:3,:3] = R
    T_lidar_cam[:3, 3] = tvec
    return model, K, D, width, height, T_lidar_cam, P


def _find_static_transform(cfg, desired_child):
    exact = None; camera_like = None; fallback = None
    for st in cfg.get("static_transforms", []):
        if st.get("parent") != "os_sensor":
            continue
        if st.get("child") == desired_child:
            exact = st; break
        if camera_like is None and ("camera" in st.get("child","") or "port_a_camera" in st.get("child","")):
            camera_like = st
        if fallback is None:
            fallback = st
    return exact or camera_like or fallback