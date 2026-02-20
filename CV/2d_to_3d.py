import time
import math
import json
import numpy as np
import cv2
import pyrealsense2 as rs
from rtde_receive import RTDEReceiveInterface

# Relative file paths for GitHub
JSON_INPUT = "CV/objects_position.json"
T_TCP_CAM_PATH = "CV/T_tcp_cam.npy"
OUTPUT_JSON = "CV/positions_3d.json"
FRAME_COUNT = 5   # Automatically save after 5 frames

robot_to_table_height_offset = 0.014  # table height above base (14 mm)


# ============================================================
# Load JSON bboxes
# ============================================================
def load_bboxes():
    with open(JSON_INPUT, "r") as f:
        data = json.load(f)

    target = data.get("target", None)
    obstacles = data.get("obstacles", [])
    return target, obstacles


# ============================================================
# Convert UR5 Pose → 4×4 Transform
# ============================================================
def pose_to_T(pose):
    x, y, z, rx, ry, rz = pose
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)

    if theta < 1e-9:
        R = np.eye(3)
    else:
        kx, ky, kz = rx / theta, ry / theta, rz / theta
        K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
        R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


# ============================================================
# Compute bbox center
# ============================================================
def bbox_center(b):
    cx = (b["x1"] + b["x2"]) / 2.0
    cy = (b["y1"] + b["y2"]) / 2.0
    return int(cx), int(cy)


# ============================================================
# Get median depth
# ============================================================
def get_depth(depth_img, cx, cy, window=5):
    """
    Returns a robust depth estimate (in raw depth units) around (cx, cy)
    by taking the median of valid pixels in a (window x window) patch.
    """
    h, w = depth_img.shape

    half = window // 2
    x1 = max(0, cx - half)
    x2 = min(w, cx + half + 1)
    y1 = max(0, cy - half)
    y2 = min(h, cy + half + 1)

    patch = depth_img[y1:y2, x1:x2]

    # keep only non-zero (valid) depths
    valid = patch[patch > 0]

    if valid.size == 0:
        return 0  # or np.nan and handle upstream

    # optional: reject extreme outliers
    low, high = np.percentile(valid, [10, 90])
    trimmed = valid[(valid >= low) & (valid <= high)]

    if trimmed.size == 0:
        trimmed = valid

    return float(np.median(trimmed))


# ============================================================
# MAIN
# ============================================================
def main():

    # Load detections
    target, obstacles = load_bboxes()
    print("\n[INFO] Loaded JSON:")
    print(" Target:", target)
    print(" Obstacles:", len(obstacles))

    # hand–eye calibration
    T_tcp_cam = np.load(T_TCP_CAM_PATH)

    # Connect UR5
    rtde = RTDEReceiveInterface("192.168.1.5")

    # Setup RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    # For averaging over frames
    target_points = []
    obstacle_points = {}

    frames_collected = 0

    print(f"\n[INFO] Collecting {FRAME_COUNT} frames...\n")

    while frames_collected < FRAME_COUNT:

        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        depth_img = np.asanyarray(depth_frame.get_data())

        # Camera intrinsics
        intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        fx, fy = intr.fx, intr.fy
        cx0, cy0 = intr.ppx, intr.ppy
        depth_scale = depth_frame.get_units()

        # Robot transform
        tcp_pose = rtde.getActualTCPPose()
        T_base_tcp = pose_to_T(tcp_pose)
        T_base_cam = T_base_tcp @ T_tcp_cam

        # ============================
        # TARGET
        # ============================
        if isinstance(target, dict) and "avg_bbox" in target:
            cx, cy = bbox_center(target["avg_bbox"])
            depth = get_depth(depth_img, cx, cy, window=5) * depth_scale

            if depth > 0:
                X_cam = (cx - cx0) * depth / fx
                Y_cam = (cy - cy0) * depth / fy
                Z_cam = depth

                p_cam = np.array([[X_cam], [Y_cam], [Z_cam], [1.0]])
                p_base = T_base_cam @ p_cam

                Xb, Yb, Zb = p_base[0].item(), p_base[1].item(), p_base[2].item()

                # Height above table
                obj_height = Zb - robot_to_table_height_offset

                xyzh = [Xb, Yb, Zb, obj_height]
                target_points.append(xyzh)

        # ============================
        # OBSTACLES
        # ============================
        for obs in obstacles:
            oid = obs["id"]
            b = obs["avg_bbox"]
            cx, cy = bbox_center(b)

            depth = get_depth(depth_img, cx, cy, window=5) * depth_scale

            if depth > 0:

                X_cam = (cx - cx0) * depth / fx
                Y_cam = (cy - cy0) * depth / fy
                Z_cam = depth

                p_cam = np.array([[X_cam], [Y_cam], [Z_cam], [1.0]])
                p_base = T_base_cam @ p_cam

                Xb, Yb, Zb = p_base[0].item(), p_base[1].item(), p_base[2].item()

                obj_height = Zb - robot_to_table_height_offset

                xyzh = [Xb, Yb, Zb, obj_height]

                if oid not in obstacle_points:
                    obstacle_points[oid] = []
                obstacle_points[oid].append(xyzh)

        frames_collected += 1
        print(f"[INFO] Frame {frames_collected}/{FRAME_COUNT} collected.")
        time.sleep(0.05)

    pipeline.stop()
    cv2.destroyAllWindows()

    # =======================================================
    # SAVE AVERAGED 3D POSITIONS
    # =======================================================
    def avg(points):
        if len(points) == 0:
            return None
        arr = np.array(points)
        return arr.mean(axis=0).tolist()

    output = {
        "target": avg(target_points),
        "obstacles": {}
    }

    for oid, pts in obstacle_points.items():
        output["obstacles"][oid] = avg(pts)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=4)

    print("\n[SAVED] 3D positions →", OUTPUT_JSON)
    print(json.dumps(output, indent=4))


if __name__ == "__main__":
    main()

