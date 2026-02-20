import time
import torch
import numpy as np
import json
import pyrealsense2 as rs
import cv2
from PIL import Image
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

from ppo import ActorCritic
from plot_trajectory import add_state, plot_trajectory, save_logs_to_excel, save_trajectory_csv

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from florence_vision import load_florence_vision
from vision_ood_detector import FlorenceVisionOOD
from ppo_ood_detector import PPOControlOOD


# =============================== ARGUMENT PARSER ==============================
parser = argparse.ArgumentParser()
parser.add_argument("--record_normal_visual_latents", action="store_true",
                    help="Record Florence2 latents during full safe cycle.")
parser.add_argument("--record_normal_ppo_latents", action="store_true",
                    help="Record PPO numeric latents (128-D) during nominal runs")
args = parser.parse_args()

RECORDING_MODE = args.record_normal_visual_latents or args.record_normal_ppo_latents


# =============================== CONFIGURATION ================================
ROBOT_IP = "192.168.1.5"
MODEL_PATH = "saved_rl_models/best_____model_precise_seed_987.pth"
JSON_3D_PATH = "CV/positions_3d.json"

# --- Action scaling (cartesian deltas) ---
XY_SCALE_FAR = 0.005
XY_SCALE_NEAR = 0.003
Z_SCALE_FAR = 0.002
Z_SCALE_NEAR = 0.001

# Servo control parameters
SERVO_TIME = 1 / 120.0
MAX_DPOS = 0.0016
SERVO_LOOKAHEAD = 0.03
SERVO_GAIN = 200

MAX_SPEED = 0.08
MAX_ACCEL = 0.06
WARMUP_STEPS = 10
MIN_SPEED = 0.01
MIN_ACCEL = 0.03

# OOD scoring warmup
WARMUP_OOD_STEPS = 5

# JSON poll steps
JSON_POLL_INTERVAL = 5

# Action smoothing
ACTION_SMOOTH_ALPHA = 0.7

# Safety workspace
SAFE_X = (0.20, 0.70)
SAFE_Y = (-0.65, -0.25)
SAFE_Z = (0.20, 0.55)

SCANNER_MEASUREMENT_RANGE = 0.260
XY_TOL = 0.015
Z_TOL = 0.005

# Caption throttling
CAPTION_XY_TOL = 0.30
CAPTION_Z_TOL = 0.30
CAPTION_STEP_INTERVAL = 10

# State normalization
MID_POS = np.array([0.45, -0.45, 0.375], dtype=np.float32)
SCALE_POS = np.array([0.25, 0.20, 0.175], dtype=np.float32)
VEL_SCALE = 0.2

CLEARANCE_MIN = 0.05
CLEARANCE_MAX = 0.50

# Recording mode controls
RECORD_FOR_PCA_VISION = False
RECORD_FOR_PCA_PPO = False

# Slope thresholds
VIS_SLOPE_THR = 0.10
PPO_SLOPE_THR = 0.10


# =============================== OOD / SAFETY ================================
ALPHA_SMOOTH = 0.25      # EWMA base alpha
ID_PERSIST = 5           # resume needs N consecutive ID steps

FUSED_PERSIST = 2        # fused OOD needs N consecutive steps
HARD_RATIO = 2.5         # immediate hard-stop in ratio-space
GATE_KAPPA = 10          # gating sharpness

RESUME_WARMUP_STEPS = 10


def set_pub_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 11,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "patch.linewidth": 0.5,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "figure.dpi": 100,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "savefig.format": "png",
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fancybox": False,
        "legend.edgecolor": "black",
        "legend.facecolor": "white",
        "legend.borderpad": 0.4,
        "legend.columnspacing": 1.0,
        "legend.handlelength": 1.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "text.usetex": False,
        "mathtext.fontset": "stix",
    })


set_pub_style()


# ====================== DEVICE & ROBOT SETUP ================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

rtde_c = RTDEControlInterface(ROBOT_IP)
rtde_r = RTDEReceiveInterface(ROBOT_IP)
print("[INFO] Connected to UR5e.")


# =============================== EWMA HELPERS ================================
smooth_vis_z = None     # EWMA in z-space
smooth_ppo_z = None
smooth_fused = None     # EWMA in ratio-space
smooth_vis_sigma = None
smooth_ppo_sigma = None

# Ground truth plot helpers
prev_vis_smooth = None
prev_ppo_smooth = None

vis_ood_marked = False
vis_id_marked = False
ppo_ood_marked = False
ppo_id_marked = False

# Track first valid data points to check initial OOD state
vis_first_valid_seen = False
ppo_first_valid_seen = False


def ewma_unc(prev, x, sigma, alpha0):
    """
    Uncertainty-aware EWMA: alpha = alpha0 / (1 + uncertainty), clipped.
    """
    if x is None:
        return prev
    if prev is None:
        return float(x)
    if sigma is None:
        sigma = 0.0
    unc = float(sigma)
    alpha = float(alpha0) / (1.0 + unc)
    return float(alpha * float(x) + (1.0 - alpha) * float(prev))


def softmax2(a, b):
    m = max(a, b)
    ea = np.exp(a - m)
    eb = np.exp(b - m)
    s = ea + eb + 1e-12
    return float(ea / s), float(eb / s)


# ======================= OOD SCORE PLOTS ==========================
fig_ood, (ax_vis, ax_ppo) = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True)
plt.subplots_adjust(hspace=0.60, left=0.12, right=0.95, top=0.95, bottom=0.1)
plt.show(block=False)

# ======================= LIVE EVENT MARKERS ==========================
_marker_labels_seen = set()

def mark_event_on_plots(event_time, kind, target=""):
    global _marker_labels_seen

    if target == "vis":
        axes = [ax_vis]
    elif target == "ppo":
        axes = [ax_ppo]
    elif target == "both":
        axes = [ax_vis, ax_ppo]
    else:
        axes = [ax_vis, ax_ppo]

    if kind == "OOD_START":
        ls = "-."
        lw = 1.8
        alpha = 1.0
        color = "#1976D2"
        label = "OOD Onset"

    elif kind == "ID_START":
        ls = ":"
        lw = 1.8
        alpha = 1.0
        color = "#388E3C"
        label = "OOD Cleared"

    else:
        ls = "-"
        lw = 1.2
        label = kind

    add_label = label not in _marker_labels_seen
    lab = label if add_label else None

    for ax in axes:
        if kind in ["OOD_START", "ID_START"]:
            ax.axvline(event_time, linestyle=ls, linewidth=lw, alpha=alpha, color=color, label=lab, zorder=5)
        else:
            ax.axvline(event_time, linestyle=ls, linewidth=lw, alpha=alpha, label=lab, zorder=5)

    if add_label:
        _marker_labels_seen.add(label)
        for ax in axes:
            ax.legend(loc="upper right", frameon=True, fancybox=False, shadow=False, edgecolor="black", framealpha=0.9)
            if hasattr(ax, 'legend_') and ax.legend_ is not None:
                ax.legend_.set_visible(True)

    fig_ood.canvas.draw_idle()

# Academic-style plot setup
ax_vis.set_title("Vision Out-of-Distribution", fontweight="normal", pad=8)
ax_vis.set_xlabel("Time (s)", fontweight="normal")
ax_vis.set_ylabel("Standardized SWD (z-score)", fontweight="normal")
ax_vis.tick_params(labelbottom=True)
ax_vis.grid(False)
ax_vis.set_axisbelow(True)
(vis_raw_line,) = ax_vis.plot([], [], linewidth=1.2, alpha=0.6, color="#FF7F00", label="Raw signal", zorder=2)
(vis_smooth_line,) = ax_vis.plot([], [], linewidth=2.0, color="#2E7D32", label="Smoothed signal", zorder=3)
vis_thr_line_raw = None

ax_ppo.set_title("Proprioceptive Out-of-Distribution", fontweight="normal", pad=8)
ax_ppo.set_xlabel("Time (s)", fontweight="normal")
ax_ppo.set_ylabel("Standardized SWD (z-score)", fontweight="normal")
ax_ppo.grid(False)
ax_ppo.set_axisbelow(True)
(ppo_raw_line,) = ax_ppo.plot([], [], linewidth=1.2, alpha=0.6, color="#FF7F00", label="Raw signal", zorder=2)
(ppo_smooth_line,) = ax_ppo.plot([], [], linewidth=2.0, color="#2E7D32", label="Smoothed signal", zorder=3)
ppo_thr_line_raw = None

time_idx = []
vis_raw_scores, vis_raw_sigmas, vis_smooth_scores, vis_smooth_sigma_scores = [], [], [], []
ppo_raw_scores, ppo_raw_sigmas, ppo_smooth_scores, ppo_smooth_sigma_scores = [], [], [], []
vis_band = None
ppo_band = None
vis_band_smooth = None
ppo_band_smooth = None
vis_ood_region = None
ppo_ood_region = None


# ====================== LOAD OOD MODELS ================================

vision_ood = None
if not args.record_normal_visual_latents:
    vision_npz = "OOD/VisualOOD/vision_ood_model.npz"
    if os.path.exists(vision_npz):
        vision_ood = FlorenceVisionOOD(vision_npz, device=DEVICE)
        print(vision_ood)
        vis_thr_line_raw = ax_vis.axhline(y=float(vision_ood.threshold), color="#D32F2F", linestyle="--", linewidth=1.5, label="OOD Threshold", zorder=4)
        ax_vis.legend(loc="upper right", frameon=True, fancybox=False, shadow=False, edgecolor="black", framealpha=0.9)
    else:
        print("[WARN] Vision OOD model not found. Run fit_vision_ood.py first.")

ppo_ood = None
ppo_npz = "OOD/ProprioOOD/ppo_swd_ood_model.npz"
if not args.record_normal_ppo_latents:
    if os.path.exists(ppo_npz):
        ppo_ood = PPOControlOOD(ppo_npz, device=DEVICE)
        print(ppo_ood)
        ppo_thr_line_raw = ax_ppo.axhline(y=float(ppo_ood.threshold), color="#D32F2F", linestyle="--", linewidth=1.5, label="OOD Threshold", zorder=4)
        ax_ppo.legend(loc="upper right", frameon=True, fancybox=False, shadow=False, edgecolor="black", framealpha=0.9)
    else:
        print("[WARN] PPO OOD model not found. Run fit_ppo_swd_ood.py first.")

model, processor, vision_encoder, device = load_florence_vision()
print("[INFO] Florence2 Vision Encoder loaded.")

recorded_visual_latents = []
recorded_ppo_latents = []
manual_events = []
last_caption = ""


# ====================== REALSENSE CAMERA SETUP ===============================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
print("[INFO] RealSense RGB camera started.")


# ====================== LOAD OBJECT & OBSTACLE ===============================
def load_positions_from_json(path=JSON_3D_PATH):
    x_off = 0.030
    y_off = 0.015
    with open(path, "r") as f:
        data = json.load(f)

    raw_target = data.get("target", None)
    if raw_target is None:
        raise ValueError("Target missing in positions_3d.json!")

    Xb, Yb, _, object_height = raw_target
    Xb -= x_off
    Yb -= y_off

    goal_z = float(object_height) + SCANNER_MEASUREMENT_RANGE
    goal_pos = np.array([Xb, Yb, goal_z], dtype=np.float32)

    obstacles = {}
    obs_dict = data.get("obstacles", {})
    for key, raw_obs in obs_dict.items():
        Xo, Yo, _, height_o = raw_obs
        Xo -= x_off
        Yo -= y_off
        obs_z = float(height_o) + SCANNER_MEASUREMENT_RANGE
        obstacles[key] = np.array([Xo, Yo, obs_z], dtype=np.float32)

    return goal_pos, obstacles


GOAL_POS, OBSTACLE_POS = load_positions_from_json()


# ====================== LOAD PPO POLICY ===============================
state_dim = 10
action_dim = 3
emb_size = 128
action_std = 0.2

policy = ActorCritic(DEVICE, state_dim, emb_size, action_dim, action_std).to(DEVICE)
policy.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
policy.eval()
print(f"[INFO] Loaded trained policy: {MODEL_PATH}")


# ======================= FLORENCE LATENT EXTRACTION ==========================
def get_florence_latent(do_caption: bool = False):
    global last_caption

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None, None

    img = np.asanyarray(color_frame.get_data())  # BGR
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    inputs = processor(images=pil, text="<CAPTION>", return_tensors="pt").to(device)
    px = inputs["pixel_values"]

    if do_caption:
        with torch.no_grad():
            generated = model.generate(**inputs, max_length=64, do_sample=False)
        caption = processor.batch_decode(generated, skip_special_tokens=True)[0]
        last_caption = caption
    else:
        caption = last_caption

    with torch.no_grad():
        image_tokens, patch_feats = vision_encoder(px)

    latent = image_tokens.mean(dim=1).squeeze(0)  # (768,)

    patch_feats_cpu = patch_feats[0].detach().cpu().numpy()
    mag = np.linalg.norm(patch_feats_cpu, axis=-1)
    mag = mag - mag.min()
    if mag.max() > 0:
        mag = mag / mag.max()

    heatmap = cv2.resize(mag, (img.shape[1], img.shape[0]))
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    if caption:
        cv2.putText(overlay, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return latent, overlay


# ========================== ROBOT STATE ======================================
def get_robot_state():
    tcp_pose = np.array(rtde_r.getActualTCPPose(), dtype=np.float32)
    tcp_pos = tcp_pose[:3]
    tcp_vel = np.array(rtde_r.getActualTCPSpeed(), dtype=np.float32)[:3]
    goal_pos = GOAL_POS

    tcp_pos_norm = np.clip((tcp_pos - MID_POS) / SCALE_POS, -1.0, 1.0)
    goal_pos_norm = np.clip((goal_pos - MID_POS) / SCALE_POS, -1.0, 1.0)
    tcp_vel_norm = np.clip(tcp_vel / VEL_SCALE, -1.0, 1.0)

    if OBSTACLE_POS:
        d = [np.linalg.norm(tcp_pos - p) for p in OBSTACLE_POS.values()]
        min_clearance = float(min(d))
    else:
        min_clearance = CLEARANCE_MAX

    c = np.clip(min_clearance, CLEARANCE_MIN, CLEARANCE_MAX)
    clearance_norm = 2 * (c - CLEARANCE_MIN) / (CLEARANCE_MAX - CLEARANCE_MIN) - 1.0

    state_vec = np.concatenate((tcp_pos_norm, goal_pos_norm, tcp_vel_norm, [clearance_norm]))
    return state_vec, min_clearance, clearance_norm, tcp_pos


# ========================= ACTION APPLICATION ================================
prev_action = np.zeros(3, dtype=np.float32)


def apply_action(action, step_idx, min_clearance, clearance_norm, resume_ctr):
    global prev_action

    action = np.clip(action, -1.0, 1.0)
    alpha = ACTION_SMOOTH_ALPHA
    filtered = alpha * action + (1.0 - alpha) * prev_action
    prev_action = filtered.copy()
    action = filtered

    startup_alpha = min(1.0, step_idx / float(WARMUP_STEPS))
    resume_alpha = min(1.0, resume_ctr / float(RESUME_WARMUP_STEPS))
    warm_alpha = min(startup_alpha, resume_alpha)
    action *= warm_alpha

    cur_pose = np.array(rtde_r.getActualTCPPose(), dtype=np.float32)
    tcp_pos = cur_pose[:3]

    diff = GOAL_POS - tcp_pos
    xy_err = np.linalg.norm(diff[:2])
    z_err = abs(diff[2])

    if z_err < 0.005:
        action[2] = 0.0
        scale_xy, scale_z = XY_SCALE_NEAR, 0.0
    elif z_err < 0.02:
        scale_xy, scale_z = XY_SCALE_NEAR, Z_SCALE_NEAR
    else:
        scale_xy, scale_z = XY_SCALE_FAR, Z_SCALE_FAR

    direction = -1 if tcp_pos[2] > GOAL_POS[2] else 1

    raw_step = np.array([action[0] * scale_xy, action[1] * scale_xy, action[2] * scale_z * direction], dtype=np.float32)

    max_abs = np.max(np.abs(raw_step)) + 1e-6
    desired_step = raw_step / max_abs
    delta_pos = desired_step * MAX_DPOS

    new_pos = tcp_pos + delta_pos
    new_pos[0] = np.clip(new_pos[0], *SAFE_X)
    new_pos[1] = np.clip(new_pos[1], *SAFE_Y)
    new_pos[2] = np.clip(new_pos[2], *SAFE_Z)

    target_pose = cur_pose.copy()
    target_pose[:3] = new_pos
    target_pose[3:] = [3.1415, 0.0, 0.0]

    speed_cmd = MIN_SPEED + warm_alpha * (MAX_SPEED - MIN_SPEED)
    accel_cmd = MIN_ACCEL + warm_alpha * (MAX_ACCEL - MIN_ACCEL)

    rtde_c.servoL(
        target_pose.tolist(),
        speed_cmd,
        accel_cmd,
        SERVO_TIME,
        SERVO_LOOKAHEAD,
        SERVO_GAIN,
    )

    time.sleep(SERVO_TIME)

    new_tcp_pos = np.array(rtde_r.getActualTCPPose(), dtype=np.float32)[:3]
    new_diff = GOAL_POS - new_tcp_pos
    xy_err = np.linalg.norm(new_diff[:2])
    z_err = abs(new_diff[2])
    total_err = np.linalg.norm(new_diff)

    tcp_vel = np.array(rtde_r.getActualTCPSpeed(), dtype=np.float32)[:3]
    dt = SERVO_TIME

    add_state(
        step_idx,
        action,
        new_tcp_pos,
        tcp_vel,
        GOAL_POS,
        xy_err,
        z_err,
        total_err,
        dt,
        clearance=min_clearance,
        c_norm=clearance_norm,
    )

    print(
        f"Step {step_idx:03d}: Action={np.round(action, 4)} | TCP={np.round(new_tcp_pos, 4)} "
        f"| XY_err={xy_err:.4f} | Z_err={z_err:.4f} | Dist={total_err:.4f} "
        f"| clearance={min_clearance*100:.1f} | c_norm={clearance_norm:.3f} "
        f"| v_limit={speed_cmd:.3f} | a_limit={accel_cmd:.3f}"
    )

    return xy_err, z_err


# ============================ MAIN LOOP ======================================
def run_policy():
    global GOAL_POS, OBSTACLE_POS
    global smooth_vis_z, smooth_ppo_z, smooth_fused, smooth_vis_sigma, smooth_ppo_sigma
    global vis_band, ppo_band, prev_action
    global prev_vis_smooth, prev_ppo_smooth, vis_ood_marked, vis_id_marked, ppo_ood_marked, ppo_id_marked
    global vis_ood_region, ppo_ood_region
    global vis_first_valid_seen, ppo_first_valid_seen

    print(f"[INFO] Moving toward goal {GOAL_POS} until XY<{XY_TOL} and Z<{Z_TOL} ...")

    step = 0
    fused_streak = 0

    PAUSE_MODE = False
    ID_STREAK = 0
    PAUSE_SLEEP = 1 / 30.0

    resume_ctr = 0
    start_time = time.time()
    try:
        last_json_mtime = os.stat(JSON_3D_PATH).st_mtime_ns
    except Exception:
        last_json_mtime = 0

    while True:
        # LIVE JSON MONITOR
        if (step % JSON_POLL_INTERVAL) == 0:
            try:
                mtime = os.stat(JSON_3D_PATH).st_mtime_ns
                if mtime != last_json_mtime:
                    last_json_mtime = mtime
                    new_goal, new_obs = load_positions_from_json(JSON_3D_PATH)
                    GOAL_POS = new_goal
                    OBSTACLE_POS = new_obs
                    print("[JSON] positions_3d.json updated -> refreshed GOAL/OBSTACLES.")

            except Exception as e:
                print(f"[JSON] Could not read {JSON_3D_PATH}: {e}")

        state, min_clear, clearance_norm, tcp_pos = get_robot_state()
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            ppo_latent = policy.get_latent(state_t).squeeze(0)

        diff = GOAL_POS - tcp_pos
        xy_err_est = np.linalg.norm(diff[:2])
        z_err_est = abs(diff[2])

        do_caption = (
            xy_err_est < CAPTION_XY_TOL and
            z_err_est < CAPTION_Z_TOL and
            (step % CAPTION_STEP_INTERVAL == 0)
        )

        vis_latent, overlay = get_florence_latent(do_caption=do_caption)

        # -------- RAW OOD (z + sigma) --------
        vis_z = None
        vis_sigma = None
        ppo_z = None
        ppo_sigma = None

        if (not RECORDING_MODE) and (step >= WARMUP_OOD_STEPS):
            if (vis_latent is not None) and (vision_ood is not None):
                vis_z, vis_sigma = vision_ood.update_and_score(vis_latent)
            if (ppo_latent is not None) and (ppo_ood is not None):
                ppo_z, ppo_sigma = ppo_ood.update_and_score(ppo_latent)

            global smooth_vis_z, smooth_ppo_z, smooth_vis_sigma, smooth_ppo_sigma
            smooth_vis_z = ewma_unc(smooth_vis_z, vis_z, vis_sigma, ALPHA_SMOOTH)
            smooth_vis_sigma = ewma_unc(smooth_vis_sigma, vis_sigma, 0.0, ALPHA_SMOOTH)

            smooth_ppo_z = ewma_unc(smooth_ppo_z, ppo_z, ppo_sigma, ALPHA_SMOOTH)
            smooth_ppo_sigma = ewma_unc(smooth_ppo_sigma, ppo_sigma, 0.0, ALPHA_SMOOTH)

            elapsed_time = time.time() - start_time

            # Vision: Check if first valid score is already above threshold
            if (not vis_first_valid_seen) and (smooth_vis_z is not None) and (vision_ood is not None):
                vis_first_valid_seen = True
                if smooth_vis_z > float(vision_ood.threshold):
                    mark_event_on_plots(elapsed_time, kind="OOD_START", target="vis")
                    global vis_ood_marked, vis_id_marked
                    vis_ood_marked = True
                    vis_id_marked = False

            # PPO: Check if first valid score is already above threshold
            if (not ppo_first_valid_seen) and (smooth_ppo_z is not None) and (ppo_ood is not None):
                ppo_first_valid_seen = True
                if smooth_ppo_z > float(ppo_ood.threshold):
                    mark_event_on_plots(elapsed_time, kind="OOD_START", target="ppo")
                    global ppo_ood_marked, ppo_id_marked
                    ppo_ood_marked = True
                    ppo_id_marked = False

            # ---------- SLOPE-BASED OOD ONSET ----------
            if step >= WARMUP_OOD_STEPS:

                # Vision slope
                global prev_vis_smooth, prev_ppo_smooth
                if (prev_vis_smooth is not None) and (smooth_vis_z is not None):
                    vis_slope = smooth_vis_z - prev_vis_smooth

                    # OOD onset (rising fast)
                    if (not vis_ood_marked) and (vis_slope > VIS_SLOPE_THR):
                        mark_event_on_plots(elapsed_time, kind="OOD_START", target="vis")
                        vis_ood_marked = True
                        vis_id_marked = False

                    # ID onset (falling fast)
                    elif (not vis_id_marked) and (vis_slope < -VIS_SLOPE_THR):
                        mark_event_on_plots(elapsed_time, kind="ID_START", target="vis")
                        vis_id_marked = True
                        vis_ood_marked = False

                # PPO slope
                if prev_ppo_smooth is not None and smooth_ppo_z is not None:
                    ppo_slope = smooth_ppo_z - prev_ppo_smooth

                    if (not ppo_ood_marked) and (ppo_slope > PPO_SLOPE_THR):
                        mark_event_on_plots(elapsed_time, kind="OOD_START", target="ppo")
                        ppo_ood_marked = True
                        ppo_id_marked = False

                    elif (not ppo_id_marked) and (ppo_slope < -PPO_SLOPE_THR):
                        mark_event_on_plots(elapsed_time, kind="ID_START", target="ppo")
                        ppo_id_marked = True
                        ppo_ood_marked = False

                prev_vis_smooth = smooth_vis_z
                prev_ppo_smooth = smooth_ppo_z

        # -------- Robust smoothed z (LCB) --------
        robust_vis_z = None
        robust_ppo_z = None

        if (smooth_vis_z is not None) and (smooth_vis_sigma is not None):
            robust_vis_z = float(smooth_vis_z) - float(smooth_vis_sigma)

        if (smooth_ppo_z is not None) and (smooth_ppo_sigma is not None):
            robust_ppo_z = float(smooth_ppo_z) - float(smooth_ppo_sigma)

        # -------- Plot --------
        elapsed_time = time.time() - start_time
        time_idx.append(elapsed_time)
        vis_raw_scores.append(vis_z if vis_z is not None else np.nan)
        vis_raw_sigmas.append(vis_sigma if vis_sigma is not None else 0.0)
        vis_smooth_scores.append(smooth_vis_z if smooth_vis_z is not None else np.nan)
        vis_smooth_sigma_scores.append(smooth_vis_sigma if smooth_vis_sigma is not None else np.nan)

        ppo_raw_scores.append(ppo_z if ppo_z is not None else np.nan)
        ppo_raw_sigmas.append(ppo_sigma if ppo_sigma is not None else 0.0)
        ppo_smooth_scores.append(smooth_ppo_z if smooth_ppo_z is not None else np.nan)
        ppo_smooth_sigma_scores.append(smooth_ppo_sigma if smooth_ppo_sigma is not None else np.nan)

        try:
            vis_raw_line.set_data(time_idx, vis_raw_scores)
            vis_smooth_line.set_data(time_idx, vis_smooth_scores)
            ppo_raw_line.set_data(time_idx, ppo_raw_scores)
            ppo_smooth_line.set_data(time_idx, ppo_smooth_scores)

            global vis_band, ppo_band, vis_band_smooth, ppo_band_smooth, vis_ood_region, ppo_ood_region

            if vis_ood_region is not None:
                vis_ood_region.remove()
                vis_ood_region = None

            if ppo_ood_region is not None:
                ppo_ood_region.remove()
                ppo_ood_region = None

            if vis_band_smooth is not None:
                vis_band_smooth.remove()
                vis_band_smooth = None

            if ppo_band_smooth is not None:
                ppo_band_smooth.remove()
                ppo_band_smooth = None

            vis_s_sig = np.array(vis_smooth_sigma_scores, dtype=np.float64)
            vis_s = np.array(vis_smooth_scores, dtype=np.float64)

            t = np.asarray(time_idx, dtype=np.float64)
            mask_v = ~np.isnan(vis_s) & ~np.isnan(vis_s_sig)
            if np.any(mask_v):
                lower_sv = (vis_s - vis_s_sig)[mask_v]
                upper_sv = (vis_s + vis_s_sig)[mask_v]
                vis_band_smooth = ax_vis.fill_between(t[mask_v], lower_sv, upper_sv, alpha=0.35, color='#64B5F6', zorder=1)

            if vision_ood is not None:
                vis_threshold = float(vision_ood.threshold)
                mask_ood_v = mask_v & (vis_s > vis_threshold)
                if np.any(mask_ood_v):
                    vis_ood_region = ax_vis.fill_between(
                        t[mask_ood_v],
                        vis_threshold,
                        vis_s[mask_ood_v],
                        alpha=0.20,
                        color='#EF5350',
                        zorder=1
                    )

            ppo_s_sig = np.array(ppo_smooth_sigma_scores, dtype=np.float64)
            ppo_s = np.array(ppo_smooth_scores, dtype=np.float64)

            mask_p = ~np.isnan(ppo_s) & ~np.isnan(ppo_s_sig)
            if np.any(mask_p):
                lower_sp = (ppo_s - ppo_s_sig)[mask_p]
                upper_sp = (ppo_s + ppo_s_sig)[mask_p]
                ppo_band_smooth = ax_ppo.fill_between(t[mask_p], lower_sp, upper_sp, alpha=0.35, color='#64B5F6', zorder=1)

            if ppo_ood is not None:
                ppo_threshold = float(ppo_ood.threshold)
                mask_ood_p = mask_p & (ppo_s > ppo_threshold)
                if np.any(mask_ood_p):
                    ppo_ood_region = ax_ppo.fill_between(
                        t[mask_ood_p],
                        ppo_threshold,
                        ppo_s[mask_ood_p],
                        alpha=0.20,
                        color='#EF5350',
                        zorder=1
                    )

            ax_vis.relim()
            ax_vis.autoscale_view()
            ax_ppo.relim()
            ax_ppo.autoscale_view()

            ax_vis.legend(loc="upper right", frameon=True, fancybox=False, shadow=False, edgecolor="black", framealpha=0.9)
            ax_ppo.legend(loc="upper right", frameon=True, fancybox=False, shadow=False, edgecolor="black", framealpha=0.9)

            fig_ood.canvas.draw_idle()
            plt.pause(0.001)

        except Exception as e:
            print(f"[WARN] OOD plot update failed: {e}")

        # -------- Overlay --------
        if overlay is not None:
            cv2.imshow("Florence2 Latent State Heatmap + Caption", overlay)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[INFO] 'q' pressed, stopping.")
                rtde_c.servoStop()
                time.sleep(0.3)
                return

        # -------- Recording mode --------
        if args.record_normal_visual_latents and (vis_latent is not None):
            z = vis_latent.detach().cpu().numpy().astype(np.float32)
            if not RECORD_FOR_PCA_VISION:
                z = z / (np.linalg.norm(z) + 1e-12)
            recorded_visual_latents.append(z)

        if args.record_normal_ppo_latents and (ppo_latent is not None):
            z = ppo_latent.detach().cpu().numpy().astype(np.float32)
            if not RECORD_FOR_PCA_PPO:
                z = z / (np.linalg.norm(z) + 1e-12)
            recorded_ppo_latents.append(z)

        # -------- Safety gate: HARD + fused-streak --------
        if (not RECORDING_MODE) and (step >= WARMUP_OOD_STEPS):
            v_ratio = None
            p_ratio = None
            v_sig_ratio = None
            p_sig_ratio = None

            if (vision_ood is not None) and (robust_vis_z is not None) and (vis_sigma is not None):
                v_thr = float(vision_ood.threshold) + 1e-12
                v_ratio = float(robust_vis_z) / v_thr
                v_sig_ratio = float(smooth_vis_sigma) / v_thr

            if (ppo_ood is not None) and (robust_ppo_z is not None) and (ppo_sigma is not None):
                p_thr = float(ppo_ood.threshold) + 1e-12
                p_ratio = float(robust_ppo_z) / p_thr
                p_sig_ratio = float(smooth_ppo_sigma) / p_thr

            hard_stop = False
            if (v_ratio is not None) and (v_ratio >= HARD_RATIO):
                hard_stop = True
            if (p_ratio is not None) and (p_ratio >= HARD_RATIO):
                hard_stop = True

            a_v = None
            a_p = None
            if (v_ratio is not None) and (p_ratio is not None):
                a_v, a_p = softmax2(GATE_KAPPA * (v_ratio - 1.0),
                                    GATE_KAPPA * (p_ratio - 1.0))
                fused = a_v * v_ratio + a_p * p_ratio
            else:
                fused = v_ratio if (p_ratio is None) else p_ratio

            sig_candidates = [x for x in [v_sig_ratio, p_sig_ratio] if x is not None]
            fused_sigma = max(sig_candidates) if len(sig_candidates) else 0.0

            global smooth_fused
            smooth_fused = ewma_unc(smooth_fused, fused, fused_sigma, ALPHA_SMOOTH)
            fused_dec = smooth_fused if (smooth_fused is not None) else fused
            fused_robust = (fused_dec - fused_sigma) if (fused_dec is not None) else None

            if (fused_robust is not None) and (fused_robust > 1.0):
                fused_streak += 1
            else:
                fused_streak = 0

            pause_trigger = bool(hard_stop) or (fused_streak >= FUSED_PERSIST)
            if pause_trigger and (not PAUSE_MODE):
                print("\n====================== SAFETY PAUSE ======================")
                print(f"[PAUSE] hard_stop={hard_stop} v_ratio={v_ratio} p_ratio={p_ratio} "
                      f"fused={fused} smooth_fused={smooth_fused} fused_robust={fused_robust}")
                rtde_c.servoStop()
                time.sleep(0.3)
                PAUSE_MODE = True
                ID_STREAK = 0

        # -------- Pause/resume loop --------
        if PAUSE_MODE:
            try:
                mtime = os.stat(JSON_3D_PATH).st_mtime_ns
                if mtime != last_json_mtime:
                    last_json_mtime = mtime
                    new_goal, new_obs = load_positions_from_json(JSON_3D_PATH)
                    GOAL_POS = new_goal
                    OBSTACLE_POS = new_obs
                    print("[PAUSE] positions_3d.json updated -> refreshed GOAL/OBSTACLES.")

            except Exception as e:
                print(f"[PAUSE] Could not read {JSON_3D_PATH}: {e}")

            ppo_safe = (ppo_ood is None) or (robust_ppo_z is not None and robust_ppo_z <= float(ppo_ood.threshold))
            vis_safe = (vision_ood is None) or (robust_vis_z is not None and robust_vis_z <= float(vision_ood.threshold))
            is_id = bool(ppo_safe and vis_safe)

            ID_STREAK = ID_STREAK + 1 if is_id else 0
            print(f"[PAUSE] is_id={is_id} ID_STREAK={ID_STREAK}/{ID_PERSIST}")

            if ID_STREAK >= ID_PERSIST:
                print(f"[RESUME] Back to ID for {ID_PERSIST} steps. Resuming motion.")
                PAUSE_MODE = False
                ID_STREAK = 0
                prev_action[:] = 0.0
                resume_ctr = 0
            else:
                time.sleep(PAUSE_SLEEP)
                step += 1
                continue

        # -------- PPO action --------
        resume_ctr += 1
        with torch.no_grad():
            action = policy.actor(state_t).cpu().numpy().flatten()

        xy_err, z_err = apply_action(action, step, min_clear, clearance_norm, resume_ctr)

        if xy_err < XY_TOL and z_err < Z_TOL:
            print(f"[SUCCESS] Reached goal at step {step} → XY_err={xy_err:.4f}, Z_err={z_err:.4f}")
            print(f"[TCP Pose] {np.round(rtde_r.getActualTCPPose(), 5)}")
            break

        if step > 1000:
            print("[WARN] Max steps reached, stopping.")
            break

        step += 1

    print("[INFO] Finished policy execution.")


# ============================= ENTRY POINT ===================================
if __name__ == "__main__":
    try:
        run_policy()

    except KeyboardInterrupt:
        print("\n[ABORTED] Stopped by user.")

    finally:
        try:
            rtde_c.stopScript()
        except Exception:
            pass
        try:
            pipeline.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # Save latents if recording
        if args.record_normal_visual_latents and len(recorded_visual_latents) > 0:
            os.makedirs("OOD/VisualOOD", exist_ok=True)
            save_path = "OOD/VisualOOD/vision_latents_normal.npy" if RECORD_FOR_PCA_VISION else "OOD/VisualOOD/vision_latents_L2_normal.npy"
            new_arr = np.stack(recorded_visual_latents, axis=0)

            if os.path.exists(save_path):
                try:
                    old_arr = np.load(save_path)
                    combined = np.concatenate([old_arr, new_arr], axis=0)
                    np.save(save_path, combined)
                    print(f"[INFO] Appended {new_arr.shape[0]} visual latents. Total now: {combined.shape[0]}")
                except Exception:
                    np.save(save_path, new_arr)
            else:
                np.save(save_path, new_arr)
                print(f"[INFO] Saved {new_arr.shape[0]} visual latents to new file.")

        if args.record_normal_ppo_latents and len(recorded_ppo_latents) > 0:
            os.makedirs("OOD/ProprioOOD", exist_ok=True)
            save_path = "OOD/ProprioOOD/ppo_latents_normal.npy" if RECORD_FOR_PCA_PPO else "OOD/ProprioOOD/ppo_latents_L2_normal.npy"
            new_arr = np.stack(recorded_ppo_latents, axis=0)

            if os.path.exists(save_path):
                try:
                    old_arr = np.load(save_path)
                    combined = np.concatenate([old_arr, new_arr], axis=0)
                    np.save(save_path, combined)
                    print(f"[INFO] Appended {new_arr.shape[0]} PPO latents. Total now: {combined.shape[0]}")
                except Exception:
                    np.save(save_path, new_arr)
            else:
                np.save(save_path, new_arr)
                print(f"[INFO] Saved {new_arr.shape[0]} PPO latents to new file.")

        print("[INFO] Robot script stopped safely.")

        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            os.makedirs("OOD", exist_ok=True)
            save_path = f"OOD/ood_scores_{timestamp}.png"
            fig_ood.savefig(save_path, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
            print(f"[INFO] OOD subplot figure saved to: {save_path}")
        except Exception as e:
            print(f"[WARN] Failed to save OOD subplot figure: {e}")

