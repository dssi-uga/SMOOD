import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

LOG_DIR = "sim2real_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Global logs
TCP_HISTORY = []
VEL_HISTORY = []
ACC_HISTORY = []
STEP_LOGS = []


def add_state(step, action, tcp_pos, tcp_vel, goal, xy_err, z_err, dist, dt, clearance=None, c_norm=None):
    """
    Add full state including position, velocity, acceleration.
    Acceleration is computed automatically.
    """
    tcp_pos = np.array(tcp_pos).copy()
    tcp_vel = np.array(tcp_vel).copy()

    if len(VEL_HISTORY) == 0:
        acc = np.zeros(3)
    else:
        acc = (tcp_vel - VEL_HISTORY[-1]) / max(dt, 1e-6)

    TCP_HISTORY.append(tcp_pos)
    VEL_HISTORY.append(tcp_vel)
    ACC_HISTORY.append(acc)

    log_entry = {
        "Step": step,
        "Action": action.tolist(),
        "TCP_X": float(tcp_pos[0]),
        "TCP_Y": float(tcp_pos[1]),
        "TCP_Z": float(tcp_pos[2]),
        "VEL_X": float(tcp_vel[0]),
        "VEL_Y": float(tcp_vel[1]),
        "VEL_Z": float(tcp_vel[2]),
        "ACC_X": float(acc[0]),
        "ACC_Y": float(acc[1]),
        "ACC_Z": float(acc[2]),
        "Goal_X": float(goal[0]),
        "Goal_Y": float(goal[1]),
        "Goal_Z": float(goal[2]),
        "XY_err": float(xy_err),
        "Z_err": float(z_err),
        "Dist": float(dist),
    }

    if clearance is not None:
        log_entry["Clearance"] = float(clearance)
    if c_norm is not None:
        log_entry["C_norm"] = float(c_norm)

    STEP_LOGS.append(log_entry)


def save_logs_to_excel():
    if not STEP_LOGS:
        print("[WARN] No logs to save.")
        return
    df = pd.DataFrame(STEP_LOGS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, f"trajectory_logs_{timestamp}.xlsx")
    df.to_excel(path, index=False)
    print(f"[INFO] Excel log saved to: {path}")


def save_trajectory_csv():
    if len(TCP_HISTORY) < 2:
        print("[WARN] Not enough TCP points for CSV.")
        return

    hist = np.array(TCP_HISTORY)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, f"tcp_trajectory_{timestamp}.csv")

    pd.DataFrame(hist, columns=["X", "Y", "Z"]).to_csv(path, index=False)
    print(f"[INFO] CSV saved to: {path}")


def plot_trajectory():
    if len(TCP_HISTORY) < 2:
        print("[WARN] Not enough points to plot.")
        return

    tcp = np.array(TCP_HISTORY)
    vel = np.array(VEL_HISTORY)
    acc = np.array(ACC_HISTORY)

    steps = np.arange(len(tcp))

    fig = plt.figure(figsize=(16, 20))

    # Position X,Y,Z
    for i, label in enumerate(["X", "Y", "Z"]):
        ax = fig.add_subplot(4, 3, i + 1)
        ax.plot(steps, tcp[:, i], marker='o')
        ax.set_title(f"{label} Position (m)")
        ax.set_xlabel("Step")
        ax.grid(True)

    # Velocity X,Y,Z
    for i, label in enumerate(["X", "Y", "Z"]):
        ax = fig.add_subplot(4, 3, i + 4)
        ax.plot(steps, vel[:, i], marker='o')
        ax.set_title(f"{label} Velocity (m/s)")
        ax.set_xlabel("Step")
        ax.grid(True)

    # Acceleration X,Y,Z
    for i, label in enumerate(["X", "Y", "Z"]):
        ax = fig.add_subplot(4, 3, i + 7)
        ax.plot(steps, acc[:, i], marker='o')
        ax.set_title(f"{label} Acceleration (m/s²)")
        ax.set_xlabel("Step")
        ax.grid(True)

    # 3D Trajectory
    ax = fig.add_subplot(4, 3, 10, projection='3d')
    ax.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2], marker='o')
    ax.scatter(tcp[0, 0], tcp[0, 1], tcp[0, 2], c='green', s=80, label="Start")
    ax.scatter(tcp[-1, 0], tcp[-1, 1], tcp[-1, 2], c='red', s=80, label="End")
    ax.set_title("3D TCP Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(LOG_DIR, f"trajectory_plot_{timestamp}.png")
    plt.savefig(save_path)
    print(f"[INFO] Trajectory plot saved: {save_path}")

    plt.show()

