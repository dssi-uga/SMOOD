## SMOOD: Safe Model-Based Out-of-Distribution Detection for UR5e Scanning

This folder contains a self-contained subset of the full project that you can publish and run from GitHub.  
It covers:

- **PyBullet simulation** of a UR5e scanning a cube on a table.
- **PPO training** for Cartesian control of the end-effector.
- **Florence-2 based vision** for object detection and visual OOD monitoring.
- **Proprioceptive OOD monitoring** for the PPO latent state.
- **Sim-to-real execution script** that uses the trained policy and OOD models on a real UR5e with an RGB‑D camera.

All paths are **relative** and no machine-specific absolute paths are exposed.

---

## Repository layout (relevant files)

- `SMOOD/`
  - `core/`
    - `gym_env.py` &mdash; PyBullet UR5e environment (`ur5GymEnv`).
    - `ppo.py` &mdash; PPO implementation and `ActorCritic` network (also exposes `get_latent` for OOD).
  - `training/`
    - `train_rl.py` &mdash; PPO training loop for the simulation environment.
  - `vision/`
    - `florence_vision.py` &mdash; Florence-2 model + vision encoder wrapper (with optional LoRA adapter).
    - `vision_ood_detector.py` &mdash; Visual OOD detector (sliced Wasserstein distance with uncertainty).
  - `ood/`
    - `ppo_ood_detector.py` &mdash; Proprioceptive OOD detector for PPO latents.
    - `fit_vision_ood.py` &mdash; Fits OOD model for the vision latent space from recorded latents.
    - `fit_ppo_swd_ood.py` &mdash; Fits OOD model for the PPO latent space from recorded latents.
  - `sim2real_runner/`
    - `sim2real.py` &mdash; Real-robot execution + online OOD monitoring and safety.
    - `plot_trajectory.py` &mdash; Logging and plotting utilities for sim-to-real trajectories.
  - `CV/`
    - `inference_OBD.py` &mdash; Florence-2 based object detection (image or live RealSense).
    - `2d_to_3d.py` &mdash; Converts 2D detections + depth to 3D positions in the robot base frame.
    - `positions_3d.json` &mdash; Example output of `2d_to_3d.py` (goal and obstacle positions).
  - `requirements.txt` &mdash; Python dependencies for this subset.

Additionally, at the **project root** (one level above `SMOOD/`) you must have:

- `ur_e_description/` &mdash; URDF + meshes for UR5e and the table:
  - `ur_e_description/urdf/ur5e.urdf`
  - `ur_e_description/urdf/table.urdf`
  - `ur_e_description/meshes/...` (robot and table meshes)
- (Optional but recommended) `saved_rl_models/` &mdash; trained PPO checkpoints (e.g. `model_precise.pth`).
- (Optional) OOD model artifacts:
  - `OOD/VisualOOD/vision_latents_L2_normal.npy`
  - `OOD/VisualOOD/vision_ood_model.npz`
  - `OOD/ProprioOOD/ppo_latents_L2_normal.npy`
  - `OOD/ProprioOOD/ppo_swd_ood_model.npz`

All scripts in `SMOOD/` assume they are run with the **project root** as the current working directory, for example:

```bash
python training/train_rl.py
```

---

## 1. Prerequisites

### 1.1 Hardware (for real-world execution)

To run the **full sim-to-real pipeline** (not just simulation), you need:

- A **UR5e** (or compatible UR robot) reachable via Ethernet.
- A **RealSense RGB‑D camera** (e.g. D435/D455), mounted and calibrated relative to the UR5e TCP.
- Hand–eye calibration matrix `T_tcp_cam.npy` in `CV/`:
  - This is a \(4 \times 4\) homogeneous transform from the robot TCP frame to the camera frame.
- A machine that can:
  - Reach the UR robot IP (e.g. `192.168.1.5`).
  - Connect to the RealSense camera.

If you only want to run **simulation and PPO training**, you do **not** need the real robot or camera.

### 1.2 Software

- Python **3.10+** (3.9–3.11 should also work).
- A working C++ toolchain for some dependencies (e.g. `pybullet`, `pyrealsense2` if built locally).
- For real-robot control:
  - Universal Robots RTDE interface enabled on the controller.
  - Network connectivity from the host running Python to the UR controller.

---

## 2. Installation

From the project root, create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate    # Windows PowerShell: .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

> Note: `pyrealsense2` and `ur-rtde` may require platform-specific wheels; if `pip` fails, follow Intel RealSense and UR RTDE official installation instructions and re‑run the command without those packages in `requirements.txt`.

---

## 3. Running the PyBullet simulation and PPO training

### 3.1 Verify URDF models

Make sure the URDFs and meshes exist at:

- `ur_e_description/urdf/ur5e.urdf`
- `ur_e_description/urdf/table.urdf`

These are referenced by `gym_env.py` using **relative paths**:

- `ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e.urdf"`
- `TABLE_URDF_PATH = "./ur_e_description/urdf/table.urdf"`

If your URDFs live in a different folder, either:

- Move them into `ur_e_description/`, or
- Adjust the two constants at the top of `SMOOD_GitHub/gym_env.py`.

### 3.2 Quick rollout test in simulation

You can smoke‑test the environment (without training) by importing and stepping it manually, e.g. from a Python shell:

```python
from core.gym_env import ur5GymEnv

env = ur5GymEnv(renders=True, maxSteps=100, actionRepeat=2, randObjPos=False)
obs, info = env.reset()
for _ in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

### 3.3 Training PPO in simulation

To train a PPO agent in the PyBullet environment:

```bash
python training/train_rl.py --render --randObjPos
```

Important command‑line arguments:

- `--render` &mdash; enables PyBullet GUI.
- `--randObjPos` &mdash; randomizes object XY position over episodes.
- `--mel` &mdash; max episode length (default 150).
- `--repeat` &mdash; action repeat (default 2).
- `--tol_xy`, `--tol_z` &mdash; success tolerances used in the reward and termination.

During training:

- Checkpoints are saved to `saved_rl_models/`:
  - Periodic snapshots: `saved_rl_models/model_epoch_*.pth`
  - Final high‑precision model (if reached): `saved_rl_models/model_precise.pth`
- TensorBoard logs are written under `saved_rl_models/tensorboard_logs/`.

You can later point `sim2real.py` to one of these `.pth` files (see below).

---

## 4. Vision pipeline: detection and 3D localization

This section describes the **camera + detection + 3D position** pipeline.  
You can use it either independently or as the front‑end for the sim‑to‑real scanning controller.

### 4.1 Florence-2 object detection (`CV/inference_OBD.py`)

This script supports:

- A **static image path** (`--image_path`), or
- **Live RealSense RGB stream** (`--live_cam`).

Key configuration inside `CV/inference_OBD.py`:

- `DEFAULT_IMAGE_PATH = "Florence2/test_images/overlap_NIST.jpg"`
- `DEFAULT_MODEL_PATH = "Florence2/Saved_Models/epoch_12"`

These are relative placeholders; you should:

1. Place your test images under `Florence2/test_images/`.
2. Place your Florence LoRA adapter (if any) under `Florence2/Saved_Models/`.
3. Or override them via command line.

**Example: static image mode**

```bash
python CV/inference_OBD.py \
  --image_path Florence2/test_images/overlap_NIST.jpg \
  --target_object "calibration cube" \
  --model_path Florence2/Saved_Models/epoch_12
```

**Example: RealSense live mode (for first 10 frames)**

```bash
python CV/inference_OBD.py \
  --live_cam \
  --target_object "calibration cube" \
  --model_path Florence2/Saved_Models/epoch_12
```

The live mode will:

- Open the RGB stream from the default RealSense camera.
- Track the target object and obstacles over a short window.
- Save averaged 2D bounding boxes to `CV/objects_position.json` (relative path).

### 4.2 2D → 3D coordinate conversion (`CV/2d_to_3d.py`)

`2d_to_3d.py` takes the 2D bounding boxes and depth data and outputs **3D coordinates in the robot base frame**.

Requirements:

- `CV/objects_position.json` produced by `inference_OBD.py` in live mode.
- Hand–eye transform `CV/T_tcp_cam.npy` (4×4 numpy array).
- A running RealSense depth stream.
- RTDE connection to the UR robot (same IP as in your setup).

The script uses **relative paths**:

- Input detections: `JSON_INPUT = "CV/objects_position.json"`
- Hand–eye: `T_TCP_CAM_PATH = "CV/T_tcp_cam.npy"`
- Output: `OUTPUT_JSON = "CV/positions_3d.json"`

Run:

```bash
python CV/2d_to_3d.py
```

It will:

- Poll synchronized depth + color frames.
- For each bounding box (target and obstacles), estimate median depth.
- Back‑project to camera coordinates and transform into the robot base frame.
- Save averaged results to `CV/positions_3d.json`.

The resulting JSON provides:

- `"target": [Xb, Yb, Zb, object_height]`
- `"obstacles": { "<id>": [Xb, Yb, Zb, object_height], ... }`

`sim2real.py` uses this file to define the **goal** and **obstacle locations**.

---

## 5. Fitting OOD models (optional but recommended)

The OOD detectors use **sliced Wasserstein distance (SWD)** with uncertainty, calibrated on in‑distribution data.
You typically:

1. Record “normal” latents during safe runs.
2. Fit the OOD models from those latents.
3. Use the resulting `.npz` models in `sim2real.py`.

### 5.1 Recording visual and PPO latents

Latents are recorded via `sim2real.py` in **recording mode**.  
This assumes you can run the real‑robot controller safely under nominal conditions.

**Record visual latents only:**

```bash
python sim2real.py --record_normal_visual_latents
```

**Record PPO latents only:**

```bash
python sim2real.py --record_normal_ppo_latents
```

Each mode will append to:

- Visual: `OOD/VisualOOD/vision_latents_L2_normal.npy`
- PPO: `OOD/ProprioOOD/ppo_latents_L2_normal.npy`

> Note: When running in recording mode, OOD gating is not active; you are expected to execute only safe trajectories.

### 5.2 Fit the visual OOD model

Once you have a sufficient number of normal visual latents:

```bash
python ood/fit_vision_ood.py
```

This script:

- Loads `OOD/VisualOOD/vision_latents_L2_normal.npy`.
- Builds random projections and empirical reference distributions.
- Calibrates z‑scores and a threshold at a chosen percentile.
- Saves the OOD model to `OOD/VisualOOD/vision_ood_model.npz`.

`sim2real.py` will automatically load this file (if present) via `FlorenceVisionOOD`.

### 5.3 Fit the PPO OOD model

Similarly, for the PPO latent space:

```bash
python ood/fit_ppo_swd_ood.py
```

This script:

- Loads `OOD/ProprioOOD/ppo_latents_L2_normal.npy`.
- Performs the same SWD + standardization pipeline.
- Saves `OOD/ProprioOOD/ppo_swd_ood_model.npz`.

`sim2real.py` will load this via `PPOControlOOD`.

---

## 6. Sim‑to‑real execution with OOD safety (`sim2real_runner/sim2real.py`)

Once you have:

- A trained PPO model saved under `saved_rl_models/` (for example `model_precise.pth`).
- A valid `CV/positions_3d.json` (goal and obstacle geometry).
- Fitted OOD models (optional but recommended).

You can run:

```bash
python sim2real_runner/sim2real.py
```

Key configuration at the top of `sim2real_runner/sim2real.py`:

- `ROBOT_IP` &mdash; IP/hostname of your UR controller (default `"192.168.1.5"`).  
  **Edit this to match your setup.**
- `MODEL_PATH` &mdash; path to the PPO checkpoint (default:
  `"saved_rl_models/best_____model_precise_seed_987.pth"`).  
  Replace this with one of the models produced by `train_rl.py`.
- `JSON_3D_PATH` &mdash; path to `CV/positions_3d.json` (default is already relative).

During execution:

- The UR5e moves towards `GOAL_POS` derived from `positions_3d.json`.
- OOD scores are computed for:
  - Vision latents (`FlorenceVisionOOD`),
  - PPO latents (`PPOControlOOD`),
  - And an adaptively fused score.
- If OOD becomes significant:
  - The controller **pauses** motion and waits until multi‑step in‑distribution behavior is re‑established.
- A live Matplotlib window shows:
  - Raw and smoothed z‑scores,
  - OOD thresholds,
  - Uncertainty bands,
  - Shaded regions when OOD is detected.
- `plot_trajectory.py` is used internally to log step‑wise motion and can later generate trajectory plots and tables.

On exit, `sim2real.py`:

- Writes final OOD score plots to `OOD/ood_scores_<timestamp>.png`.
- If recording mode was enabled, appends latents to the appropriate `.npy` files.

---

## 7. Reproducibility and tips

- Always run commands from the **project root**, so that all relative paths resolve correctly.
- For simulation‑only experiments:
  - You can ignore `sim2real.py`, `2d_to_3d.py`, and the RealSense/UR dependencies.
  - Focus on `gym_env.py`, `ppo.py`, and `train_rl.py`.
- For real‑robot experiments:
  - Verify that RTDE (`ur-rtde`) can connect to your UR controller using a simple test script.
  - Verify RealSense connectivity using Intel’s examples before running the full pipeline.
  - Carefully check workspace limits in `sim2real.py` (`SAFE_X`, `SAFE_Y`, `SAFE_Z`) before running on hardware.

If you keep this folder intact and follow the steps above, another user should be able to:

1. Install dependencies from `requirements.txt`.
2. Train PPO in PyBullet.
3. Run Florence‑based detection and 2D→3D localization.
4. Fit OOD models from recorded latents.
5. Execute the full sim‑to‑real scanning behavior with OOD‑aware safety gating.

