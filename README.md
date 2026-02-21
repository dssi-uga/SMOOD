# SMOOD

### Safety-Aware Multi-Modal Out-of-Distribution Detection for Reinforcement Learning-Based Robotic Inspection

SMOOD is a lightweight research framework for **safe robotic
inspection** that combines reinforcement learning, vision perception,
and multi-modal out-of-distribution (OOD) monitoring.

The system enables a robot to execute learned policies while
continuously checking whether its **visual observations** and **control
behavior** remain within known safe operating conditions.

This repository provides a **minimal, reproducible subset** of the full
research system, including simulation, training, perception, and
optional sim-to-real execution.

------------------------------------------------------------------------

## Overview

SMOOD integrates three main components:

-   **Control** --- PPO policy controlling a UR5e end-effector\
-   **Perception** --- Florence-2 vision encoder for object
    understanding\
-   **Safety** --- multi-modal OOD detection using latent
    representations

The framework pauses robot motion when abnormal conditions are detected
and resumes execution once normal operation is restored.

------------------------------------------------------------------------

## What This Repository Supports

  Mode              Description                          Hardware Needed
  ----------------- ------------------------------------ -----------------
  Simulation        Train and test PPO in PyBullet       x
  Vision Pipeline   Object detection + 3D localization   RGB-D camera
  Full Pipeline     OOD-aware real robot execution       UR5e + camera

Most users should begin with **simulation only**.

------------------------------------------------------------------------

## Repository Structure

    SMOOD/
    │
    ├── core/
    │   ├── gym_env.py
    │   └── ppo.py
    │
    ├── training/
    │   └── train_rl.py
    │
    ├── vision/
    │   ├── florence_vision.py
    │   └── vision_ood_detector.py
    │
    ├── ood/
    │   ├── ppo_ood_detector.py
    │   ├── fit_vision_ood.py
    │   └── fit_ppo_swd_ood.py
    │
    ├── sim2real_runner/
    │   ├── sim2real.py
    │   └── plot_trajectory.py
    │
    ├── CV/
    │   ├── inference_OBD.py
    │   ├── 2d_to_3d.py
    │   └── positions_3d.json
    │
    └── requirements.txt

Project root should also contain:

    ur_e_description/
    saved_rl_models/
    OOD/

All paths are relative for portability and reproducibility.

------------------------------------------------------------------------

## Installation

``` bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Running the Simulation

### Quick Environment Test

``` python
from core.gym_env import ur5GymEnv

env = ur5GymEnv(renders=True)
obs, info = env.reset()

for _ in range(50):
    obs, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )

env.close()
```

### Train PPO Policy

``` bash
python training/train_rl.py --render
```

------------------------------------------------------------------------

## Vision Pipeline

### Object Detection

``` bash
python CV/inference_OBD.py --live_cam --target_object "calibration cube"
```

Output:

    CV/objects_position.json

### Convert 2D → 3D Coordinates

``` bash
python CV/2d_to_3d.py
```

Output:

    CV/positions_3d.json

------------------------------------------------------------------------

## Training OOD Models (Optional)

Record normal latents:

``` bash
python sim2real_runner/sim2real.py --record_normal_visual_latents
python sim2real_runner/sim2real.py --record_normal_ppo_latents
```

Fit models:

``` bash
python ood/fit_vision_ood.py
python ood/fit_ppo_swd_ood.py
```

------------------------------------------------------------------------

## Sim-to-Real Execution (Optional)

``` bash
python sim2real_runner/sim2real.py
```

The robot pauses automatically when OOD conditions are detected and
resumes once behavior returns to normal.

------------------------------------------------------------------------

## Typical Workflow

1.  Install dependencies\
2.  Run simulation test\
3.  Train PPO policy\
4.  Test vision detection\
5.  Fit OOD models (optional)\
6.  Run sim-to-real execution (optional)

------------------------------------------------------------------------

## Research Context

This repository contains a reproducible subset of a larger robotic
inspection framework focused on safe reinforcement learning and
distribution-shift awareness in manufacturing environments.

------------------------------------------------------------------------

## License

This project is released under the Apache License 2.0. See the LICENSE file for details.

------------------------------------------------------------------------

## Citation

``` bibtex
@inproceedings{smood2026,
  title  = {SMOOD: Safety-Aware Multi-Modal Out-of-Distribution Detection for Reinforcement-Learning-Based Robotic Inspection},
  author = {Silwal, Nitesh and Sun, Hongyue},
  year   = {2026}
}
```
