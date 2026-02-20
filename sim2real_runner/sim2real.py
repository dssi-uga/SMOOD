import os
import sys

# Allow imports from SMOOD_GitHub/* when running this script directly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
import torch
import numpy as np
import json
import pyrealsense2 as rs
import cv2
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl

from core.ppo import ActorCritic
from sim2real_runner.plot_trajectory import add_state, plot_trajectory, save_logs_to_excel, save_trajectory_csv

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from vision.florence_vision import load_florence_vision
from vision.vision_ood_detector import FlorenceVisionOOD
from ood.ppo_ood_detector import PPOControlOOD

... (rest of the original sim2real.py content remains unchanged, using the same configuration and logic) ...

