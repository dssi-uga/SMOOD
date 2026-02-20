import random
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict, Box
import os
import math
import pybullet
import pybullet_data
from collections import namedtuple

ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e.urdf"
TABLE_URDF_PATH = "./ur_e_description/urdf/table.urdf"
CUBE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "cube_small.urdf")


def goal_distance(goal_a, goal_b):
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def goal_distance2d(goal_a, goal_b):
    return np.linalg.norm(goal_a[0:2] - goal_b[0:2], axis=-1)


class ur5GymEnv(gym.Env):
    def __init__(self,
                 camera_attached=False,
                 actionRepeat=1,
                 renders=False,
                 maxSteps=100,
                 simulatedGripper=False,
                 randObjPos=False):

        self.renders = renders
        self.actionRepeat = actionRepeat
        self.camera_attached = camera_attached
        self.maxSteps = maxSteps
        self.randObjPos = randObjPos
        self.simulatedGripper = simulatedGripper

        # Success tolerances
        self.tol_xy = 0.01  # 1 cm
        self.tol_z = 0.01   # 1 cm

        # Camera parameters
        self.image_width = 640
        self.image_height = 480

        # PyBullet setup
        if self.renders:
            pybullet.connect(pybullet.GUI)
        else:
            pybullet.connect(pybullet.DIRECT)

        pybullet.setTimeStep(1. / 240.)
        pybullet.setGravity(0, 0, -9.81)
        pybullet.setRealTimeSimulation(False)
        pybullet.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=60, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

        # Load table and robot
        self.table = pybullet.loadURDF(TABLE_URDF_PATH, [0.30, -0.390, 0], [0, 0, 0, 1], useFixedBase=True)
        flags = pybullet.URDF_USE_SELF_COLLISION
        self.ur5 = pybullet.loadURDF(ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags)

        # Define joints
        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])
        joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joints = {}

        for i in range(pybullet.getNumJoints(self.ur5)):
            info = pybullet.getJointInfo(self.ur5, i)
            jname = info[1].decode("utf-8")
            jtype = joint_type_list[info[2]]
            jinfo = self.joint_info(info[0], jname, jtype, info[8], info[9], info[10], info[11], jname in self.control_joints)
            if jinfo.type == "REVOLUTE":
                pybullet.setJointMotorControl2(self.ur5, jinfo.id, pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[jname] = jinfo

        # Find link indices
        self.end_effector_index = 7
        self.tool0_link_index = -1
        self.camera_link_index = -1
        for i in range(pybullet.getNumJoints(self.ur5)):
            link_name = pybullet.getJointInfo(self.ur5, i)[12].decode('utf-8')
            if link_name == "tool0":
                self.tool0_link_index = i
            elif link_name == "camera_link":
                self.camera_link_index = i

        # Load object
        self.initial_obj_pos = [0.45, -0.35, 0.0]
        self.obj = pybullet.loadURDF(CUBE_URDF_PATH, self.initial_obj_pos)
        self.object_height = 0.05

        # Action space: Δx, Δy, Δz
        self.action_dim = 3
        self._action_bound = 1.0
        high = np.array([self._action_bound] * self.action_dim)
        self.action_space = spaces.Box(-high, high, dtype=np.float32)

        # Observation space: [tool_pos(3), goal_pos(3), tcp_vel(3), clearance(1)]
        high_obs = np.array([2.0] * 10)
        if self.camera_attached:
            self.observation_space = Dict({
                "state": spaces.Box(-high_obs, high_obs, dtype=np.float32),
                "cam_image": Box(low=0, high=255, shape=(self.image_height, self.image_width, 1), dtype=np.uint8)
            })
        else:
            self.observation_space = spaces.Box(-high_obs, high_obs, dtype=np.float32)

        # Internal vars
        self.stepCounter = 0
        self.terminated = False
        self.prev_distance = None
        self.prev_xy_dist = None
        self._seed = None
        self.tcp_axis_lines = []

        self.reset()

    # ========== Utility ==========

    def seed(self, seed=None):
        self._seed = seed
        return [seed]

    def close(self):
        pybullet.disconnect()

    def get_joint_angles(self):
        joint_ids = [self.joints[name].id for name in self.control_joints]
        return np.array([i[0] for i in pybullet.getJointStates(self.ur5, joint_ids)])

    def set_joint_angles(self, joint_angles):
        ids = [self.joints[name].id for name in self.control_joints]
        forces = [self.joints[name].maxForce for name in self.control_joints]
        pybullet.setJointMotorControlArray(self.ur5, ids, pybullet.POSITION_CONTROL, targetPositions=joint_angles, positionGains=[0.05]*6, forces=forces)

    def get_current_pose(self):
        state = pybullet.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        return np.array(state[0]), state[1]

    def calculate_ik(self, pos, ori):
        quat = pybullet.getQuaternionFromEuler(ori)
        rest = [-1.157, -1.554, 1.753, -1.587, -1.587, -3.14]
        return pybullet.calculateInverseKinematics(self.ur5, self.end_effector_index, pos, quat, jointDamping=[0.01]*6, restPoses=rest)

    def check_collisions(self):
        return len(pybullet.getContactPoints()) > 0

    # ========== Core Env ==========

    def reset(self, seed=None, options=None):
        if seed is None and self._seed is not None:
            seed = self._seed
        super().reset(seed=seed)
        self.stepCounter = 0
        self.terminated = False
        self.prev_distance = None
        self.prev_xy_dist = None
        self.prev_xy_err = None
        self.ur5_or = [0.0, math.pi / 2, 0.0]

        self.episode_count = getattr(self, "episode_count", 0) + 1

        # Randomize object XY position
        if self.randObjPos:
            if self.episode_count < 500:
                x_min, x_max, y_min, y_max = 0.42, 0.47, -0.50, -0.45
            elif self.episode_count < 1500:
                x_min, x_max, y_min, y_max = 0.38, 0.52, -0.55, -0.40
            else:
                x_min, x_max, y_min, y_max = 0.33, 0.57, -0.60, -0.35

            if not hasattr(self, "_goal_refresh_count"):
                self._goal_refresh_count = 0
            self._goal_refresh_count += 1

            if self._goal_refresh_count % 5 == 0:
                x, y = random.uniform(x_min, x_max), random.uniform(y_min, y_max)
                self.last_obj_pos = [x, y, 0.0]
            else:
                if hasattr(self, "last_obj_pos"):
                    x, y = self.last_obj_pos[:2]
                else:
                    x, y = 0.45, -0.35

            self.initial_obj_pos = [x, y, 0.0]
        else:
            self.initial_obj_pos = [0.45, -0.35, 0.0]

        # Randomize object height (5–15 cm)
        if not hasattr(self, "_height_refresh_count"):
            self._height_refresh_count = 0
        self._height_refresh_count += 1

        if self._height_refresh_count % 5 == 0:
            self.object_height = random.uniform(0.05, 0.15)

        # Remove old object and reload
        pybullet.removeBody(self.obj)
        self.obj = pybullet.loadURDF(CUBE_URDF_PATH, self.initial_obj_pos, globalScaling=1.0, flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL)

        # Set object color to WHITE
        pybullet.changeVisualShape(self.obj, -1, rgbaColor=[1, 1, 1, 1])

        # Position cube so top remains on table
        pybullet.resetBasePositionAndOrientation(self.obj, [self.initial_obj_pos[0], self.initial_obj_pos[1], self.object_height / 2], [0, 0, 0, 1])

        # Show height in GUI
        pybullet.addUserDebugText(
            text=f"Height: {self.object_height*100:.1f} cm",
            textPosition=[self.initial_obj_pos[0], self.initial_obj_pos[1], self.object_height + 0.05],
            textColorRGB=[1, 0, 0],
            textSize=1.4,
            lifeTime=0
        )

        # Reset robot joints
        reset_angles = (-1.157, -1.554, 1.753, -1.587, -1.587, -3.14)
        for i, n in enumerate(self.control_joints):
            j = self.joints[n]
            pybullet.resetJointState(self.ur5, j.id, reset_angles[i], 0)
            pybullet.setJointMotorControl2(self.ur5, j.id, pybullet.POSITION_CONTROL, targetPosition=reset_angles[i], force=j.maxForce)

        for _ in range(100):
            pybullet.stepSimulation()

        self.getExtendedObservation()
        self._draw_tcp_axes()

        return self.observation, {}

    def step(self, action):
        action = np.clip(np.array(action), -1, 1)
        cur_pos, _ = self.get_current_pose()
        obj_pos, _ = pybullet.getBasePositionAndOrientation(self.obj)
        measurement_offset = 0.310
        goal_pos = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + self.object_height + measurement_offset])

        # Get distances
        z_dist = abs(cur_pos[2] - goal_pos[2])

        # Per-axis adaptive scaling
        xy_scale_far, xy_scale_near = 0.25, 0.12
        z_scale_far, z_scale_near = 0.08, 0.01
        if z_dist < 0.03:
            scale_xy, scale_z = xy_scale_near, z_scale_near
        else:
            scale_xy, scale_z = xy_scale_far, z_scale_far

        scaled_action = np.array([action[0] * scale_xy, action[1] * scale_xy, action[2] * scale_z])
        new_pos = cur_pos + scaled_action

        # Clip workspace
        new_pos[0] = np.clip(new_pos[0], 0.2, 0.7)
        new_pos[1] = np.clip(new_pos[1], -0.65, -0.25)
        new_pos[2] = np.clip(new_pos[2], 0.2, 0.55)

        self.set_joint_angles(self.calculate_ik(new_pos, self.ur5_or))

        for _ in range(self.actionRepeat):
            pybullet.stepSimulation()
            if self.renders:
                time.sleep(1. / 240.)

        self.getExtendedObservation()
        reward = self.compute_reward(self.achieved_goal, self.desired_goal, None)
        done = self.terminated or self.stepCounter >= self.maxSteps
        self.stepCounter += 1
        self._draw_tcp_axes()

        tcp_pos = self.achieved_goal[-3:]
        xy_dist = goal_distance2d(tcp_pos[:2], self.desired_goal[:2])
        z_dist = abs(tcp_pos[2] - self.desired_goal[2])

        info = {"is_success": self.terminated, "xy_dist": xy_dist, "z_dist": z_dist}
        return self.observation, reward, done, False, info

    # ========== Observation & Reward ==========

    def _render_camera_from_link(self, body_id, link_index):
        """Render image from specified robot link (e.g., tool0 or end-effector)."""
        if link_index == -1:
            link_state = pybullet.getLinkState(body_id, self.end_effector_index, computeForwardKinematics=True)
        else:
            link_state = pybullet.getLinkState(body_id, link_index, computeForwardKinematics=True)

        link_pos = link_state[0]
        link_ori = link_state[1]

        obj_pos, _ = pybullet.getBasePositionAndOrientation(self.obj)

        # Camera positioned at the link (scanner position)
        cam_eye = list(link_pos)

        cam_target = [
            obj_pos[0],
            obj_pos[1],
            obj_pos[2] + 0.05
        ]

        rot_matrix = pybullet.getMatrixFromQuaternion(link_ori)
        up = [rot_matrix[1], rot_matrix[4], rot_matrix[7]]

        view_dir = np.array(cam_target) - np.array(cam_eye)
        view_dir_norm = view_dir / (np.linalg.norm(view_dir) + 1e-6)
        if abs(view_dir_norm[2]) > 0.9:
            up = [0, 1, 0]

        view = pybullet.computeViewMatrix(cam_eye, cam_target, up)
        proj = pybullet.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.image_width / self.image_height,
            nearVal=0.01,
            farVal=2.0
        )

        w, h, rgba, depth, seg = pybullet.getCameraImage(
            self.image_width,
            self.image_height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL if self.renders else pybullet.ER_TINY_RENDERER
        )

        rgba_img = np.reshape(rgba, (h, w, 4))
        gray = np.dot(rgba_img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        gray = np.expand_dims(gray, axis=2)
        return gray

    def getExtendedObservation(self):
        link_state = pybullet.getLinkState(self.ur5, self.end_effector_index,
                                           computeForwardKinematics=True, computeLinkVelocity=True)
        tcp_pos = np.array(link_state[0])
        tcp_vel = np.array(link_state[6])

        obj_pos, _ = pybullet.getBasePositionAndOrientation(self.obj)
        measurement_offset = 0.310
        goal_pos = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + self.object_height + measurement_offset])

        mid = np.array([0.45, -0.45, 0.375])
        scale = np.array([0.25, 0.20, 0.175])

        tcp_pos_norm = np.clip((tcp_pos - mid) / scale, -1, 1)
        goal_pos_norm = np.clip((goal_pos - mid) / scale, -1, 1)
        tcp_vel_norm = np.clip(tcp_vel / 0.2, -1, 1)

        CLEARANCE_MIN = 0.20
        CLEARANCE_MAX = 0.95

        min_clearance = CLEARANCE_MAX
        c = np.clip(min_clearance, CLEARANCE_MIN, CLEARANCE_MAX)
        clearance_norm = 2.0 * (c - CLEARANCE_MIN) / (CLEARANCE_MAX - CLEARANCE_MIN) - 1.0
        clearance_norm += np.random.uniform(-0.2, 0.2)

        state_obs = np.concatenate((tcp_pos_norm, goal_pos_norm, tcp_vel_norm, [clearance_norm]))

        self.achieved_goal = np.concatenate((obj_pos, tcp_pos))
        self.desired_goal = goal_pos
        self.tcp_vel = tcp_vel

        if self.camera_attached:
            if self.camera_link_index != -1:
                camera_link = self.camera_link_index
            elif self.tool0_link_index != -1:
                camera_link = self.tool0_link_index
            else:
                camera_link = self.end_effector_index
            img = self._render_camera_from_link(self.ur5, camera_link)
            self.observation = {"state": state_obs, "cam_image": img}
        else:
            self.observation = state_obs

    def compute_reward(self, achieved_goal, desired_goal, info):
        tcp = np.array(achieved_goal[-3:])
        goal = np.array(desired_goal[-3:])
        vel = np.array(self.tcp_vel) if hasattr(self, 'tcp_vel') else np.zeros(3)

        pos_err = tcp - goal
        dist = np.linalg.norm(pos_err)
        xy_err, z_err = np.linalg.norm(pos_err[:2]), abs(pos_err[2])

        xy_norm = np.clip(xy_err / 0.14, 0, 1)
        z_norm = np.clip(z_err / 0.10, 0, 1)

        # Base pose rewards
        r_xy = -3.0 * (1 - np.exp(-35.0 * xy_norm))
        r_z = -2.0 * (1 - np.exp(-8.0 * z_norm))
        r_pose = r_xy + r_z

        # Progress shaping
        if hasattr(self, "prev_xy_err") and self.prev_xy_err is not None:
            r_prog = 3.0 * (self.prev_xy_err - xy_err)
        else:
            r_prog = 0.0
        self.prev_xy_err = xy_err

        # Stabilization near final position
        r_stab = 6.0 * np.exp(-10.0 * dist)

        # Velocity penalty
        r_vel = -0.05 * np.linalg.norm(vel)

        # Weighted XY reward gated by Z
        r_goal = 2.0 * np.exp(-8.0 * xy_norm) * (1 - z_norm)

        # Bonus for correct scanning height
        if z_err < 0.01:
            r_z_bonus = 1.5 * np.exp(-20.0 * xy_norm)
        else:
            r_z_bonus = 0.0

        # Terminal success bonus
        success = (xy_err < self.tol_xy) and (z_err < self.tol_z)
        r_succ = 20.0 if success else 0.0
        if success:
            self.terminated = True

        # Directional reward
        rel_vec = goal - tcp
        if hasattr(self, 'last_tcp'):
            move_vec = tcp - self.last_tcp
            if np.linalg.norm(move_vec) > 1e-4 and np.linalg.norm(rel_vec) > 1e-4:
                r_directional = 2.0 * np.dot(move_vec, rel_vec) / (
                    np.linalg.norm(move_vec) * np.linalg.norm(rel_vec)
                )
            else:
                r_directional = 0.0
        else:
            r_directional = 0.0
        self.last_tcp = tcp.copy()

        # Collision penalty
        r_col = -1.0 if self.check_collisions() else 0.0

        reward = r_pose + r_stab + r_vel + r_succ + r_prog + r_goal + r_directional + r_col + r_z_bonus
        return float(reward)

    def _draw_tcp_axes(self, axis_length=0.15):
        for line in self.tcp_axis_lines:
            pybullet.removeUserDebugItem(line)
        self.tcp_axis_lines = []
        if self.tool0_link_index == -1:
            return
        s = pybullet.getLinkState(self.ur5, self.tool0_link_index, computeForwardKinematics=True)
        pos, ori = s[0], s[1]
        rot = np.array(pybullet.getMatrixFromQuaternion(ori)).reshape(3, 3)
        origin = np.array(pos)
        ends = [origin + axis_length * rot[:, i] for i in range(3)]
        if self.renders:
            self.tcp_axis_lines += [
                pybullet.addUserDebugLine(origin, ends[0], [1, 0, 0], 2),
                pybullet.addUserDebugLine(origin, ends[1], [0, 1, 0], 2),
                pybullet.addUserDebugLine(origin, ends[2], [0, 0, 1], 2)
            ]

