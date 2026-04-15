import pybullet as p
import pybullet_data
import numpy as np
import yaml
import os

class MobileWarehouseEnvV2:
    def __init__(self, config_path="configs/config_mobile.yaml", render=False,
                 curriculum_stage=3):
        """
        curriculum_stage controls starting distance:
        0 = 1m from shelf (easy)
        1 = 2m from shelf (medium)
        2 = 3m from shelf (hard)
        3 = full random (complete task)
        """
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        self.render           = render
        self.env_cfg          = self.cfg['environment']
        self.step_count       = 0
        self.curriculum_stage = curriculum_stage

        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            # Try GPU-accelerated rendering via EGL (10-50x faster on Linux+NVIDIA)
            try:
                egl = p.loadPlugin("eglRendererPlugin")
                if egl >= 0:
                    print("EGL GPU rendering enabled!")
                else:
                    print("EGL plugin not available, using CPU rendering")
            except Exception:
                print("EGL not available, using CPU rendering")

        self.task_instructions = [
            "navigate to shelf one and pick up the red box",
            "navigate to shelf one and pick up the blue box",
            "navigate to shelf two and pick up the green box",
            "navigate to shelf two and pick up the yellow box",
            "pick up the red box and carry it to the dropoff zone",
            "pick up the blue box and deliver it to the station",
            "go to shelf one get the red box and place it at dropoff",
            "go to shelf two get the green box and place it at dropoff",
        ]

        self.current_instruction  = None
        self.husky_id             = None
        self.panda_id             = None
        self.object_ids           = []
        self.shelf_ids            = []
        self.target_object_idx    = 0
        self.grasp_constraint     = None

        # Phase tracking for reward shaping
        self.reached_shelf    = False
        self.reached_object   = False
        self.grasped_object   = False
        self.delivered_object = False
        self.lifted_object    = False
        self.phase_bonuses    = 0.0

    def setup_world(self):
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.env_cfg['sim_timestep'])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")
        self._create_warehouse_structure()

    def _create_warehouse_structure(self):
        room_size = self.env_cfg['workspace_size']
        wall_configs = [
            ([room_size, 0.1, 1.0], [0,  room_size, 1.0]),
            ([room_size, 0.1, 1.0], [0, -room_size, 1.0]),
            ([0.1, room_size, 1.0], [ room_size, 0, 1.0]),
            ([0.1, room_size, 1.0], [-room_size, 0, 1.0]),
        ]
        for half_extents, pos in wall_configs:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                      rgbaColor=[0.7, 0.7, 0.7, 1])
            p.createMultiBody(0, col, vis, pos)

        self.shelf_ids = []
        shelf_positions = self.env_cfg['shelf_positions']
        for shelf_pos in shelf_positions:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.3, 0.02])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.3, 0.02],
                                      rgbaColor=[0.5, 0.35, 0.1, 1])
            sid = p.createMultiBody(0, col, vis,
                                    [shelf_pos[0], shelf_pos[1], 0.5])
            self.shelf_ids.append(sid)
            for lx, ly in [(-0.5,-0.25),(0.5,-0.25),(-0.5,0.25),(0.5,0.25)]:
                lc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03,0.03,0.25])
                lv = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03,0.03,0.25],
                                         rgbaColor=[0.3,0.3,0.3,1])
                p.createMultiBody(0, lc, lv,
                                  [shelf_pos[0]+lx, shelf_pos[1]+ly, 0.25])

        dropoff = self.env_cfg['dropoff_position']
        dc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.01])
        dv = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.01],
                                  rgbaColor=[0.0, 0.8, 0.0, 0.5])
        self.dropoff_id = p.createMultiBody(0, dc, dv,
                                             [dropoff[0], dropoff[1], 0.01])

    def load_robot(self):
        self.husky_id = p.loadURDF(
            "husky/husky.urdf",
            basePosition=[0, 0, 0.15],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False
        )
        self.panda_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0.65],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False
        )
        self.attach_constraint = p.createConstraint(
            self.husky_id, -1, self.panda_id, -1,
            p.JOINT_FIXED, [0,0,0], [0,0,0.5], [0,0,0]
        )
        # Allow constraint to move with high force
        p.changeConstraint(self.attach_constraint, maxForce=10000)
        self.wheel_joints = []
        for j in range(p.getNumJoints(self.husky_id)):
            info = p.getJointInfo(self.husky_id, j)
            if 'wheel' in info[1].decode('utf-8').lower():
                self.wheel_joints.append(j)
        self.arm_joints    = list(range(7))
        self.gripper_joints = [9, 10]
        home = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        for i, pos in enumerate(home):
            p.resetJointState(self.panda_id, i, pos)

    def load_objects(self):
        colors = [
            [1,0,0,1], [0,0,1,1],
            [0,1,0,1], [1,1,0,1],
        ]
        shelf_positions = self.env_cfg['shelf_positions']
        object_positions = [
            [shelf_positions[0][0]-0.2, shelf_positions[0][1], 0.58],
            [shelf_positions[0][0]+0.2, shelf_positions[0][1], 0.58],
            [shelf_positions[1][0]-0.2, shelf_positions[1][1], 0.58],
            [shelf_positions[1][0]+0.2, shelf_positions[1][1], 0.58],
        ]
        self.object_ids = []
        for i in range(self.env_cfg['num_objects']):
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.04,0.04,0.04])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.04,0.04,0.04],
                                       rgbaColor=colors[i])
            oid = p.createMultiBody(0.1, col, vis, object_positions[i])
            self.object_ids.append(oid)

    def get_camera_image(self):
        w = self.env_cfg['camera_width']
        h = self.env_cfg['camera_height']
        husky_pos, husky_orn = p.getBasePositionAndOrientation(self.husky_id)
        euler   = p.getEulerFromQuaternion(husky_orn)
        yaw     = euler[2]
        cam_pos = [husky_pos[0] - 0.5*np.cos(yaw),
                   husky_pos[1] - 0.5*np.sin(yaw),
                   husky_pos[2] + 1.2]
        cam_target = [husky_pos[0] + np.cos(yaw),
                      husky_pos[1] + np.sin(yaw),
                      husky_pos[2] + 0.5]
        view_matrix = p.computeViewMatrix(cam_pos, cam_target, [0,0,1])
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=80, aspect=w/h, nearVal=0.1, farVal=20
        )
        _, _, rgb, _, _ = p.getCameraImage(w, h, view_matrix, proj_matrix)
        return np.array(rgb, dtype=np.uint8).reshape(h, w, 4)[:,:,:3]

    def compute_reward(self):
        metrics = self._update_task_status()

        reward = 0.0
        reward -= 0.35 * metrics["dist_to_shelf"]
        reward -= 0.55 * metrics["dist_to_obj"]
        reward -= 0.05 * metrics["dist_dropoff"]
        reward -= 0.01

        if metrics["dist_to_shelf"] < 1.2:
            reward += 2.0
        if metrics["dist_to_obj"] < 0.30:
            reward += 3.0
        if metrics["dist_to_obj"] < 0.18:
            reward += 5.0
        if metrics["dist_to_obj"] < 0.10 and metrics["gripper_closed"]:
            reward += 8.0
        if metrics["attached"]:
            reward += 15.0
        if metrics["obj_z"] > 0.70:
            reward += 20.0
        if metrics["obj_speed"] > 2.5:
            reward -= 6.0
        if metrics["obj_z"] > 0.75 and not metrics["attached"]:
            reward -= 10.0

        return reward

    def _release_grasp_constraint(self):
        if self.grasp_constraint is not None:
            try:
                p.removeConstraint(self.grasp_constraint)
            except Exception:
                pass
            self.grasp_constraint = None

    def get_target_metrics(self):
        husky_pos, _ = p.getBasePositionAndOrientation(self.husky_id)
        husky_xy = np.array(husky_pos[:2])
        shelf_positions = getattr(self, 'current_shelf_positions',
                                  self.env_cfg['shelf_positions'])
        shelf_idx = self.target_object_idx // 2
        target_shelf = np.array(shelf_positions[shelf_idx])

        gripper_pos = np.array(p.getLinkState(self.panda_id, 11)[0])
        obj_pos, _ = p.getBasePositionAndOrientation(
            self.object_ids[self.target_object_idx]
        )
        obj_pos = np.array(obj_pos)
        obj_vel, _ = p.getBaseVelocity(self.object_ids[self.target_object_idx])
        obj_speed = np.linalg.norm(obj_vel)
        current_dropoff = getattr(self, 'current_dropoff',
                                  self.env_cfg['dropoff_position'])
        dropoff = np.array(current_dropoff + [0.05])
        gripper_opening = p.getJointState(self.panda_id, 9)[0]

        return {
            "husky_xy": husky_xy,
            "target_shelf": target_shelf,
            "gripper_pos": gripper_pos,
            "obj_pos": obj_pos,
            "obj_z": float(obj_pos[2]),
            "obj_speed": float(obj_speed),
            "dist_to_shelf": float(np.linalg.norm(husky_xy - target_shelf)),
            "dist_to_obj": float(np.linalg.norm(gripper_pos - obj_pos)),
            "dist_dropoff": float(np.linalg.norm(obj_pos - dropoff)),
            "gripper_closed": bool(gripper_opening < 0.02),
            "attached": bool(self.grasp_constraint is not None),
        }

    def _update_task_status(self):
        metrics = self.get_target_metrics()

        if not metrics["gripper_closed"]:
            self._release_grasp_constraint()
            metrics["attached"] = False

        if self.grasp_constraint is None:
            should_attach = (
                metrics["dist_to_obj"] < 0.13 and
                metrics["gripper_closed"] and
                metrics["obj_z"] < 0.70 and
                metrics["obj_speed"] < 1.5
            )
            if should_attach:
                target_id = self.object_ids[self.target_object_idx]
                self.grasp_constraint = p.createConstraint(
                    self.panda_id, 11,
                    target_id, -1,
                    p.JOINT_FIXED,
                    [0, 0, 0],
                    [0, 0, 0.04],
                    [0, 0, 0]
                )
                p.changeConstraint(self.grasp_constraint, maxForce=300)
                metrics["attached"] = True

        metrics["attached"] = bool(self.grasp_constraint is not None)

        if not self.reached_shelf and metrics["dist_to_shelf"] < 1.5:
            self.reached_shelf = True
            self.phase_bonuses += 2.0
        if not self.reached_object and metrics["dist_to_obj"] < 0.25:
            self.reached_object = True
            self.phase_bonuses += 4.0
        if not self.grasped_object and metrics["attached"]:
            self.grasped_object = True
            self.phase_bonuses += 8.0
        if not self.lifted_object and metrics["attached"] and metrics["obj_z"] > 0.66:
            self.lifted_object = True
            self.phase_bonuses += 25.0
        if not self.delivered_object and metrics["attached"] and metrics["dist_dropoff"] < 0.5:
            self.delivered_object = True
            self.phase_bonuses += 30.0

        metrics["success"] = self.lifted_object
        return metrics

    def _get_curriculum_start(self):
        """Return robot start position based on curriculum stage"""
        shelf_positions = self.env_cfg['shelf_positions']
        target_shelf    = shelf_positions[self.target_object_idx // 2]

        if self.curriculum_stage == 0:
            # Start very close to the target object, facing it.
            target_x = target_shelf[0] + (-0.2 if self.target_object_idx % 2 == 0 else 0.2)
            target_y = target_shelf[1]
            direction = np.array([-target_x, -target_y])
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            return [target_x + direction[0]*0.55,
                    target_y + direction[1]*0.55, 0.15]
        elif self.curriculum_stage == 1:
            # 1.0m from shelf
            angle = np.random.uniform(-0.6, 0.6)
            direction = np.array([-target_shelf[0], -target_shelf[1]])
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            rot = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]
            ])
            direction = rot @ direction
            return [target_shelf[0] + direction[0]*1.0,
                    target_shelf[1] + direction[1]*1.0, 0.15]
        elif self.curriculum_stage == 2:
            # 2m from shelf
            angle = np.random.uniform(0, 2*np.pi)
            return [target_shelf[0] + 2*np.cos(angle),
                    target_shelf[1] + 2*np.sin(angle), 0.15]
        else:
            # Full random within workspace
            ws = self.env_cfg['workspace_size'] - 0.5
            return [np.random.uniform(-ws, ws),
                    np.random.uniform(-ws, ws), 0.15]

    def reset(self):
        self.step_count       = 0
        self.reached_shelf    = False
        self.reached_object   = False
        self.grasped_object   = False
        self.delivered_object = False
        self.lifted_object    = False
        self.phase_bonuses    = 0.0
        self._release_grasp_constraint()

        # Randomize shelf positions every episode
        sx = np.random.uniform(1.5, 3.5)
        sy = np.random.uniform(-1.5, 1.5)
        self.current_shelf_positions = [
            [sx,  sy],   # shelf 1
            [-sx, -sy],  # shelf 2 opposite side
        ]

        # Randomize dropoff position
        self.current_dropoff = [
            np.random.uniform(-1.5, 1.5),
            np.random.uniform(-1.5, 1.5)
        ]

        # Pick target object
        self.target_object_idx = np.random.randint(0, self.env_cfg['num_objects'])

        # Curriculum-aware robot start
        start_x, start_y, _ = self._get_curriculum_start()
        target_shelf = np.array(self.current_shelf_positions[self.target_object_idx // 2])
        start_yaw = float(np.arctan2(target_shelf[1] - start_y, target_shelf[0] - start_x))
        p.resetBasePositionAndOrientation(
            self.husky_id,
            [start_x, start_y, 0.15],
            p.getQuaternionFromEuler([0, 0, start_yaw])
        )
        p.resetBaseVelocity(self.husky_id, [0,0,0], [0,0,0])

        # Reset arm
        home = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        for i, pos in enumerate(home):
            p.resetJointState(self.panda_id, i, pos)

        # Place objects on randomized shelves with noise
        base_positions = [
            [self.current_shelf_positions[0][0]-0.2,
             self.current_shelf_positions[0][1], 0.58],
            [self.current_shelf_positions[0][0]+0.2,
             self.current_shelf_positions[0][1], 0.58],
            [self.current_shelf_positions[1][0]-0.2,
             self.current_shelf_positions[1][1], 0.58],
            [self.current_shelf_positions[1][0]+0.2,
             self.current_shelf_positions[1][1], 0.58],
        ]
        for i, obj_id in enumerate(self.object_ids):
            if self.curriculum_stage == 0:
                noise = np.zeros(2)
            elif self.curriculum_stage == 1:
                noise = np.random.uniform(-0.015, 0.015, 2)
            else:
                noise = np.random.uniform(-0.05, 0.05, 2)
            pos   = [base_positions[i][0]+noise[0],
                     base_positions[i][1]+noise[1],
                     base_positions[i][2]]
            p.resetBasePositionAndOrientation(obj_id, pos, [0,0,0,1])
            p.resetBaseVelocity(obj_id, [0,0,0], [0,0,0])

        # Pick instruction that matches target object color
        color_instructions = {
            0: ["pick up the red box and deliver it",
                "navigate to shelf and pick up the red box",
                "get the red box and place it at dropoff"],
            1: ["pick up the blue box and deliver it",
                "navigate to shelf and pick up the blue box",
                "get the blue box and place it at dropoff"],
            2: ["pick up the green box and deliver it",
                "navigate to shelf and pick up the green box",
                "get the green box and place it at dropoff"],
            3: ["pick up the yellow box and deliver it",
                "navigate to shelf and pick up the yellow box",
                "get the yellow box and place it at dropoff"],
        }
        self.current_instruction = np.random.choice(
            color_instructions[self.target_object_idx]
        )
        return self.get_camera_image(), self.current_instruction

    def reset_state_only(self):
        """Reset without paying the camera-render cost."""
        self.reset()
        return self.current_instruction

    def step(self, action):
        # Navigation
        # Move Husky directly via position update (avoids constraint instability)
        vx = float(action[0]) * 0.05   # meters per step
        wz = float(action[2]) * 0.05   # radians per step

        husky_pos, husky_orn = p.getBasePositionAndOrientation(self.husky_id)
        current_yaw = p.getEulerFromQuaternion(husky_orn)[2]

        # Update heading
        new_yaw = current_yaw + wz
        new_orn = p.getQuaternionFromEuler([0, 0, new_yaw])

        # Move in current heading direction
        new_x = husky_pos[0] + vx * np.cos(new_yaw)
        new_y = husky_pos[1] + vx * np.sin(new_yaw)
        new_pos = [new_x, new_y, husky_pos[2]]

        p.resetBasePositionAndOrientation(self.husky_id, new_pos, new_orn)

        # Move Panda with Husky
        panda_pos, panda_orn = p.getBasePositionAndOrientation(self.panda_id)
        p.resetBasePositionAndOrientation(
            self.panda_id,
            [new_x, new_y, new_pos[2] + 0.5],
            new_orn
        )

        # Arm
        p.setJointMotorControlArray(
            self.panda_id, self.arm_joints[:6],
            p.POSITION_CONTROL,
            targetPositions=action[3:9],
            forces=[87]*6
        )

        # Gripper
        gpos = 0.04 if float(action[9]) > 0.5 else 0.0
        for gj in self.gripper_joints:
            p.setJointMotorControl2(self.panda_id, gj,
                p.POSITION_CONTROL, targetPosition=gpos, force=10)

        p.stepSimulation()
        self.step_count += 1

        obs    = self.get_camera_image()
        reward = self.compute_reward() + self.phase_bonuses
        phase_bonus = self.phase_bonuses
        self.phase_bonuses = 0.0
        metrics = self._update_task_status()
        done   = metrics["success"] or self.step_count >= self.env_cfg['max_episode_steps']

        return obs, reward, done, {
            'instruction':    self.current_instruction,
            'phase_bonuses':  phase_bonus,
            'reached_shelf':  self.reached_shelf,
            'reached_object': self.reached_object,
            'grasped':        self.grasped_object,
            'lifted':         self.lifted_object,
            'delivered':      self.delivered_object,
            'attached':       metrics["attached"],
            'dist_to_shelf':  metrics["dist_to_shelf"],
            'dist_to_obj':    metrics["dist_to_obj"],
            'dist_dropoff':   metrics["dist_dropoff"],
            'obj_z':          metrics["obj_z"],
            'success':        metrics["success"],
        }

    def step_state_only(self, action):
        """State-only step for fast RL training without camera rendering."""
        vx = float(action[0]) * 0.05
        wz = float(action[2]) * 0.05

        husky_pos, husky_orn = p.getBasePositionAndOrientation(self.husky_id)
        current_yaw = p.getEulerFromQuaternion(husky_orn)[2]
        new_yaw = current_yaw + wz
        new_orn = p.getQuaternionFromEuler([0, 0, new_yaw])
        new_x = husky_pos[0] + vx * np.cos(new_yaw)
        new_y = husky_pos[1] + vx * np.sin(new_yaw)
        new_pos = [new_x, new_y, husky_pos[2]]

        p.resetBasePositionAndOrientation(self.husky_id, new_pos, new_orn)
        p.resetBasePositionAndOrientation(
            self.panda_id,
            [new_x, new_y, new_pos[2] + 0.5],
            new_orn
        )

        p.setJointMotorControlArray(
            self.panda_id, self.arm_joints[:6],
            p.POSITION_CONTROL,
            targetPositions=action[3:9],
            forces=[87] * 6
        )

        gpos = 0.04 if float(action[9]) > 0.5 else 0.0
        for gj in self.gripper_joints:
            p.setJointMotorControl2(
                self.panda_id, gj,
                p.POSITION_CONTROL, targetPosition=gpos, force=10
            )

        p.stepSimulation()
        self.step_count += 1

        reward = self.compute_reward()
        phase_bonus = self.phase_bonuses
        self.phase_bonuses = 0.0
        metrics = self._update_task_status()
        done = metrics["success"] or self.step_count >= self.env_cfg['max_episode_steps']

        return reward + phase_bonus, done, {
            'instruction':    self.current_instruction,
            'phase_bonuses':  phase_bonus,
            'reached_shelf':  self.reached_shelf,
            'reached_object': self.reached_object,
            'grasped':        self.grasped_object,
            'lifted':         self.lifted_object,
            'delivered':      self.delivered_object,
            'attached':       metrics["attached"],
            'dist_to_shelf':  metrics["dist_to_shelf"],
            'dist_to_obj':    metrics["dist_to_obj"],
            'dist_dropoff':   metrics["dist_dropoff"],
            'obj_z':          metrics["obj_z"],
            'success':        metrics["success"],
        }

    def initialize(self):
        self.setup_world()
        self.load_robot()
        self.load_objects()
        self.target_object_idx   = 0
        self.current_instruction = np.random.choice(self.task_instructions)
        print(f"Mobile V2 environment initialized! (curriculum stage: {self.curriculum_stage})")
        return self.get_camera_image(), self.current_instruction

    def close(self):
        p.disconnect()


class SimpleRewardWrapper:
    """
    Wraps MobileWarehouseEnvV2 with a much simpler reward:
    Just negative distance from gripper to target object.
    Same as Phase 1 but with navigation built in.
    This gives a clean, dense reward signal RL can learn from immediately.
    """
    def __init__(self, env):
        self.env = env
        # Copy all attributes
        self.husky_id    = None
        self.panda_id    = None
        self.object_ids  = []
        self.target_object_idx = 0
        self.current_instruction = None

    def initialize(self):
        result = self.env.initialize()
        self._sync()
        return result

    def _sync(self):
        self.husky_id    = self.env.husky_id
        self.panda_id    = self.env.panda_id
        self.object_ids  = self.env.object_ids
        self.target_object_idx = self.env.target_object_idx
        self.current_instruction = self.env.current_instruction

    def reset(self):
        result = self.env.reset()
        self._sync()
        return result

    def step(self, action):
        import pybullet as p
        import numpy as np
        obs, _, done, info = self.env.step(action)
        self._sync()

        # Simple reward: negative distance from gripper to object
        gripper_state = p.getLinkState(self.env.panda_id, 11)
        gripper_pos   = np.array(gripper_state[0])
        obj_pos, _    = p.getBasePositionAndOrientation(
            self.env.object_ids[self.env.target_object_idx]
        )
        dist   = np.linalg.norm(gripper_pos - np.array(obj_pos))
        reward = -dist

        # Bonus for getting very close
        if dist < 0.15:
            reward += 2.0
        if dist < 0.08:
            reward += 3.0

        return obs, reward, done, info

    def close(self):
        self.env.close()
