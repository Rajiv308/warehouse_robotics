import pybullet as p
import pybullet_data
import numpy as np
import yaml
import os

class MobileWarehouseEnvV2:
    def __init__(self, config_path="configs/config_mobile.yaml", render=False,
                 curriculum_stage=3, success_mode="delivery"):
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
        self.success_mode     = success_mode

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
        self.shelf_part_ids       = []
        self.shelf_blocker_ids    = []
        self.wall_ids             = []
        self.target_object_idx    = 0
        self.grasp_constraint     = None
        self.base_collision_flag  = False
        self.base_collision_penalty = 0.0
        self.robot_mount_height   = 0.52
        self.shelf_object_front_offset = -0.12
        self.arm_home             = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        self.ee_workspace_min     = np.array([-3.2, -1.8, 0.48], dtype=np.float32)
        self.ee_workspace_max     = np.array([ 3.2,  1.8, 1.10], dtype=np.float32)

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
        self.wall_ids = []
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
            wall_id = p.createMultiBody(0, col, vis, pos)
            self.wall_ids.append(wall_id)

        self.shelf_ids = []
        self.shelf_part_ids = []
        self.shelf_blocker_ids = []
        shelf_positions = self.env_cfg['shelf_positions']
        for shelf_pos in shelf_positions:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.3, 0.02])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.3, 0.02],
                                      rgbaColor=[0.5, 0.35, 0.1, 1])
            sid = p.createMultiBody(0, col, vis,
                                    [shelf_pos[0], shelf_pos[1], 0.5])
            self.shelf_ids.append(sid)
            self.shelf_part_ids.append(sid)
            for lx, ly in [(-0.5,-0.25),(0.5,-0.25),(-0.5,0.25),(0.5,0.25)]:
                lc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03,0.03,0.25])
                lv = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03,0.03,0.25],
                                         rgbaColor=[0.3,0.3,0.3,1])
                leg_id = p.createMultiBody(0, lc, lv,
                                           [shelf_pos[0]+lx, shelf_pos[1]+ly, 0.25])
                self.shelf_part_ids.append(leg_id)

            blocker_col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.58, 0.28, 0.24]
            )
            blocker_vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.58, 0.28, 0.24],
                rgbaColor=[0.5, 0.35, 0.1, 0.08]
            )
            blocker_id = p.createMultiBody(
                0, blocker_col, blocker_vis,
                [shelf_pos[0], shelf_pos[1], 0.24]
            )
            self.shelf_blocker_ids.append(blocker_id)
            self.shelf_part_ids.append(blocker_id)

        dropoff = self.env_cfg['dropoff_position']
        dc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.01])
        dv = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.01],
                                  rgbaColor=[0.0, 0.8, 0.0, 0.5])
        self.dropoff_id = p.createMultiBody(0, dc, dv,
                                             [dropoff[0], dropoff[1], 0.01])

    def load_robot(self):
        self.husky_id = p.loadURDF(
            "husky/husky.urdf",
            basePosition=[0, 0, 0.02],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False
        )
        self.panda_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, self.robot_mount_height],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True
        )
        self.wheel_joints = []
        for j in range(p.getNumJoints(self.husky_id)):
            info = p.getJointInfo(self.husky_id, j)
            if 'wheel' in info[1].decode('utf-8').lower():
                self.wheel_joints.append(j)
        self.arm_joints    = list(range(7))
        self.gripper_joints = [9, 10]
        for i, pos in enumerate(self.arm_home):
            p.resetJointState(self.panda_id, i, pos)
        self._sync_panda_to_husky([0.0, 0.0, 0.02], 0.0)

    def _sync_panda_to_husky(self, husky_pos, yaw):
        panda_pos = [husky_pos[0], husky_pos[1], husky_pos[2] + self.robot_mount_height]
        panda_orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.panda_id, panda_pos, panda_orn)

    def get_pick_pose(self):
        """
        Return a reasonable base pickup pose in front of the target object.
        """
        metrics = self.get_target_metrics()
        target_shelf = metrics["target_shelf"]
        obj_pos = metrics["obj_pos"]
        pick_xy = np.array([target_shelf[0], target_shelf[1] - 1.02], dtype=np.float32)
        pick_yaw = float(np.arctan2(obj_pos[1] - pick_xy[1], obj_pos[0] - pick_xy[0]))
        return pick_xy, pick_yaw

    def apply_cartesian_action(self, action):
        """
        Apply an end-effector delta action for the mounted Panda arm.
        action = [dx, dy, dz, grip]
        grip > 0 opens, else closes.
        """
        action = np.asarray(action, dtype=np.float32)
        if action.shape[0] < 4:
            raise ValueError("Cartesian action must have 4 dims: dx, dy, dz, grip")

        ee_pos = np.array(p.getLinkState(self.panda_id, 11)[0], dtype=np.float32)
        delta = np.clip(action[:3], -1.0, 1.0) * np.array([0.025, 0.025, 0.020], dtype=np.float32)
        target_pos = np.clip(ee_pos + delta, self.ee_workspace_min, self.ee_workspace_max)

        joint_angles = p.calculateInverseKinematics(
            self.panda_id,
            11,
            target_pos.tolist(),
            maxNumIterations=140,
            residualThreshold=1e-4,
        )
        joint_targets = np.array(joint_angles[:7], dtype=np.float32)
        p.setJointMotorControlArray(
            self.panda_id,
            self.arm_joints,
            p.POSITION_CONTROL,
            targetPositions=joint_targets.tolist(),
            forces=[87] * len(self.arm_joints),
        )

        gripper_pos = 0.04 if float(action[3]) > 0.0 else 0.0
        for gj in self.gripper_joints:
            p.setJointMotorControl2(
                self.panda_id,
                gj,
                p.POSITION_CONTROL,
                targetPosition=gripper_pos,
                force=25,
            )

    def _is_base_pose_valid(self):
        p.performCollisionDetection()
        for wall_id in getattr(self, "wall_ids", []):
            if p.getContactPoints(self.husky_id, wall_id):
                return False
        for shelf_id in getattr(self, "shelf_part_ids", []):
            if p.getContactPoints(self.husky_id, shelf_id):
                return False
        return True

    def _set_mobile_pose(self, new_pos, new_yaw):
        old_pos, old_orn = p.getBasePositionAndOrientation(self.husky_id)
        old_yaw = p.getEulerFromQuaternion(old_orn)[2]
        new_orn = p.getQuaternionFromEuler([0, 0, new_yaw])

        p.resetBasePositionAndOrientation(self.husky_id, new_pos, new_orn)
        self._sync_panda_to_husky(new_pos, new_yaw)

        if self._is_base_pose_valid():
            self.base_collision_flag = False
            self.base_collision_penalty = 0.0
            return True

        p.resetBasePositionAndOrientation(self.husky_id, old_pos, old_orn)
        self._sync_panda_to_husky(old_pos, old_yaw)
        self.base_collision_flag = True
        self.base_collision_penalty = -12.0
        return False

    def load_objects(self):
        colors = [
            [1,0,0,1], [0,0,1,1],
            [0,1,0,1], [1,1,0,1],
        ]
        shelf_positions = self.env_cfg['shelf_positions']
        object_positions = [
            [shelf_positions[0][0]-0.2, shelf_positions[0][1] + self.shelf_object_front_offset, 0.58],
            [shelf_positions[0][0]+0.2, shelf_positions[0][1] + self.shelf_object_front_offset, 0.58],
            [shelf_positions[1][0]-0.2, shelf_positions[1][1] + self.shelf_object_front_offset, 0.58],
            [shelf_positions[1][0]+0.2, shelf_positions[1][1] + self.shelf_object_front_offset, 0.58],
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
        reward += self.base_collision_penalty

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
        if self.base_collision_flag:
            reward -= 8.0

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

        if self.grasp_constraint is None and getattr(self, "auto_weld", True):
            should_attach = (
                metrics["dist_to_obj"] < 0.13 and
                metrics["gripper_closed"] and
                metrics["obj_z"] < 0.70 and
                metrics["obj_speed"] < 1.5
            )
            if should_attach:
                target_id = self.object_ids[self.target_object_idx]
                # Weld the object at its current relative pose to the EE so the
                # box stays physically where it is instead of snapping to a
                # hardcoded offset.
                ee_state = p.getLinkState(self.panda_id, 11)
                ee_pos_world, ee_orn_world = ee_state[0], ee_state[1]
                obj_pos_world, obj_orn_world = p.getBasePositionAndOrientation(target_id)
                inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos_world, ee_orn_world)
                rel_pos, rel_orn = p.multiplyTransforms(
                    inv_ee_pos, inv_ee_orn,
                    obj_pos_world, obj_orn_world,
                )
                self.grasp_constraint = p.createConstraint(
                    parentBodyUniqueId=self.panda_id,
                    parentLinkIndex=11,
                    childBodyUniqueId=target_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=list(rel_pos),
                    childFramePosition=[0, 0, 0],
                    parentFrameOrientation=list(rel_orn),
                    childFrameOrientation=[0, 0, 0, 1],
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

        if self.success_mode == "pickup":
            metrics["success"] = self.lifted_object
        else:
            metrics["success"] = self.delivered_object
        return metrics

    def _get_curriculum_start(self):
        """Return robot start position based on curriculum stage"""
        shelf_positions = self.env_cfg['shelf_positions']
        target_shelf    = shelf_positions[self.target_object_idx // 2]

        if self.curriculum_stage == 0:
            # Start in front of the shelf, not on top of it.
            return [target_shelf[0], target_shelf[1] - 1.25, 0.02]
        elif self.curriculum_stage == 1:
            return [
                target_shelf[0] + np.random.uniform(-0.25, 0.25),
                target_shelf[1] - 1.35 + np.random.uniform(-0.15, 0.15),
                0.02,
            ]
        elif self.curriculum_stage == 2:
            return [
                target_shelf[0] + np.random.uniform(-0.6, 0.6),
                target_shelf[1] - 2.0 + np.random.uniform(-0.3, 0.3),
                0.02,
            ]
        else:
            # Full random within workspace
            ws = self.env_cfg['workspace_size'] - 0.5
            return [np.random.uniform(-ws, ws),
                    np.random.uniform(-ws, ws), 0.02]

    def _reset_common(self):
        self.step_count       = 0
        self.reached_shelf    = False
        self.reached_object   = False
        self.grasped_object   = False
        self.delivered_object = False
        self.lifted_object    = False
        self.phase_bonuses    = 0.0
        self.base_collision_flag = False
        self.base_collision_penalty = 0.0
        self._release_grasp_constraint()

        # Keep the world geometry fixed so visuals and physics match the task.
        self.current_shelf_positions = [list(pos) for pos in self.env_cfg['shelf_positions']]
        self.current_dropoff = list(self.env_cfg['dropoff_position'])

        # Pick target object
        self.target_object_idx = np.random.randint(0, self.env_cfg['num_objects'])

        # Curriculum-aware robot start
        start_x, start_y, _ = self._get_curriculum_start()
        target_shelf = np.array(self.current_shelf_positions[self.target_object_idx // 2])
        start_yaw = float(np.arctan2(target_shelf[1] - start_y, target_shelf[0] - start_x))
        p.resetBasePositionAndOrientation(
            self.husky_id,
            [start_x, start_y, 0.02],
            p.getQuaternionFromEuler([0, 0, start_yaw])
        )
        p.resetBaseVelocity(self.husky_id, [0,0,0], [0,0,0])
        self._sync_panda_to_husky([start_x, start_y, 0.02], start_yaw)

        # Reset arm
        for i, pos in enumerate(self.arm_home):
            p.resetJointState(self.panda_id, i, pos)

        # Place objects on randomized shelves with noise
        base_positions = [
            [self.current_shelf_positions[0][0]-0.2,
             self.current_shelf_positions[0][1] + self.shelf_object_front_offset, 0.58],
            [self.current_shelf_positions[0][0]+0.2,
             self.current_shelf_positions[0][1] + self.shelf_object_front_offset, 0.58],
            [self.current_shelf_positions[1][0]-0.2,
             self.current_shelf_positions[1][1] + self.shelf_object_front_offset, 0.58],
            [self.current_shelf_positions[1][0]+0.2,
             self.current_shelf_positions[1][1] + self.shelf_object_front_offset, 0.58],
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

    def reset_pickup_task(
        self,
        target_idx=None,
        distractors=False,
        base_noise=0.03,
        obj_noise=0.01,
        ready_y_offset=-0.08,
        ready_z_clearance=0.18,
        ready_y_jitter=0.015,
        ready_z_jitter=0.02,
    ):
        """
        Reset a simplified shelf-front pickup task for learning arm pickup in isolation.
        - base starts already at a valid pickup pose
        - target object remains on the shelf
        - optional distractors can be hidden away to keep the task focused
        """
        self.step_count = 0
        self.reached_shelf = False
        self.reached_object = False
        self.grasped_object = False
        self.delivered_object = False
        self.lifted_object = False
        self.phase_bonuses = 0.0
        self.base_collision_flag = False
        self.base_collision_penalty = 0.0
        self._release_grasp_constraint()

        self.current_shelf_positions = [list(pos) for pos in self.env_cfg['shelf_positions']]
        self.current_dropoff = list(self.env_cfg['dropoff_position'])
        if target_idx is None:
            target_idx = np.random.randint(0, self.env_cfg['num_objects'])
        self.target_object_idx = int(target_idx)

        base_positions = [
            [self.current_shelf_positions[0][0]-0.2, self.current_shelf_positions[0][1] + self.shelf_object_front_offset, 0.58],
            [self.current_shelf_positions[0][0]+0.2, self.current_shelf_positions[0][1] + self.shelf_object_front_offset, 0.58],
            [self.current_shelf_positions[1][0]-0.2, self.current_shelf_positions[1][1] + self.shelf_object_front_offset, 0.58],
            [self.current_shelf_positions[1][0]+0.2, self.current_shelf_positions[1][1] + self.shelf_object_front_offset, 0.58],
        ]
        for i, obj_id in enumerate(self.object_ids):
            if i == self.target_object_idx or distractors:
                noise = np.random.uniform(-obj_noise, obj_noise, 2)
                pos = [base_positions[i][0] + noise[0], base_positions[i][1] + noise[1], base_positions[i][2]]
            else:
                pos = [base_positions[i][0], base_positions[i][1] + 1.6, 0.05]
            p.resetBasePositionAndOrientation(obj_id, pos, [0, 0, 0, 1])
            p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0])

        pick_xy, pick_yaw = self.get_pick_pose()
        start_xy = pick_xy + np.random.uniform(-base_noise, base_noise, 2).astype(np.float32)
        yaw = float(pick_yaw + np.random.uniform(-0.06, 0.06))
        p.resetBasePositionAndOrientation(
            self.husky_id,
            [float(start_xy[0]), float(start_xy[1]), 0.02],
            p.getQuaternionFromEuler([0, 0, yaw]),
        )
        p.resetBaseVelocity(self.husky_id, [0, 0, 0], [0, 0, 0])
        self._sync_panda_to_husky([float(start_xy[0]), float(start_xy[1]), 0.02], yaw)

        self.set_pickup_ready_pose(
            y_offset=float(ready_y_offset + np.random.uniform(-ready_y_jitter, ready_y_jitter)),
            z_clearance=float(ready_z_clearance + np.random.uniform(-ready_z_jitter, ready_z_jitter)),
        )

        color_instructions = {
            0: "pick up the red box from the shelf",
            1: "pick up the blue box from the shelf",
            2: "pick up the green box from the shelf",
            3: "pick up the yellow box from the shelf",
        }
        self.current_instruction = color_instructions[self.target_object_idx]
        return self.current_instruction

    def set_pickup_ready_pose(self, y_offset=-0.08, z_clearance=0.18, open_gripper=True):
        """
        Move the arm to the shelf-front pre-hover pose used by the isolated pickup module.
        This is the bridge state between navigation and learned pickup.
        """
        target_obj_pos, _ = p.getBasePositionAndOrientation(self.object_ids[self.target_object_idx])
        target_obj_pos = np.array(target_obj_pos, dtype=np.float32)
        prehover = target_obj_pos + np.array([0.0, y_offset, z_clearance], dtype=np.float32)
        joint_angles = p.calculateInverseKinematics(
            self.panda_id,
            11,
            prehover.tolist(),
            maxNumIterations=140,
            residualThreshold=1e-4,
        )
        for i, pos in enumerate(joint_angles[:7]):
            p.resetJointState(self.panda_id, i, pos)
        grip_pos = 0.04 if open_gripper else 0.0
        for gj in self.gripper_joints:
            p.resetJointState(self.panda_id, gj, grip_pos)

    def get_pickup_ready_joint_targets(self, y_offset=-0.08, z_clearance=0.18):
        """
        Return the 7 Panda joint targets for the shelf-front pickup-ready pre-hover pose.
        """
        target_obj_pos, _ = p.getBasePositionAndOrientation(self.object_ids[self.target_object_idx])
        target_obj_pos = np.array(target_obj_pos, dtype=np.float32)
        prehover = target_obj_pos + np.array([0.0, y_offset, z_clearance], dtype=np.float32)
        joint_angles = p.calculateInverseKinematics(
            self.panda_id,
            11,
            prehover.tolist(),
            maxNumIterations=140,
            residualThreshold=1e-4,
        )
        return np.array(joint_angles[:7], dtype=np.float32)

    def animate_pickup_ready_pose(self, y_offset=-0.08, z_clearance=0.18, open_gripper=True, steps=80):
        """
        Smoothly move the arm into the pickup-ready pre-hover pose using normal motor control.
        This avoids the abrupt reset used for training/task setup.
        """
        target_obj_pos, _ = p.getBasePositionAndOrientation(self.object_ids[self.target_object_idx])
        target_obj_pos = np.array(target_obj_pos, dtype=np.float32)
        prehover = target_obj_pos + np.array([0.0, y_offset, z_clearance], dtype=np.float32)
        joint_angles = p.calculateInverseKinematics(
            self.panda_id,
            11,
            prehover.tolist(),
            maxNumIterations=140,
            residualThreshold=1e-4,
        )
        target_joints = np.array(joint_angles[:7], dtype=np.float32)
        current_joints = np.array([p.getJointState(self.panda_id, j)[0] for j in self.arm_joints], dtype=np.float32)
        grip_pos = 0.04 if open_gripper else 0.0

        for k in range(steps):
            alpha = float(k + 1) / float(max(steps, 1))
            interp = (1.0 - alpha) * current_joints + alpha * target_joints
            for j, val in enumerate(interp):
                p.resetJointState(self.panda_id, self.arm_joints[j], float(val))
            for gj in self.gripper_joints:
                p.resetJointState(self.panda_id, gj, grip_pos)
            p.stepSimulation()

    def servo_pickup_ready_pose(self, y_offset=-0.08, z_clearance=0.18, open_gripper=True, steps=60, tolerance=0.02):
        """
        Smoothly servo the end effector into the pickup-ready pre-hover pose using
        Cartesian delta control until the target is actually reached.
        """
        target_obj_pos, _ = p.getBasePositionAndOrientation(self.object_ids[self.target_object_idx])
        target_obj_pos = np.array(target_obj_pos, dtype=np.float32)
        prehover = target_obj_pos + np.array([0.0, y_offset, z_clearance], dtype=np.float32)
        grip = 1.0 if open_gripper else -1.0

        for _ in range(steps):
            ee = np.array(p.getLinkState(self.panda_id, 11)[0], dtype=np.float32)
            delta = np.clip((prehover - ee) / np.array([0.025, 0.025, 0.020], dtype=np.float32), -1.0, 1.0)
            self.apply_cartesian_action(np.array([delta[0], delta[1], delta[2], grip], dtype=np.float32))
            p.stepSimulation()
            ee_after = np.array(p.getLinkState(self.panda_id, 11)[0], dtype=np.float32)
            if np.linalg.norm(ee_after - prehover) < tolerance:
                break

    def step_pickup_cartesian(self, action):
        """
        Pickup-only control step used to train the isolated arm skill.
        The mobile base stays fixed at the shelf-front pickup pose.
        """
        self.base_collision_flag = False
        self.base_collision_penalty = 0.0
        self.apply_cartesian_action(action)
        p.stepSimulation()
        self.step_count += 1

        metrics = self._update_task_status()
        hover_target = metrics["obj_pos"] + np.array([0.0, 0.0, 0.14], dtype=np.float32)
        pregrasp_target = metrics["obj_pos"] + np.array([0.0, 0.0, 0.035], dtype=np.float32)
        hover_dist = float(np.linalg.norm(metrics["gripper_pos"] - hover_target))
        pregrasp_dist = float(np.linalg.norm(metrics["gripper_pos"] - pregrasp_target))
        xy_dist = float(np.linalg.norm(metrics["gripper_pos"][:2] - metrics["obj_pos"][:2]))
        z_gap = float(metrics["gripper_pos"][2] - metrics["obj_pos"][2])

        reward = -2.5 * hover_dist - 2.0 * xy_dist - 0.01
        if float(action[3]) > 0.0:
            reward += 0.4
        if xy_dist < 0.05:
            reward += 2.0
        if xy_dist < 0.03 and 0.07 <= z_gap <= 0.18 and float(action[3]) > 0.0:
            reward += 5.0
        if xy_dist < 0.035 and -0.01 <= z_gap <= 0.07 and float(action[3]) <= 0.0:
            reward += 8.0
        if metrics["attached"]:
            reward += 20.0
        if metrics["obj_z"] > 0.66:
            reward += 30.0
        if metrics["obj_speed"] > 2.0:
            reward -= 4.0

        done = bool(metrics["success"]) or self.step_count >= min(self.env_cfg["max_episode_steps"], 120)
        info = {
            "instruction": self.current_instruction,
            "grasped": self.grasped_object,
            "lifted": self.lifted_object,
            "attached": metrics["attached"],
            "dist_to_obj": metrics["dist_to_obj"],
            "obj_z": metrics["obj_z"],
            "hover_dist": hover_dist,
            "pregrasp_dist": pregrasp_dist,
            "xy_dist": xy_dist,
            "z_gap": z_gap,
            "success": metrics["success"],
        }
        return reward, done, info

    def reset(self):
        self._reset_common()
        return self.get_camera_image(), self.current_instruction

    def reset_state_only(self):
        """Reset without paying the camera-render cost."""
        self._reset_common()
        return self.current_instruction

    def step(self, action):
        # ===== ACTION SCALING FIX =====
        action = np.array(action).copy()

        # Navigation (keep small)
        action[0] *= 1.0   # forward velocity scale (you already multiply by 0.05 later)
        action[2] *= 1.0   # angular velocity scale

        # Arm joints (CRITICAL — expand range)
        action[3:9] *= 2.5   # allows joints to move beyond [-1,1]

        # Gripper (binary)
        action[9] = 1.0 if action[9] > 0 else 0.0
        # =============================
        # Navigation
        # Kinematic base update with collision rejection
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

        self._set_mobile_pose(new_pos, new_yaw)

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

        self._set_mobile_pose(new_pos, new_yaw)

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
            'base_collision': self.base_collision_flag,
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
