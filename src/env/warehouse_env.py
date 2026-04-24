import pybullet as p
import pybullet_data
import numpy as np
import time
import yaml
import os

class WarehouseEnv:
    def __init__(self, config_path="configs/config.yaml", render=False):
        # Load config
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        self.render = render
        self.env_cfg = self.cfg['environment']
        self.step_count = 0
        
        # Connect to PyBullet
        if render:
            self.physics_client = p.connect(p.GUI)   # opens a visual window
        else:
            self.physics_client = p.connect(p.DIRECT) # headless, no window (faster for training)
        
        # Task instructions the robot can receive
        self.task_instructions = [
            "pick up the red box",
            "pick up the blue box", 
            "pick up the green box",
            "place object on shelf",
            "move to the target location"
        ]
        
        self.current_instruction = None
        self.robot_id = None
        self.object_ids = []
        self.grasp_constraint = None
        self.object_half_extent = 0.03
        self.attach_dist_threshold = 0.075
        self.attach_height_threshold = 0.09
        self.attach_speed_threshold = 1.5
        self.success_lift_height = 0.09
        self.success_hold_steps = 4
        self.max_success_obj_speed = 1.5
        self.post_grasp_target_height = 0.16
        self.gripper_closed_threshold = 0.024
        self.attach_center_offset_threshold = 0.05
        self.attach_symmetry_threshold = 0.018
        self.ee_workspace_min = np.array([0.28, -0.45, 0.02], dtype=np.float32)
        self.ee_workspace_max = np.array([0.72, 0.45, 0.42], dtype=np.float32)
        
    def setup_world(self):
        """Set up physics, gravity, and load the ground plane"""
        p.setGravity(0, 0, -9.81)  # real world gravity
        p.setTimeStep(self.env_cfg['sim_timestep'])
        
        # PyBullet comes with built-in assets (plane, objects etc.)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load warehouse walls (simple boxes)
        self._create_warehouse_walls()
        
    def _create_warehouse_walls(self):
        """Create a simple rectangular warehouse boundary"""
        wall_thickness = 0.1
        wall_height = 2.0
        room_size = self.env_cfg['workspace_size']
        
        # Wall visual and collision shape
        wall_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[room_size, wall_thickness, wall_height/2]
        )
        wall_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[room_size, wall_thickness, wall_height/2],
            rgbaColor=[0.8, 0.8, 0.8, 1]  # grey walls
        )
        
        # Create 4 walls around the workspace
        wall_positions = [
            [0, room_size, wall_height/2],   # front
            [0, -room_size, wall_height/2],  # back
        ]
        self.wall_ids = []
        for pos in wall_positions:
            wid = p.createMultiBody(
                baseMass=0,  # mass=0 means static (immovable)
                baseCollisionShapeIndex=wall_shape,
                baseVisualShapeIndex=wall_visual,
                basePosition=pos
            )
            self.wall_ids.append(wid)

    def load_robot(self):
        """Load the Panda arm (we use this as our manipulator)"""
        # Panda is included in pybullet_data
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True  # fixed base since we're not using mobile base yet
        )
        
        # Get joint info
        self.num_joints = p.getNumJoints(self.robot_id)
        self.arm_joints = list(range(7))   # joints 0-6 are the arm
        self.gripper_joints = [9, 10]      # joints 9,10 are the gripper fingers
        
        # Set initial pose (home position)
        home_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        for i, pos in enumerate(home_positions):
            p.resetJointState(self.robot_id, i, pos)
            
        print(f"Robot loaded with {self.num_joints} joints")

    def load_objects(self):
        """Load colored boxes as warehouse objects"""
        self.object_ids = []
        colors = [
            [1, 0, 0, 1],   # red
            [0, 0, 1, 1],   # blue
            [0, 1, 0, 1],   # green
        ]
        positions = [
            [0.5, 0.0, 0.05],
            [0.5, 0.3, 0.05],
            [0.5, -0.3, 0.05],
        ]
        
        self.object_ids = []
        for i in range(self.env_cfg['num_objects']):
            col_shape = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[self.object_half_extent] * 3
            )
            vis_shape = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[self.object_half_extent] * 3, rgbaColor=colors[i]
            )
            obj_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=positions[i]
            )
            self.object_ids.append(obj_id)
        print(f"Loaded {len(self.object_ids)} objects")

    def get_camera_image(self):
        """Capture RGB image from robot's perspective"""
        w = self.env_cfg['camera_width']
        h = self.env_cfg['camera_height']
        
        # Camera positioned above and in front of robot (simulates wrist camera)
        cam_pos = [0.5, 0, 0.8]
        cam_target = [0.5, 0, 0]
        
        view_matrix = p.computeViewMatrix(cam_pos, cam_target, [0, 1, 0])
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=w/h, nearVal=0.1, farVal=10
        )
        
        _, _, rgb, _, _ = p.getCameraImage(w, h, view_matrix, proj_matrix)
        
        # Convert to numpy array and drop alpha channel
        rgb_array = np.array(rgb, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        return rgb_array

    def get_robot_state(self):
        """Get current joint positions and velocities"""
        positions, velocities = [], []
        for j in self.arm_joints:
            state = p.getJointState(self.robot_id, j)
            positions.append(state[0])   # joint angle
            velocities.append(state[1])  # joint velocity
        return np.array(positions), np.array(velocities)

    def apply_action(self, action):
        """Apply action: either 6-or-7 arm joints plus gripper."""
        action = np.asarray(action, dtype=np.float32)
        if action.shape[0] >= 8:
            arm_action = action[:7]
            gripper_action = action[7]
            controlled_joints = self.arm_joints[:7]
            forces = [87] * 7
        else:
            arm_action = action[:6]
            gripper_action = action[6]
            controlled_joints = self.arm_joints[:6]
            forces = [87] * 6
        
        # Move arm joints using position control
        p.setJointMotorControlArray(
            self.robot_id,
            controlled_joints,
            p.POSITION_CONTROL,
            targetPositions=arm_action,
            forces=forces  # max force in Newtons
        )
        
        # Open/close gripper
        gripper_pos = 0.04 if gripper_action > 0.5 else 0.0  # 0.04=open, 0=closed
        for gj in self.gripper_joints:
            p.setJointMotorControl2(self.robot_id, gj, p.POSITION_CONTROL,
                                     targetPosition=gripper_pos, force=25)

    def apply_cartesian_action(self, action):
        """
        Apply action in end-effector delta space:
        action = [dx, dy, dz, grip]
        where dx/dy/dz are small deltas and grip > 0 => open, else close.
        """
        action = np.asarray(action, dtype=np.float32)
        if action.shape[0] < 4:
            raise ValueError("Cartesian action must have 4 dims: dx, dy, dz, grip")

        ee_pos = np.array(p.getLinkState(self.robot_id, 11)[0], dtype=np.float32)
        delta = np.clip(action[:3], -1.0, 1.0) * np.array([0.03, 0.03, 0.025], dtype=np.float32)
        target_pos = np.clip(ee_pos + delta, self.ee_workspace_min, self.ee_workspace_max)

        joint_angles = p.calculateInverseKinematics(
            self.robot_id,
            11,
            target_pos.tolist(),
            maxNumIterations=120,
            residualThreshold=1e-4,
        )
        joint_targets = np.array(joint_angles[:7], dtype=np.float32)
        full_action = np.zeros(8, dtype=np.float32)
        full_action[:7] = joint_targets
        full_action[7] = 1.0 if action[3] > 0.0 else 0.0
        self.apply_action(full_action)

    def execute_pick_macro(self, hover_clearance=0.14, grasp_height=0.012, lift_height=0.20, settle_steps=10):
        """
        Deterministic grasp macro used once the policy has aligned cleanly over the target.
        Returns True only if the environment's normal success logic is satisfied.
        """
        target_id = self.object_ids[getattr(self, "_target_idx", 0)]
        obj_pos, _ = p.getBasePositionAndOrientation(target_id)
        obj_pos = np.array(obj_pos, dtype=np.float32)

        waypoints = [
            (obj_pos + np.array([0.0, 0.0, hover_clearance], dtype=np.float32), 1.0, 20),
            (obj_pos + np.array([0.0, 0.0, grasp_height], dtype=np.float32), 1.0, 24),
            (obj_pos + np.array([0.0, 0.0, grasp_height], dtype=np.float32), -1.0, 18),
            (obj_pos + np.array([0.0, 0.0, lift_height], dtype=np.float32), -1.0, 30),
        ]

        for target_pos, grip, steps in waypoints:
            for _ in range(steps):
                ee_pos = np.array(p.getLinkState(self.robot_id, 11)[0], dtype=np.float32)
                delta = np.clip(
                    (target_pos - ee_pos) / np.array([0.03, 0.03, 0.025], dtype=np.float32),
                    -1.0,
                    1.0,
                )
                action = np.array([delta[0], delta[1], delta[2], grip], dtype=np.float32)
                self.apply_cartesian_action(action)
                p.stepSimulation()
                self.step_count += 1
                self.update_success_state()

            # After the close segment, enforce a clean centered attachment if the box is in place.
            if grip < 0.0 and self.grasp_constraint is None:
                metrics = self.get_target_metrics()
                xy_dist = float(np.linalg.norm(metrics["gripper_pos"][:2] - metrics["obj_pos"][:2]))
                z_gap = float(metrics["gripper_pos"][2] - metrics["obj_pos"][2])
                if (
                    xy_dist < 0.035 and
                    0.0 <= z_gap <= 0.09 and
                    metrics["obj_z"] < 0.09
                ):
                    try:
                        # Weld at the live relative pose (including orientation)
                        # so the box doesn't snap or dangle.
                        ee_state = p.getLinkState(self.robot_id, 11)
                        obj_pos_w, obj_orn_w = p.getBasePositionAndOrientation(
                            metrics["target_id"]
                        )
                        inv_ee_pos, inv_ee_orn = p.invertTransform(
                            ee_state[0], ee_state[1]
                        )
                        rel_pos, rel_orn = p.multiplyTransforms(
                            inv_ee_pos, inv_ee_orn, obj_pos_w, obj_orn_w,
                        )
                        self.grasp_constraint = p.createConstraint(
                            parentBodyUniqueId=self.robot_id,
                            parentLinkIndex=11,
                            childBodyUniqueId=metrics["target_id"],
                            childLinkIndex=-1,
                            jointType=p.JOINT_FIXED,
                            jointAxis=[0, 0, 0],
                            parentFramePosition=list(rel_pos),
                            childFramePosition=[0, 0, 0],
                            parentFrameOrientation=list(rel_orn),
                            childFrameOrientation=[0, 0, 0, 1],
                        )
                        p.changeConstraint(self.grasp_constraint, maxForce=220)
                    except Exception:
                        self.grasp_constraint = None

        for _ in range(settle_steps):
            hold_action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
            self.apply_cartesian_action(hold_action)
            p.stepSimulation()
            self.step_count += 1
            success, _ = self.update_success_state()
            if success:
                return True

        success, _ = self.update_success_state()
        return bool(success)

    def reset_simple_task(self, target_idx=0, distractors=False, position_noise=0.015):
        """
        Reset a simplified pick task for Cartesian training.
        - one target color/instruction
        - optionally hides distractors away from workspace
        - keeps object positions tight and learnable
        """
        self.step_count = 0
        self._release_grasp_constraint()
        self._near_object = False
        self._grasped = False
        self._lift_count = 0
        self._prev_action = None
        self._current_action = None

        home_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        for i, pos in enumerate(home_positions):
            p.resetJointState(self.robot_id, i, pos)
        for gj in self.gripper_joints:
            p.resetJointState(self.robot_id, gj, 0.04)

        self._target_idx = int(target_idx)
        color_names = ["red", "blue", "green"]
        self.current_instruction = f"pick up the {color_names[self._target_idx]} box"

        target_base = np.array([0.50, 0.0, 0.05], dtype=np.float32)
        noise = np.random.uniform(-position_noise, position_noise, 2)
        target_pos = target_base.copy()
        target_pos[:2] += noise

        for i, obj_id in enumerate(self.object_ids):
            if i == self._target_idx:
                pos = target_pos.tolist()
            elif distractors:
                offset_y = 0.22 if i == 1 else -0.22
                pos = [0.50, offset_y, 0.05]
            else:
                pos = [1.8 + 0.2 * i, 1.8 + 0.2 * i, 0.05]
            p.resetBasePositionAndOrientation(obj_id, pos, [0, 0, 0, 1])
            p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0])

        return self.get_camera_image(), self.current_instruction

    def _release_grasp_constraint(self):
        if self.grasp_constraint is not None:
            try:
                p.removeConstraint(self.grasp_constraint)
            except Exception:
                pass
            self.grasp_constraint = None

    def _update_grasp_constraint(self):
        """Attach the target object when the fingers close around it."""
        metrics = self.get_target_metrics()

        if not metrics["gripper_closed"]:
            self._release_grasp_constraint()
            return metrics

        if self.grasp_constraint is None:
            close_enough = (
                metrics["dist"] < self.attach_dist_threshold and
                metrics["obj_z"] < self.attach_height_threshold and
                metrics["obj_speed"] < self.attach_speed_threshold and
                metrics["left_contacts"] > 0 and
                metrics["right_contacts"] > 0 and
                metrics["center_offset"] < self.attach_center_offset_threshold and
                metrics["finger_symmetry"] < self.attach_symmetry_threshold
            )
            if close_enough:
                target_id = metrics["target_id"]
                # Weld at the live relative pose so the box doesn't dangle.
                ee_state = p.getLinkState(self.robot_id, 11)
                obj_pos_w, obj_orn_w = p.getBasePositionAndOrientation(target_id)
                inv_ee_pos, inv_ee_orn = p.invertTransform(ee_state[0], ee_state[1])
                rel_pos, rel_orn = p.multiplyTransforms(
                    inv_ee_pos, inv_ee_orn, obj_pos_w, obj_orn_w,
                )
                self.grasp_constraint = p.createConstraint(
                    parentBodyUniqueId=self.robot_id,
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
                p.changeConstraint(self.grasp_constraint, maxForce=200)

        metrics["attached"] = self.grasp_constraint is not None
        return metrics

    def _has_physical_hold(self, metrics):
        return (
            metrics["gripper_closed"] and
            metrics["left_contacts"] > 0 and
            metrics["right_contacts"] > 0 and
            metrics["center_offset"] < 0.05 and
            metrics["finger_symmetry"] < 0.025
        )

    def compute_reward(self):
        """Reward for deliberate grasping — not flinging."""
        gripper_state = p.getLinkState(self.robot_id, 11)
        gripper_pos = np.array(gripper_state[0])

        # Target object (set during reset)
        target_id = self.object_ids[getattr(self, '_target_idx', 0)]
        obj_pos, _ = p.getBasePositionAndOrientation(target_id)
        obj_pos = np.array(obj_pos)
        obj_vel, _ = p.getBaseVelocity(target_id)
        obj_speed = np.linalg.norm(obj_vel)
        left_finger_pos = np.array(p.getLinkState(self.robot_id, 9)[0])
        right_finger_pos = np.array(p.getLinkState(self.robot_id, 10)[0])
        finger_mid = 0.5 * (left_finger_pos + right_finger_pos)
        center_offset = np.linalg.norm(finger_mid - obj_pos)
        finger_symmetry = abs(
            np.linalg.norm(left_finger_pos - obj_pos) -
            np.linalg.norm(right_finger_pos - obj_pos)
        )

        dist = np.linalg.norm(gripper_pos - obj_pos)
        obj_z = obj_pos[2]
        gripper_opening = p.getJointState(self.robot_id, 9)[0]  # 0=closed, 0.04=open
        gripper_closed = gripper_opening < 0.02

        reward = 0.0

        # 1. Approach target — dense
        reward -= dist * 2.0

        # 2. Proximity milestones
        if dist < 0.15: reward += 1.0
        if dist < 0.08: reward += 3.0
        if dist < 0.05: reward += 5.0

        # 3. Gripper OPEN during approach (so it can wrap around object)
        if dist > 0.08 and not gripper_closed:
            reward += 1.0  # keep open while approaching

        # 4. Encourage approaching the center between the fingers.
        if dist < 0.10:
            reward += max(0.0, 0.05 - center_offset) * 60.0
            reward += max(0.0, 0.03 - finger_symmetry) * 40.0

        # 5. Gripper CLOSE only when centered and very close.
        if dist < 0.06 and gripper_closed and center_offset < 0.04 and finger_symmetry < 0.025:
            reward += 8.0

        if self.grasp_constraint is not None:
            reward += 12.0
        elif self._has_physical_hold(self.get_target_metrics()):
            reward += 10.0

        # 6. REAL grasp: lifted while either physically held or constraint-held.
        self._grasped = False
        self._near_object = dist < 0.08
        metrics = self.get_target_metrics()
        physically_held = self._has_physical_hold(metrics)
        if dist < 0.08 and obj_z > self.success_lift_height and gripper_closed and obj_speed < 1.0 and (physically_held or self.grasp_constraint is not None):
            self._grasped = True
            reward += 30.0
            reward += obj_z * 40.0  # lift higher = better

        # 6b. Encourage actually holding the object up, not just barely lifting it.
        if (self.grasp_constraint is not None or physically_held) and gripper_closed:
            reward += max(0.0, obj_z - self.success_lift_height) * 60.0
            reward -= max(0.0, self.post_grasp_target_height - obj_z) * 10.0

        # 7. Penalty for violence — object moving fast means flinging not grasping
        if obj_speed > 2.0:
            reward -= 10.0
        if obj_z > 0.10 and dist > 0.15:
            reward -= 10.0  # object airborne but far from gripper = fling

        # 8. Smooth motion bonus — penalize jerky actions
        if self._prev_action is not None and self._current_action is not None:
            action_delta = np.linalg.norm(
                np.array(self._current_action[:6]) - np.array(self._prev_action[:6]))
            if action_delta < 0.3:
                reward += 0.5  # smooth motion bonus
            elif action_delta > 1.5:
                reward -= 1.0  # jerk penalty

        # 9. Time penalty
        reward -= 0.01

        return reward

    def reset(self):
        """Reset environment for a new episode"""
        self.step_count = 0
        
        # Reset robot to home position
        home_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        for i, pos in enumerate(home_positions):
            p.resetJointState(self.robot_id, i, pos)
        
        # Randomize object positions slightly
        for i, obj_id in enumerate(self.object_ids):
            noise = np.random.uniform(-0.05, 0.05, 2)
            base_pos = [0.5 + noise[0], (i-1)*0.3 + noise[1], 0.05]
            p.resetBasePositionAndOrientation(obj_id, base_pos, [0,0,0,1])
            p.resetBaseVelocity(obj_id, [0,0,0], [0,0,0])

        # Reset internal state tracking
        self._release_grasp_constraint()
        self._near_object = False
        self._grasped = False
        self._lift_count = 0
        self._prev_action = None
        self._current_action = None

        # Pick a specific target object
        self._target_idx = np.random.randint(0, len(self.object_ids))
        color_names = ["red", "blue", "green"]
        self.current_instruction = f"pick up the {color_names[self._target_idx]} box"

        return self.get_camera_image(), self.current_instruction

    def get_target_metrics(self):
        """Return the current target-object metrics used by reward/success logic."""
        target_id = self.object_ids[getattr(self, '_target_idx', 0)]
        obj_pos, _ = p.getBasePositionAndOrientation(target_id)
        obj_pos = np.array(obj_pos)
        obj_vel, _ = p.getBaseVelocity(target_id)
        obj_speed = np.linalg.norm(obj_vel)
        gripper_pos = np.array(p.getLinkState(self.robot_id, 11)[0])
        gripper_orn = p.getLinkState(self.robot_id, 11)[1]
        left_finger_pos = np.array(p.getLinkState(self.robot_id, 9)[0])
        right_finger_pos = np.array(p.getLinkState(self.robot_id, 10)[0])
        finger_mid = 0.5 * (left_finger_pos + right_finger_pos)
        left_dist = float(np.linalg.norm(left_finger_pos - obj_pos))
        right_dist = float(np.linalg.norm(right_finger_pos - obj_pos))
        center_offset = float(np.linalg.norm(finger_mid - obj_pos))
        finger_symmetry = float(abs(left_dist - right_dist))
        inv_gripper_pos, inv_gripper_orn = p.invertTransform(
            gripper_pos.tolist(), gripper_orn
        )
        local_obj_pos, _ = p.multiplyTransforms(
            inv_gripper_pos, inv_gripper_orn, obj_pos.tolist(), [0, 0, 0, 1]
        )
        left_contacts = len(p.getContactPoints(self.robot_id, target_id, linkIndexA=9))
        right_contacts = len(p.getContactPoints(self.robot_id, target_id, linkIndexA=10))
        dist = np.linalg.norm(gripper_pos - obj_pos)
        gripper_opening = p.getJointState(self.robot_id, 9)[0]
        gripper_closed = gripper_opening < self.gripper_closed_threshold
        return {
            "target_id": target_id,
            "obj_pos": obj_pos,
            "obj_z": float(obj_pos[2]),
            "obj_speed": float(obj_speed),
            "gripper_pos": gripper_pos,
            "dist": float(dist),
            "gripper_closed": bool(gripper_closed),
            "left_contacts": int(left_contacts),
            "right_contacts": int(right_contacts),
            "center_offset": center_offset,
            "finger_symmetry": finger_symmetry,
            "local_obj_pos": list(local_obj_pos),
            "attached": bool(self.grasp_constraint is not None),
        }

    def update_success_state(self):
        """Update lift counters and return whether the task is successful."""
        metrics = self._update_grasp_constraint()
        physically_held = self._has_physical_hold(metrics)
        stable_hold = (
            (metrics.get("attached", False) or physically_held) and
            metrics["obj_z"] > self.success_lift_height and
            metrics["gripper_closed"] and
            metrics["obj_speed"] < self.max_success_obj_speed
        )
        self._grasped = stable_hold
        if stable_hold:
            self._lift_count += 1
        else:
            self._lift_count = 0
        success = self._lift_count >= self.success_hold_steps
        return success, metrics

    def step(self, action):
        """Run one simulation step"""
        self._prev_action = self._current_action
        self._current_action = action
        self.apply_action(action)
        p.stepSimulation()
        self.step_count += 1

        obs = self.get_camera_image()
        reward = self.compute_reward()

        success, metrics = self.update_success_state()
        done = success or self.step_count >= self.env_cfg['max_episode_steps']

        if success:
            reward += 100.0  # large terminal bonus for a stable held lift

        return obs, reward, done, {
            "instruction": self.current_instruction,
            "success": success,
            "target_dist": metrics["dist"],
            "target_obj_z": metrics["obj_z"],
            "target_obj_speed": metrics["obj_speed"],
            "gripper_closed": metrics["gripper_closed"],
        }

    def initialize(self):
        """Full initialization sequence"""
        self.setup_world()
        self.load_robot()
        self.load_objects()
        self.current_instruction = np.random.choice(self.task_instructions)
        print("Environment initialized successfully!")
        return self.get_camera_image(), self.current_instruction

    def close(self):
        """Disconnect from PyBullet"""
        p.disconnect()
