import pybullet as p
import pybullet_data
import numpy as np
import time
import yaml
import os

class MobileWarehouseEnv:
    def __init__(self, config_path="configs/config_mobile.yaml", render=False):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        self.render = render
        self.env_cfg = self.cfg['environment']
        self.step_count = 0

        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        # Richer instructions for mobile manipulation
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

        self.current_instruction = None
        self.husky_id = None
        self.panda_id = None
        self.object_ids = []
        self.shelf_ids = []
        self.target_object_idx = 0
        self.task_stage = 0  # 0=navigate, 1=reach, 2=grasp, 3=deliver

    def setup_world(self):
        """Set up physics and load ground"""
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.env_cfg['sim_timestep'])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")
        self._create_warehouse_structure()

    def _create_warehouse_structure(self):
        """Create warehouse walls and shelves"""
        room_size = self.env_cfg['workspace_size']

        # Walls
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

        # Shelves at configured positions
        self.shelf_ids = []
        shelf_positions = self.env_cfg['shelf_positions']
        for shelf_pos in shelf_positions:
            # Shelf base
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.3, 0.02])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.3, 0.02],
                                      rgbaColor=[0.5, 0.35, 0.1, 1])
            sid = p.createMultiBody(0, col, vis,
                                    [shelf_pos[0], shelf_pos[1], 0.5])
            self.shelf_ids.append(sid)

            # Shelf legs
            for lx, ly in [(-0.5, -0.25), (0.5, -0.25), (-0.5, 0.25), (0.5, 0.25)]:
                lc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.25])
                lv = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.25],
                                         rgbaColor=[0.3, 0.3, 0.3, 1])
                p.createMultiBody(0, lc, lv,
                                  [shelf_pos[0]+lx, shelf_pos[1]+ly, 0.25])

        # Dropoff zone marker (flat green pad)
        dropoff = self.env_cfg['dropoff_position']
        dc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.01])
        dv = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.01],
                                  rgbaColor=[0.0, 0.8, 0.0, 0.5])
        self.dropoff_id = p.createMultiBody(0, dc, dv,
                                             [dropoff[0], dropoff[1], 0.01])

    def load_robot(self):
        """
        Load Husky base with Panda arm mounted on top.
        Since we don't have a combined URDF, we load them separately
        and use a constraint to attach the arm to the base.
        """
        # Load Husky mobile base
        self.husky_id = p.loadURDF(
            "husky/husky.urdf",
            basePosition=[0, 0, 0.15],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False  # mobile! can drive around
        )

        # Load Panda arm positioned on top of Husky
        self.panda_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0.65],  # 0.65m above ground = on top of Husky
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False
        )

        # Attach Panda to Husky with a fixed constraint
        # This makes them move together as one robot
        self.attach_constraint = p.createConstraint(
            self.husky_id,
            -1,       # -1 = base link
            self.panda_id,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0.5],  # top of Husky
            [0, 0, 0]       # base of Panda
        )

        # Get Husky wheel joints for driving
        self.wheel_joints = []
        for j in range(p.getNumJoints(self.husky_id)):
            info = p.getJointInfo(self.husky_id, j)
            joint_name = info[1].decode('utf-8')
            if 'wheel' in joint_name.lower():
                self.wheel_joints.append(j)

        # Panda arm joints
        self.arm_joints = list(range(7))
        self.gripper_joints = [9, 10]

        # Set Panda home position
        home = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        for i, pos in enumerate(home):
            p.resetJointState(self.panda_id, i, pos)

        print(f"Husky loaded with {len(self.wheel_joints)} wheel joints")
        print(f"Panda loaded with 7 arm joints")
        print(f"Robots attached via fixed constraint")

    def load_objects(self):
        """Load colored boxes on shelves"""
        colors = [
            [1, 0, 0, 1],      # red   - shelf 1
            [0, 0, 1, 1],      # blue  - shelf 1
            [0, 1, 0, 1],      # green - shelf 2
            [1, 1, 0, 1],      # yellow- shelf 2
        ]
        shelf_positions = self.env_cfg['shelf_positions']

        # Place 2 objects per shelf, on top of shelf surface
        object_positions = [
            [shelf_positions[0][0] - 0.2, shelf_positions[0][1], 0.58],
            [shelf_positions[0][0] + 0.2, shelf_positions[0][1], 0.58],
            [shelf_positions[1][0] - 0.2, shelf_positions[1][1], 0.58],
            [shelf_positions[1][0] + 0.2, shelf_positions[1][1], 0.58],
        ]

        self.object_ids = []
        for i in range(self.env_cfg['num_objects']):
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.04])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.04],
                                       rgbaColor=colors[i])
            oid = p.createMultiBody(0.1, col, vis, object_positions[i])
            self.object_ids.append(oid)

        print(f"Loaded {len(self.object_ids)} objects across {len(shelf_positions)} shelves")

    def get_camera_image(self):
        """Camera mounted on Panda wrist"""
        w = self.env_cfg['camera_width']
        h = self.env_cfg['camera_height']

        # Get Husky base position to move camera with robot
        husky_pos, husky_orn = p.getBasePositionAndOrientation(self.husky_id)
        
        # Camera above and slightly in front of robot
        cam_pos = [husky_pos[0], husky_pos[1], husky_pos[2] + 1.5]
        cam_target = [husky_pos[0] + 1.0, husky_pos[1], husky_pos[2]]

        view_matrix = p.computeViewMatrix(cam_pos, cam_target, [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=80, aspect=w/h, nearVal=0.1, farVal=20
        )
        _, _, rgb, _, _ = p.getCameraImage(w, h, view_matrix, proj_matrix)
        rgb_array = np.array(rgb, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        return rgb_array

    def get_robot_state(self):
        """Get full robot state: base pose + arm joints"""
        # Husky base position and velocity
        base_pos, base_orn = p.getBasePositionAndOrientation(self.husky_id)
        base_vel, base_ang_vel = p.getBaseVelocity(self.husky_id)

        # Panda arm joint states
        arm_positions, arm_velocities = [], []
        for j in self.arm_joints:
            state = p.getJointState(self.panda_id, j)
            arm_positions.append(state[0])
            arm_velocities.append(state[1])

        return {
            'base_pos': np.array(base_pos),
            'base_orn': np.array(base_orn),
            'base_vel': np.array(base_vel),
            'arm_positions': np.array(arm_positions),
            'arm_velocities': np.array(arm_velocities)
        }

    def apply_action(self, action):
        """
        Apply 10-dim action:
        action[0:2] = base velocity (vx, vy)
        action[2]   = base rotation (wz)
        action[3:9] = arm joint angles
        action[9]   = gripper
        """
        # Base navigation: differential drive
        vx = float(action[0]) * 2.0   # forward/backward, scaled to 2 m/s max
        vy = float(action[1]) * 2.0   # lateral
        wz = float(action[2]) * 1.0   # rotation

        # Convert to wheel velocities (differential drive)
        left_vel  = (vx - wz * 0.3)   # 0.3 = half wheel base width
        right_vel = (vx + wz * 0.3)

        if self.wheel_joints:
            # Apply to all left/right wheels
            num_wheels = len(self.wheel_joints)
            half = num_wheels // 2
            for j in self.wheel_joints[:half]:
                p.setJointMotorControl2(self.husky_id, j,
                    p.VELOCITY_CONTROL, targetVelocity=left_vel, force=100)
            for j in self.wheel_joints[half:]:
                p.setJointMotorControl2(self.husky_id, j,
                    p.VELOCITY_CONTROL, targetVelocity=right_vel, force=100)

        # Arm control
        arm_action = action[3:9]
        p.setJointMotorControlArray(
            self.panda_id,
            self.arm_joints[:6],
            p.POSITION_CONTROL,
            targetPositions=arm_action,
            forces=[87] * 6
        )

        # Gripper
        gripper_pos = 0.04 if float(action[9]) > 0.5 else 0.0
        for gj in self.gripper_joints:
            p.setJointMotorControl2(self.panda_id, gj,
                p.POSITION_CONTROL, targetPosition=gripper_pos, force=10)

    def compute_reward(self):
        """
        Multi-stage reward:
        Stage 0 (navigate): reward for getting Husky close to target shelf
        Stage 1 (reach):    reward for getting gripper close to target object
        Stage 2 (grasp):    reward for maintaining grasp while moving
        Stage 3 (deliver):  reward for getting object to dropoff zone
        """
        husky_pos, _ = p.getBasePositionAndOrientation(self.husky_id)
        husky_pos = np.array(husky_pos[:2])  # x,y only

        shelf_positions = self.env_cfg['shelf_positions']
        target_shelf = shelf_positions[self.target_object_idx // 2]
        target_shelf = np.array(target_shelf)

        dist_to_shelf = np.linalg.norm(husky_pos - target_shelf)

        # Navigation reward
        nav_reward = -dist_to_shelf

        # Gripper to object reward
        gripper_state = p.getLinkState(self.panda_id, 11)
        gripper_pos = np.array(gripper_state[0])
        obj_pos, _ = p.getBasePositionAndOrientation(
            self.object_ids[self.target_object_idx]
        )
        obj_pos = np.array(obj_pos)
        dist_to_obj = np.linalg.norm(gripper_pos - obj_pos)
        reach_reward = -dist_to_obj

        # Dropoff reward
        dropoff = np.array(self.env_cfg['dropoff_position'] + [0.1])
        dist_to_dropoff = np.linalg.norm(obj_pos - dropoff)
        deliver_reward = -dist_to_dropoff

        # Weighted combination
        total_reward = (0.3 * nav_reward +
                        0.4 * reach_reward +
                        0.3 * deliver_reward)

        return total_reward

    def reset(self):
        """Reset for new episode"""
        self.step_count = 0
        self.task_stage = 0

        # Reset Husky to center
        p.resetBasePositionAndOrientation(
            self.husky_id,
            [0, 0, 0.15],
            p.getQuaternionFromEuler([0, 0, 0])
        )
        p.resetBaseVelocity(self.husky_id, [0,0,0], [0,0,0])

        # Reset Panda arm
        home = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        for i, pos in enumerate(home):
            p.resetJointState(self.panda_id, i, pos)

        # Reset objects on shelves with small noise
        shelf_positions = self.env_cfg['shelf_positions']
        base_positions = [
            [shelf_positions[0][0] - 0.2, shelf_positions[0][1], 0.58],
            [shelf_positions[0][0] + 0.2, shelf_positions[0][1], 0.58],
            [shelf_positions[1][0] - 0.2, shelf_positions[1][1], 0.58],
            [shelf_positions[1][0] + 0.2, shelf_positions[1][1], 0.58],
        ]
        for i, obj_id in enumerate(self.object_ids):
            noise = np.random.uniform(-0.02, 0.02, 2)
            pos = [base_positions[i][0] + noise[0],
                   base_positions[i][1] + noise[1],
                   base_positions[i][2]]
            p.resetBasePositionAndOrientation(obj_id, pos, [0,0,0,1])

        # Pick random target object and matching instruction
        self.target_object_idx = np.random.randint(0, self.env_cfg['num_objects'])
        self.current_instruction = np.random.choice(self.task_instructions)

        return self.get_camera_image(), self.current_instruction

    def step(self, action):
        """Step the simulation"""
        self.apply_action(action)
        p.stepSimulation()
        self.step_count += 1

        obs = self.get_camera_image()
        reward = self.compute_reward()
        done = self.step_count >= self.env_cfg['max_episode_steps']

        return obs, reward, done, {
            "instruction": self.current_instruction,
            "stage": self.task_stage
        }

    def initialize(self):
        """Full initialization"""
        self.setup_world()
        self.load_robot()
        self.load_objects()
        self.current_instruction = np.random.choice(self.task_instructions)
        print("Mobile environment initialized!")
        return self.get_camera_image(), self.current_instruction

    def close(self):
        p.disconnect()
