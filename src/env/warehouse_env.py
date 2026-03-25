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
            col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.04])
            vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.04], rgbaColor=colors[i])
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
        """Apply 7-dim action: 6 joint angles + 1 gripper"""
        arm_action = action[:6]
        gripper_action = action[6]
        
        # Move arm joints using position control
        p.setJointMotorControlArray(
            self.robot_id,
            self.arm_joints[:6],
            p.POSITION_CONTROL,
            targetPositions=arm_action,
            forces=[87] * 6  # max force in Newtons
        )
        
        # Open/close gripper
        gripper_pos = 0.04 if gripper_action > 0.5 else 0.0  # 0.04=open, 0=closed
        for gj in self.gripper_joints:
            p.setJointMotorControl2(self.robot_id, gj, p.POSITION_CONTROL,
                                     targetPosition=gripper_pos, force=10)

    def compute_reward(self):
        """Simple reward: how close is the gripper to the nearest object?"""
        # Get gripper position
        gripper_state = p.getLinkState(self.robot_id, 11)
        gripper_pos = np.array(gripper_state[0])
        
        # Find closest object
        min_dist = float('inf')
        for obj_id in self.object_ids:
            obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
            dist = np.linalg.norm(gripper_pos - np.array(obj_pos))
            min_dist = min(min_dist, dist)
        
        # Reward is negative distance (closer = higher reward)
        return -min_dist

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
        
        # Pick a random instruction
        self.current_instruction = np.random.choice(self.task_instructions)
        
        return self.get_camera_image(), self.current_instruction

    def step(self, action):
        """Run one simulation step"""
        self.apply_action(action)
        p.stepSimulation()
        self.step_count += 1
        
        obs = self.get_camera_image()
        reward = self.compute_reward()
        done = self.step_count >= self.env_cfg['max_episode_steps']
        
        return obs, reward, done, {"instruction": self.current_instruction}

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
