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

        # Phase tracking for reward shaping
        self.reached_shelf    = False
        self.reached_object   = False
        self.grasped_object   = False
        self.delivered_object = False
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
        """
        Shaped reward with phase bonuses.
        Each milestone gives a one-time bonus + continuous distance shaping.
        """
        husky_pos, _ = p.getBasePositionAndOrientation(self.husky_id)
        husky_xy     = np.array(husky_pos[:2])

        shelf_positions = self.env_cfg['shelf_positions']
        shelf_idx    = self.target_object_idx // 2
        target_shelf = np.array(shelf_positions[shelf_idx])
        dist_to_shelf = np.linalg.norm(husky_xy - target_shelf)

        gripper_state = p.getLinkState(self.panda_id, 11)
        gripper_pos   = np.array(gripper_state[0])
        obj_pos, _    = p.getBasePositionAndOrientation(
            self.object_ids[self.target_object_idx]
        )
        obj_pos      = np.array(obj_pos)
        dist_to_obj  = np.linalg.norm(gripper_pos - obj_pos)

        dropoff      = np.array(self.env_cfg['dropoff_position'] + [0.1])
        dist_dropoff = np.linalg.norm(obj_pos - dropoff)

        # Continuous shaping
        reward = -0.3 * dist_to_shelf - 0.4 * dist_to_obj - 0.3 * dist_dropoff

        # Phase bonuses — one time only
        if not self.reached_shelf and dist_to_shelf < 1.5:
            self.reached_shelf = True
            reward += 1.0
            self.phase_bonuses += 1.0

        if not self.reached_object and dist_to_obj < 0.25:
            self.reached_object = True
            reward += 2.0
            self.phase_bonuses += 2.0

        if not self.grasped_object and dist_to_obj < 0.08:
            self.grasped_object = True
            reward += 3.0
            self.phase_bonuses += 3.0

        if not self.delivered_object and dist_dropoff < 0.3:
            self.delivered_object = True
            reward += 5.0
            self.phase_bonuses += 5.0

        return reward

    def _get_curriculum_start(self):
        """Return robot start position based on curriculum stage"""
        shelf_positions = self.env_cfg['shelf_positions']
        target_shelf    = shelf_positions[self.target_object_idx // 2]

        if self.curriculum_stage == 0:
            # 0.4m from shelf - very close
            angle = np.random.uniform(-0.3, 0.3)  # narrow angle toward center
            direction = np.array([-target_shelf[0], -target_shelf[1]])
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            return [target_shelf[0] + direction[0]*0.4,
                    target_shelf[1] + direction[1]*0.4, 0.15]
        elif self.curriculum_stage == 1:
            # 2m from shelf
            angle = np.random.uniform(0, 2*np.pi)
            return [target_shelf[0] + 2*np.cos(angle),
                    target_shelf[1] + 2*np.sin(angle), 0.15]
        elif self.curriculum_stage == 2:
            # 3m from shelf
            angle = np.random.uniform(0, 2*np.pi)
            return [target_shelf[0] + 3*np.cos(angle),
                    target_shelf[1] + 3*np.sin(angle), 0.15]
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
        self.phase_bonuses    = 0.0

        # Pick target object first so curriculum can use shelf position
        self.target_object_idx = np.random.randint(0, self.env_cfg['num_objects'])

        # Curriculum-based start position
        start_pos = self._get_curriculum_start()
        p.resetBasePositionAndOrientation(
            self.husky_id, start_pos,
            p.getQuaternionFromEuler([0, 0, np.random.uniform(0, 2*np.pi)])
        )
        p.resetBaseVelocity(self.husky_id, [0,0,0], [0,0,0])

        # Reset arm
        home = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        for i, pos in enumerate(home):
            p.resetJointState(self.panda_id, i, pos)

        # Reset objects
        shelf_positions = self.env_cfg['shelf_positions']
        base_positions  = [
            [shelf_positions[0][0]-0.2, shelf_positions[0][1], 0.58],
            [shelf_positions[0][0]+0.2, shelf_positions[0][1], 0.58],
            [shelf_positions[1][0]-0.2, shelf_positions[1][1], 0.58],
            [shelf_positions[1][0]+0.2, shelf_positions[1][1], 0.58],
        ]
        for i, obj_id in enumerate(self.object_ids):
            noise = np.random.uniform(-0.02, 0.02, 2)
            pos   = [base_positions[i][0]+noise[0],
                     base_positions[i][1]+noise[1],
                     base_positions[i][2]]
            p.resetBasePositionAndOrientation(obj_id, pos, [0,0,0,1])

        self.current_instruction = np.random.choice(self.task_instructions)
        return self.get_camera_image(), self.current_instruction

    def step(self, action):
        # Navigation
        vx = float(action[0]) * 2.0
        wz = float(action[2]) * 1.0
        left_vel  = vx - wz * 0.3
        right_vel = vx + wz * 0.3
        half = len(self.wheel_joints) // 2
        for j in self.wheel_joints[:half]:
            p.setJointMotorControl2(self.husky_id, j,
                p.VELOCITY_CONTROL, targetVelocity=left_vel, force=500)
        for j in self.wheel_joints[half:]:
            p.setJointMotorControl2(self.husky_id, j,
                p.VELOCITY_CONTROL, targetVelocity=right_vel, force=500)

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
        reward = self.compute_reward()
        done   = self.step_count >= self.env_cfg['max_episode_steps']

        return obs, reward, done, {
            'instruction':    self.current_instruction,
            'phase_bonuses':  self.phase_bonuses,
            'reached_shelf':  self.reached_shelf,
            'reached_object': self.reached_object,
            'grasped':        self.grasped_object,
            'delivered':      self.delivered_object
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
