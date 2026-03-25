import pybullet as p
import numpy as np
import pickle
import os
import yaml
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.env.warehouse_env_mobile import MobileWarehouseEnv


class MobileExpertController:
    """
    5-phase expert controller for mobile manipulation:
    Phase 0: Navigate Husky to target shelf
    Phase 1: Align and stabilize near shelf
    Phase 2: IK reach to target object
    Phase 3: Close gripper and grasp
    Phase 4: Navigate to dropoff zone
    """
    def __init__(self, husky_id, panda_id, cfg):
        self.husky_id = husky_id
        self.panda_id = panda_id
        self.cfg = cfg
        self.phase = 0
        self.phase_steps = 0
        self.target_pos = None
        self.target_shelf = None
        self.gripper_link = 11

    def reset(self, target_object_idx, object_ids):
        self.phase = 0
        self.phase_steps = 0
        self.target_object_idx = target_object_idx

        # Get target object position
        obj_pos, _ = p.getBasePositionAndOrientation(object_ids[target_object_idx])
        self.target_pos = np.array(obj_pos)

        # Get target shelf position
        shelf_positions = self.cfg['environment']['shelf_positions']
        shelf_idx = target_object_idx // 2
        self.target_shelf = np.array(shelf_positions[shelf_idx] + [0.0])

        # Dropoff position
        self.dropoff_pos = np.array(self.cfg['environment']['dropoff_position'] + [0.1])

    def compute_ik(self, position):
        """Compute IK for Panda arm to reach position"""
        joint_angles = p.calculateInverseKinematics(
            self.panda_id,
            self.gripper_link,
            position,
            maxNumIterations=100,
            residualThreshold=0.001
        )
        return np.array(joint_angles[:7])

    def navigate_to(self, target_xy):
        """
        Proportional controller for navigation.
        Returns (vx, vy, wz) normalized to [-1, 1]
        """
        husky_pos, husky_orn = p.getBasePositionAndOrientation(self.husky_id)
        husky_pos = np.array(husky_pos[:2])

        # Get robot's current heading angle
        euler = p.getEulerFromQuaternion(husky_orn)
        heading = euler[2]  # yaw angle

        # Vector to target
        diff = np.array(target_xy) - husky_pos
        dist = np.linalg.norm(diff)

        if dist < 0.05:
            return 0.0, 0.0, 0.0, True  # reached target

        # Desired heading toward target
        desired_heading = np.arctan2(diff[1], diff[0])
        heading_error = desired_heading - heading

        # Normalize heading error to [-pi, pi]
        while heading_error > np.pi:  heading_error -= 2 * np.pi
        while heading_error < -np.pi: heading_error += 2 * np.pi

        # Proportional control
        vx = np.clip(dist * 0.5, 0, 1.0)           # forward speed
        wz = np.clip(heading_error * 1.0, -1.0, 1.0) # rotation

        # Slow down if we need to turn a lot
        if abs(heading_error) > 0.5:
            vx *= 0.3

        return float(vx), 0.0, float(wz), False

    def get_action(self, object_ids):
        """Return 10-dim action based on current phase"""
        # Default: arm home position, gripper open
        home_joints = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        action = np.zeros(10)
        action[3:9] = home_joints[:6]
        action[9] = 1.0  # gripper open

        tx, ty, tz = self.target_pos
        dropoff = self.dropoff_pos

        if self.phase == 0:
            # Navigate to shelf vicinity (1m away from shelf)
            nav_target = [self.target_shelf[0] * 0.5, self.target_shelf[1] * 0.5]
            vx, vy, wz, reached = self.navigate_to(nav_target)
            action[0] = vx
            action[1] = vy
            action[2] = wz
            action[3:9] = home_joints[:6]
            action[9] = 1.0

            if reached or self.phase_steps >= 80:
                self.phase = 1
                self.phase_steps = 0

        elif self.phase == 1:
            # Fine alignment - get closer to shelf
            nav_target = [self.target_shelf[0] * 0.7, self.target_shelf[1] * 0.7]
            vx, vy, wz, reached = self.navigate_to(nav_target)
            action[0] = vx * 0.5  # slower for fine alignment
            action[1] = vy
            action[2] = wz * 0.5
            action[3:9] = home_joints[:6]
            action[9] = 1.0

            if reached or self.phase_steps >= 60:
                self.phase = 2
                self.phase_steps = 0

        elif self.phase == 2:
            # Stop moving, reach arm above object
            action[0] = 0.0
            action[1] = 0.0
            action[2] = 0.0

            # Get current Husky position for IK offset
            husky_pos, _ = p.getBasePositionAndOrientation(self.husky_id)
            
            # Hover above object
            above_pos = [tx, ty, tz + 0.25]
            joints = self.compute_ik(above_pos)
            action[3:9] = joints[:6]
            action[9] = 1.0  # open gripper

            if self.phase_steps >= 50:
                self.phase = 3
                self.phase_steps = 0

        elif self.phase == 3:
            # Lower to object
            action[0] = 0.0
            action[1] = 0.0
            action[2] = 0.0

            reach_pos = [tx, ty, tz + 0.05]
            joints = self.compute_ik(reach_pos)
            action[3:9] = joints[:6]
            action[9] = 1.0  # open gripper

            if self.phase_steps >= 40:
                self.phase = 4
                self.phase_steps = 0

        elif self.phase == 4:
            # Close gripper
            action[0] = 0.0
            action[1] = 0.0
            action[2] = 0.0

            reach_pos = [tx, ty, tz + 0.05]
            joints = self.compute_ik(reach_pos)
            action[3:9] = joints[:6]
            action[9] = 0.0  # close gripper

            if self.phase_steps >= 30:
                self.phase = 5
                self.phase_steps = 0

        elif self.phase == 5:
            # Lift up then navigate to dropoff
            lift_pos = [tx, ty, tz + 0.4]
            joints = self.compute_ik(lift_pos)

            nav_target = [dropoff[0], dropoff[1]]
            vx, vy, wz, reached = self.navigate_to(nav_target)

            action[0] = vx * 0.5
            action[1] = vy
            action[2] = wz * 0.5
            action[3:9] = joints[:6]
            action[9] = 0.0  # keep gripper closed

            if reached or self.phase_steps >= 100:
                self.phase = 6
                self.phase_steps = 0

        elif self.phase == 6:
            # At dropoff - open gripper to place
            action[0] = 0.0
            action[1] = 0.0
            action[2] = 0.0

            drop_pos = [dropoff[0], dropoff[1], dropoff[2] + 0.1]
            joints = self.compute_ik(drop_pos)
            action[3:9] = joints[:6]
            action[9] = 1.0  # open gripper to release

        self.phase_steps += 1
        return action


def collect_mobile_demonstrations(num_demos=150, save_dir="data/demos_mobile"):
    os.makedirs(save_dir, exist_ok=True)

    with open("configs/config_mobile.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    env = MobileWarehouseEnv(render=False)
    env.initialize()

    expert = MobileExpertController(env.husky_id, env.panda_id, cfg)
    demonstrations = []
    successful_demos = 0

    print(f"Collecting {num_demos} mobile demonstrations...")

    for demo_idx in tqdm(range(num_demos)):
        obs, instruction = env.reset()
        expert.reset(env.target_object_idx, env.object_ids)

        demo_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'instruction': instruction,
            'done': False
        }

        max_steps = cfg['environment']['max_episode_steps']

        for step in range(max_steps):
            action = expert.get_action(env.object_ids)
            demo_data['observations'].append(obs.copy())
            demo_data['actions'].append(action.copy())

            obs, reward, done, info = env.step(action)
            demo_data['rewards'].append(reward)

            if done:
                demo_data['done'] = True
                break

        best_reward = max(demo_data['rewards']) if demo_data['rewards'] else -999
        if best_reward > -2.0:
            demonstrations.append(demo_data)
            successful_demos += 1

    env.close()

    # Save demonstrations
    save_path = os.path.join(save_dir, 'demonstrations.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(demonstrations, f)

    print(f"\nCollection complete!")
    print(f"Successful: {successful_demos}/{num_demos}")
    print(f"Saved to: {save_path}")

    if demonstrations:
        metadata = {
            'num_demonstrations': len(demonstrations),
            'avg_episode_length': float(np.mean([len(d['actions']) for d in demonstrations])),
            'instruction_distribution': {}
        }
        for d in demonstrations:
            instr = d['instruction']
            metadata['instruction_distribution'][instr] = \
                metadata['instruction_distribution'].get(instr, 0) + 1

        with open(os.path.join(save_dir, 'metadata.yaml'), 'w') as f:
            yaml.dump(metadata, f)

        print(f"Avg episode length: {metadata['avg_episode_length']:.1f}")
        print(f"Instruction distribution: {metadata['instruction_distribution']}")
    else:
        if demo_data['rewards']:
            print(f"No successful demos. Best reward seen: {max(demo_data['rewards']):.3f}")

    return demonstrations


def create_mobile_dataset(demo_path="data/demos_mobile/demonstrations.pkl"):
    with open(demo_path, 'rb') as f:
        demos = pickle.load(f)

    if not demos:
        print("No demonstrations found!")
        return [], []

    # Subsample every 5th step to keep dataset manageable
    all_samples = []
    for demo in demos:
        for i in range(0, len(demo['actions']), 5):
            all_samples.append({
                'image': demo['observations'][i],
                'action': demo['actions'][i],
                'instruction': demo['instruction']
            })

    np.random.shuffle(all_samples)
    split = int(0.8 * len(all_samples))
    train_data = all_samples[:split]
    val_data = all_samples[split:]

    with open("data/demos_mobile/train_data.pkl", 'wb') as f:
        pickle.dump(train_data, f)
    with open("data/demos_mobile/val_data.pkl", 'wb') as f:
        pickle.dump(val_data, f)

    print(f"Dataset created!")
    print(f"Total samples: {len(all_samples)}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    return train_data, val_data


if __name__ == "__main__":
    demos = collect_mobile_demonstrations(num_demos=300)
    if demos:
        create_mobile_dataset()
