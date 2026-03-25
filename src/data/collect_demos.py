import pybullet as p
import numpy as np
import pickle
import os
import yaml
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.env.warehouse_env import WarehouseEnv

class IKExpertController:
    """
    Expert that uses PyBullet's built-in Inverse Kinematics (IK).
    IK takes a target 3D position and computes the exact joint angles
    needed to put the gripper there. Much more reliable than hardcoded angles.
    
    4 phases:
    0: Move above object (approach)
    1: Lower to object (reach)
    2: Close gripper (grasp)
    3: Lift up (retract)
    """
    def __init__(self, robot_id, gripper_link=11):
        self.robot_id = robot_id
        self.gripper_link = gripper_link  # link 11 = gripper tip on Panda
        self.phase = 0
        self.phase_steps = 0
        self.target_pos = None

    def reset(self, target_pos):
        self.phase = 0
        self.phase_steps = 0
        self.target_pos = np.array(target_pos)

    def compute_ik(self, position):
        """Ask PyBullet to compute joint angles for a given gripper position"""
        joint_angles = p.calculateInverseKinematics(
            self.robot_id,
            self.gripper_link,
            position,
            maxNumIterations=100,
            residualThreshold=0.001
        )
        return np.array(joint_angles[:7])  # first 7 = arm joints

    def get_action(self):
        """Return 7-dim action based on current phase"""
        tx, ty, tz = self.target_pos

        if self.phase == 0:
            # Hover 0.3m above object
            above_pos = [tx, ty, tz + 0.3]
            joints = self.compute_ik(above_pos)
            action = np.append(joints[:6], 1.0)  # open gripper
            if self.phase_steps >= 40:
                self.phase = 1
                self.phase_steps = 0

        elif self.phase == 1:
            # Move to object height
            reach_pos = [tx, ty, tz + 0.05]
            joints = self.compute_ik(reach_pos)
            action = np.append(joints[:6], 1.0)  # open gripper
            if self.phase_steps >= 40:
                self.phase = 2
                self.phase_steps = 0

        elif self.phase == 2:
            # Close gripper
            reach_pos = [tx, ty, tz + 0.05]
            joints = self.compute_ik(reach_pos)
            action = np.append(joints[:6], 0.0)  # close gripper
            if self.phase_steps >= 30:
                self.phase = 3
                self.phase_steps = 0

        elif self.phase == 3:
            # Lift up
            lift_pos = [tx, ty, tz + 0.4]
            joints = self.compute_ik(lift_pos)
            action = np.append(joints[:6], 0.0)  # keep closed
            # Stay in phase 3

        self.phase_steps += 1
        return action


def collect_demonstrations(num_demos=200, save_dir="data/demos"):
    os.makedirs(save_dir, exist_ok=True)

    with open("configs/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    env = WarehouseEnv(render=False)
    env.initialize()
    expert = IKExpertController(env.robot_id)

    demonstrations = []
    successful_demos = 0

    print(f"Collecting {num_demos} demonstrations...")

    for demo_idx in tqdm(range(num_demos)):
        obs, instruction = env.reset()

        # Get target object position
        target_id = env.object_ids[0]
        target_pos, _ = p.getBasePositionAndOrientation(target_id)
        expert.reset(target_pos)

        demo_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'instruction': instruction,
            'done': False
        }

        episode_reward = 0
        max_steps = cfg['environment']['max_episode_steps']

        for step in range(max_steps):
            action = expert.get_action()
            demo_data['observations'].append(obs.copy())
            demo_data['actions'].append(action.copy())

            obs, reward, done, info = env.step(action)
            demo_data['rewards'].append(reward)
            episode_reward += reward

            if done:
                demo_data['done'] = True
                break

        # Relaxed threshold - check best reward achieved
        best_reward = max(demo_data['rewards']) if demo_data['rewards'] else -999
        if best_reward > -0.3:
            demonstrations.append(demo_data)
            successful_demos += 1

    env.close()

    save_path = os.path.join(save_dir, 'demonstrations.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(demonstrations, f)

    print(f"\nCollection complete!")
    print(f"Successful demos: {successful_demos}/{num_demos}")
    print(f"Saved to: {save_path}")

    if demonstrations:
        metadata = {
            'num_demonstrations': len(demonstrations),
            'instruction_distribution': {},
            'avg_episode_length': float(np.mean([len(d['actions']) for d in demonstrations]))
        }
        for d in demonstrations:
            instr = d['instruction']
            metadata['instruction_distribution'][instr] = \
                metadata['instruction_distribution'].get(instr, 0) + 1

        meta_path = os.path.join(save_dir, 'metadata.yaml')
        with open(meta_path, 'w') as f:
            yaml.dump(metadata, f)

        print(f"Average episode length: {metadata['avg_episode_length']:.1f} steps")
        print(f"Instruction distribution: {metadata['instruction_distribution']}")
    else:
        print("No successful demos - will debug reward values")
        # Print reward stats from last episode to help diagnose
        if demo_data['rewards']:
            print(f"Last episode - best reward: {max(demo_data['rewards']):.3f}, final reward: {demo_data['rewards'][-1]:.3f}")

    return demonstrations


def create_dataset(demo_path="data/demos/demonstrations.pkl"):
    with open(demo_path, 'rb') as f:
        demos = pickle.load(f)

    if not demos:
        print("No demonstrations to create dataset from!")
        return [], []

    all_samples = []
    for demo in demos:
        for i in range(len(demo['actions'])):
            all_samples.append({
                'image': demo['observations'][i],
                'action': demo['actions'][i],
                'instruction': demo['instruction']
            })

    np.random.shuffle(all_samples)
    split = int(0.8 * len(all_samples))
    train_data = all_samples[:split]
    val_data = all_samples[split:]

    with open("data/demos/train_data.pkl", 'wb') as f:
        pickle.dump(train_data, f)
    with open("data/demos/val_data.pkl", 'wb') as f:
        pickle.dump(val_data, f)

    print(f"Dataset created!")
    print(f"Total samples: {len(all_samples)}")
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    return train_data, val_data


if __name__ == "__main__":
    demos = collect_demonstrations(num_demos=200)
    if demos:
        train_data, val_data = create_dataset()
