import pybullet as p
import numpy as np
import pickle
import os
import yaml
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.env.warehouse_env_mobile_v2 import MobileWarehouseEnvV2


class ImprovedExpert:
    def __init__(self, husky_id, panda_id, cfg):
        self.husky_id     = husky_id
        self.panda_id     = panda_id
        self.cfg          = cfg
        self.phase        = 0
        self.phase_steps  = 0
        self.gripper_link = 11

    def reset(self, target_object_idx, object_ids):
        self.phase        = 0
        self.phase_steps  = 0
        obj_pos, _        = p.getBasePositionAndOrientation(object_ids[target_object_idx])
        self.target_pos   = np.array(obj_pos)
        self.dropoff_pos  = np.array(self.cfg['environment']['dropoff_position'] + [0.15])

        # Approach from origin (0,0) toward object
        # This is always a safe direction since origin is center of warehouse
        tx, ty = obj_pos[0], obj_pos[1]
        diff = np.array([0.0 - tx, 0.0 - ty])
        dist = np.linalg.norm(diff)
        if dist < 0.01:
            diff = np.array([1.0, 0.0])
        else:
            diff = diff / dist
        # Park 0.35m from object, approaching from warehouse center
        self.approach_pos = [tx + diff[0] * 0.35, ty + diff[1] * 0.35]

    def compute_ik(self, position):
        joints = p.calculateInverseKinematics(
            self.panda_id, self.gripper_link, position,
            maxNumIterations=200, residualThreshold=0.0001
        )
        return np.array(joints[:7])

    def navigate_to(self, target_xy, speed=1.0):
        """Navigate Husky to target_xy using proportional control"""
        husky_pos, husky_orn = p.getBasePositionAndOrientation(self.husky_id)
        husky_xy  = np.array(husky_pos[:2])
        heading   = p.getEulerFromQuaternion(husky_orn)[2]
        diff      = np.array(target_xy) - husky_xy
        dist      = np.linalg.norm(diff)
        if dist < 0.1:
            return 0.0, 0.0, 0.0, True
        desired = np.arctan2(diff[1], diff[0])
        err     = desired - heading
        while err >  np.pi: err -= 2*np.pi
        while err < -np.pi: err += 2*np.pi
        vx = np.clip(dist * 0.5 * speed, 0, speed)
        wz = np.clip(err * 1.5, -1.0, 1.0)
        if abs(err) > 0.4:
            vx *= 0.2
        return float(vx), 0.0, float(wz), False

    def get_approach_position(self):
        """Return the fixed approach position computed at reset time"""
        return self.approach_pos

    def get_action(self, object_ids):
        home   = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        action = np.zeros(10)
        action[3:9] = home[:6]
        action[9]   = 1.0

        tx, ty, tz = self.target_pos
        dropoff    = self.dropoff_pos

        if self.phase == 0:
            # Navigate directly toward object position
            approach = self.get_approach_position()
            vx, vy, wz, reached = self.navigate_to(approach, speed=1.0)
            action[0], action[1], action[2] = vx, vy, wz
            action[3:9] = home[:6]
            action[9]   = 1.0
            if reached or self.phase_steps >= 40:
                self.phase = 1
                self.phase_steps = 0

        elif self.phase == 1:
            # Fine approach — stop very close to object
            approach = self.get_approach_position()
            vx, vy, wz, reached = self.navigate_to(approach, speed=0.4)
            action[0], action[1], action[2] = vx*0.4, vy, wz*0.4
            action[3:9] = home[:6]
            action[9]   = 1.0
            if reached or self.phase_steps >= 30:
                self.phase = 2
                self.phase_steps = 0

        elif self.phase == 2:
            # Stop driving, hover arm above object
            action[0:3] = 0.0
            joints = self.compute_ik([tx, ty, tz + 0.3])
            action[3:9] = joints[:6]
            action[9]   = 1.0
            if self.phase_steps >= 60:
                self.phase = 3
                self.phase_steps = 0

        elif self.phase == 3:
            # Lower arm to object
            action[0:3] = 0.0
            joints = self.compute_ik([tx, ty, tz + 0.06])
            action[3:9] = joints[:6]
            action[9]   = 1.0
            if self.phase_steps >= 50:
                self.phase = 4
                self.phase_steps = 0

        elif self.phase == 4:
            # Close gripper
            action[0:3] = 0.0
            joints = self.compute_ik([tx, ty, tz + 0.06])
            action[3:9] = joints[:6]
            action[9]   = 0.0
            if self.phase_steps >= 40:
                self.phase = 5
                self.phase_steps = 0

        elif self.phase == 5:
            # Lift up
            action[0:3] = 0.0
            joints = self.compute_ik([tx, ty, tz + 0.45])
            action[3:9] = joints[:6]
            action[9]   = 0.0
            if self.phase_steps >= 40:
                self.phase = 6
                self.phase_steps = 0

        elif self.phase == 6:
            # Navigate to dropoff
            vx, vy, wz, reached = self.navigate_to(
                [dropoff[0], dropoff[1]], speed=0.6
            )
            joints = self.compute_ik([tx, ty, tz + 0.45])
            action[0], action[1], action[2] = vx*0.5, vy, wz*0.5
            action[3:9] = joints[:6]
            action[9]   = 0.0
            if reached or self.phase_steps >= 40:
                self.phase = 7
                self.phase_steps = 0

        elif self.phase == 7:
            # Place at dropoff
            action[0:3] = 0.0
            joints = self.compute_ik([dropoff[0], dropoff[1], dropoff[2]+0.1])
            action[3:9] = joints[:6]
            action[9]   = 1.0

        self.phase_steps += 1
        return action


def collect_v2_demonstrations(num_demos=150, save_dir="data/demos_mobile"):
    os.makedirs(save_dir, exist_ok=True)
    with open("configs/config_mobile.yaml") as f:
        cfg = yaml.safe_load(f)

    env    = MobileWarehouseEnvV2(render=False, curriculum_stage=0)
    env.initialize()
    expert = ImprovedExpert(env.husky_id, env.panda_id, cfg)

    demonstrations = []
    successful     = 0
    max_steps      = cfg['environment']['max_episode_steps']

    print(f"Collecting {num_demos} high-quality demonstrations...")

    for demo_idx in tqdm(range(num_demos)):
        obs, instruction = env.reset()
        expert.reset(env.target_object_idx, env.object_ids)

        demo_data = {
            'observations':   [],
            'actions':        [],
            'rewards':        [],
            'instruction':    instruction,
            'phases_reached': set()
        }

        for step in range(max_steps):
            action = expert.get_action(env.object_ids)
            # Downsample image to 84x84 to save disk space (7x smaller)
            import cv2
            small_obs = cv2.resize(obs, (84, 84))
            demo_data['observations'].append(small_obs)
            demo_data['actions'].append(action.copy())
            obs, reward, done, info = env.step(action)
            demo_data['rewards'].append(reward)

            if info['reached_shelf']:  demo_data['phases_reached'].add('shelf')
            if info['reached_object']: demo_data['phases_reached'].add('object')
            if info['grasped']:        demo_data['phases_reached'].add('grasp')
            if info['delivered']:      demo_data['phases_reached'].add('deliver')
            if done:
                break

        phases = demo_data['phases_reached']
        if 'shelf' in phases:
            demo_data['phases_reached'] = list(phases)
            demonstrations.append(demo_data)
            successful += 1

            if successful % 10 == 0:
                with open(f"{save_dir}/demonstrations.pkl", 'wb') as f:
                    pickle.dump(demonstrations, f)
                avg_steps = np.mean([len(d['actions']) for d in demonstrations])
                tqdm.write(
                    f"Checkpoint {successful} | "
                    f"shelf={sum(1 for d in demonstrations if 'shelf'   in d['phases_reached'])} "
                    f"object={sum(1 for d in demonstrations if 'object' in d['phases_reached'])} "
                    f"grasp={sum(1 for d in demonstrations if 'grasp'   in d['phases_reached'])} "
                    f"deliver={sum(1 for d in demonstrations if 'deliver' in d['phases_reached'])} "
                    f"avg_steps={avg_steps:.0f}"
                )

    env.close()

    with open(f"{save_dir}/demonstrations.pkl", 'wb') as f:
        pickle.dump(demonstrations, f)

    print(f"\nCollection complete! Successful: {successful}/{num_demos}")

    if demonstrations:
        phases_stats = {
            'shelf':   sum(1 for d in demonstrations if 'shelf'   in d['phases_reached']),
            'object':  sum(1 for d in demonstrations if 'object'  in d['phases_reached']),
            'grasp':   sum(1 for d in demonstrations if 'grasp'   in d['phases_reached']),
            'deliver': sum(1 for d in demonstrations if 'deliver' in d['phases_reached']),
        }
        print("Phase completion:")
        for phase, count in phases_stats.items():
            print(f"  {phase:10s}: {count}/{successful} ({100*count/successful:.1f}%)")

        all_samples = []
        for demo in demonstrations:
            for i in range(0, len(demo['actions']), 5):
                all_samples.append({
                    'image':       demo['observations'][i],
                    'action':      demo['actions'][i],
                    'instruction': demo['instruction']
                })

        np.random.shuffle(all_samples)
        split = int(0.8 * len(all_samples))
        with open(f"{save_dir}/train_data.pkl", 'wb') as f:
            pickle.dump(all_samples[:split], f)
        with open(f"{save_dir}/val_data.pkl", 'wb') as f:
            pickle.dump(all_samples[split:], f)

        print(f"\nDataset: {len(all_samples)} samples "
              f"({len(all_samples[:split])} train, {len(all_samples[split:])} val)")

    return demonstrations


if __name__ == "__main__":
    collect_v2_demonstrations(num_demos=300)
# patch applied via append - ignore
