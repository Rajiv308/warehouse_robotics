import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pybullet as p
import numpy as np
import pickle
import yaml
from tqdm import tqdm
from src.env.warehouse_env_mobile import MobileWarehouseEnv
from src.data.collect_demos_mobile import MobileExpertController

def collect_safe(num_demos=150, save_dir="data/demos_mobile"):
    os.makedirs(save_dir, exist_ok=True)

    with open("configs/config_mobile.yaml") as f:
        cfg = yaml.safe_load(f)

    env = MobileWarehouseEnv(render=False)
    env.initialize()
    expert = MobileExpertController(env.husky_id, env.panda_id, cfg)

    demonstrations = []
    successful = 0
    max_steps = cfg['environment']['max_episode_steps']

    print(f"Collecting {num_demos} demos (saving incrementally)...")

    for demo_idx in tqdm(range(num_demos)):
        obs, instruction = env.reset()
        expert.reset(env.target_object_idx, env.object_ids)

        demo_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'instruction': instruction
        }

        for step in range(max_steps):
            action = expert.get_action(env.object_ids)
            demo_data['observations'].append(obs.copy())
            demo_data['actions'].append(action.copy())
            obs, reward, done, info = env.step(action)
            demo_data['rewards'].append(reward)
            if done:
                break

        best_reward = max(demo_data['rewards']) if demo_data['rewards'] else -999
        if best_reward > -2.0:
            demonstrations.append(demo_data)
            successful += 1

            # Save after every 10 successful demos
            if successful % 10 == 0:
                with open(f"{save_dir}/demonstrations.pkl", 'wb') as f:
                    pickle.dump(demonstrations, f)
                tqdm.write(f"Checkpoint: {successful} demos saved")

    env.close()

    # Final save
    with open(f"{save_dir}/demonstrations.pkl", 'wb') as f:
        pickle.dump(demonstrations, f)

    print(f"\nDone! Successful: {successful}/{num_demos}")

    if demonstrations:
        # Build dataset
        all_samples = []
        for demo in demonstrations:
            for i in range(0, len(demo['actions']), 5):
                all_samples.append({
                    'image': demo['observations'][i],
                    'action': demo['actions'][i],
                    'instruction': demo['instruction']
                })

        np.random.shuffle(all_samples)
        split = int(0.8 * len(all_samples))

        with open(f"{save_dir}/train_data.pkl", 'wb') as f:
            pickle.dump(all_samples[:split], f)
        with open(f"{save_dir}/val_data.pkl", 'wb') as f:
            pickle.dump(all_samples[split:], f)

        print(f"Total samples: {len(all_samples)}")
        print(f"Train: {len(all_samples[:split])}, Val: {len(all_samples[split:])}")

if __name__ == "__main__":
    collect_safe()
