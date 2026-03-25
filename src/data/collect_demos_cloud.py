"""
Cloud version of data collection:
- 1000 demos instead of 300
- Full image resolution (no downsampling needed, disk is cheap)
- Better expert with tighter tolerances
- Saves every 20 demos
"""
import pybullet as p
import numpy as np
import pickle
import os
import yaml
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.env.warehouse_env_mobile_v2 import MobileWarehouseEnvV2
from src.data.collect_demos_mobile_v2 import ImprovedExpert


def collect_cloud_demonstrations(num_demos=1000, save_dir="data/demos_mobile"):
    os.makedirs(save_dir, exist_ok=True)

    with open("configs/config_cloud.yaml") as f:
        cfg = yaml.safe_load(f)

    # Override max steps for collection
    cfg['environment']['max_episode_steps'] = 600

    env    = MobileWarehouseEnvV2(
        config_path="configs/config_mobile.yaml",
        render=False,
        curriculum_stage=0
    )
    env.initialize()
    expert = ImprovedExpert(env.husky_id, env.panda_id, cfg)

    demonstrations = []
    successful     = 0
    max_steps      = cfg['environment']['max_episode_steps']

    print(f"Collecting {num_demos} cloud demonstrations...")
    print(f"Full resolution images, tight IK tolerances\n")

    for demo_idx in tqdm(range(num_demos)):
        obs, instruction = env.reset()
        expert.reset(env.target_object_idx, env.object_ids, env=env)

        demo_data = {
            'observations':   [],
            'actions':        [],
            'rewards':        [],
            'states':         [],
            'instruction':    instruction,
            'phases_reached': set()
        }

        for step in range(max_steps):
            action = expert.get_action(env.object_ids)
            # Downsample to save disk
            import cv2
            small_obs = cv2.resize(obs, (84, 84))
            import pybullet as p_s
            base_pos, base_orn = p_s.getBasePositionAndOrientation(env.husky_id)
            yaw = p_s.getEulerFromQuaternion(base_orn)[2]
            arm_j = [p_s.getJointState(env.panda_id, j)[0] for j in range(6)]
            import numpy as np_s
            robot_state = np_s.array([base_pos[0], base_pos[1], yaw] + arm_j, dtype=np_s.float32)
            demo_data['observations'].append(small_obs)
            demo_data['actions'].append(action.copy())
            demo_data['states'].append(robot_state)
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
            demo_data['quality'] = (
                3 if 'deliver' in phases else
                2 if 'grasp'   in phases else
                1 if 'object'  in phases else
                0
            )
            demonstrations.append(demo_data)
            successful += 1

            if successful % 20 == 0:
                with open(f"{save_dir}/demonstrations.pkl", 'wb') as f:
                    pickle.dump(demonstrations, f)
                tqdm.write(
                    f"Checkpoint {successful} | "
                    f"shelf={sum(1 for d in demonstrations if 'shelf'   in d['phases_reached'])} "
                    f"object={sum(1 for d in demonstrations if 'object' in d['phases_reached'])} "
                    f"grasp={sum(1 for d in demonstrations if 'grasp'   in d['phases_reached'])} "
                    f"deliver={sum(1 for d in demonstrations if 'deliver' in d['phases_reached'])}"
                )

    env.close()

    with open(f"{save_dir}/demonstrations.pkl", 'wb') as f:
        pickle.dump(demonstrations, f)

    print(f"\nCollection complete! Successful: {successful}/{num_demos}")

    if demonstrations:
        phases_stats = {k: sum(1 for d in demonstrations if k in d['phases_reached'])
                       for k in ['shelf','object','grasp','deliver']}
        print("Phase completion:")
        for phase, count in phases_stats.items():
            print(f"  {phase:10s}: {count}/{successful} ({100*count/successful:.1f}%)")

        # Subsample every 3rd step (less aggressive than local)
        all_samples = []
        for demo in demonstrations:
            quality = demo.get('quality', 0)
            repeat = {3: 6, 2: 4, 1: 2, 0: 1}.get(quality, 1)
            for _ in range(repeat):
                for i in range(0, len(demo['actions']), 3):
                    all_samples.append({
                        'image':       demo['observations'][i],
                        'action':      demo['actions'][i],
                        'instruction': demo['instruction'],
                        'state':       demo['states'][i] if 'states' in demo else None
                    })
        quality_counts = {}
        for d in demonstrations:
            q = d.get('quality', 0)
            quality_counts[q] = quality_counts.get(q, 0) + 1
        print(f"Quality distribution: {quality_counts}")

        np.random.shuffle(all_samples)
        split = int(0.8 * len(all_samples))
        with open(f"{save_dir}/train_data.pkl", 'wb') as f:
            pickle.dump(all_samples[:split], f)
        with open(f"{save_dir}/val_data.pkl", 'wb') as f:
            pickle.dump(all_samples[split:], f)

        print(f"\nDataset: {len(all_samples)} samples")
        print(f"Train: {len(all_samples[:split])}, Val: {len(all_samples[split:])}")

    return demonstrations


if __name__ == "__main__":
    collect_cloud_demonstrations(num_demos=1000)
