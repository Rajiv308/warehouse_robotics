"""
Cloud RL training - full mobile manipulation with navigation.
Key improvements over local version:
- CUDA GPU acceleration
- 2000 episodes with curriculum learning
- Larger rollouts (512 steps)
- Real robot state (not zeros)
- Full reward: navigate + reach + grasp + deliver
- Curriculum: starts easy, gets harder every 500 episodes
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import numpy as np
import yaml
import pybullet as p
from collections import deque
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.env.warehouse_env_mobile_v2 import MobileWarehouseEnvV2
from src.models.vla_model_mobile import MobileVLAModel, freeze_language_encoder


class CloudPPOMemory:
    def __init__(self):
        self.images = []
        self.instructions = []
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


class CloudVLAPPOPolicy(nn.Module):
    def __init__(self, config_path="configs/config_cloud.yaml"):
        super().__init__()
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        mc = cfg['model']

        from src.models.vla_model_mobile import (
            VisionEncoder, LanguageEncoder,
            RobotStateEncoder, CrossAttentionFusion
        )
        self.vision_encoder   = VisionEncoder(output_dim=mc['vision_features'])
        self.language_encoder = LanguageEncoder(output_dim=mc['language_features'])
        self.state_encoder    = RobotStateEncoder(state_dim=9, output_dim=64)
        self.fusion = CrossAttentionFusion(
            vision_dim=mc['vision_features'],
            language_dim=mc['language_features'],
            state_dim=64,
            fusion_dim=mc['fusion_dim']
        )

        fusion_dim = mc['fusion_dim']
        action_dim = mc['action_dim']

        self.actor_mean = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.ReLU(),
            nn.Linear(512, 256),        nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

        self.critic = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.ReLU(),
            nn.Linear(512, 256),        nn.ReLU(),
            nn.Linear(256, 1)
        )

    def get_features(self, images, instructions, states):
        v = self.vision_encoder(images)
        l = self.language_encoder(instructions)
        s = self.state_encoder(states)
        return self.fusion(v, l, s)

    def get_action_and_value(self, images, instructions, states, action=None):
        features    = self.get_features(images, instructions, states)
        action_mean = self.actor_mean(features)
        action_std  = self.actor_log_std.exp()
        dist        = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy  = dist.entropy().sum(-1)
        value    = self.critic(features).squeeze(-1)
        return action, log_prob, entropy, value

    def get_value(self, images, instructions, states):
        return self.critic(self.get_features(images, instructions, states)).squeeze(-1)

    def load_bc_weights(self, bc_path, device):
        ckpt      = torch.load(bc_path, map_location=device, weights_only=False)
        bc_state  = ckpt['model_state_dict']
        prefixes  = ['vision_encoder', 'language_encoder', 'state_encoder', 'fusion']
        ppo_state = self.state_dict()
        loaded    = 0
        for key, val in bc_state.items():
            for prefix in prefixes:
                if key.startswith(prefix) and key in ppo_state:
                    if ppo_state[key].shape == val.shape:
                        ppo_state[key] = val
                        loaded += 1
        self.load_state_dict(ppo_state)
        print(f"Loaded {loaded} layers from BC checkpoint")


def get_robot_state(env):
    base_pos, base_orn = p.getBasePositionAndOrientation(env.husky_id)
    yaw = p.getEulerFromQuaternion(base_orn)[2]
    arm_joints = [p.getJointState(env.panda_id, j)[0] for j in range(6)]
    return np.array([base_pos[0], base_pos[1], yaw] + arm_joints, dtype=np.float32)


def preprocess(obs_np, state_np, device):
    img   = torch.FloatTensor(obs_np).permute(2,0,1).unsqueeze(0) / 255.0
    state = torch.FloatTensor(state_np).unsqueeze(0)
    return img.to(device), state.to(device)


def compute_shaped_reward(env, prev_dist_to_shelf, prev_dist_to_obj):
    """
    Dense reward shaping for full task:
    1. Progress toward shelf
    2. Progress toward object
    3. Progress toward dropoff
    Plus phase bonuses
    """
    husky_pos, _ = p.getBasePositionAndOrientation(env.husky_id)
    husky_xy     = np.array(husky_pos[:2])

    shelf_positions = env.env_cfg['shelf_positions']
    shelf_idx    = env.target_object_idx // 2
    target_shelf = np.array(shelf_positions[shelf_idx])
    dist_to_shelf = np.linalg.norm(husky_xy - target_shelf)

    gripper_state = p.getLinkState(env.panda_id, 11)
    gripper_pos   = np.array(gripper_state[0])
    obj_pos, _    = p.getBasePositionAndOrientation(
        env.object_ids[env.target_object_idx]
    )
    obj_pos      = np.array(obj_pos)
    dist_to_obj  = np.linalg.norm(gripper_pos - obj_pos)

    dropoff      = np.array(env.env_cfg['dropoff_position'] + [0.1])
    dist_dropoff = np.linalg.norm(obj_pos - dropoff)

    # Progress rewards (positive when getting closer)
    shelf_progress = prev_dist_to_shelf - dist_to_shelf
    obj_progress   = prev_dist_to_obj   - dist_to_obj

    reward = (
        2.0 * shelf_progress +   # navigate to shelf
        3.0 * obj_progress   +   # reach object
        -0.1 * dist_dropoff      # delivery shaping
    )

    # Phase bonuses
    if dist_to_shelf < 1.2:  reward += 0.5
    if dist_to_obj   < 0.2:  reward += 1.0
    if dist_to_obj   < 0.1:  reward += 2.0
    if dist_dropoff  < 0.3:  reward += 5.0

    return reward, dist_to_shelf, dist_to_obj


def train_cloud_ppo(config_path="configs/config_cloud.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tc     = cfg['training']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cloud RL Training on: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # PPO hyperparameters - optimized for cloud
    num_episodes      = tc['rl_episodes']      # 2000
    steps_per_rollout = 512                     # larger rollouts
    ppo_epochs        = 6                       # more updates per rollout
    clip_epsilon      = 0.2
    gamma             = tc['gamma']
    gae_lambda        = 0.95
    vf_coef           = 0.5
    ent_coef          = 0.02                    # slightly more exploration
    max_grad_norm     = 0.5
    lr                = 1e-4

    # Curriculum: increase difficulty every N episodes
    curriculum_schedule = {
        0:    0,   # episodes 0-499:    stage 0 (close start)
        500:  1,   # episodes 500-999:  stage 1 (medium)
        1000: 2,   # episodes 1000-1499: stage 2 (far)
        1500: 3,   # episodes 1500-2000: stage 3 (full random)
    }

    current_stage = 0
    env = MobileWarehouseEnvV2(
        config_path=config_path,
        render=False,
        curriculum_stage=current_stage
    )
    env.initialize()

    policy = CloudVLAPPOPolicy(config_path).to(device)
    freeze_language_encoder(policy)

    if os.path.exists("checkpoints/best_mobile_model.pth"):
        policy.load_bc_weights("checkpoints/best_mobile_model.pth", device)
        print("BC weights loaded!")
    else:
        print("No BC checkpoint - training from scratch")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=lr, eps=1e-5
    )
    # Cosine LR decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_episodes, eta_min=1e-6
    )

    writer = SummaryWriter(log_dir="logs/cloud_rl")
    os.makedirs("checkpoints", exist_ok=True)

    memory           = CloudPPOMemory()
    episode_rewards  = deque(maxlen=50)
    episode_lengths  = deque(maxlen=50)
    phase_successes  = {'shelf': deque(maxlen=50), 'object': deque(maxlen=50),
                        'grasp': deque(maxlen=50), 'deliver': deque(maxlen=50)}
    best_mean_reward = -float('inf')

    obs, instruction = env.reset()
    robot_state      = get_robot_state(env)
    episode_reward   = 0
    episode_length   = 0
    total_episodes   = 0
    global_step      = 0

    # Initial distances for progress reward
    husky_pos, _ = p.getBasePositionAndOrientation(env.husky_id)
    shelf_positions = env.env_cfg['shelf_positions']
    shelf_idx = env.target_object_idx // 2
    prev_dist_shelf = np.linalg.norm(
        np.array(husky_pos[:2]) - np.array(shelf_positions[shelf_idx])
    )
    gripper_state = p.getLinkState(env.panda_id, 11)
    obj_pos, _ = p.getBasePositionAndOrientation(env.object_ids[env.target_object_idx])
    prev_dist_obj = np.linalg.norm(np.array(gripper_state[0]) - np.array(obj_pos))

    ep_phases = set()

    print(f"\nStarting Cloud PPO - {num_episodes} episodes")
    print(f"Curriculum: 4 stages, switching every 500 episodes\n")

    pbar = tqdm(total=num_episodes, desc="Episodes", unit="ep")

    while total_episodes < num_episodes:
        # Update curriculum
        for threshold, stage in sorted(curriculum_schedule.items()):
            if total_episodes >= threshold:
                if stage != current_stage:
                    current_stage = stage
                    env.curriculum_stage = stage
                    pbar.write(f"\n>>> Curriculum stage {stage} activated at episode {total_episodes}")

        policy.eval()
        memory.clear()

        for _ in range(steps_per_rollout):
            img, state = preprocess(obs, robot_state, device)

            with torch.no_grad():
                action, log_prob, _, value = policy.get_action_and_value(
                    img, [instruction], state
                )

            action_np  = action.squeeze().cpu().numpy()
            next_obs, _, done, info = env.step(action_np)
            next_state = get_robot_state(env)

            # Compute shaped reward
            reward, prev_dist_shelf, prev_dist_obj = compute_shaped_reward(
                env, prev_dist_shelf, prev_dist_obj
            )

            # Track phases
            if info.get('reached_shelf'):  ep_phases.add('shelf')
            if info.get('reached_object'): ep_phases.add('object')
            if info.get('grasped'):        ep_phases.add('grasp')
            if info.get('delivered'):      ep_phases.add('deliver')

            memory.images.append(img.squeeze(0))
            memory.instructions.append(instruction)
            memory.states.append(state.squeeze(0))
            memory.actions.append(action.squeeze())
            memory.log_probs.append(log_prob.squeeze())
            memory.rewards.append(torch.FloatTensor([reward]))
            memory.values.append(value.squeeze())
            memory.dones.append(torch.FloatTensor([1.0 if done else 0.0]))

            episode_reward += reward
            episode_length += 1
            global_step    += 1
            obs         = next_obs
            robot_state = next_state

            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                for phase in ['shelf','object','grasp','deliver']:
                    phase_successes[phase].append(1 if phase in ep_phases else 0)

                total_episodes += 1
                episode_reward  = 0
                episode_length  = 0
                ep_phases       = set()
                obs, instruction = env.reset()
                robot_state      = get_robot_state(env)

                # Reset distances
                husky_pos, _ = p.getBasePositionAndOrientation(env.husky_id)
                shelf_idx = env.target_object_idx // 2
                prev_dist_shelf = np.linalg.norm(
                    np.array(husky_pos[:2]) - np.array(shelf_positions[shelf_idx])
                )
                gripper_state = p.getLinkState(env.panda_id, 11)
                obj_pos, _ = p.getBasePositionAndOrientation(
                    env.object_ids[env.target_object_idx]
                )
                prev_dist_obj = np.linalg.norm(
                    np.array(gripper_state[0]) - np.array(obj_pos)
                )

                pbar.update(1)
                scheduler.step()

                if total_episodes % 20 == 0:
                    mean_r = np.mean(episode_rewards)
                    rates  = {k: np.mean(v)*100 for k,v in phase_successes.items()
                              if len(v) > 0}
                    pbar.write(
                        f"Ep {total_episodes:4d} | "
                        f"Reward: {mean_r:.2f} | "
                        f"Stage: {current_stage} | "
                        f"Shelf: {rates.get('shelf',0):.0f}% | "
                        f"Object: {rates.get('object',0):.0f}% | "
                        f"Grasp: {rates.get('grasp',0):.0f}% | "
                        f"Deliver: {rates.get('deliver',0):.0f}%"
                    )
                    writer.add_scalar('CloudRL/mean_reward',   mean_r, total_episodes)
                    writer.add_scalar('CloudRL/shelf_rate',    rates.get('shelf',0),   total_episodes)
                    writer.add_scalar('CloudRL/object_rate',   rates.get('object',0),  total_episodes)
                    writer.add_scalar('CloudRL/grasp_rate',    rates.get('grasp',0),   total_episodes)
                    writer.add_scalar('CloudRL/deliver_rate',  rates.get('deliver',0), total_episodes)
                    writer.add_scalar('CloudRL/curriculum_stage', current_stage, total_episodes)

                    if mean_r > best_mean_reward:
                        best_mean_reward = mean_r
                        torch.save({
                            'episode':          total_episodes,
                            'model_state_dict': policy.state_dict(),
                            'mean_reward':      best_mean_reward,
                            'phase_rates':      rates,
                        }, "checkpoints/best_cloud_rl_model.pth")
                        pbar.write(f"  ✓ Best model saved (reward={best_mean_reward:.2f})")

                if (total_episodes % 100 == 0):
                    torch.save({
                        'episode':          total_episodes,
                        'model_state_dict': policy.state_dict(),
                        'mean_reward':      np.mean(episode_rewards),
                    }, f"checkpoints/cloud_rl_ep_{total_episodes}.pth")

        # GAE
        with torch.no_grad():
            img, state = preprocess(obs, robot_state, device)
            next_value = policy.get_value(img, [instruction], state).squeeze()

        rewards    = torch.cat(memory.rewards).to(device)
        dones      = torch.cat(memory.dones).to(device)
        values     = torch.stack(memory.values).to(device)

        advantages = torch.zeros_like(rewards)
        last_gae   = 0
        for t in reversed(range(len(rewards))):
            next_val  = next_value if t == len(rewards)-1 else values[t+1]
            delta     = rewards[t] + gamma * next_val * (1-dones[t]) - values[t]
            last_gae  = delta + gamma * gae_lambda * (1-dones[t]) * last_gae
            advantages[t] = last_gae

        returns    = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        policy.train()
        imgs_batch    = torch.stack(memory.images).to(device)
        states_batch  = torch.stack(memory.states).to(device)
        actions_batch = torch.stack(memory.actions).to(device)
        old_log_probs = torch.stack(memory.log_probs).detach().to(device)

        for _ in range(ppo_epochs):
            indices         = torch.randperm(len(rewards))
            mini_batch_size = steps_per_rollout // 8  # smaller minibatches

            for start in range(0, len(rewards), mini_batch_size):
                end    = start + mini_batch_size
                mb_idx = indices[start:end]

                _, new_lp, entropy, new_vals = policy.get_action_and_value(
                    imgs_batch[mb_idx],
                    [memory.instructions[i] for i in mb_idx],
                    states_batch[mb_idx],
                    actions_batch[mb_idx]
                )

                ratio    = (new_lp - old_log_probs[mb_idx]).exp()
                pg_loss  = torch.max(
                    -advantages[mb_idx] * ratio,
                    -advantages[mb_idx] * torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon)
                ).mean()
                v_loss   = nn.MSELoss()(new_vals, returns[mb_idx])
                ent_loss = -entropy.mean()
                loss     = pg_loss + vf_coef * v_loss + ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        writer.add_scalar('CloudRL/policy_loss', pg_loss.item(), global_step)
        writer.add_scalar('CloudRL/value_loss',  v_loss.item(),  global_step)

    pbar.close()
    env.close()
    writer.close()
    print(f"\nCloud RL Training complete!")
    print(f"Best mean reward: {best_mean_reward:.3f}")
    print(f"Saved to checkpoints/best_cloud_rl_model.pth")


if __name__ == "__main__":
    train_cloud_ppo()
