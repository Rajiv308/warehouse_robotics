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


class MobilePPOMemory:
    def __init__(self):
        self.images       = []
        self.instructions = []
        self.states       = []
        self.actions      = []
        self.log_probs    = []
        self.rewards      = []
        self.values       = []
        self.dones        = []

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


class MobileVLAPPOPolicy(nn.Module):
    def __init__(self, config_path="configs/config_mobile.yaml"):
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
        action_dim = mc['action_dim']  # 10

        self.actor_mean = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),        nn.ReLU(),
            nn.Linear(128, action_dim), nn.Tanh()
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)

        self.critic = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),        nn.ReLU(),
            nn.Linear(128, 1)
        )

    def get_features(self, images, instructions, states):
        v = self.vision_encoder(images)
        l = self.language_encoder(instructions)
        s = self.state_encoder(states)
        return self.fusion(v, l, s)

    def get_action_and_value(self, images, instructions, states, action=None):
        features   = self.get_features(images, instructions, states)
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
        features = self.get_features(images, instructions, states)
        return self.critic(features).squeeze(-1)

    def load_bc_weights(self, bc_path, device):
        ckpt     = torch.load(bc_path, map_location=device, weights_only=False)
        bc_state = ckpt['model_state_dict']
        prefixes = ['vision_encoder', 'language_encoder', 'state_encoder', 'fusion']
        ppo_state = self.state_dict()
        loaded = 0
        for key, val in bc_state.items():
            for prefix in prefixes:
                if key.startswith(prefix) and key in ppo_state:
                    if ppo_state[key].shape == val.shape:
                        ppo_state[key] = val
                        loaded += 1
        self.load_state_dict(ppo_state)
        print(f"Loaded {loaded} layers from BC checkpoint")


def get_robot_state(env):
    """
    Extract 9-dim robot state from environment:
    [base_x, base_y, base_yaw, j0, j1, j2, j3, j4, j5]
    """
    base_pos, base_orn = p.getBasePositionAndOrientation(env.husky_id)
    yaw = p.getEulerFromQuaternion(base_orn)[2]
    arm_joints = []
    for j in range(6):
        state = p.getJointState(env.panda_id, j)
        arm_joints.append(state[0])
    return np.array([base_pos[0], base_pos[1], yaw] + arm_joints, dtype=np.float32)


def preprocess(obs_np, state_np, device):
    img   = torch.FloatTensor(obs_np).permute(2,0,1).unsqueeze(0) / 255.0
    state = torch.FloatTensor(state_np).unsqueeze(0)
    return img.to(device), state.to(device)


def train_mobile_ppo(config_path="configs/config_mobile.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tc     = cfg['training']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Mobile RL Training on: {device}")

    # PPO hyperparameters
    num_episodes      = tc['rl_episodes']
    steps_per_rollout = 256
    ppo_epochs        = 4
    clip_epsilon      = 0.2
    gamma             = tc['gamma']
    gae_lambda        = 0.95
    vf_coef           = 0.5
    ent_coef          = 0.01
    max_grad_norm     = 0.5
    lr                = 1e-4

    from src.env.warehouse_env_mobile_v2 import SimpleRewardWrapper
    env    = SimpleRewardWrapper(MobileWarehouseEnvV2(render=False, curriculum_stage=0))
    env.initialize()

    policy = MobileVLAPPOPolicy(config_path).to(device)
    freeze_language_encoder(policy)

    if os.path.exists("checkpoints/best_mobile_model.pth"):
        policy.load_bc_weights("checkpoints/best_mobile_model.pth", device)
        print("Mobile BC weights loaded!")
    else:
        print("No BC checkpoint found - training from scratch")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=lr, eps=1e-5
    )

    writer = SummaryWriter(log_dir="logs/mobile_rl")
    os.makedirs("checkpoints", exist_ok=True)

    memory           = MobilePPOMemory()
    episode_rewards  = deque(maxlen=20)
    episode_lengths  = deque(maxlen=20)
    best_mean_reward = -float('inf')

    obs, instruction = env.reset()
    robot_state      = get_robot_state(env)
    episode_reward   = 0
    episode_length   = 0
    total_episodes   = 0
    global_step      = 0

    print(f"\nStarting Mobile PPO for {num_episodes} episodes...")
    print(f"Steps per rollout: {steps_per_rollout}, PPO epochs: {ppo_epochs}\n")

    pbar = tqdm(total=num_episodes, desc="Episodes", unit="ep")

    while total_episodes < num_episodes:
        policy.eval()
        memory.clear()

        for _ in range(steps_per_rollout):
            img, state = preprocess(obs, robot_state, device)

            with torch.no_grad():
                action, log_prob, _, value = policy.get_action_and_value(
                    img, [instruction], state
                )

            action_np   = action.squeeze().numpy()
            next_obs, reward, done, info = env.step(action_np)
            next_state  = get_robot_state(env)

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
                total_episodes += 1
                episode_reward  = 0
                episode_length  = 0
                obs, instruction = env.reset()
                robot_state      = get_robot_state(env)
                pbar.update(1)

                if total_episodes % 20 == 0:
                    mean_r = np.mean(episode_rewards)
                    mean_l = np.mean(episode_lengths)
                    # Normalize by episode length for cleaner tracking
                    mean_r_norm = mean_r / mean_l if mean_l > 0 else mean_r
                    pbar.write(
                        f"Episode {total_episodes:4d}/{num_episodes} | "
                        f"Mean Reward: {mean_r:.1f} | "
                        f"Per-Step: {mean_r_norm:.3f} | "
                        f"Mean Length: {mean_l:.1f} | "
                        f"Best: {best_mean_reward:.1f}"
                    )
                    writer.add_scalar('MobileRL/mean_reward', mean_r, total_episodes)
                    writer.add_scalar('MobileRL/mean_length', mean_l, total_episodes)

                    if mean_r > best_mean_reward:
                        best_mean_reward = mean_r
                        torch.save({
                            'episode': total_episodes,
                            'model_state_dict': policy.state_dict(),
                            'mean_reward': best_mean_reward,
                        }, "checkpoints/best_mobile_rl_model.pth")
                        pbar.write(f"  ✓ Best model saved (mean_reward={best_mean_reward:.3f})")

        # GAE
        with torch.no_grad():
            img, state  = preprocess(obs, robot_state, device)
            next_value  = policy.get_value(img, [instruction], state).squeeze()

        rewards    = torch.cat(memory.rewards)
        dones      = torch.cat(memory.dones)
        values     = torch.stack(memory.values)

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
        imgs_batch    = torch.stack(memory.images)
        states_batch  = torch.stack(memory.states)
        actions_batch = torch.stack(memory.actions)
        old_log_probs = torch.stack(memory.log_probs).detach()

        for _ in range(ppo_epochs):
            indices         = torch.randperm(len(rewards))
            mini_batch_size = steps_per_rollout // 4

            for start in range(0, len(rewards), mini_batch_size):
                end    = start + mini_batch_size
                mb_idx = indices[start:end]

                mb_imgs   = imgs_batch[mb_idx]
                mb_states = states_batch[mb_idx]
                mb_instrs = [memory.instructions[i] for i in mb_idx]
                mb_acts   = actions_batch[mb_idx]
                mb_old_lp = old_log_probs[mb_idx]
                mb_adv    = advantages[mb_idx]
                mb_ret    = returns[mb_idx]

                _, new_lp, entropy, new_vals = policy.get_action_and_value(
                    mb_imgs, mb_instrs, mb_states, mb_acts
                )

                ratio    = (new_lp - mb_old_lp).exp()
                pg_loss  = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon)
                ).mean()
                v_loss   = nn.MSELoss()(new_vals, mb_ret)
                ent_loss = -entropy.mean()
                loss     = pg_loss + vf_coef * v_loss + ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        writer.add_scalar('MobileRL/policy_loss', pg_loss.item(), global_step)
        writer.add_scalar('MobileRL/value_loss',  v_loss.item(),  global_step)
        writer.add_scalar('MobileRL/entropy',    -ent_loss.item(), global_step)

    pbar.close()
    env.close()
    writer.close()
    print(f"\nMobile RL Training complete!")
    print(f"Best mean reward: {best_mean_reward:.3f}")
    print(f"Saved to checkpoints/best_mobile_rl_model.pth")


if __name__ == "__main__":
    train_mobile_ppo()
