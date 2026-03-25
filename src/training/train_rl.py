import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import numpy as np
import yaml
from collections import deque
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.env.warehouse_env import WarehouseEnv
from src.models.vla_model import VLAModel, freeze_language_encoder


class PPOMemory:
    def __init__(self):
        self.images = []
        self.instructions = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


class VLAPPOPolicy(nn.Module):
    def __init__(self, config_path="configs/config.yaml"):
        super().__init__()
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        mc = cfg['model']

        from src.models.vla_model import VisionEncoder, LanguageEncoder, CrossAttentionFusion
        self.vision_encoder = VisionEncoder(output_dim=mc['vision_features'])
        self.language_encoder = LanguageEncoder(output_dim=mc['language_features'])
        self.fusion = CrossAttentionFusion(
            vision_dim=mc['vision_features'],
            language_dim=mc['language_features'],
            fusion_dim=mc['fusion_dim']
        )

        fusion_dim = mc['fusion_dim']
        action_dim = mc['action_dim']

        self.actor_mean = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Tanh()
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)

        self.critic = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def get_features(self, images, instructions):
        v = self.vision_encoder(images)
        l = self.language_encoder(instructions)
        return self.fusion(v, l)

    def get_action_and_value(self, images, instructions, action=None):
        features = self.get_features(images, instructions)
        action_mean = self.actor_mean(features)
        action_std = self.actor_log_std.exp()
        dist = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(features).squeeze(-1)
        return action, log_prob, entropy, value

    def get_value(self, images, instructions):
        features = self.get_features(images, instructions)
        return self.critic(features).squeeze(-1)

    def load_bc_weights(self, bc_checkpoint_path, device):
        checkpoint = torch.load(bc_checkpoint_path, map_location=device, weights_only=False)
        bc_state = checkpoint['model_state_dict']
        mapping = {
            'vision_encoder': 'vision_encoder',
            'language_encoder': 'language_encoder',
            'fusion': 'fusion'
        }
        ppo_state = self.state_dict()
        loaded = 0
        for key, val in bc_state.items():
            for bc_prefix, ppo_prefix in mapping.items():
                if key.startswith(bc_prefix):
                    new_key = key.replace(bc_prefix, ppo_prefix, 1)
                    if new_key in ppo_state and ppo_state[new_key].shape == val.shape:
                        ppo_state[new_key] = val
                        loaded += 1
        self.load_state_dict(ppo_state)
        print(f"Loaded {loaded} layers from BC checkpoint")


def preprocess_image(obs_np, device):
    img = torch.FloatTensor(obs_np).permute(2, 0, 1).unsqueeze(0) / 255.0
    return img.to(device)


def train_ppo(config_path="configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tc = cfg['training']
    device = torch.device("cpu")
    print(f"RL Training on: {device}")

    num_episodes    = tc['rl_episodes']
    steps_per_rollout = 128
    ppo_epochs      = 4
    clip_epsilon    = 0.2
    gamma           = tc['gamma']
    gae_lambda      = 0.95
    vf_coef         = 0.5
    ent_coef        = 0.01
    max_grad_norm   = 0.5
    lr              = 1e-4

    env = WarehouseEnv(render=False)
    env.initialize()

    policy = VLAPPOPolicy(config_path).to(device)
    freeze_language_encoder(policy)

    if os.path.exists("checkpoints/best_model.pth"):
        policy.load_bc_weights("checkpoints/best_model.pth", device)
        print("BC weights loaded - starting from pretrained policy")
    else:
        print("No BC checkpoint found - training from scratch")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=lr, eps=1e-5
    )

    writer = SummaryWriter(log_dir="logs/rl")
    os.makedirs("checkpoints", exist_ok=True)

    memory = PPOMemory()
    episode_rewards = deque(maxlen=20)
    episode_lengths = deque(maxlen=20)
    best_mean_reward = -float('inf')

    obs, instruction = env.reset()
    episode_reward = 0
    episode_length = 0
    total_episodes = 0
    global_step = 0

    print(f"\nStarting PPO fine-tuning for {num_episodes} episodes...")
    print(f"Steps per rollout: {steps_per_rollout}, PPO epochs: {ppo_epochs}\n")

    # Progress bar tracks episodes
    pbar = tqdm(total=num_episodes, desc="Episodes", unit="ep")

    while total_episodes < num_episodes:
        policy.eval()
        memory.clear()

        for _ in range(steps_per_rollout):
            img = preprocess_image(obs, device)

            with torch.no_grad():
                action, log_prob, _, value = policy.get_action_and_value(img, [instruction])

            action_np = action.squeeze().numpy()
            next_obs, reward, done, info = env.step(action_np)

            memory.images.append(img.squeeze(0))
            memory.instructions.append(instruction)
            memory.actions.append(action.squeeze())
            memory.log_probs.append(log_prob.squeeze())
            memory.rewards.append(torch.FloatTensor([reward]))
            memory.values.append(value.squeeze())
            memory.dones.append(torch.FloatTensor([1.0 if done else 0.0]))

            episode_reward += reward
            episode_length += 1
            global_step += 1
            obs = next_obs

            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                total_episodes += 1
                episode_reward = 0
                episode_length = 0
                obs, instruction = env.reset()
                pbar.update(1)

                if total_episodes % 20 == 0:
                    mean_r = np.mean(episode_rewards)
                    mean_l = np.mean(episode_lengths)
                    pbar.write(
                        f"Episode {total_episodes:4d}/{num_episodes} | "
                        f"Mean Reward: {mean_r:.3f} | "
                        f"Mean Length: {mean_l:.1f} | "
                        f"Best: {best_mean_reward:.3f}"
                    )
                    writer.add_scalar('RL/mean_reward', mean_r, total_episodes)
                    writer.add_scalar('RL/mean_length', mean_l, total_episodes)

                    if mean_r > best_mean_reward:
                        best_mean_reward = mean_r
                        torch.save({
                            'episode': total_episodes,
                            'model_state_dict': policy.state_dict(),
                            'mean_reward': best_mean_reward,
                        }, "checkpoints/best_rl_model.pth")
                        pbar.write(f"  ✓ Best RL model saved (mean_reward={best_mean_reward:.3f})")

        # GAE
        with torch.no_grad():
            img = preprocess_image(obs, device)
            next_value = policy.get_value(img, [instruction]).squeeze()

        rewards = torch.cat(memory.rewards)
        dones   = torch.cat(memory.dones)
        values  = torch.stack(memory.values)

        advantages = torch.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_val  = next_value if t == len(rewards) - 1 else values[t + 1]
            next_done = dones[t]
            delta     = rewards[t] + gamma * next_val * (1 - next_done) - values[t]
            last_gae  = delta + gamma * gae_lambda * (1 - next_done) * last_gae
            advantages[t] = last_gae

        returns     = advantages + values
        advantages  = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        policy.train()
        imgs_batch    = torch.stack(memory.images)
        actions_batch = torch.stack(memory.actions)
        old_log_probs = torch.stack(memory.log_probs).detach()

        for _ in range(ppo_epochs):
            indices        = torch.randperm(len(rewards))
            mini_batch_size = steps_per_rollout // 4

            for start in range(0, len(rewards), mini_batch_size):
                end    = start + mini_batch_size
                mb_idx = indices[start:end]

                mb_imgs         = imgs_batch[mb_idx]
                mb_instructions = [memory.instructions[i] for i in mb_idx]
                mb_actions      = actions_batch[mb_idx]
                mb_old_lp       = old_log_probs[mb_idx]
                mb_adv          = advantages[mb_idx]
                mb_returns      = returns[mb_idx]

                _, new_lp, entropy, new_values = policy.get_action_and_value(
                    mb_imgs, mb_instructions, mb_actions
                )

                ratio    = (new_lp - mb_old_lp).exp()
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()
                v_loss   = nn.MSELoss()(new_values, mb_returns)
                ent_loss = -entropy.mean()
                loss     = pg_loss + vf_coef * v_loss + ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        writer.add_scalar('RL/policy_loss', pg_loss.item(), global_step)
        writer.add_scalar('RL/value_loss',  v_loss.item(),  global_step)
        writer.add_scalar('RL/entropy',    -ent_loss.item(), global_step)

    pbar.close()
    env.close()
    writer.close()
    print(f"\nRL Training complete!")
    print(f"Best mean reward: {best_mean_reward:.3f}")
    print(f"Best RL model saved to checkpoints/best_rl_model.pth")


if __name__ == "__main__":
    train_ppo()
