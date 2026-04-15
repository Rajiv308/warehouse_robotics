"""
Phase 2 state-based RL on MobileWarehouseEnvV2.

Goal for this overnight run:
- navigate to the correct shelf,
- reach the correct object,
- close the gripper,
- lift the object.

This intentionally optimizes pick-and-lift before full delivery because that is
the highest-probability path to a convincing demo under the time constraint.
"""
import os
import sys
import time
from collections import deque

import numpy as np
import pybullet as p
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.collect_demos_mobile_v2 import ImprovedExpert
from src.env.warehouse_env_mobile_v2 import MobileWarehouseEnvV2


class MobileStatePolicy(nn.Module):
    def __init__(self, state_dim=21, action_dim=10):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim) - 2.4)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action=None):
        mean = self.actor_mean(state)
        std = self.actor_log_std.exp().clamp(min=0.01, max=0.06)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        return (
            action,
            dist.log_prob(action).sum(-1),
            dist.entropy().sum(-1),
            self.critic(state).squeeze(-1),
        )


def get_state(env):
    husky_pos, husky_orn = p.getBasePositionAndOrientation(env.husky_id)
    yaw = p.getEulerFromQuaternion(husky_orn)[2]
    gripper_pos = np.array(p.getLinkState(env.panda_id, 11)[0], dtype=np.float32)
    obj_pos, _ = p.getBasePositionAndOrientation(env.object_ids[env.target_object_idx])
    obj_pos = np.array(obj_pos, dtype=np.float32)
    target_shelf = np.array(env.current_shelf_positions[env.target_object_idx // 2], dtype=np.float32)
    joints = np.array([p.getJointState(env.panda_id, j)[0] for j in range(6)], dtype=np.float32)
    gripper_opening = np.float32(p.getJointState(env.panda_id, 9)[0])
    robot_xy = np.array(husky_pos[:2], dtype=np.float32)
    dist_shelf = np.float32(np.linalg.norm(robot_xy - target_shelf))
    dist_obj = np.float32(np.linalg.norm(gripper_pos - obj_pos))
    return np.concatenate([
        np.array([husky_pos[0], husky_pos[1], yaw], dtype=np.float32),
        gripper_pos,
        obj_pos,
        target_shelf,
        joints,
        np.array([gripper_opening, dist_shelf, dist_obj, obj_pos[2]], dtype=np.float32),
    ]).astype(np.float32)


def collect_expert_dataset(policy, env, num_demos=150, max_steps=180):
    expert = ImprovedExpert(env.husky_id, env.panda_id, env.cfg)
    states = []
    actions = []

    print(f"Collecting {num_demos} expert mobile rollouts for warm start...", flush=True)
    for _ in range(num_demos):
        env.reset_state_only()
        expert.reset(env.target_object_idx, env.object_ids, env=env)
        for _ in range(max_steps):
            states.append(get_state(env))
            action = expert.get_action(env.object_ids).astype(np.float32)
            actions.append(action)
            reward, done, info = env.step_state_only(action)
            if info.get("success"):
                break
            if done:
                break

    return (
        torch.FloatTensor(np.array(states, dtype=np.float32)),
        torch.FloatTensor(np.array(actions, dtype=np.float32)),
    )


def pretrain_actor(policy, expert_states, expert_actions, epochs=10, batch_size=256):
    optimizer = torch.optim.Adam(policy.actor_mean.parameters(), lr=1e-3)
    num_samples = expert_states.shape[0]
    print(f"Pretraining actor on {num_samples:,} mobile expert pairs...", flush=True)

    for epoch in range(epochs):
        perm = torch.randperm(num_samples)
        losses = []
        for start in range(0, num_samples, batch_size):
            idx = perm[start:start + batch_size]
            pred = policy.actor_mean(expert_states[idx])
            nav_loss = nn.MSELoss()(pred[:, :3], expert_actions[idx][:, :3])
            arm_loss = nn.MSELoss()(pred[:, 3:9], expert_actions[idx][:, 3:9])
            grip_loss = nn.MSELoss()(pred[:, 9:], expert_actions[idx][:, 9:])
            loss = 2.0 * nav_loss + arm_loss + 5.0 * grip_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.actor_mean.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        print(f"  Warm start epoch {epoch + 1}/{epochs} | actor_loss={np.mean(losses):.4f}", flush=True)


def rollout_step(env, action_np):
    reward, done, info = env.step_state_only(action_np)
    if info.get("success"):
        reward += 150.0
    return reward, done, info


def evaluate_policy(policy, env, episodes=20):
    policy.eval()
    device = next(policy.parameters()).device
    rewards = []
    success = 0
    grasp = 0
    shelf = 0
    max_zs = []

    with torch.no_grad():
        for _ in range(episodes):
            env.reset_state_only()
            state = get_state(env)
            ep_reward = 0.0
            max_z = 0.0

            for _ in range(env.env_cfg["max_episode_steps"]):
                st = torch.FloatTensor(state).unsqueeze(0).to(device)
                action_np = policy.actor_mean(st).squeeze().cpu().numpy()
                reward, done, info = rollout_step(env, action_np)
                ep_reward += reward
                max_z = max(max_z, info.get("obj_z", 0.0))
                state = get_state(env)
                if info.get("reached_shelf"):
                    shelf = shelf + 1
                if info.get("grasped"):
                    grasp = grasp + 1
                if done:
                    if info.get("success"):
                        success += 1
                    break

            rewards.append(ep_reward)
            max_zs.append(max_z)

    return {
        "success_rate": 100.0 * success / max(episodes, 1),
        "grasp_rate": 100.0 * grasp / max(episodes, 1),
        "shelf_rate": 100.0 * shelf / max(episodes, 1),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_max_z": float(np.mean(max_zs)) if max_zs else 0.0,
    }


if __name__ == "__main__":
    NUM_EPISODES = 30000
    MAX_STEPS = 220
    EVAL_INTERVAL = 250
    EVAL_EPISODES = 20
    PPO_EPOCHS = 4
    BC_ANCHOR = 0.05

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 2 state RL on: {device}", flush=True)

    env = MobileWarehouseEnvV2(
        config_path="configs/config_cloud.yaml",
        render=False,
        curriculum_stage=0,
    )
    env.initialize()
    env.env_cfg["max_episode_steps"] = MAX_STEPS

    policy = MobileStatePolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    print(f"Params: {sum(pp.numel() for pp in policy.parameters()):,}", flush=True)

    warm_ckpt = "checkpoints/phase2_state_bc_init_v2.pth"
    if os.path.exists(warm_ckpt):
        policy.load_state_dict(torch.load(warm_ckpt, map_location=device, weights_only=True))
        print(f"Loaded warm-start state policy from {warm_ckpt}", flush=True)
        expert_states, expert_actions = collect_expert_dataset(policy, env, num_demos=100, max_steps=160)
    else:
        expert_states, expert_actions = collect_expert_dataset(policy, env, num_demos=180, max_steps=160)
        pretrain_actor(policy, expert_states.to(device), expert_actions.to(device), epochs=10)
        torch.save(policy.state_dict(), warm_ckpt)
        print(f"Saved warm-start state policy to {warm_ckpt}", flush=True)

    expert_states = expert_states.to(device)
    expert_actions = expert_actions.to(device)

    episode_rewards = deque(maxlen=200)
    successes = deque(maxlen=200)
    best_eval_success = -1.0
    best_eval_reward = -float("inf")
    total_successes = 0
    start = time.time()

    os.makedirs("checkpoints", exist_ok=True)

    for episode in range(NUM_EPISODES):
        if episode < 5000:
            env.curriculum_stage = 0
        elif episode < 12000:
            env.curriculum_stage = 1
        elif episode < 20000:
            env.curriculum_stage = 2
        else:
            env.curriculum_stage = 3

        env.reset_state_only()
        state = get_state(env)
        states_b, actions_b, lps_b, rewards_b, values_b = [], [], [], [], []
        ep_reward = 0.0
        ep_success = False

        for _ in range(MAX_STEPS):
            st = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action, lp, _, val = policy(st)
            action_np = action.squeeze().cpu().numpy()
            reward, done, info = rollout_step(env, action_np)

            states_b.append(st.squeeze(0).cpu())
            actions_b.append(action.squeeze().cpu())
            lps_b.append(lp.squeeze().cpu())
            rewards_b.append(reward)
            values_b.append(val.squeeze().cpu())
            ep_reward += reward
            state = get_state(env)

            if done:
                ep_success = bool(info.get("success"))
                break

        if ep_success:
            total_successes += 1
        episode_rewards.append(ep_reward)
        successes.append(1 if ep_success else 0)

        rewards_t = torch.FloatTensor(rewards_b).to(device)
        values_t = torch.stack([v.to(device) for v in values_b])
        advantages = torch.zeros_like(rewards_t)
        last_gae = 0.0
        for t in reversed(range(len(rewards_t))):
            next_value = values_t[t + 1] if t < len(rewards_t) - 1 else torch.tensor(0.0, device=device)
            delta = rewards_t[t] + 0.99 * next_value - values_t[t]
            last_gae = delta + 0.99 * 0.95 * last_gae
            advantages[t] = last_gae
        returns = advantages + values_t.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        S = torch.stack(states_b).to(device)
        A = torch.stack(actions_b).to(device)
        old_lp = torch.stack(lps_b).detach().to(device)

        for _ in range(PPO_EPOCHS):
            _, new_lp, ent, new_values = policy(S, A)
            ratio = (new_lp - old_lp).exp()
            pg_loss = -torch.min(
                advantages * ratio,
                advantages * ratio.clamp(0.85, 1.15)
            ).mean()
            value_loss = nn.MSELoss()(new_values, returns)

            bc_idx = torch.randint(0, expert_states.shape[0], (128,), device=device)
            bc_pred = policy.actor_mean(expert_states[bc_idx])
            nav_loss = nn.MSELoss()(bc_pred[:, :3], expert_actions[bc_idx][:, :3])
            arm_loss = nn.MSELoss()(bc_pred[:, 3:9], expert_actions[bc_idx][:, 3:9])
            grip_loss = nn.MSELoss()(bc_pred[:, 9:], expert_actions[bc_idx][:, 9:])
            bc_loss = 2.0 * nav_loss + arm_loss + 5.0 * grip_loss

            loss = pg_loss + 0.5 * value_loss - 0.0002 * ent.mean() + BC_ANCHOR * bc_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        if episode > 0 and episode % EVAL_INTERVAL == 0:
            eval_metrics = evaluate_policy(policy, env, episodes=EVAL_EPISODES)
            print(
                f"[Eval @ {episode:6d}] "
                f"Success={eval_metrics['success_rate']:5.1f}% | "
                f"Grasp={eval_metrics['grasp_rate']:5.1f}% | "
                f"Shelf={eval_metrics['shelf_rate']:5.1f}% | "
                f"Reward={eval_metrics['mean_reward']:7.1f} | "
                f"MeanMaxZ={eval_metrics['mean_max_z']:.3f}",
                flush=True
            )
            better = (
                eval_metrics["success_rate"] > best_eval_success or
                (eval_metrics["success_rate"] == best_eval_success and
                 eval_metrics["mean_reward"] > best_eval_reward)
            )
            if better:
                best_eval_success = eval_metrics["success_rate"]
                best_eval_reward = eval_metrics["mean_reward"]
                torch.save(policy.state_dict(), "checkpoints/phase2_state_policy_v2_best.pth")
                print(
                    f"  ✓ Saved eval-best checkpoint "
                    f"(success={best_eval_success:.1f}%, reward={best_eval_reward:.1f})",
                    flush=True
                )

        if episode % 500 == 0:
            mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            train_success = 100.0 * np.mean(successes) if successes else 0.0
            eps = (episode + 1) / max(time.time() - start, 1e-6)
            print(
                f"Ep {episode:6d} | R: {mean_reward:7.1f} | "
                f"Success: {train_success:5.1f}% | "
                f"EvalBest: {max(best_eval_success, 0.0):.1f}% | "
                f"Stage: {env.curriculum_stage} | "
                f"Total: {total_successes} | {eps:.1f} ep/s",
                flush=True
            )

    env.close()
