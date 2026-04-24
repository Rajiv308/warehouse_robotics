"""
Phase 1 Cartesian pick training.

This is the emergency simplification path:
- one fixed target color at a time
- end-effector delta control instead of raw joint-space control
- explicit expert warm start
- deterministic eval on the exact pickup task we want
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

from src.env.warehouse_env import WarehouseEnv


class CartesianPickPolicy(nn.Module):
    def __init__(self, state_dim=11, action_dim=4):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.actor_log_std = nn.Parameter(torch.tensor([-2.8, -2.8, -2.9, -3.0], dtype=torch.float32))
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def get_state(self, env):
        ee = np.array(p.getLinkState(env.robot_id, 11)[0], dtype=np.float32)
        target_id = env.object_ids[getattr(env, "_target_idx", 0)]
        obj_pos, _ = p.getBasePositionAndOrientation(target_id)
        obj_pos = np.array(obj_pos, dtype=np.float32)
        rel = obj_pos - ee
        opening = np.float32(p.getJointState(env.robot_id, 9)[0])
        obj_vel, _ = p.getBaseVelocity(target_id)
        obj_speed = np.float32(np.linalg.norm(obj_vel))
        state = np.concatenate([
            ee,
            obj_pos,
            rel,
            np.array([opening, obj_speed], dtype=np.float32),
        ]).astype(np.float32)
        return state

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


class CartesianExpert:
    def __init__(self):
        self.phase = 0
        self.phase_steps = 0

    def reset(self):
        self.phase = 0
        self.phase_steps = 0

    def get_action(self, env):
        ee = np.array(p.getLinkState(env.robot_id, 11)[0], dtype=np.float32)
        target_id = env.object_ids[getattr(env, "_target_idx", 0)]
        obj_pos, _ = p.getBasePositionAndOrientation(target_id)
        obj_pos = np.array(obj_pos, dtype=np.float32)

        hover = obj_pos + np.array([0.0, 0.0, 0.18], dtype=np.float32)
        pregrasp = obj_pos + np.array([0.0, 0.0, 0.015], dtype=np.float32)
        lift = obj_pos + np.array([0.0, 0.0, 0.22], dtype=np.float32)

        if self.phase == 0:
            target = hover
            grip = 1.0
            if np.linalg.norm(ee - hover) < 0.03 or self.phase_steps > 25:
                self.phase = 1
                self.phase_steps = 0
        elif self.phase == 1:
            target = pregrasp
            grip = 1.0
            if np.linalg.norm(ee - pregrasp) < 0.02 or self.phase_steps > 28:
                self.phase = 2
                self.phase_steps = 0
        elif self.phase == 2:
            target = pregrasp
            grip = -1.0
            if self.phase_steps > 18:
                self.phase = 3
                self.phase_steps = 0
        else:
            target = lift
            grip = -1.0

        delta = np.clip((target - ee) / np.array([0.03, 0.03, 0.025], dtype=np.float32), -1.0, 1.0)
        action = np.array([delta[0], delta[1], delta[2], grip], dtype=np.float32)
        self.phase_steps += 1
        return action


def get_alignment_metrics(env):
    metrics = env.get_target_metrics()
    obj_pos = metrics["obj_pos"]
    gripper_pos = metrics["gripper_pos"]
    hover_target = obj_pos + np.array([0.0, 0.0, 0.12], dtype=np.float32)
    hover_dist = float(np.linalg.norm(gripper_pos - hover_target))
    xy_dist = float(np.linalg.norm(gripper_pos[:2] - obj_pos[:2]))
    z_gap = float(gripper_pos[2] - obj_pos[2])
    aligned = (
        xy_dist < 0.025 and
        0.09 <= z_gap <= 0.15 and
        not metrics["gripper_closed"]
    )
    return metrics, hover_dist, xy_dist, z_gap, aligned


def rollout_step(env, action_np):
    env.apply_cartesian_action(action_np)
    p.stepSimulation()
    env.step_count += 1
    metrics, hover_dist, xy_dist, z_gap, aligned = get_alignment_metrics(env)

    reward = 0.0
    reward -= 4.0 * hover_dist
    reward -= 1.5 * xy_dist
    reward -= 0.01

    if action_np[3] > 0.0:
        reward += 0.6  # keep open while approaching
    else:
        reward -= 1.0  # premature close is bad during alignment stage

    if xy_dist < 0.05:
        reward += 2.0
    if xy_dist < 0.03:
        reward += 4.0
    if aligned:
        reward += 12.0
        success = env.execute_pick_macro()
        metrics = env.get_target_metrics()
        if success:
            reward += 150.0
            return reward, True, True, metrics
        reward -= 5.0

    success, metrics = env.update_success_state()
    done = success or env.step_count >= env.env_cfg["max_episode_steps"]
    return reward, done, success, metrics


def collect_expert_dataset(policy, env, target_idx, num_demos=160, steps_per_demo=90):
    expert = CartesianExpert()
    states = []
    actions = []
    print(f"Collecting {num_demos} Cartesian expert rollouts...", flush=True)
    for _ in range(num_demos):
        env.reset_simple_task(target_idx=target_idx, distractors=False, position_noise=0.01)
        expert.reset()
        for _ in range(steps_per_demo):
            states.append(policy.get_state(env))
            action = expert.get_action(env)
            actions.append(action)
            _, done, success, _ = rollout_step(env, action)
            if success or done:
                break
    return (
        torch.FloatTensor(np.array(states, dtype=np.float32)),
        torch.FloatTensor(np.array(actions, dtype=np.float32)),
    )


def pretrain_actor(policy, expert_states, expert_actions, epochs=12, batch_size=256):
    opt = torch.optim.Adam(policy.actor_mean.parameters(), lr=1e-3)
    n = expert_states.shape[0]
    print(f"Pretraining on {n:,} Cartesian pairs...", flush=True)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        losses = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            pred = policy.actor_mean(expert_states[idx])
            move_loss = nn.MSELoss()(pred[:, :3], expert_actions[idx][:, :3])
            grip_loss = nn.MSELoss()(pred[:, 3:], expert_actions[idx][:, 3:])
            loss = move_loss + 6.0 * grip_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.actor_mean.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        print(f"  Warm start epoch {epoch + 1}/{epochs} | actor_loss={np.mean(losses):.4f}", flush=True)


def evaluate_policy(policy, env, target_idx, episodes=20):
    policy.eval()
    rewards = []
    successes = 0
    max_zs = []
    with torch.no_grad():
        for _ in range(episodes):
            env.reset_simple_task(target_idx=target_idx, distractors=False, position_noise=0.012)
            state = policy.get_state(env)
            ep_reward = 0.0
            max_z = 0.0
            for _ in range(env.env_cfg["max_episode_steps"]):
                st = torch.FloatTensor(state).unsqueeze(0)
                action_np = policy.actor_mean(st).squeeze().cpu().numpy()
                reward, done, success, metrics = rollout_step(env, action_np)
                ep_reward += reward
                max_z = max(max_z, metrics["obj_z"])
                state = policy.get_state(env)
                if done:
                    if success:
                        successes += 1
                    break
            rewards.append(ep_reward)
            max_zs.append(max_z)
    return {
        "success_rate": 100.0 * successes / max(episodes, 1),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_max_z": float(np.mean(max_zs)) if max_zs else 0.0,
    }


if __name__ == "__main__":
    target_idx = int(os.environ.get("PHASE1_TARGET_IDX", "0"))
    num_episodes = int(os.environ.get("PHASE1_CART_NUM_EPISODES", "30000"))
    eval_interval = int(os.environ.get("PHASE1_CART_EVAL_INTERVAL", "250"))
    eval_episodes = int(os.environ.get("PHASE1_CART_EVAL_EPISODES", "20"))
    warm_demos = int(os.environ.get("PHASE1_CART_WARM_DEMOS", "160"))
    max_steps = int(os.environ.get("PHASE1_CART_MAX_STEPS", "140"))
    resume_ckpt = os.environ.get("PHASE1_CART_RESUME_CKPT")

    ckpt_prefix = os.environ.get("PHASE1_CART_CKPT_PREFIX", f"phase1_cartesian_target{target_idx}")
    warm_ckpt = f"checkpoints/{ckpt_prefix}_bc_init.pth"
    best_ckpt = f"checkpoints/{ckpt_prefix}_best.pth"
    latest_ckpt = f"checkpoints/{ckpt_prefix}_latest.pth"

    print(
        f"Phase 1 Cartesian Pick | target_idx={target_idx} episodes={num_episodes} "
        f"eval_interval={eval_interval}",
        flush=True,
    )

    env = WarehouseEnv(render=False)
    env.initialize()
    env.env_cfg["max_episode_steps"] = max_steps

    policy = CartesianPickPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
    print(f"Params: {sum(pp.numel() for pp in policy.parameters()):,}", flush=True)

    if resume_ckpt and os.path.exists(resume_ckpt):
        policy.load_state_dict(torch.load(resume_ckpt, map_location="cpu", weights_only=True))
        print(f"Loaded resume checkpoint from {resume_ckpt}", flush=True)
        expert_states, expert_actions = collect_expert_dataset(policy, env, target_idx, num_demos=warm_demos)
    elif os.path.exists(warm_ckpt):
        policy.load_state_dict(torch.load(warm_ckpt, map_location="cpu", weights_only=True))
        print(f"Loaded warm start from {warm_ckpt}", flush=True)
        expert_states, expert_actions = collect_expert_dataset(policy, env, target_idx, num_demos=warm_demos)
    else:
        expert_states, expert_actions = collect_expert_dataset(policy, env, target_idx, num_demos=warm_demos)
        pretrain_actor(policy, expert_states, expert_actions)
        torch.save(policy.state_dict(), warm_ckpt)
        print(f"Saved warm start to {warm_ckpt}", flush=True)

    episode_rewards = deque(maxlen=200)
    successes = deque(maxlen=200)
    best_eval_success = -1.0
    best_eval_reward = -float("inf")
    total_successes = 0
    start = time.time()

    for episode in range(num_episodes):
        distractors = episode >= 8000
        noise = 0.01 if episode < 4000 else 0.02 if episode < 12000 else 0.03
        env.reset_simple_task(target_idx=target_idx, distractors=distractors, position_noise=noise)
        state = policy.get_state(env)
        states_b, actions_b, lps_b, rewards_b, values_b = [], [], [], [], []
        ep_reward = 0.0
        ep_success = False

        for _ in range(max_steps):
            st = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, lp, _, val = policy(st)
            action_np = action.squeeze().cpu().numpy()
            reward, done, success, _ = rollout_step(env, action_np)

            states_b.append(st.squeeze(0))
            actions_b.append(action.squeeze().cpu())
            lps_b.append(lp.squeeze().cpu())
            rewards_b.append(reward)
            values_b.append(val.squeeze().cpu())
            ep_reward += reward
            state = policy.get_state(env)
            if done:
                ep_success = bool(success)
                break

        if ep_success:
            total_successes += 1
        episode_rewards.append(ep_reward)
        successes.append(1 if ep_success else 0)

        rewards_t = torch.FloatTensor(rewards_b)
        values_t = torch.stack(values_b)
        advantages = torch.zeros_like(rewards_t)
        last_gae = 0.0
        for t in reversed(range(len(rewards_t))):
            next_value = values_t[t + 1] if t < len(rewards_t) - 1 else torch.tensor(0.0)
            delta = rewards_t[t] + 0.99 * next_value - values_t[t]
            last_gae = delta + 0.99 * 0.95 * last_gae
            advantages[t] = last_gae
        returns = advantages + values_t.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        S = torch.stack(states_b)
        A = torch.stack(actions_b)
        old_lp = torch.stack(lps_b).detach()

        for _ in range(4):
            _, new_lp, ent, new_values = policy(S, A)
            ratio = (new_lp - old_lp).exp()
            pg_loss = -torch.min(
                advantages * ratio,
                advantages * ratio.clamp(0.85, 1.15),
            ).mean()
            value_loss = nn.MSELoss()(new_values, returns)

            bc_idx = torch.randint(0, expert_states.shape[0], (128,))
            bc_pred = policy.actor_mean(expert_states[bc_idx])
            move_loss = nn.MSELoss()(bc_pred[:, :3], expert_actions[bc_idx][:, :3])
            grip_loss = nn.MSELoss()(bc_pred[:, 3:], expert_actions[bc_idx][:, 3:])
            bc_loss = move_loss + 6.0 * grip_loss

            loss = pg_loss + 0.5 * value_loss - 0.0002 * ent.mean() + 0.18 * bc_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()

        if episode > 0 and episode % eval_interval == 0:
            eval_metrics = evaluate_policy(policy, env, target_idx, episodes=eval_episodes)
            print(
                f"[Eval @ {episode:6d}] Success={eval_metrics['success_rate']:5.1f}% | "
                f"Reward={eval_metrics['mean_reward']:7.1f} | "
                f"MeanMaxZ={eval_metrics['mean_max_z']:.3f}",
                flush=True,
            )
            better = (
                eval_metrics["success_rate"] > best_eval_success or
                (
                    eval_metrics["success_rate"] == best_eval_success and
                    eval_metrics["mean_reward"] > best_eval_reward
                )
            )
            if better:
                best_eval_success = eval_metrics["success_rate"]
                best_eval_reward = eval_metrics["mean_reward"]
                torch.save(policy.state_dict(), best_ckpt)
                print(
                    f"  ✓ Saved eval-best checkpoint to {best_ckpt} "
                    f"(success={best_eval_success:.1f}%, reward={best_eval_reward:.1f})",
                    flush=True,
                )

        if episode % 500 == 0:
            mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            train_success = 100.0 * np.mean(successes) if successes else 0.0
            eps = (episode + 1) / max(time.time() - start, 1e-6)
            print(
                f"Ep {episode:6d} | R: {mean_reward:7.1f} | "
                f"Success: {train_success:5.1f}% | "
                f"EvalBest: {max(best_eval_success, 0.0):.1f}% | "
                f"Total: {total_successes} | {eps:.1f} ep/s",
                flush=True,
            )
            torch.save(policy.state_dict(), latest_ckpt)

    torch.save(policy.state_dict(), latest_ckpt)
    env.close()
