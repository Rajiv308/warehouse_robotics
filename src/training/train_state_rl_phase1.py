"""
Phase 1 state-based RL with expert-guided warm start and deterministic eval.

Why this exists:
- Pure RL from scratch collapsed into a low-motion local minimum.
- Old vision BC was not a good control prior.
- The fastest path to a presentable pick-and-lift demo is:
  1. learn a clean state policy from the working IK expert,
  2. fine-tune with PPO in the real environment,
  3. choose checkpoints by deterministic eval, not sampled rollout luck.
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

from src.data.collect_demos import IKExpertController
from src.env.warehouse_env import WarehouseEnv


class StatePolicy(nn.Module):
    def __init__(self, state_dim=15, action_dim=7):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim) - 2.6)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def get_state(self, env):
        gripper = np.array(p.getLinkState(env.robot_id, 11)[0], dtype=np.float32)
        target_id = env.object_ids[getattr(env, "_target_idx", 0)]
        target_pos, _ = p.getBasePositionAndOrientation(target_id)
        target_pos = np.array(target_pos, dtype=np.float32)
        joints = np.array([p.getJointState(env.robot_id, j)[0] for j in range(7)], dtype=np.float32)
        gripper_opening = np.float32(p.getJointState(env.robot_id, 9)[0])
        dist = np.float32(np.linalg.norm(gripper - target_pos))
        return np.concatenate([gripper, target_pos, joints, [gripper_opening, dist]]).astype(np.float32)

    def forward(self, state, action=None):
        mean = self.actor_mean(state)
        std = self.actor_log_std.exp().clamp(min=0.01, max=0.05)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        return (
            action,
            dist.log_prob(action).sum(-1),
            dist.entropy().sum(-1),
            self.critic(state).squeeze(-1),
        )


def collect_expert_dataset(policy, env, num_demos=200, steps_per_demo=110):
    expert = IKExpertController(env.robot_id)
    states = []
    actions = []

    print(f"Collecting {num_demos} expert rollouts for warm start...", flush=True)
    for _ in range(num_demos):
        env.reset()
        target_id = env.object_ids[getattr(env, "_target_idx", 0)]
        target_pos, _ = p.getBasePositionAndOrientation(target_id)
        expert.reset(target_pos)

        for _ in range(steps_per_demo):
            states.append(policy.get_state(env))
            action = expert.get_action().astype(np.float32)
            actions.append(action)
            env.step(action)

    states_t = torch.FloatTensor(np.array(states, dtype=np.float32))
    actions_t = torch.FloatTensor(np.array(actions, dtype=np.float32))
    return states_t, actions_t


def pretrain_actor(policy, expert_states, expert_actions, epochs=10, batch_size=256):
    optimizer = torch.optim.Adam(policy.actor_mean.parameters(), lr=1e-3)
    num_samples = expert_states.shape[0]
    print(f"Pretraining actor on {num_samples:,} expert state-action pairs...", flush=True)

    for epoch in range(epochs):
        perm = torch.randperm(num_samples)
        losses = []
        for start in range(0, num_samples, batch_size):
            idx = perm[start:start + batch_size]
            pred = policy.actor_mean(expert_states[idx])
            joint_loss = nn.MSELoss()(pred[:, :-1], expert_actions[idx][:, :-1])
            gripper_loss = nn.MSELoss()(pred[:, -1:], expert_actions[idx][:, -1:])
            loss = joint_loss + 5.0 * gripper_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.actor_mean.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        print(f"  Warm start epoch {epoch + 1}/{epochs} | actor_loss={np.mean(losses):.4f}", flush=True)


def apply_curriculum_reset(env, episode):
    """
    Start easy and gradually reintroduce randomness.
    The task is still pick-and-lift, but the policy first learns the basic
    motion pattern before dealing with larger pose variation.
    """
    env.reset()

    if episode < 2000:
        noise_scale = 0.005
    elif episode < 6000:
        noise_scale = 0.02
    elif episode < 12000:
        noise_scale = 0.035
    else:
        return

    for i, obj_id in enumerate(env.object_ids):
        noise = np.random.uniform(-noise_scale, noise_scale, 2)
        base_pos = [0.5 + noise[0], (i - 1) * 0.3 + noise[1], 0.05]
        p.resetBasePositionAndOrientation(obj_id, base_pos, [0, 0, 0, 1])
        p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0])


def rollout_step(env, action_np):
    env.apply_action(action_np)
    p.stepSimulation()
    env.step_count += 1

    reward = env.compute_reward()
    success, metrics = env.update_success_state()
    done = success or env.step_count >= env.env_cfg["max_episode_steps"]

    if success:
        reward += 120.0
    elif done and metrics["obj_z"] > 0.07:
        reward += 15.0

    return reward, done, success, metrics


def evaluate_policy(policy, env, episodes=20):
    policy.eval()
    rewards = []
    successes = 0
    max_zs = []
    deltas = []

    with torch.no_grad():
        for _ in range(episodes):
            env.reset()
            state = policy.get_state(env)
            ep_reward = 0.0
            max_z = 0.0
            prev_action = None

            for _ in range(env.env_cfg["max_episode_steps"]):
                st = torch.FloatTensor(state).unsqueeze(0)
                action_np = policy.actor_mean(st).squeeze().cpu().numpy()
                reward, done, success, metrics = rollout_step(env, action_np)
                ep_reward += reward
                max_z = max(max_z, metrics["obj_z"])
                if prev_action is not None:
                    deltas.append(float(np.linalg.norm(action_np[:6] - prev_action[:6])))
                prev_action = action_np
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
        "mean_action_delta": float(np.mean(deltas)) if deltas else 0.0,
    }


if __name__ == "__main__":
    polish_mode = os.environ.get("PHASE1_POLISH", "0") == "1"
    full_arm_mode = os.environ.get("PHASE1_FULL_ARM", "0") == "1"
    NUM_EPISODES = 20000 if polish_mode else 100000
    MAX_STEPS = 180
    EVAL_INTERVAL = 250
    EVAL_EPISODES = 20
    PPO_EPOCHS = 4
    LR = 5e-5 if polish_mode else 1e-4
    ENT_COEF = 0.00005 if polish_mode else 0.0002
    BC_ANCHOR_COEF = 0.15 if polish_mode else 0.10
    pretrain_ckpt = os.environ.get(
        "PHASE1_INIT_CKPT",
        (
            "checkpoints/phase1_state_policy_eval_best.pth"
            if polish_mode and not full_arm_mode else
            "checkpoints/phase1_state_bc_init_fullarm.pth"
            if full_arm_mode else
            "checkpoints/phase1_state_bc_init.pth"
        ),
    )
    train_ckpt = os.environ.get(
        "PHASE1_TRAIN_CKPT",
        (
            "checkpoints/phase1_state_policy_polish.pth"
            if polish_mode and not full_arm_mode else
            "checkpoints/phase1_state_policy_fullarm.pth"
            if full_arm_mode else
            "checkpoints/phase1_state_policy.pth"
        ),
    )
    eval_ckpt = os.environ.get(
        "PHASE1_EVAL_CKPT",
        (
            "checkpoints/phase1_state_policy_polish_eval_best.pth"
            if polish_mode and not full_arm_mode else
            "checkpoints/phase1_state_policy_fullarm_eval_best.pth"
            if full_arm_mode else
            "checkpoints/phase1_state_policy_eval_best.pth"
        ),
    )

    print(
        f"Phase 1 State RL"
        f"{' FullArm' if full_arm_mode else ''}"
        f"{' Polish' if polish_mode else ''} — {NUM_EPISODES} episodes",
        flush=True
    )

    policy = StatePolicy(action_dim=8 if full_arm_mode else 7)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    print(f"Params: {sum(pp.numel() for pp in policy.parameters()):,}", flush=True)

    env = WarehouseEnv(render=False)
    env.initialize()
    env.env_cfg["max_episode_steps"] = MAX_STEPS
    if polish_mode:
        env.attach_dist_threshold = 0.06
        env.success_lift_height = 0.12
        env.success_hold_steps = 6
        env.max_success_obj_speed = 1.0
        env.post_grasp_target_height = 0.18

    if os.path.exists(pretrain_ckpt):
        policy.load_state_dict(torch.load(pretrain_ckpt, map_location="cpu", weights_only=True))
        print(f"Loaded warm-start state policy from {pretrain_ckpt}", flush=True)
        expert_states, expert_actions = collect_expert_dataset(
            policy,
            env,
            num_demos=80 if polish_mode else (120 if not full_arm_mode else 160),
            steps_per_demo=120 if polish_mode else 110,
        )
    else:
        expert_states, expert_actions = collect_expert_dataset(
            policy,
            env,
            num_demos=220 if full_arm_mode else 200,
            steps_per_demo=120 if full_arm_mode else 110,
        )
        pretrain_actor(policy, expert_states, expert_actions, epochs=12)
        torch.save(policy.state_dict(), pretrain_ckpt)
        print(f"Saved warm-start state policy to {pretrain_ckpt}", flush=True)

    episode_rewards = deque(maxlen=200)
    successes = deque(maxlen=200)
    total_successes = 0
    best_train_success = 0.0
    best_eval_success = -1.0
    best_eval_reward = -float("inf")
    start = time.time()

    os.makedirs("checkpoints", exist_ok=True)

    for episode in range(NUM_EPISODES):
        apply_curriculum_reset(env, episode)
        state = policy.get_state(env)
        states_b, actions_b, lps_b, rewards_b, values_b = [], [], [], [], []
        ep_reward = 0.0
        success = False

        for _ in range(MAX_STEPS):
            st = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, lp, _, val = policy(st)
            action_np = action.squeeze().cpu().numpy()
            reward, done, success, _ = rollout_step(env, action_np)

            states_b.append(st.squeeze())
            actions_b.append(action.squeeze())
            lps_b.append(lp.squeeze())
            rewards_b.append(reward)
            values_b.append(val.squeeze())
            ep_reward += reward
            state = policy.get_state(env)

            if done:
                break

        if success:
            total_successes += 1
        episode_rewards.append(ep_reward)
        successes.append(1 if success else 0)

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

        policy.train()
        for _ in range(PPO_EPOCHS):
            _, new_lp, ent, new_values = policy(S, A)
            ratio = (new_lp - old_lp).exp()
            pg_loss = -torch.min(
                advantages * ratio,
                advantages * ratio.clamp(0.85, 1.15)
            ).mean()
            value_loss = nn.MSELoss()(new_values, returns)

            bc_idx = torch.randint(0, expert_states.shape[0], (128,))
            bc_pred = policy.actor_mean(expert_states[bc_idx])
            bc_joint = nn.MSELoss()(bc_pred[:, :-1], expert_actions[bc_idx][:, :-1])
            bc_gripper = nn.MSELoss()(bc_pred[:, -1:], expert_actions[bc_idx][:, -1:])
            bc_loss = bc_joint + 5.0 * bc_gripper

            loss = pg_loss + 0.5 * value_loss - ENT_COEF * ent.mean() + BC_ANCHOR_COEF * bc_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        train_success = 100.0 * np.mean(successes) if successes else 0.0
        if train_success > best_train_success:
            best_train_success = train_success
            torch.save(policy.state_dict(), train_ckpt)

        if episode > 0 and episode % EVAL_INTERVAL == 0:
            eval_metrics = evaluate_policy(policy, env, episodes=EVAL_EPISODES)
            eval_success = eval_metrics["success_rate"]
            eval_reward = eval_metrics["mean_reward"]
            print(
                f"[Eval @ {episode:6d}] "
                f"Success={eval_success:5.1f}% | "
                f"Reward={eval_reward:7.1f} | "
                f"MeanMaxZ={eval_metrics['mean_max_z']:.3f} | "
                f"MeanDelta={eval_metrics['mean_action_delta']:.3f}",
                flush=True
            )
            better = (
                eval_success > best_eval_success or
                (eval_success == best_eval_success and eval_reward > best_eval_reward)
            )
            if better:
                best_eval_success = eval_success
                best_eval_reward = eval_reward
                torch.save(policy.state_dict(), eval_ckpt)
                print(
                    f"  ✓ Saved eval-best checkpoint "
                    f"(success={best_eval_success:.1f}%, reward={best_eval_reward:.1f})",
                    flush=True
                )

        if episode % 500 == 0:
            mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            eps = (episode + 1) / max(time.time() - start, 1e-6)
            print(
                f"Ep {episode:6d} | R: {mean_reward:7.1f} | "
                f"Success: {train_success:5.1f}% | "
                f"Best: {best_train_success:.1f}% | "
                f"EvalBest: {max(best_eval_success, 0.0):.1f}% | "
                f"Total: {total_successes} | {eps:.1f} ep/s",
                flush=True
            )

    env.close()
    elapsed = (time.time() - start) / 60.0
    print(
        f"\nDONE in {elapsed:.1f} min | "
        f"Best train success: {best_train_success:.1f}% | "
        f"Best eval success: {max(best_eval_success, 0.0):.1f}% | "
        f"Total: {total_successes}/{NUM_EPISODES}",
        flush=True
    )
