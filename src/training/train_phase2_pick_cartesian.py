"""
Phase 2 isolated pickup training.

This stage learns only the shelf-front arm pickup skill:
- the mobile base starts already in a valid pickup pose
- the target object is already identified
- the policy controls only Cartesian end-effector deltas + gripper

The goal is to learn a real pickup module we can compose after navigation.
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

from src.env.warehouse_env_mobile_v2 import MobileWarehouseEnvV2
from src.training.train_phase2_nav_pickpose import NavPolicy, get_nav_state, nav_success, build_full_action


class MobileCartesianPickPolicy(nn.Module):
    def __init__(self, state_dim=13, action_dim=4):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.actor_log_std = nn.Parameter(torch.tensor([-2.8, -2.8, -2.9, -3.1], dtype=torch.float32))
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def get_state(self, env):
        metrics = env.get_target_metrics()
        ee = np.array(metrics["gripper_pos"], dtype=np.float32)
        obj = np.array(metrics["obj_pos"], dtype=np.float32)
        rel = obj - ee
        opening = np.float32(p.getJointState(env.panda_id, 9)[0])
        target_id = env.object_ids[env.target_object_idx]
        obj_vel, _ = p.getBaseVelocity(target_id)
        obj_speed = np.float32(np.linalg.norm(obj_vel))
        xy_dist = np.float32(np.linalg.norm(ee[:2] - obj[:2]))
        z_gap = np.float32(ee[2] - obj[2])
        return np.concatenate([
            ee,
            obj,
            rel,
            np.array([opening, obj_speed, xy_dist, z_gap], dtype=np.float32),
        ]).astype(np.float32)

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


class MobilePickupExpert:
    def __init__(self):
        self.phase = 0
        self.phase_steps = 0

    def reset(self):
        self.phase = 0
        self.phase_steps = 0

    def get_action(self, env):
        metrics = env.get_target_metrics()
        ee = np.array(metrics["gripper_pos"], dtype=np.float32)
        obj = np.array(metrics["obj_pos"], dtype=np.float32)
        xy_dist = float(np.linalg.norm(ee[:2] - obj[:2]))
        z_gap = float(ee[2] - obj[2])

        hover = obj + np.array([0.0, 0.0, 0.16], dtype=np.float32)
        pregrasp = obj + np.array([0.0, 0.0, 0.045], dtype=np.float32)
        lift = obj + np.array([0.0, 0.0, 0.26], dtype=np.float32)

        if metrics["attached"] or metrics["obj_z"] > 0.63:
            target = lift
            grip = -1.0
        elif xy_dist < 0.045 and z_gap < 0.075:
            target = pregrasp
            grip = -1.0
        elif xy_dist < 0.07:
            target = pregrasp
            grip = 1.0
        else:
            target = hover
            grip = 1.0

        delta = np.clip((target - ee) / np.array([0.025, 0.025, 0.020], dtype=np.float32), -1.0, 1.0)
        return np.array([delta[0], delta[1], delta[2], grip], dtype=np.float32)


def rollout_step(env, action_np):
    reward, done, info = env.step_pickup_cartesian(action_np)
    if info.get("success"):
        reward += 150.0
    return reward, done, info


def maybe_load_nav_policy():
    nav_ckpt = os.environ.get("PHASE2_PICK_NAV_CKPT")
    if not nav_ckpt:
        return None
    if not os.path.exists(nav_ckpt):
        print(f"WARNING: nav handoff checkpoint not found: {nav_ckpt}", flush=True)
        return None
    nav_policy = NavPolicy()
    nav_policy.load_state_dict(torch.load(nav_ckpt, map_location="cpu", weights_only=True))
    nav_policy.eval()
    print(f"Loaded nav handoff checkpoint from {nav_ckpt}", flush=True)
    return nav_policy


def reset_via_nav_handoff(env, nav_policy, target_idx=None):
    """
    Produce pickup start states from the actual navigation policy instead of a synthetic reset.
    This is the key bridge to reduce nav->pickup distribution mismatch.
    """
    if nav_policy is None:
        target_idx = np.random.randint(0, env.env_cfg["num_objects"]) if target_idx is None else target_idx
        env.reset_pickup_task(
            target_idx=target_idx,
            distractors=False,
            base_noise=0.05,
            obj_noise=0.015,
            ready_y_jitter=0.03,
            ready_z_jitter=0.035,
        )
        return True

    instruction = env.reset_state_only()
    nav_ok = False
    for _ in range(150):
        st = torch.FloatTensor(get_nav_state(env)).unsqueeze(0)
        with torch.no_grad():
            nav_action = nav_policy(st).squeeze(0).cpu().numpy()
        _, _, _ = env.step_state_only(build_full_action(nav_action))
        success, _, _ = nav_success(env)
        if success:
            nav_ok = True
            break

    if not nav_ok:
        # Fall back to the simpler pickup reset instead of poisoning the dataset.
        target_idx = np.random.randint(0, env.env_cfg["num_objects"])
        env.reset_pickup_task(
            target_idx=target_idx,
            distractors=False,
            base_noise=0.05,
            obj_noise=0.015,
            ready_y_jitter=0.03,
            ready_z_jitter=0.035,
        )
        return False

    # Keep the real nav-produced base pose and only servo the arm into a reachable pre-hover.
    env.servo_pickup_ready_pose(steps=70, tolerance=0.018)
    return True


def collect_expert_dataset(policy, env, nav_policy=None, num_demos=220, steps_per_demo=70):
    expert = MobilePickupExpert()
    states = []
    actions = []
    print(f"Collecting {num_demos} mobile pickup expert rollouts...", flush=True)
    for _ in range(num_demos):
        reset_via_nav_handoff(env, nav_policy=nav_policy)
        expert.reset()
        for _ in range(steps_per_demo):
            states.append(policy.get_state(env))
            action = expert.get_action(env)
            actions.append(action)
            _, done, info = rollout_step(env, action)
            if info.get("success") or done:
                break
    return (
        torch.FloatTensor(np.array(states, dtype=np.float32)),
        torch.FloatTensor(np.array(actions, dtype=np.float32)),
    )


def pretrain_actor(policy, expert_states, expert_actions, epochs=14, batch_size=256):
    opt = torch.optim.Adam(policy.actor_mean.parameters(), lr=1e-3)
    n = expert_states.shape[0]
    print(f"Pretraining mobile pickup actor on {n:,} pairs...", flush=True)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        losses = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            pred = policy.actor_mean(expert_states[idx])
            move_loss = nn.MSELoss()(pred[:, :3], expert_actions[idx][:, :3])
            grip_loss = nn.MSELoss()(pred[:, 3:], expert_actions[idx][:, 3:])
            loss = move_loss + 7.0 * grip_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.actor_mean.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        print(f"  Warm start epoch {epoch + 1}/{epochs} | actor_loss={np.mean(losses):.4f}", flush=True)


def evaluate_policy(policy, env, nav_policy=None, episodes=20):
    policy.eval()
    rewards = []
    successes = 0
    grasp = 0
    lift = 0
    max_zs = []
    with torch.no_grad():
        for _ in range(episodes):
            reset_via_nav_handoff(env, nav_policy=nav_policy)
            state = policy.get_state(env)
            ep_reward = 0.0
            max_z = 0.0
            ep_grasp = False
            ep_lift = False
            for _ in range(100):
                st = torch.FloatTensor(state).unsqueeze(0)
                action_np = policy.actor_mean(st).squeeze(0).cpu().numpy()
                reward, done, info = rollout_step(env, action_np)
                ep_reward += reward
                max_z = max(max_z, info["obj_z"])
                ep_grasp = ep_grasp or bool(info["grasped"])
                ep_lift = ep_lift or bool(info["lifted"])
                state = policy.get_state(env)
                if done:
                    if info.get("success"):
                        successes += 1
                    break
            rewards.append(ep_reward)
            max_zs.append(max_z)
            if ep_grasp:
                grasp += 1
            if ep_lift:
                lift += 1
    return {
        "success_rate": 100.0 * successes / max(episodes, 1),
        "grasp_rate": 100.0 * grasp / max(episodes, 1),
        "lift_rate": 100.0 * lift / max(episodes, 1),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_max_z": float(np.mean(max_zs)) if max_zs else 0.0,
    }


if __name__ == "__main__":
    num_episodes = int(os.environ.get("PHASE2_PICK_NUM_EPISODES", "18000"))
    eval_interval = int(os.environ.get("PHASE2_PICK_EVAL_INTERVAL", "250"))
    eval_episodes = int(os.environ.get("PHASE2_PICK_EVAL_EPISODES", "20"))
    warm_demos = int(os.environ.get("PHASE2_PICK_WARM_DEMOS", "220"))
    resume_ckpt = os.environ.get("PHASE2_PICK_RESUME_CKPT")
    resume_pretrain = os.environ.get("PHASE2_PICK_RESUME_PRETRAIN", "0") == "1"

    warm_ckpt = os.environ.get("PHASE2_PICK_WARM_CKPT", "checkpoints/phase2_pick_cartesian_bc_init.pth")
    best_ckpt = os.environ.get("PHASE2_PICK_BEST_CKPT", "checkpoints/phase2_pick_cartesian_best.pth")
    latest_ckpt = os.environ.get("PHASE2_PICK_LATEST_CKPT", "checkpoints/phase2_pick_cartesian_latest.pth")

    print(
        f"Phase 2 Isolated Pickup | episodes={num_episodes} "
        f"eval_interval={eval_interval} eval_episodes={eval_episodes}",
        flush=True,
    )

    env = MobileWarehouseEnvV2(
        config_path="configs/config_cloud.yaml",
        render=False,
        curriculum_stage=0,
        success_mode="pickup",
    )
    env.initialize()
    env.env_cfg["max_episode_steps"] = 120

    policy = MobileCartesianPickPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
    nav_policy = maybe_load_nav_policy()
    print(f"Params: {sum(pp.numel() for pp in policy.parameters()):,}", flush=True)

    if resume_ckpt and os.path.exists(resume_ckpt):
        policy.load_state_dict(torch.load(resume_ckpt, map_location="cpu", weights_only=True))
        print(f"Loaded resume checkpoint from {resume_ckpt}", flush=True)
        expert_states, expert_actions = collect_expert_dataset(policy, env, nav_policy=nav_policy, num_demos=warm_demos)
        if resume_pretrain:
            print("Running actor refresh on resumed checkpoint using current expert dataset...", flush=True)
            pretrain_actor(policy, expert_states, expert_actions, epochs=8)
    elif os.path.exists(warm_ckpt):
        policy.load_state_dict(torch.load(warm_ckpt, map_location="cpu", weights_only=True))
        print(f"Loaded warm start from {warm_ckpt}", flush=True)
        expert_states, expert_actions = collect_expert_dataset(policy, env, nav_policy=nav_policy, num_demos=warm_demos)
    else:
        expert_states, expert_actions = collect_expert_dataset(policy, env, nav_policy=nav_policy, num_demos=warm_demos)
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
        if nav_policy is None:
            target_idx = np.random.randint(0, env.env_cfg["num_objects"])
            distractors = episode >= 6000
            obj_noise = 0.010 if episode < 4000 else 0.02
            base_noise = 0.04 if episode < 4000 else 0.06
            ready_y_jitter = 0.02 if episode < 4000 else 0.04
            ready_z_jitter = 0.025 if episode < 4000 else 0.045
            env.reset_pickup_task(
                target_idx=target_idx,
                distractors=distractors,
                base_noise=base_noise,
                obj_noise=obj_noise,
                ready_y_jitter=ready_y_jitter,
                ready_z_jitter=ready_z_jitter,
            )
        else:
            reset_via_nav_handoff(env, nav_policy=nav_policy)
        state = policy.get_state(env)
        states_b, actions_b, lps_b, rewards_b, values_b = [], [], [], [], []
        ep_reward = 0.0
        ep_success = False

        for _ in range(100):
            st = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, lp, _, val = policy(st)
            action_np = action.squeeze(0).cpu().numpy()
            reward, done, info = rollout_step(env, action_np)

            states_b.append(st.squeeze(0))
            actions_b.append(action.squeeze(0).cpu())
            lps_b.append(lp.squeeze(0).cpu())
            rewards_b.append(reward)
            values_b.append(val.squeeze(0).cpu())
            ep_reward += reward
            state = policy.get_state(env)
            if done:
                ep_success = bool(info.get("success"))
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

            bc_idx = torch.randint(0, expert_states.shape[0], (min(128, expert_states.shape[0]),))
            bc_pred = policy.actor_mean(expert_states[bc_idx])
            move_loss = nn.MSELoss()(bc_pred[:, :3], expert_actions[bc_idx][:, :3])
            grip_loss = nn.MSELoss()(bc_pred[:, 3:], expert_actions[bc_idx][:, 3:])
            bc_loss = move_loss + 7.0 * grip_loss

            loss = pg_loss + 0.5 * value_loss - 0.0002 * ent.mean() + 0.20 * bc_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()

        if episode > 0 and episode % eval_interval == 0:
            eval_metrics = evaluate_policy(policy, env, nav_policy=nav_policy, episodes=eval_episodes)
            print(
                f"[Eval @ {episode:6d}] Success={eval_metrics['success_rate']:5.1f}% | "
                f"Grasp={eval_metrics['grasp_rate']:5.1f}% | "
                f"Lift={eval_metrics['lift_rate']:5.1f}% | "
                f"Reward={eval_metrics['mean_reward']:7.1f} | "
                f"MeanMaxZ={eval_metrics['mean_max_z']:.3f}",
                flush=True,
            )
            if (
                eval_metrics["success_rate"] > best_eval_success or
                (
                    eval_metrics["success_rate"] == best_eval_success and
                    eval_metrics["mean_reward"] > best_eval_reward
                )
            ):
                best_eval_success = eval_metrics["success_rate"]
                best_eval_reward = eval_metrics["mean_reward"]
                torch.save(policy.state_dict(), best_ckpt)
                print(
                    f"  ✓ Saved eval-best pickup checkpoint "
                    f"(success={best_eval_success:.1f}%, reward={best_eval_reward:.1f})",
                    flush=True,
                )

        if episode % 500 == 0:
            elapsed = time.time() - start
            ep_per_sec = episode / max(elapsed, 1e-6)
            print(
                f"Ep {episode:6d} | R: {np.mean(episode_rewards):7.1f} | "
                f"Success: {100.0 * np.mean(successes):5.1f}% | "
                f"EvalBest: {best_eval_success:.1f}% | "
                f"Total: {total_successes} | {ep_per_sec:.1f} ep/s",
                flush=True,
            )
            torch.save(policy.state_dict(), latest_ckpt)

    torch.save(policy.state_dict(), latest_ckpt)
    print(
        f"\nDONE | Best eval success: {best_eval_success:.1f}% | Total successes: {total_successes}",
        flush=True,
    )
    env.close()
