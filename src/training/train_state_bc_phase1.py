"""
Phase 1 state-based behavioral cloning.
Trains a deterministic policy to imitate the now-working IK expert using
compact state inputs instead of pixels. This is the fastest path to a stable,
graceful pick-and-lift demo for Phase 1.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pybullet as p

from src.env.warehouse_env import WarehouseEnv
from src.data.collect_demos import IKExpertController
from src.training.train_state_rl_phase1 import StatePolicy, collect_expert_dataset


def evaluate_actor(policy, env, episodes=20, max_steps=220):
    policy.eval()
    successes = 0
    lift_heights = []
    episode_rewards = []

    with torch.no_grad():
        for _ in range(episodes):
            env.reset()
            total_reward = 0.0
            max_z = 0.0

            for _ in range(max_steps):
                state = policy.get_state(env)
                state_t = torch.FloatTensor(state).unsqueeze(0)
                action = policy.actor_mean(state_t).squeeze().cpu().numpy()
                env.apply_action(action)
                p.stepSimulation()
                env.step_count += 1
                reward = env.compute_reward()
                success, metrics = env.update_success_state()
                total_reward += reward
                max_z = max(max_z, metrics["obj_z"])
                if success:
                    successes += 1
                    break

            lift_heights.append(max_z)
            episode_rewards.append(total_reward)

    return {
        "success_rate": successes / episodes,
        "avg_reward": float(np.mean(episode_rewards)),
        "avg_max_z": float(np.mean(lift_heights)),
        "max_z": float(np.max(lift_heights)),
    }


def train_state_bc(num_demos=300, bc_epochs=25, batch_size=256, lr=1e-3,
                   steps_per_demo=110, max_steps=220):
    device = torch.device("cpu")
    print(f"Phase 1 State BC on: {device}", flush=True)

    env = WarehouseEnv(render=False)
    env.initialize()
    env.env_cfg["max_episode_steps"] = max_steps

    policy = StatePolicy().to(device)
    optimizer = torch.optim.Adam(policy.actor_mean.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)

    expert_states, expert_actions = collect_expert_dataset(
        policy, env, num_demos=num_demos, steps_per_demo=steps_per_demo
    )
    expert_states = expert_states.to(device)
    expert_actions = expert_actions.to(device)

    best_success = -1.0
    best_loss = float("inf")
    num_samples = expert_states.shape[0]

    print(f"Training on {num_samples:,} expert state-action pairs", flush=True)
    start = time.time()

    for epoch in range(bc_epochs):
        policy.train()
        perm = torch.randperm(num_samples)
        losses = []

        for start_idx in range(0, num_samples, batch_size):
            idx = perm[start_idx:start_idx + batch_size]
            pred = policy.actor_mean(expert_states[idx])

            joint_loss = nn.MSELoss()(pred[:, :6], expert_actions[idx][:, :6])
            gripper_loss = nn.MSELoss()(pred[:, 6:], expert_actions[idx][:, 6:])
            loss = joint_loss + 4.0 * gripper_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.actor_mean.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        avg_loss = float(np.mean(losses))
        metrics = evaluate_actor(policy, env, episodes=20, max_steps=max_steps)

        print(
            f"Epoch {epoch + 1:2d}/{bc_epochs} | "
            f"loss={avg_loss:.4f} | "
            f"success={metrics['success_rate'] * 100:.1f}% | "
            f"avg_max_z={metrics['avg_max_z']:.3f} | "
            f"avg_reward={metrics['avg_reward']:.1f}",
            flush=True
        )

        if metrics["success_rate"] > best_success or (
            metrics["success_rate"] == best_success and avg_loss < best_loss
        ):
            best_success = metrics["success_rate"]
            best_loss = avg_loss
            torch.save(policy.state_dict(), "checkpoints/phase1_state_bc_policy.pth")
            print(
                f"  Saved best BC policy | success={best_success * 100:.1f}% | loss={best_loss:.4f}",
                flush=True
            )

    elapsed = (time.time() - start) / 60.0
    print(f"Done in {elapsed:.1f} min", flush=True)
    env.close()


if __name__ == "__main__":
    train_state_bc()
