"""
Demo the simplified Cartesian Phase 1 pickup policy.
"""
import os
import sys
import time

import numpy as np
import pybullet as p
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.env.warehouse_env import WarehouseEnv
from src.training.train_phase1_cartesian_pick import CartesianPickPolicy


def run_demo(num_episodes=10, slow_motion=True):
    ckpt = os.environ.get("PHASE1_CART_CKPT", "checkpoints/phase1_cartesian_target0_best.pth")
    target_idx = int(os.environ.get("PHASE1_TARGET_IDX", "0"))
    if not os.path.exists(ckpt):
        print(f"ERROR: checkpoint not found: {ckpt}")
        return

    policy = CartesianPickPolicy()
    policy.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    policy.eval()
    print(f"Loaded Cartesian checkpoint: {ckpt}")

    env = WarehouseEnv(render=True)
    env.initialize()
    p.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=40,
        cameraPitch=-32,
        cameraTargetPosition=[0.48, 0.0, 0.12],
    )

    successes = 0
    for ep in range(num_episodes):
        _, instruction = env.reset_simple_task(target_idx=target_idx, distractors=False, position_noise=0.01)
        print(f"\n--- Episode {ep + 1}/{num_episodes} ---")
        print(f"Instruction: {instruction}")
        state = policy.get_state(env)
        ep_reward = 0.0
        success = False

        for step in range(env.env_cfg["max_episode_steps"]):
            st = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = policy.actor_mean(st).squeeze(0).cpu().numpy()
            env.apply_cartesian_action(action)
            p.stepSimulation()
            env.step_count += 1
            metrics = env.get_target_metrics()
            obj_pos = metrics["obj_pos"]
            gripper_pos = metrics["gripper_pos"]
            xy_dist = float(np.linalg.norm(gripper_pos[:2] - obj_pos[:2]))
            z_gap = float(gripper_pos[2] - obj_pos[2])
            aligned = xy_dist < 0.025 and 0.09 <= z_gap <= 0.15 and not metrics["gripper_closed"]

            reward = env.compute_reward()
            success, metrics = env.update_success_state()
            if aligned:
                success = env.execute_pick_macro()
                metrics = env.get_target_metrics()
            ep_reward += reward
            state = policy.get_state(env)
            if slow_motion:
                time.sleep(0.02)
            if success:
                print(f"  SUCCESS at step {step}! Object grasped and lifted!")
                successes += 1
                break

        if not success:
            print(f"  No grasp this episode (reward: {ep_reward:.1f})")

    print("\n=== DEMO RESULTS ===")
    print(f"Success: {successes}/{num_episodes} ({100.0 * successes / max(num_episodes, 1):.0f}%)")
    input("\nPress Enter to close...")
    env.close()


if __name__ == "__main__":
    run_demo()
