"""
Phase 2 demo for the current state-policy pickup pipeline.

This showcases the mobile manipulation stack we are actually training:
- navigate to the correct shelf
- approach the target object
- grasp and lift

By default the demo uses pickup mode, which matches the current trainer.
"""
import os
import sys
import time

import pybullet as p
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.env.warehouse_env_mobile_v2 import MobileWarehouseEnvV2
from src.training.train_state_rl_phase2_v2 import MobileStatePolicy, get_state


def run_demo(num_episodes=5, slow_motion=True):
    ckpt_path = os.environ.get(
        "PHASE2_CKPT",
        "checkpoints/phase2_state_policy_stage0_best.pth",
    )
    success_mode = os.environ.get("PHASE2_SUCCESS_MODE", "pickup")
    curriculum_stage = int(os.environ.get("PHASE2_DEMO_STAGE", "0"))

    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        return

    policy = MobileStatePolicy()
    policy.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    policy.eval()
    print(f"Loaded Phase 2 checkpoint: {ckpt_path}")

    env = MobileWarehouseEnvV2(
        config_path="configs/config_cloud.yaml",
        render=True,
        curriculum_stage=curriculum_stage,
        success_mode=success_mode,
    )
    env.initialize()
    env.env_cfg["max_episode_steps"] = 220

    p.resetDebugVisualizerCamera(
        cameraDistance=3.0,
        cameraYaw=55,
        cameraPitch=-28,
        cameraTargetPosition=[0.5, 0.0, 0.5],
    )

    successes = 0
    for ep in range(num_episodes):
        instruction = env.reset_state_only()
        print(f"\n--- Episode {ep + 1}/{num_episodes} ---")
        print(f"Instruction: {instruction}")

        state = get_state(env)
        ep_reward = 0.0
        success = False

        for step in range(env.env_cfg["max_episode_steps"]):
            st = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = policy.actor_mean(st).squeeze(0).cpu().numpy()

            reward, done, info = env.step_state_only(action)
            ep_reward += reward
            state = get_state(env)

            husky_pos, _ = p.getBasePositionAndOrientation(env.husky_id)
            p.resetDebugVisualizerCamera(
                cameraDistance=3.0,
                cameraYaw=55,
                cameraPitch=-28,
                cameraTargetPosition=[husky_pos[0], husky_pos[1], 0.5],
            )

            if slow_motion:
                time.sleep(0.02)

            if step % 50 == 0:
                print(
                    f"  Step {step}: shelf={info['dist_to_shelf']:.2f} "
                    f"obj={info['dist_to_obj']:.2f} z={info['obj_z']:.2f} "
                    f"grasped={info['grasped']} lifted={info['lifted']}"
                )

            if done:
                success = bool(info.get("success"))
                if success:
                    successes += 1
                    print(f"  SUCCESS at step {step}! Pickup objective completed.")
                else:
                    print(
                        f"  No success this episode "
                        f"(reward: {ep_reward:.1f}, grasped={info['grasped']}, lifted={info['lifted']})"
                    )
                break

        if not done:
            print(f"  Timed out (reward: {ep_reward:.1f})")

    print("\n=== DEMO RESULTS ===")
    print(f"Success: {successes}/{num_episodes} ({100.0 * successes / max(num_episodes, 1):.0f}%)")
    input("\nPress Enter to close...")
    env.close()


if __name__ == "__main__":
    run_demo()
