"""
Phase 1 Demo — Watch the trained policy grasp objects in PyBullet GUI.
Run: python3 src/eval/demo_phase1.py
"""
import sys, os, torch, numpy as np, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pybullet as p
from src.env.warehouse_env import WarehouseEnv
from src.training.train_state_rl_phase1 import StatePolicy


def run_demo(num_episodes=5, slow_motion=True):
    print("Loading Phase 1 trained policy...", flush=True)
    policy = StatePolicy()
    ckpt_path = "checkpoints/phase1_state_policy.pth"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: {ckpt_path} not found!")
        return
    policy.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    policy.eval()
    print("Policy loaded!", flush=True)

    # Use GUI mode for visual rendering
    env = WarehouseEnv(render=True)
    env.initialize()

    # Set nice camera angle
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0.4, 0.0, 0.2]
    )

    successes = 0
    for ep in range(num_episodes):
        obs, instruction = env.reset()
        print(f"\n--- Episode {ep+1}/{num_episodes} ---")
        print(f"Instruction: {instruction}")

        state = policy.get_state(env)
        ep_reward = 0

        for step in range(300):
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = policy(state_t)
            action_np = action.squeeze().numpy()

            env.apply_action(action_np)
            p.stepSimulation()
            env.step_count += 1
            reward = env.compute_reward()

            if env._grasped:
                env._lift_count = getattr(env, '_lift_count', 0) + 1
            else:
                env._lift_count = 0

            success = env._lift_count >= 15
            ep_reward += reward
            state = policy.get_state(env)

            if slow_motion:
                time.sleep(0.02)  # 50fps playback

            if success:
                print(f"  SUCCESS at step {step}! Object grasped and lifted!")
                successes += 1
                # Hold for a moment so user can see
                for _ in range(50):
                    p.stepSimulation()
                    time.sleep(0.02)
                break

        if not success:
            print(f"  No grasp this episode (reward: {ep_reward:.1f})")

    print(f"\n=== DEMO RESULTS ===")
    print(f"Success: {successes}/{num_episodes} ({successes/num_episodes*100:.0f}%)")

    input("\nPress Enter to close...")
    env.close()


if __name__ == "__main__":
    run_demo(num_episodes=10, slow_motion=True)
