"""
Phase 1 Demo — Watch the trained policy grasp objects in PyBullet GUI.
Run: python3 src/eval/demo_phase1.py
"""
import sys, os, torch, numpy as np, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pybullet as p
from src.env.warehouse_env import WarehouseEnv
from src.training.train_state_rl_phase1 import StatePolicy


def run_stabilized_lift(env, action_dim, hold_steps=70, lift_delta=0.14, sleep_s=0.02):
    """Demo-only polish after success: lift a bit higher and hold."""
    gripper_state = p.getLinkState(env.robot_id, 11)
    gripper_pos = np.array(gripper_state[0], dtype=np.float32)
    target_pos = gripper_pos.copy()
    target_pos[2] = max(target_pos[2] + lift_delta, 0.24)

    num_arm_joints = max(1, action_dim - 1)
    current_joints = np.array(
        [p.getJointState(env.robot_id, j)[0] for j in range(num_arm_joints)],
        dtype=np.float32,
    )
    ik = p.calculateInverseKinematics(
        env.robot_id,
        11,
        target_pos.tolist(),
        maxNumIterations=200,
        residualThreshold=1e-4,
    )
    target_joints = np.array(ik[:num_arm_joints], dtype=np.float32)

    for t in range(hold_steps):
        alpha = min(1.0, (t + 1) / max(hold_steps * 0.5, 1))
        blended = (1.0 - alpha) * current_joints + alpha * target_joints
        action = np.zeros(action_dim, dtype=np.float32)
        action[:num_arm_joints] = blended
        action[-1] = 0.0  # keep gripper closed
        env.apply_action(action)
        p.stepSimulation()
        env.step_count += 1
        if sleep_s > 0:
            time.sleep(sleep_s)


def run_demo(num_episodes=5, slow_motion=True):
    print("Loading Phase 1 trained policy...", flush=True)
    env_ckpt = os.environ.get("PHASE1_CKPT")
    candidate_paths = [
        env_ckpt,
        "checkpoints/phase1_state_policy_fullarm_eval_best.pth",
        "checkpoints/phase1_state_policy_eval_best.pth",
        "checkpoints/phase1_state_policy_fullarm.pth",
        "checkpoints/phase1_state_policy.pth",
    ]
    ckpt_path = next((path for path in candidate_paths if path and os.path.exists(path)), None)
    if ckpt_path is None:
        print("ERROR: no Phase 1 state checkpoint found!")
        return
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    action_dim = state_dict["actor_mean.4.bias"].shape[0]
    policy = StatePolicy(action_dim=action_dim)
    policy.load_state_dict(state_dict)
    policy.eval()
    print(f"Policy loaded from {ckpt_path}! (action_dim={action_dim})", flush=True)

    # Use GUI mode for visual rendering
    env = WarehouseEnv(render=True)
    env.initialize()
    env.attach_dist_threshold = 0.075
    env.success_lift_height = 0.09
    env.success_hold_steps = 4

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
        success = False

        for step in range(300):
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_np = policy.actor_mean(state_t).squeeze().numpy()

            env.apply_action(action_np)
            p.stepSimulation()
            env.step_count += 1
            reward = env.compute_reward()

            success, metrics = env.update_success_state()
            ep_reward += reward
            state = policy.get_state(env)

            if slow_motion:
                time.sleep(0.02)  # 50fps playback

            if success:
                print(f"  SUCCESS at step {step}! Object grasped and lifted!")
                successes += 1
                run_stabilized_lift(
                    env,
                    action_dim=action_dim,
                    hold_steps=60,
                    lift_delta=0.12,
                    sleep_s=0.02 if slow_motion else 0.0,
                )
                break

        if not success:
            print(f"  No grasp this episode (reward: {ep_reward:.1f})")

    print(f"\n=== DEMO RESULTS ===")
    print(f"Success: {successes}/{num_episodes} ({successes/num_episodes*100:.0f}%)")

    input("\nPress Enter to close...")
    env.close()


if __name__ == "__main__":
    run_demo(num_episodes=10, slow_motion=True)
