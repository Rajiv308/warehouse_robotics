"""
Headless evaluation for the delivery-drive BC policy.

Resets the Husky into random starts (same distribution the BC was trained on),
picks a random target, runs the BC policy for up to a budget of frames, and
reports success rate (fraction of episodes where Husky reached within
`pos_tol` meters of the target).

Used as a gate before swapping BC into the main demo.
"""
import os
import sys

import numpy as np
import pybullet as p
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.env.warehouse_env_mobile_v2 import MobileWarehouseEnvV2
from src.data.collect_delivery_bc_demos import (
    WHEEL_VEL_NORM, body_frame_state, discover_wheels, reset_husky,
    sample_start_pose, sample_target, set_wheel_vels,
)
from src.training.train_bc_delivery import DeliveryDrivePolicy


def main():
    ckpt_path = os.environ.get(
        "PHASE2_DELIVERY_BC_CKPT", "checkpoints/phase2_delivery_bc_best.pth"
    )
    num_episodes = int(os.environ.get("PHASE2_DELIVERY_BC_EVAL_EPISODES", "30"))
    pos_tol = float(os.environ.get("PHASE2_DELIVERY_BC_EVAL_TOL", "0.15"))
    max_frames = int(os.environ.get("PHASE2_DELIVERY_BC_EVAL_MAX_FRAMES", "800"))

    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        return

    policy = DeliveryDrivePolicy()
    policy.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    policy.eval()

    env = MobileWarehouseEnvV2(
        config_path="configs/config_cloud.yaml",
        render=False,
        curriculum_stage=0,
        success_mode="pickup",
    )
    env.initialize()

    left_wheels, right_wheels = discover_wheels(env)
    wheel_joints = left_wheels + right_wheels

    try:
        p.setCollisionFilterPair(env.husky_id, env.dropoff_id, -1, -1, enableCollision=0)
    except Exception:
        pass

    successes = 0
    frame_totals = []
    for ep in range(num_episodes):
        sx, sy, syaw = sample_start_pose()
        tx, ty = sample_target()
        reset_husky(env, sx, sy, syaw, wheel_joints)

        reached = False
        used_frames = 0
        for f in range(max_frames):
            hp, ho = p.getBasePositionAndOrientation(env.husky_id)
            hy = p.getEulerFromQuaternion(ho)[2]
            state = body_frame_state(hp[:2], hy, (tx, ty))
            if state[2] < pos_tol:
                reached = True
                used_frames = f
                break
            st_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                norm_action = policy(st_tensor).squeeze(0).cpu().numpy()
            lvel = float(norm_action[0]) * WHEEL_VEL_NORM
            rvel = float(norm_action[1]) * WHEEL_VEL_NORM
            set_wheel_vels(env, left_wheels, right_wheels, lvel, rvel)
            p.stepSimulation()
            used_frames = f

        set_wheel_vels(env, left_wheels, right_wheels, 0.0, 0.0)
        for _ in range(5):
            p.stepSimulation()

        if reached:
            successes += 1
            frame_totals.append(used_frames)
        print(f"  Ep {ep+1:2d}: start=({sx:.2f},{sy:.2f}) target=({tx:.2f},{ty:.2f})  "
              f"{'REACHED' if reached else 'TIMEOUT'} in {used_frames} frames")

    print("\n=== DELIVERY BC EVAL ===")
    print(f"Success: {successes}/{num_episodes} ({100.0 * successes / num_episodes:.0f}%)")
    if frame_totals:
        print(f"Avg frames to reach: {np.mean(frame_totals):.0f}")
    env.close()


if __name__ == "__main__":
    main()
