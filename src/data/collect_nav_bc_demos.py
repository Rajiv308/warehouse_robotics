"""
Behavior-cloning demonstration collector for Phase 2 navigation over a broad
spawn distribution.

The original nav BC checkpoint was trained only on curriculum stage 0
(Husky already shelf-proximal). For a stronger demo, we want nav that
generalizes to arbitrary starts — center of the aisle, random locations,
etc. This collector runs the existing scripted `expert_nav_action`
P-controller from random starts over the whole accessible workspace and
saves (state, action) pairs from successful rollouts.

Output: demos/nav_bc.npz
"""
import os
import sys

import numpy as np
import pybullet as p

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.env.warehouse_env_mobile_v2 import MobileWarehouseEnvV2
from src.training.train_phase2_nav_pickpose import (
    NavPolicy, get_nav_state, nav_success, build_full_action,
)


def smart_expert_nav_action(env):
    """Turn-then-drive P-controller. The original expert in
    train_phase2_nav_pickpose.py only aligns yaw to pick_yaw, which fails for
    lateral starts. This one rotates toward the target direction first,
    drives when heading is aligned, and only aligns to pick_yaw at the end."""
    hp, ho = p.getBasePositionAndOrientation(env.husky_id)
    yaw = p.getEulerFromQuaternion(ho)[2]
    pick_xy, pick_yaw = env.get_pick_pose()
    diff = pick_xy - np.array(hp[:2], dtype=np.float32)
    dist = float(np.linalg.norm(diff))

    def wrap(a):
        while a > np.pi:
            a -= 2 * np.pi
        while a < -np.pi:
            a += 2 * np.pi
        return a

    if dist < 0.08:
        yerr = wrap(pick_yaw - yaw)
        return np.array([0.0, 0.0, np.clip(1.8 * yerr, -1.0, 1.0)], dtype=np.float32)

    desired_heading = float(np.arctan2(diff[1], diff[0]))
    herr = wrap(desired_heading - yaw)
    if abs(herr) > 0.3:
        return np.array([0.0, 0.0, np.clip(1.8 * herr, -1.0, 1.0)], dtype=np.float32)
    vx = float(np.clip(dist * 1.8, 0.05, 1.0))
    wz = float(np.clip(1.5 * herr, -1.0, 1.0))
    return np.array([vx, 0.0, wz], dtype=np.float32)


expert_nav_action = smart_expert_nav_action  # local override for this file


SPAWN_X_RANGE = (-2.4, 2.4)
SPAWN_Y_RANGE = (-3.0, -0.4)
SHELF_CLEAR = 0.7  # keep away from the shelf footprint


def sample_valid_start():
    for _ in range(50):
        x = float(np.random.uniform(*SPAWN_X_RANGE))
        y = float(np.random.uniform(*SPAWN_Y_RANGE))
        if abs(abs(x) - 2.0) < 0.6 + SHELF_CLEAR and abs(y) < 0.3 + SHELF_CLEAR:
            continue
        yaw = float(np.random.uniform(-np.pi, np.pi))
        return x, y, yaw
    return 0.0, -1.5, float(np.pi / 2)


def collect_episode(env, max_steps=300):
    env.reset_state_only()
    x, y, yaw = sample_valid_start()
    p.resetBasePositionAndOrientation(
        env.husky_id, [x, y, 0.02],
        p.getQuaternionFromEuler([0, 0, yaw]),
    )
    p.resetBaseVelocity(env.husky_id, [0, 0, 0], [0, 0, 0])
    env._sync_panda_to_husky([x, y, 0.02], yaw)

    trajectory = []
    for _ in range(max_steps):
        state = get_nav_state(env)
        action = expert_nav_action(env)
        trajectory.append((state, action))
        env.step_state_only(build_full_action(action))
        ok, _, _ = nav_success(env)
        if ok:
            return trajectory
    return []


def main():
    num_episodes = int(os.environ.get("PHASE2_NAV_BC_NUM_EPISODES", "400"))
    out_path = os.environ.get("PHASE2_NAV_BC_DEMO_PATH", "demos/nav_bc.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    env = MobileWarehouseEnvV2(
        config_path="configs/config_cloud.yaml",
        render=False, curriculum_stage=0, success_mode="pickup",
    )
    env.initialize()
    try:
        p.setCollisionFilterPair(env.husky_id, env.dropoff_id, -1, -1, enableCollision=0)
    except Exception:
        pass

    all_states, all_actions = [], []
    successes, total = 0, 0
    while successes < num_episodes:
        total += 1
        traj = collect_episode(env)
        if traj:
            successes += 1
            for s, a in traj:
                all_states.append(s)
                all_actions.append(a)
            if successes % 25 == 0:
                print(f"  Collected {successes} episodes "
                      f"({successes}/{total} success rate)")
        if total > num_episodes * 4:
            print(f"  WARNING: too many failures ({successes}/{total}), stopping")
            break

    s_arr = np.stack(all_states).astype(np.float32)
    a_arr = np.stack(all_actions).astype(np.float32)
    np.savez(out_path, states=s_arr, actions=a_arr)
    print(f"\nSaved {s_arr.shape[0]} (state, action) pairs from "
          f"{successes}/{total} episodes to {out_path}")
    env.close()


if __name__ == "__main__":
    main()
