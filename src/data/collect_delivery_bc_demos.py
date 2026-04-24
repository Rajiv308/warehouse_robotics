"""
Collect behavior-cloning demonstrations for the Phase 2 delivery navigation
module.

The scripted diff-drive P-controller (same logic used in
src/eval/demo_phase2_hybrid.py) is the expert. At each frame we record
  - state:  target position in the Husky body frame + heading error
  - action: (left_wheel_velocity, right_wheel_velocity)

Episodes are reset with random Husky start poses and random dropoff targets
(drawn from the same ranges the demo randomizes). Only successful episodes
(Husky reaches within 0.12 m of the target) contribute to the dataset.

Output: demos/delivery_bc.npz
"""
import os
import sys

import numpy as np
import pybullet as p

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.env.warehouse_env_mobile_v2 import MobileWarehouseEnvV2


WHEEL_RADIUS = 0.17
WHEEL_FORCE = 50.0
# Soft action bound used for normalization: max_drive / wheel_radius ~ 11.8,
# max_turn likewise. We normalize with 15 rad/s to keep actions within [-1, 1].
WHEEL_VEL_NORM = 15.0

# Starting positions the Husky might realistically be at after pickup. We
# sample a mix of shelf-front poses plus broader starts so the BC policy
# generalizes to new approach situations.
HUSKY_START_RANGES = [
    # (x_min, x_max, y_min, y_max, yaw_min, yaw_max)
    ( 1.6,  2.4, -2.2, -0.8, np.pi / 3.0,  2 * np.pi / 3.0),   # right shelf area
    (-2.4, -1.6, -2.2, -0.8, np.pi / 3.0,  2 * np.pi / 3.0),   # left shelf area
    (-1.5,  1.5, -2.5, -0.5, -np.pi,       np.pi),             # open center
]


def body_frame_state(husky_pos, husky_yaw, target_xy):
    dx = float(target_xy[0] - husky_pos[0])
    dy = float(target_xy[1] - husky_pos[1])
    dist = float(np.hypot(dx, dy))
    tx = dx * np.cos(husky_yaw) + dy * np.sin(husky_yaw)
    ty = -dx * np.sin(husky_yaw) + dy * np.cos(husky_yaw)
    heading_err = float(np.arctan2(dy, dx) - husky_yaw)
    while heading_err > np.pi:
        heading_err -= 2 * np.pi
    while heading_err < -np.pi:
        heading_err += 2 * np.pi
    return np.array([
        float(tx), float(ty), dist,
        float(np.sin(heading_err)), float(np.cos(heading_err)),
    ], dtype=np.float32)


def short_angle(a, b):
    d = b - a
    while d > np.pi:
        d -= 2 * np.pi
    while d < -np.pi:
        d += 2 * np.pi
    return d


def scripted_wheel_vels(husky_pos, husky_yaw, target_xy,
                        max_drive=2.0, max_turn=2.0, pos_tol=0.12):
    """Matches the demo's drive_to_physics controller. Returns left_vel,
    right_vel in rad/s (not normalized)."""
    dx = target_xy[0] - husky_pos[0]
    dy = target_xy[1] - husky_pos[1]
    dist = float(np.hypot(dx, dy))
    if dist < pos_tol:
        return 0.0, 0.0
    desired_heading = float(np.arctan2(dy, dx))
    heading_err = short_angle(husky_yaw, desired_heading)
    if abs(heading_err) > 0.25:
        sign = 1.0 if heading_err > 0 else -1.0
        wv = max_turn / WHEEL_RADIUS
        return -sign * wv, sign * wv
    speed = min(dist * 1.5, max_drive)
    fwd = speed / WHEEL_RADIUS
    correction = heading_err * 1.8 / WHEEL_RADIUS
    return fwd - correction, fwd + correction


def set_wheel_vels(env, left_wheels, right_wheels, left_v, right_v):
    for wj in left_wheels:
        p.setJointMotorControl2(
            env.husky_id, wj, p.VELOCITY_CONTROL,
            targetVelocity=left_v, force=WHEEL_FORCE,
        )
    for wj in right_wheels:
        p.setJointMotorControl2(
            env.husky_id, wj, p.VELOCITY_CONTROL,
            targetVelocity=right_v, force=WHEEL_FORCE,
        )


def discover_wheels(env):
    left_wheels = []
    right_wheels = []
    for j in range(p.getNumJoints(env.husky_id)):
        name = p.getJointInfo(env.husky_id, j)[1].decode().lower()
        if "wheel" in name:
            if "left" in name:
                left_wheels.append(j)
            elif "right" in name:
                right_wheels.append(j)
    return left_wheels, right_wheels


def reset_husky(env, start_x, start_y, start_yaw, wheel_joints):
    p.resetBasePositionAndOrientation(
        env.husky_id,
        [start_x, start_y, 0.02],
        p.getQuaternionFromEuler([0, 0, start_yaw]),
    )
    p.resetBaseVelocity(env.husky_id, [0, 0, 0], [0, 0, 0])
    for wj in wheel_joints:
        p.resetJointState(env.husky_id, wj, 0.0, targetVelocity=0.0)


def sample_start_pose():
    r = HUSKY_START_RANGES[np.random.randint(len(HUSKY_START_RANGES))]
    return (
        float(np.random.uniform(r[0], r[1])),
        float(np.random.uniform(r[2], r[3])),
        float(np.random.uniform(r[4], r[5])),
    )


def sample_target():
    return (
        float(np.random.uniform(-1.3, 1.3)),
        float(np.random.uniform(1.2, 2.7)),
    )


def collect_episode(env, left_wheels, right_wheels, wheel_joints,
                    max_frames=600, pos_tol=0.12):
    start_x, start_y, start_yaw = sample_start_pose()
    target_x, target_y = sample_target()
    reset_husky(env, start_x, start_y, start_yaw, wheel_joints)

    trajectory = []
    reached = False
    for _ in range(max_frames):
        hp, ho = p.getBasePositionAndOrientation(env.husky_id)
        hy = p.getEulerFromQuaternion(ho)[2]

        state = body_frame_state(hp[:2], hy, (target_x, target_y))
        lvel, rvel = scripted_wheel_vels(hp[:2], hy, (target_x, target_y))
        trajectory.append((state, np.array([lvel, rvel], dtype=np.float32)))

        if state[2] < pos_tol:
            reached = True
            break

        set_wheel_vels(env, left_wheels, right_wheels, lvel, rvel)
        p.stepSimulation()

    # Stop wheels before returning (clean state for next episode).
    set_wheel_vels(env, left_wheels, right_wheels, 0.0, 0.0)
    for _ in range(10):
        p.stepSimulation()

    return trajectory if reached else []


def main():
    num_episodes = int(os.environ.get("PHASE2_DELIVERY_BC_NUM_EPISODES", "400"))
    out_path = os.environ.get("PHASE2_DELIVERY_BC_DEMO_PATH", "demos/delivery_bc.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    env = MobileWarehouseEnvV2(
        config_path="configs/config_cloud.yaml",
        render=False,
        curriculum_stage=0,
        success_mode="pickup",
    )
    env.initialize()

    left_wheels, right_wheels = discover_wheels(env)
    wheel_joints = left_wheels + right_wheels

    # Disable physical collision between the Husky and the dropoff pad, matching
    # the demo behavior so data distribution matches deployment.
    try:
        p.setCollisionFilterPair(env.husky_id, env.dropoff_id, -1, -1, enableCollision=0)
    except Exception:
        pass

    all_states = []
    all_actions = []
    successes = 0
    total = 0
    while successes < num_episodes:
        total += 1
        trajectory = collect_episode(env, left_wheels, right_wheels, wheel_joints)
        if trajectory:
            successes += 1
            for state, action in trajectory:
                all_states.append(state)
                all_actions.append(action / WHEEL_VEL_NORM)  # normalize
            if successes % 25 == 0:
                print(f"  Collected {successes} episodes "
                      f"({successes}/{total} success rate)")
        if total > num_episodes * 4:
            print(f"  WARNING: too many failures ({successes}/{total}), stopping early")
            break

    states_arr = np.stack(all_states).astype(np.float32)
    actions_arr = np.stack(all_actions).astype(np.float32)
    np.savez(out_path, states=states_arr, actions=actions_arr)
    print(f"\nSaved {states_arr.shape[0]} (state, action) pairs from "
          f"{successes}/{total} episodes to {out_path}")
    env.close()


if __name__ == "__main__":
    main()
