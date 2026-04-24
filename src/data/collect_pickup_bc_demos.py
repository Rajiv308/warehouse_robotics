"""
Collect behavior-cloning demonstrations for the Phase 2 pickup module.

Strategy
--------
The hybrid demo already uses a deterministic grasp finalizer (open gripper ->
IK hover above object -> IK descend -> close gripper -> verify two-finger
contact -> weld at live relative pose -> IK lift). This script reproduces the
same finalizer logic but expresses each step as a normalized Cartesian action
in the pickup policy's action space (the same action shape as
MobileCartesianPickPolicy / step_pickup_cartesian), records (state, action)
pairs from successful episodes, and saves them for BC training.

Environment auto-weld is disabled so the expert only "succeeds" when it
achieves a genuine two-finger contact grasp. Failed episodes are discarded.
"""
import os
import sys

import numpy as np
import pybullet as p

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.env.warehouse_env_mobile_v2 import MobileWarehouseEnvV2
from src.training.train_phase2_pick_cartesian import MobileCartesianPickPolicy


ACTION_SCALE = np.array([0.025, 0.025, 0.020], dtype=np.float32)


def cartesian_delta_action(env, target_xyz, grip):
    ee_pos = np.array(p.getLinkState(env.panda_id, 11)[0], dtype=np.float32)
    delta = (np.asarray(target_xyz, dtype=np.float32) - ee_pos) / ACTION_SCALE
    delta = np.clip(delta, -1.0, 1.0)
    return np.array([delta[0], delta[1], delta[2], grip], dtype=np.float32)


def two_finger_contact(env):
    target_id = env.object_ids[env.target_object_idx]
    left = p.getContactPoints(env.panda_id, target_id, linkIndexA=9) or []
    right = p.getContactPoints(env.panda_id, target_id, linkIndexA=10) or []
    return len(left) > 0 and len(right) > 0


def collect_episode(env, policy_state_fn, max_steps=160, verbose=False):
    """
    Runs the scripted pickup expert for one episode, recording (state, action)
    pairs at every env step. Returns the trajectory if the episode ends with
    real two-finger contact and a successful lift, otherwise an empty list.
    """
    env.auto_weld = False
    env._release_grasp_constraint()
    env.reset_pickup_task()
    initial_obj_z = float(p.getBasePositionAndOrientation(
        env.object_ids[env.target_object_idx]
    )[0][2])

    target_id = env.object_ids[env.target_object_idx]
    trajectory = []

    # Phase 1: drive to hover pose above the object with gripper open.
    obj_pos = np.array(p.getBasePositionAndOrientation(target_id)[0], dtype=np.float32)
    hover_xyz = obj_pos + np.array([0.0, 0.0, 0.10], dtype=np.float32)
    for _ in range(60):
        state = policy_state_fn(env)
        action = cartesian_delta_action(env, hover_xyz, grip=1.0)
        trajectory.append((state, action))
        env.step_pickup_cartesian(action)
        ee = np.array(p.getLinkState(env.panda_id, 11)[0], dtype=np.float32)
        if np.linalg.norm(ee - hover_xyz) < 0.015:
            break

    # Phase 2: descend to the object center with gripper open. The policy
    # action space caps horizontal/vertical delta at 2.5cm/2.0cm per step, so
    # we allow many frames for the arm to physically reach the object.
    for _ in range(80):
        obj_pos = np.array(p.getBasePositionAndOrientation(target_id)[0], dtype=np.float32)
        grasp_xyz = obj_pos.copy()
        state = policy_state_fn(env)
        action = cartesian_delta_action(env, grasp_xyz, grip=1.0)
        trajectory.append((state, action))
        env.step_pickup_cartesian(action)
        ee = np.array(p.getLinkState(env.panda_id, 11)[0], dtype=np.float32)
        if ee[2] - grasp_xyz[2] < 0.01 and np.linalg.norm(ee[:2] - grasp_xyz[:2]) < 0.02:
            break

    # Phase 3: close the gripper while holding horizontal alignment on the
    # object. Keeping a small corrective xy delta helps the fingers center.
    for _ in range(60):
        obj_pos = np.array(p.getBasePositionAndOrientation(target_id)[0], dtype=np.float32)
        state = policy_state_fn(env)
        action = cartesian_delta_action(env, obj_pos, grip=-1.0)
        trajectory.append((state, action))
        env.step_pickup_cartesian(action)

    if verbose:
        ee = np.array(p.getLinkState(env.panda_id, 11)[0])
        obj = np.array(p.getBasePositionAndOrientation(target_id)[0])
        left = p.getContactPoints(env.panda_id, target_id, linkIndexA=9) or []
        right = p.getContactPoints(env.panda_id, target_id, linkIndexA=10) or []
        print(f"    post-close: ee={ee} obj={obj} "
              f"xy_dist={np.linalg.norm(ee[:2]-obj[:2]):.3f} "
              f"z_gap={ee[2]-obj[2]:.3f} L={len(left)} R={len(right)}")

    # Contact gate: discard this episode if fingers are not truly on the box.
    if not two_finger_contact(env):
        return []

    # Manual weld (matches the demo's grasp finalizer logic) so the lift phase
    # actually carries the box. This also trains the BC model to command lift
    # actions while holding the object.
    ee_state = p.getLinkState(env.panda_id, 11)
    obj_pos_w, obj_orn_w = p.getBasePositionAndOrientation(target_id)
    inv_ee_pos, inv_ee_orn = p.invertTransform(ee_state[0], ee_state[1])
    rel_pos, rel_orn = p.multiplyTransforms(
        inv_ee_pos, inv_ee_orn, obj_pos_w, obj_orn_w,
    )
    env.grasp_constraint = p.createConstraint(
        parentBodyUniqueId=env.panda_id,
        parentLinkIndex=11,
        childBodyUniqueId=target_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=list(rel_pos),
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=list(rel_orn),
        childFrameOrientation=[0, 0, 0, 1],
    )
    p.changeConstraint(env.grasp_constraint, maxForce=10000)

    # Phase 4: lift up.
    obj_pos = np.array(p.getBasePositionAndOrientation(target_id)[0], dtype=np.float32)
    lift_xyz = obj_pos + np.array([0.0, 0.0, 0.22], dtype=np.float32)
    for _ in range(40):
        state = policy_state_fn(env)
        action = cartesian_delta_action(env, lift_xyz, grip=-1.0)
        trajectory.append((state, action))
        env.step_pickup_cartesian(action)
        ee = np.array(p.getLinkState(env.panda_id, 11)[0], dtype=np.float32)
        if ee[2] >= lift_xyz[2] - 0.01:
            break

    # Success check: did the box actually come up off the shelf?
    final_obj_z = float(p.getBasePositionAndOrientation(target_id)[0][2])
    if final_obj_z < initial_obj_z + 0.05:
        return []

    # Clean up the manual weld before returning.
    env._release_grasp_constraint()
    return trajectory


def main():
    num_episodes = int(os.environ.get("PHASE2_BC_NUM_EPISODES", "200"))
    out_path = os.environ.get("PHASE2_BC_DEMO_PATH", "demos/pickup_bc.npz")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    env = MobileWarehouseEnvV2(
        config_path="configs/config_cloud.yaml",
        render=False,
        curriculum_stage=0,
        success_mode="pickup",
    )
    env.initialize()
    env.env_cfg["max_episode_steps"] = 400
    env.auto_weld = False

    # Build a temporary policy instance just for its state_fn (shared with the
    # training-time state representation so BC and inference match).
    dummy_policy = MobileCartesianPickPolicy()
    state_fn = dummy_policy.get_state

    all_states = []
    all_actions = []
    successes = 0
    total = 0
    while successes < num_episodes:
        total += 1
        verbose = total <= 3
        trajectory = collect_episode(env, state_fn, verbose=verbose)
        if trajectory:
            successes += 1
            for state, action in trajectory:
                all_states.append(state)
                all_actions.append(action)
            if successes % 20 == 0:
                print(f"  Collected {successes} successful episodes "
                      f"({successes}/{total} success rate)")
        if total > num_episodes * 4:
            print(f"  WARNING: too many failures ({successes}/{total}), stopping early")
            break

    states_arr = np.stack(all_states).astype(np.float32)
    actions_arr = np.stack(all_actions).astype(np.float32)
    np.savez(out_path, states=states_arr, actions=actions_arr)
    print(f"\nSaved {states_arr.shape[0]} (state, action) pairs from "
          f"{successes} episodes to {out_path}")
    env.close()


if __name__ == "__main__":
    main()
