"""
Phase 2 navigation-only training.

This stage learns only one skill:
- move the mobile base to a valid pickup pose in front of the correct object

The arm stays in a fixed home pose with the gripper open.
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


class NavPolicy(nn.Module):
    def __init__(self, state_dim=12, action_dim=3):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.actor(x)


def get_nav_state(env):
    husky_pos, husky_orn = p.getBasePositionAndOrientation(env.husky_id)
    yaw = p.getEulerFromQuaternion(husky_orn)[2]
    target_metrics = env.get_target_metrics()
    obj_xy = target_metrics["obj_pos"][:2].astype(np.float32)
    shelf_xy = target_metrics["target_shelf"][:2].astype(np.float32)
    robot_xy = np.array(husky_pos[:2], dtype=np.float32)
    pick_xy, pick_yaw = env.get_pick_pose()

    rel_pick = pick_xy - robot_xy
    rel_obj = obj_xy - robot_xy
    yaw_err = pick_yaw - yaw
    while yaw_err > np.pi:
        yaw_err -= 2 * np.pi
    while yaw_err < -np.pi:
        yaw_err += 2 * np.pi

    return np.concatenate([
        robot_xy,
        np.array([yaw], dtype=np.float32),
        shelf_xy,
        obj_xy,
        rel_pick.astype(np.float32),
        rel_obj.astype(np.float32),
        np.array([yaw_err], dtype=np.float32),
    ]).astype(np.float32)


def expert_nav_action(env):
    husky_pos, husky_orn = p.getBasePositionAndOrientation(env.husky_id)
    yaw = p.getEulerFromQuaternion(husky_orn)[2]
    robot_xy = np.array(husky_pos[:2], dtype=np.float32)
    pick_xy, pick_yaw = env.get_pick_pose()
    diff = pick_xy - robot_xy
    dist = np.linalg.norm(diff)

    if dist < 0.08:
        vx = 0.0
    else:
        desired = np.arctan2(diff[1], diff[0])
        err = desired - yaw
        while err > np.pi:
            err -= 2 * np.pi
        while err < -np.pi:
            err += 2 * np.pi
        vx = np.clip(dist * 1.2, 0.0, 1.0)
        if abs(err) > 0.5:
            vx *= 0.25

    yaw_err = pick_yaw - yaw
    while yaw_err > np.pi:
        yaw_err -= 2 * np.pi
    while yaw_err < -np.pi:
        yaw_err += 2 * np.pi

    wz = np.clip(1.6 * yaw_err, -1.0, 1.0)
    return np.array([vx, 0.0, wz], dtype=np.float32)


def build_full_action(nav_action):
    full = np.zeros(10, dtype=np.float32)
    full[:3] = nav_action
    home = np.array([0, -0.785, 0, -2.356, 0, 1.571], dtype=np.float32)
    full[3:9] = home
    full[9] = 1.0
    return full


def nav_success(env):
    husky_pos, husky_orn = p.getBasePositionAndOrientation(env.husky_id)
    yaw = p.getEulerFromQuaternion(husky_orn)[2]
    pick_xy, pick_yaw = env.get_pick_pose()
    robot_xy = np.array(husky_pos[:2], dtype=np.float32)
    pos_err = float(np.linalg.norm(robot_xy - pick_xy))
    yaw_err = pick_yaw - yaw
    while yaw_err > np.pi:
        yaw_err -= 2 * np.pi
    while yaw_err < -np.pi:
        yaw_err += 2 * np.pi
    return pos_err < 0.12 and abs(yaw_err) < 0.25, pos_err, abs(yaw_err)


def collect_dataset(env, num_demos=200, max_steps=80):
    states = []
    actions = []
    print(f"Collecting {num_demos} navigation expert rollouts...", flush=True)
    for _ in range(num_demos):
        env.reset_state_only()
        for _ in range(max_steps):
            states.append(get_nav_state(env))
            nav_action = expert_nav_action(env)
            actions.append(nav_action)
            _, done, info = env.step_state_only(build_full_action(nav_action))
            success, _, _ = nav_success(env)
            if success:
                break
            if done:
                break
    return (
        torch.FloatTensor(np.array(states, dtype=np.float32)),
        torch.FloatTensor(np.array(actions, dtype=np.float32)),
    )


def evaluate(policy, env, episodes=25):
    policy.eval()
    success_count = 0
    pos_errs = []
    yaw_errs = []
    with torch.no_grad():
        for _ in range(episodes):
            env.reset_state_only()
            final_pos_err = None
            final_yaw_err = None
            for _ in range(100):
                state = torch.FloatTensor(get_nav_state(env)).unsqueeze(0)
                nav_action = policy(state).squeeze(0).cpu().numpy()
                _, done, _ = env.step_state_only(build_full_action(nav_action))
                success, pos_err, yaw_err = nav_success(env)
                final_pos_err = pos_err
                final_yaw_err = yaw_err
                if success:
                    success_count += 1
                    pos_errs.append(pos_err)
                    yaw_errs.append(yaw_err)
                    break
                if done:
                    pos_errs.append(pos_err)
                    yaw_errs.append(yaw_err)
                    break
            else:
                if final_pos_err is not None and final_yaw_err is not None:
                    pos_errs.append(final_pos_err)
                    yaw_errs.append(final_yaw_err)
    return {
        "success_rate": 100.0 * success_count / max(episodes, 1),
        "mean_pos_err": float(np.mean(pos_errs)) if pos_errs else 0.0,
        "mean_yaw_err": float(np.mean(yaw_errs)) if yaw_errs else 0.0,
    }


if __name__ == "__main__":
    num_epochs = int(os.environ.get("PHASE2_NAV_EPOCHS", "40"))
    num_demos = int(os.environ.get("PHASE2_NAV_DEMOS", "220"))
    nav_stage = int(os.environ.get("PHASE2_NAV_STAGE", "0"))
    ckpt = os.environ.get(
        "PHASE2_NAV_CKPT",
        f"checkpoints/phase2_nav_pickpose_stage{nav_stage}_best.pth",
    )

    env = MobileWarehouseEnvV2(
        config_path="configs/config_cloud.yaml",
        render=False,
        curriculum_stage=nav_stage,
        success_mode="pickup",
    )
    env.initialize()
    env.env_cfg["max_episode_steps"] = 120

    policy = NavPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    states, actions = collect_dataset(env, num_demos=num_demos)
    print(f"Dataset: {states.shape[0]:,} samples", flush=True)

    best_success = -1.0
    for epoch in range(num_epochs):
        perm = torch.randperm(states.shape[0])
        losses = []
        for start in range(0, states.shape[0], 256):
            idx = perm[start:start + 256]
            pred = policy(states[idx])
            loss = nn.MSELoss()(pred, actions[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        metrics = evaluate(policy, env, episodes=20)
        print(
            f"Epoch {epoch + 1:02d}/{num_epochs} | "
            f"loss={np.mean(losses):.4f} | "
            f"success={metrics['success_rate']:.1f}% | "
            f"pos_err={metrics['mean_pos_err']:.3f} | "
            f"yaw_err={metrics['mean_yaw_err']:.3f}",
            flush=True,
        )
        if metrics["success_rate"] > best_success:
            best_success = metrics["success_rate"]
            torch.save(policy.state_dict(), ckpt)
            print(f"  ✓ Saved best nav checkpoint to {ckpt}", flush=True)

    env.close()
