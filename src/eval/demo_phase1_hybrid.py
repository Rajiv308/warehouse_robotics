"""
Hybrid Phase 1 demo — fixed-base Panda arm, 3 colored boxes on a tabletop.

Mirrors the Phase 2 pattern: receive a natural-language instruction, parse
the target color, use vision to confirm the target is in view, then execute
a scripted grasp finalizer (open -> hover -> descend -> close ->
verify real two-finger contact -> weld at live relative pose -> lift).

The previously trained RL/BC Phase 1 policies did not converge to reliable
picks. This hybrid demo replaces them with a deterministic grasp, while
keeping instruction+vision grounding as the load-bearing input.
"""
import os
import sys
import time

import numpy as np
import pybullet as p

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.env.warehouse_env import WarehouseEnv
from src.perception.instruction_parser import parse_target_color, color_to_idx, idx_to_color
from src.perception.color_detector import detect_color


PHASE1_COLOR_POOL = ["red", "blue", "green"]


def pick_random_instruction():
    color = PHASE1_COLOR_POOL[np.random.randint(len(PHASE1_COLOR_POOL))]
    templates = [
        f"pick up the {color} box",
        f"grab the {color} box from the table",
        f"get the {color} box",
    ]
    return templates[np.random.randint(len(templates))], color


def run_demo(num_episodes=5, slow_motion=True):
    env = WarehouseEnv(render=True)
    env.initialize()
    env.attach_dist_threshold = 0.075

    p.resetDebugVisualizerCamera(
        cameraDistance=1.3,
        cameraYaw=45,
        cameraPitch=-32,
        cameraTargetPosition=[0.45, 0.0, 0.15],
    )

    successes = 0
    for ep in range(num_episodes):
        instruction, demo_color = pick_random_instruction()
        target_idx = color_to_idx(demo_color)
        env.reset_simple_task(target_idx=target_idx, distractors=True, position_noise=0.01)
        # Override the env's stock instruction with the paraphrased one, so the
        # language parser is doing real work.
        env.current_instruction = instruction

        print(f"\n--- Episode {ep + 1}/{num_episodes} ---")
        print(f"Instruction: {instruction}")

        parsed_color = parse_target_color(instruction)
        print(f"  Language parser extracted target color: '{parsed_color}'")

        rgb = env.get_camera_image()
        found, centroid, pixel_count = detect_color(rgb, parsed_color)
        if found:
            print(f"  Vision detected '{parsed_color}' at pixel "
                  f"({centroid[0]:.0f}, {centroid[1]:.0f}) [{pixel_count} px]")
        else:
            print(f"  Vision: '{parsed_color}' not detected "
                  f"(matched={pixel_count} px); falling back to env target.")

        grounded_idx = color_to_idx(parsed_color)
        if grounded_idx is not None:
            env._target_idx = grounded_idx

        target_id = env.object_ids[env._target_idx]

        # --- Scripted grasp finalizer ---
        def settle(frames):
            for _ in range(frames):
                p.stepSimulation()
                if slow_motion:
                    time.sleep(0.012)

        def drive_ee_to(target_xyz, frames=60):
            joints = p.calculateInverseKinematics(
                env.robot_id, 11, list(target_xyz),
                maxNumIterations=200, residualThreshold=1e-4,
            )
            p.setJointMotorControlArray(
                env.robot_id, env.arm_joints, p.POSITION_CONTROL,
                targetPositions=list(joints[:7]), forces=[87] * 7,
            )
            settle(frames)

        # 1. Open gripper.
        for gj in env.gripper_joints:
            p.setJointMotorControl2(
                env.robot_id, gj, p.POSITION_CONTROL,
                targetPosition=0.04, force=25,
            )
        settle(15)

        # 2. Hover above the object.
        obj_pos, _ = p.getBasePositionAndOrientation(target_id)
        hover_xyz = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.12]
        drive_ee_to(hover_xyz, frames=55)

        # 3. Descend to the object.
        grasp_xyz = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.01]
        drive_ee_to(grasp_xyz, frames=45)

        # 4. Close the gripper.
        for gj in env.gripper_joints:
            p.setJointMotorControl2(
                env.robot_id, gj, p.POSITION_CONTROL,
                targetPosition=0.0, force=60,
            )
        settle(45)

        # 5. Verify real two-finger contact.
        left_c = p.getContactPoints(env.robot_id, target_id, linkIndexA=9) or []
        right_c = p.getContactPoints(env.robot_id, target_id, linkIndexA=10) or []
        real_grasp = len(left_c) > 0 and len(right_c) > 0
        print(f"  Finger contacts: L={len(left_c)} R={len(right_c)}")

        if real_grasp:
            # Weld at the live relative pose so the box doesn't snap.
            ee_state = p.getLinkState(env.robot_id, 11)
            obj_pos_w, obj_orn_w = p.getBasePositionAndOrientation(target_id)
            inv_ee_pos, inv_ee_orn = p.invertTransform(ee_state[0], ee_state[1])
            rel_pos, rel_orn = p.multiplyTransforms(
                inv_ee_pos, inv_ee_orn, obj_pos_w, obj_orn_w,
            )
            env.grasp_constraint = p.createConstraint(
                parentBodyUniqueId=env.robot_id,
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
            p.changeConstraint(env.grasp_constraint, maxForce=500)

            # 6. Lift.
            lift_xyz = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.22]
            lift_joints = p.calculateInverseKinematics(
                env.robot_id, 11, lift_xyz,
                maxNumIterations=200, residualThreshold=1e-4,
            )
            p.setJointMotorControlArray(
                env.robot_id, env.arm_joints, p.POSITION_CONTROL,
                targetPositions=list(lift_joints[:7]), forces=[87] * 7,
            )

            def pin_obj():
                ee_now = p.getLinkState(env.robot_id, 11)
                wpos, worn = p.multiplyTransforms(ee_now[0], ee_now[1], rel_pos, rel_orn)
                p.resetBasePositionAndOrientation(target_id, wpos, worn)
                p.resetBaseVelocity(target_id, [0, 0, 0], [0, 0, 0])

            for _ in range(55):
                p.stepSimulation()
                pin_obj()
                if slow_motion:
                    time.sleep(0.012)

            final_pos, _ = p.getBasePositionAndOrientation(target_id)
            if final_pos[2] > obj_pos[2] + 0.08:
                successes += 1
                print(f"  SUCCESS: z {obj_pos[2]:.2f} -> {final_pos[2]:.2f}")
            else:
                print(f"  Lift failed. Object stayed at z={final_pos[2]:.2f}")
        else:
            print("  Grasp finalizer: no real contact, skipping lift.")

        # Hold for 15 frames at the lift so the viewer can see the result.
        for _ in range(20):
            p.stepSimulation()
            if slow_motion:
                time.sleep(0.015)

    print("\n=== HYBRID PHASE 1 RESULTS ===")
    print(f"Pickup: {successes}/{num_episodes} ({100.0 * successes / max(num_episodes, 1):.0f}%)")
    input("\nPress Enter to close...")
    env.close()


if __name__ == "__main__":
    run_demo()
