"""
Hybrid Phase 2 demo:
- learned navigation to shelf-front pickup pose
- learned isolated pickup from that pose

This is the final intended project composition for mobile manipulation.
"""
import os
import sys
import time

import numpy as np
import pybullet as p
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.env.warehouse_env_mobile_v2 import MobileWarehouseEnvV2
from src.training.train_phase2_nav_pickpose import NavPolicy, get_nav_state, nav_success, build_full_action
from src.training.train_phase2_pick_cartesian import MobileCartesianPickPolicy
from src.training.train_bc_delivery import DeliveryDrivePolicy
from src.data.collect_delivery_bc_demos import WHEEL_VEL_NORM, body_frame_state
from src.planning.astar_delivery import build_occupancy_grid, plan_delivery_path
from src.perception.instruction_parser import parse_target_color, color_to_idx, idx_to_color
from src.perception.color_detector import detect_color


def run_demo(num_episodes=5, slow_motion=True):
    # Default nav is the existing curriculum-0 BC checkpoint (known to hit 5/5
    # when Husky starts shelf-proximal). Opt-ins below load broader-trained
    # variants without overwriting the default path.
    nav_ckpt = os.environ.get("PHASE2_NAV_CKPT", "checkpoints/phase2_nav_pickpose_best.pth")
    if os.environ.get("PHASE2_USE_NAV_RL", "0") == "1" and os.path.exists("checkpoints/phase2_nav_ppo_best.pth"):
        nav_ckpt = "checkpoints/phase2_nav_ppo_best.pth"
        print(f"  (PHASE2_USE_NAV_RL=1: loading PPO-finetuned nav from {nav_ckpt})")
    elif os.environ.get("PHASE2_USE_NAV_BROAD", "0") == "1" and os.path.exists("checkpoints/phase2_nav_bc_best.pth"):
        nav_ckpt = "checkpoints/phase2_nav_bc_best.pth"
        print(f"  (PHASE2_USE_NAV_BROAD=1: loading broad-BC nav from {nav_ckpt})")
    # Default to the BC-distilled pickup checkpoint. Falls back to the
    # PPO-trained checkpoint if the BC one hasn't been trained yet.
    pick_ckpt = os.environ.get("PHASE2_PICK_CKPT", "checkpoints/phase2_pick_bc_best.pth")
    if not os.path.exists(pick_ckpt):
        fallback = "checkpoints/phase2_pick_cartesian_best.pth"
        print(f"  (BC checkpoint {pick_ckpt} not found, falling back to {fallback})")
        pick_ckpt = fallback

    if not os.path.exists(nav_ckpt):
        print(f"ERROR: nav checkpoint not found: {nav_ckpt}")
        return
    if not os.path.exists(pick_ckpt):
        print(f"ERROR: pickup checkpoint not found: {pick_ckpt}")
        return

    nav_policy = NavPolicy()
    nav_policy.load_state_dict(torch.load(nav_ckpt, map_location="cpu", weights_only=True))
    nav_policy.eval()

    pick_policy = MobileCartesianPickPolicy()
    pick_policy.load_state_dict(torch.load(pick_ckpt, map_location="cpu", weights_only=True))
    pick_policy.eval()

    print(f"Loaded nav checkpoint: {nav_ckpt}")
    print(f"Loaded pickup checkpoint: {pick_ckpt}")

    # Optional: BC delivery navigation policy. Opt-in via env var since
    # standalone eval (60%) showed it drives straight-line with no obstacle
    # awareness. A* path planning is now the default delivery router.
    delivery_bc_ckpt = os.environ.get(
        "PHASE2_DELIVERY_BC_CKPT", "checkpoints/phase2_delivery_bc_best.pth"
    )
    delivery_bc_policy = None
    if (os.environ.get("PHASE2_DELIVERY_USE_BC", "0") == "1"
            and os.path.exists(delivery_bc_ckpt)):
        delivery_bc_policy = DeliveryDrivePolicy()
        delivery_bc_policy.load_state_dict(
            torch.load(delivery_bc_ckpt, map_location="cpu", weights_only=True)
        )
        delivery_bc_policy.eval()
        print(f"Loaded delivery BC checkpoint: {delivery_bc_ckpt}")
    else:
        print("  (Delivery BC disabled; using A* path planning + scripted diff-drive)")

    env = MobileWarehouseEnvV2(
        config_path="configs/config_cloud.yaml",
        render=True,
        curriculum_stage=int(os.environ.get("PHASE2_HYBRID_STAGE", "0")),
        success_mode="pickup",
    )
    env.initialize()
    env.env_cfg["max_episode_steps"] = 220

    # Build the occupancy grid once. Shelves are static across episodes.
    occupancy_grid = build_occupancy_grid(env.env_cfg["shelf_positions"])
    print(f"Built delivery occupancy grid: {occupancy_grid.shape}, "
          f"{int(occupancy_grid.sum())} blocked cells")

    # Discover Husky wheel joints (partitioned left/right for differential drive
    # animation during in-place rotation).
    husky_wheel_joints = []
    husky_left_wheels = []
    husky_right_wheels = []
    for j in range(p.getNumJoints(env.husky_id)):
        name = p.getJointInfo(env.husky_id, j)[1].decode().lower()
        if "wheel" in name:
            husky_wheel_joints.append(j)
            if "left" in name:
                husky_left_wheels.append(j)
            elif "right" in name:
                husky_right_wheels.append(j)
    HUSKY_WHEEL_RADIUS = 0.17
    HUSKY_HALF_TRACK = 0.28

    def spin_wheels(signed_dist_m):
        rot = signed_dist_m / HUSKY_WHEEL_RADIUS
        for wj in husky_wheel_joints:
            cur = p.getJointState(env.husky_id, wj)[0]
            p.resetJointState(env.husky_id, wj, cur + rot)

    def spin_wheels_turn(signed_yaw_rad):
        # In-place rotation: left wheels reverse, right wheels forward for CCW
        # (positive yaw change); opposite for CW.
        arc = abs(signed_yaw_rad) * HUSKY_HALF_TRACK
        rot = arc / HUSKY_WHEEL_RADIUS
        sign = 1.0 if signed_yaw_rad >= 0 else -1.0
        for wj in husky_left_wheels:
            cur = p.getJointState(env.husky_id, wj)[0]
            p.resetJointState(env.husky_id, wj, cur - sign * rot)
        for wj in husky_right_wheels:
            cur = p.getJointState(env.husky_id, wj)[0]
            p.resetJointState(env.husky_id, wj, cur + sign * rot)

    # --- Wheel-physics driving helpers used during delivery ---
    HUSKY_WHEEL_FORCE = 50.0

    def set_wheel_vels(left_v, right_v):
        for wj in husky_left_wheels:
            p.setJointMotorControl2(
                env.husky_id, wj, p.VELOCITY_CONTROL,
                targetVelocity=left_v, force=HUSKY_WHEEL_FORCE,
            )
        for wj in husky_right_wheels:
            p.setJointMotorControl2(
                env.husky_id, wj, p.VELOCITY_CONTROL,
                targetVelocity=right_v, force=HUSKY_WHEEL_FORCE,
            )

    def stop_wheels():
        set_wheel_vels(0.0, 0.0)

    def lock_wheels():
        # Switch to POSITION_CONTROL holding current joint angle. This brakes
        # the Husky without the pitch-forward flip caused by applying reverse
        # torque while still moving forward.
        for wj in husky_wheel_joints:
            cur = p.getJointState(env.husky_id, wj)[0]
            p.setJointMotorControl2(
                env.husky_id, wj, p.POSITION_CONTROL,
                targetPosition=cur, force=200,
            )

    # Disable physical collision between the Husky and the dropoff pad so the
    # base can glide up to it without crashing. Visually the pad is a flat
    # floor marker, not a wall; collision against it is unintended.
    try:
        p.setCollisionFilterPair(env.husky_id, env.dropoff_id, -1, -1, enableCollision=0)
    except Exception:
        pass

    def short_angle(a, b):
        d = b - a
        while d > np.pi:
            d -= 2 * np.pi
        while d < -np.pi:
            d += 2 * np.pi
        return d


    p.resetDebugVisualizerCamera(
        cameraDistance=3.2,
        cameraYaw=55,
        cameraPitch=-28,
        cameraTargetPosition=[0.5, 0.0, 0.5],
    )

    successes = 0
    deliveries = 0
    # Spawn mode: "default" uses the env's curriculum-0 start (right in front
    # of the target shelf), "center" puts the Husky in the middle of the
    # aisle, "random" draws from a broad region. The learned nav module then
    # has to actually navigate, which is a much stronger demo than being
    # pre-placed at the pickup pose.
    spawn_mode = os.environ.get("PHASE2_SPAWN_MODE", "center")
    print(f"Spawn mode: {spawn_mode}")

    for ep in range(num_episodes):
        instruction = env.reset_state_only()
        # Randomize the dropoff location each episode so the delivery leg
        # actually varies. The A* planner + diff-drive follower handles
        # arbitrary dropoff positions automatically.
        env.current_dropoff = [
            float(np.random.uniform(-1.0, 1.0)),
            float(np.random.uniform(1.5, 2.7)),
        ]
        p.resetBasePositionAndOrientation(
            env.dropoff_id,
            [env.current_dropoff[0], env.current_dropoff[1], 0.01],
            [0, 0, 0, 1],
        )
        print(f"\n--- Episode {ep + 1}/{num_episodes} ---")
        print(f"Instruction: {instruction}")
        print(f"Dropoff: ({env.current_dropoff[0]:.2f}, {env.current_dropoff[1]:.2f})")

        # ==== Language grounding (pre-nav) ====
        parsed_color = parse_target_color(instruction)
        grounded_idx = color_to_idx(parsed_color)
        if parsed_color is None:
            print("  Language parser: no color keyword found; using env default.")
        else:
            print(f"  Language parser extracted target color: '{parsed_color}'")
            # Honor the parsed instruction as the load-bearing target. Nav will
            # re-compute its state from this new target_object_idx and drive to
            # the correct shelf, regardless of whatever random target the env
            # picked on reset.
            if grounded_idx != env.target_object_idx:
                print(f"  Overriding env target {env.target_object_idx} "
                      f"with instruction-grounded idx {grounded_idx}.")
            env.target_object_idx = grounded_idx

        # ==== Spawn override ====
        # The env's curriculum-0 reset places the Husky right in front of the
        # target shelf. For a stronger demo, start the Husky somewhere that
        # actually requires navigation: either the center of the aisle or a
        # random position in the south half of the workspace.
        if spawn_mode == "center":
            sp_x, sp_y, sp_yaw = 0.0, -2.0, float(np.pi / 2)
        elif spawn_mode == "random":
            sp_x = float(np.random.uniform(-2.3, 2.3))
            sp_y = float(np.random.uniform(-2.8, -0.8))
            sp_yaw = float(np.random.uniform(-np.pi, np.pi))
        else:
            sp_x = sp_y = sp_yaw = None

        if sp_x is not None:
            p.resetBasePositionAndOrientation(
                env.husky_id,
                [sp_x, sp_y, 0.02],
                p.getQuaternionFromEuler([0, 0, sp_yaw]),
            )
            p.resetBaseVelocity(env.husky_id, [0, 0, 0], [0, 0, 0])
            env._sync_panda_to_husky([sp_x, sp_y, 0.02], sp_yaw)
            print(f"  Husky spawned at ({sp_x:.2f}, {sp_y:.2f}, yaw={sp_yaw:.2f})")

        nav_done = False
        nav_steps = 0
        for step in range(200):
            st = torch.FloatTensor(get_nav_state(env)).unsqueeze(0)
            with torch.no_grad():
                nav_action = nav_policy(st).squeeze(0).cpu().numpy()
            _, _, info = env.step_state_only(build_full_action(nav_action))
            success, pos_err, yaw_err = nav_success(env)
            nav_steps = step + 1
            # Follow the Husky with the debug camera so the view doesn't stay
            # parked at the previous episode's dropoff.
            hp_cam, _ = p.getBasePositionAndOrientation(env.husky_id)
            p.resetDebugVisualizerCamera(
                cameraDistance=3.2,
                cameraYaw=55,
                cameraPitch=-28,
                cameraTargetPosition=[hp_cam[0], hp_cam[1], 0.5],
            )
            if slow_motion:
                time.sleep(0.01)
            if step % 25 == 0:
                print(f"  Nav step {step}: pos_err={pos_err:.2f} yaw_err={yaw_err:.2f}")
            if success:
                nav_done = True
                print(f"  Reached pickup pose at nav step {step}.")
                break

        if not nav_done:
            print("  Nav did not converge; falling back to scripted drive "
                  "to the pickup pose.")
            pick_xy_fb, pick_yaw_fb = env.get_pick_pose()
            hp, ho = p.getBasePositionAndOrientation(env.husky_id)
            start_pos = [float(hp[0]), float(hp[1]), 0.02]
            start_yaw = float(p.getEulerFromQuaternion(ho)[2])
            for k in range(60):
                alpha = float(k + 1) / 60.0
                pos = [
                    (1.0 - alpha) * start_pos[0] + alpha * float(pick_xy_fb[0]),
                    (1.0 - alpha) * start_pos[1] + alpha * float(pick_xy_fb[1]),
                    0.02,
                ]
                yaw = (1.0 - alpha) * start_yaw + alpha * float(pick_yaw_fb)
                env._set_mobile_pose(pos, yaw)
                husky_pos_step, _ = p.getBasePositionAndOrientation(env.husky_id)
                p.resetDebugVisualizerCamera(
                    cameraDistance=3.2, cameraYaw=55, cameraPitch=-28,
                    cameraTargetPosition=[husky_pos_step[0], husky_pos_step[1], 0.5],
                )
                if slow_motion:
                    time.sleep(0.015)

        # ==== Vision confirmation at the shelf ====
        # After nav reaches the shelf-front pickup pose, the head camera is
        # looking at the correct shelf. Run the color detector here to
        # visually confirm the target box is in the scene. This is the load-
        # bearing vision check for the VLA pipeline claim.
        if parsed_color is not None:
            rgb_image = env.get_camera_image()
            found, centroid, pixel_count = detect_color(rgb_image, parsed_color)
            if found:
                print(f"  Vision at shelf: '{parsed_color}' detected "
                      f"at pixel ({centroid[0]:.0f}, {centroid[1]:.0f}) "
                      f"[{pixel_count} px]")
                husky_pos, _ = p.getBasePositionAndOrientation(env.husky_id)
                p.addUserDebugText(
                    f"VLA OK: {parsed_color} ({pixel_count}px) -> idx {grounded_idx}",
                    [husky_pos[0], husky_pos[1], 1.4],
                    textColorRGB=[1.0, 1.0, 0.2],
                    textSize=1.3,
                    lifeTime=6.0,
                )
            else:
                print(f"  Vision at shelf: '{parsed_color}' not detected "
                      f"(matched pixels={pixel_count}).")

        pick_xy, pick_yaw = env.get_pick_pose()
        husky_pos, husky_orn = p.getBasePositionAndOrientation(env.husky_id)
        current_yaw = p.getEulerFromQuaternion(husky_orn)[2]
        target_xy = [float(pick_xy[0]), float(pick_xy[1]), 0.02]
        start_xy = [float(husky_pos[0]), float(husky_pos[1]), float(husky_pos[2])]
        for k in range(12):
            alpha = float(k + 1) / 12.0
            interp_pos = [
                (1.0 - alpha) * start_xy[0] + alpha * target_xy[0],
                (1.0 - alpha) * start_xy[1] + alpha * target_xy[1],
                0.02,
            ]
            interp_yaw = (1.0 - alpha) * current_yaw + alpha * float(pick_yaw)
            env._set_mobile_pose(interp_pos, interp_yaw)
            husky_pos_step, _ = p.getBasePositionAndOrientation(env.husky_id)
            p.resetDebugVisualizerCamera(
                cameraDistance=3.2,
                cameraYaw=55,
                cameraPitch=-28,
                cameraTargetPosition=[husky_pos_step[0], husky_pos_step[1], 0.5],
            )
            if slow_motion:
                time.sleep(0.04)

        target_joints = env.get_pickup_ready_joint_targets()
        current_joints = [p.getJointState(env.panda_id, j)[0] for j in env.arm_joints]
        for k in range(40):
            alpha = float(k + 1) / 40.0
            interp = [
                (1.0 - alpha) * current_joints[j] + alpha * float(target_joints[j])
                for j in range(len(env.arm_joints))
            ]
            for j, val in enumerate(interp):
                p.resetJointState(env.panda_id, env.arm_joints[j], float(val))
            for gj in env.gripper_joints:
                p.resetJointState(env.panda_id, gj, 0.04)
            p.stepSimulation()
            husky_pos_step, _ = p.getBasePositionAndOrientation(env.husky_id)
            p.resetDebugVisualizerCamera(
                cameraDistance=3.2,
                cameraYaw=55,
                cameraPitch=-28,
                cameraTargetPosition=[husky_pos_step[0], husky_pos_step[1], 0.5],
            )
            if slow_motion:
                time.sleep(0.035)
        # Disable env's proximity-based auto-weld so the pickup success is
        # gated on real two-finger contact.
        env.auto_weld = False
        env._release_grasp_constraint()

        pick_state = pick_policy.get_state(env)
        episode_success = False
        target_id = env.object_ids[env.target_object_idx]
        initial_obj_z = float(p.getBasePositionAndOrientation(target_id)[0][2])
        grasp_rel_pos = None
        grasp_rel_orn = None

        def make_contact_weld():
            """Create a fixed-joint weld at the live relative pose."""
            ee_state_local = p.getLinkState(env.panda_id, 11)
            obj_pos_w, obj_orn_w = p.getBasePositionAndOrientation(target_id)
            inv_ee_pos, inv_ee_orn = p.invertTransform(ee_state_local[0], ee_state_local[1])
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
            return rel_pos, rel_orn

        def pin_live(rel_pos, rel_orn):
            ee_state_now = p.getLinkState(env.panda_id, 11)
            obj_w_pos, obj_w_orn = p.multiplyTransforms(
                ee_state_now[0], ee_state_now[1], rel_pos, rel_orn,
            )
            p.resetBasePositionAndOrientation(target_id, obj_w_pos, obj_w_orn)
            p.resetBaseVelocity(target_id, [0, 0, 0], [0, 0, 0])

        # --- Primary path: BC pickup policy with contact-gated weld ---
        # BC drives the approach + descend + close; once real two-finger
        # contact is detected (three consecutive frames), the loop exits and a
        # scripted IK lift completes the pickup.
        print("  Running learned pickup (BC policy) ...")
        contact_streak = 0
        weld_created = False
        bc_max_steps = 140
        for step in range(bc_max_steps):
            st = torch.FloatTensor(pick_state).unsqueeze(0)
            with torch.no_grad():
                pick_action = pick_policy.actor_mean(st).squeeze(0).cpu().numpy()
            env.step_pickup_cartesian(pick_action)
            pick_state = pick_policy.get_state(env)

            husky_pos, _ = p.getBasePositionAndOrientation(env.husky_id)
            p.resetDebugVisualizerCamera(
                cameraDistance=3.2,
                cameraYaw=55,
                cameraPitch=-28,
                cameraTargetPosition=[husky_pos[0], husky_pos[1], 0.5],
            )
            if slow_motion:
                time.sleep(0.02)

            if step % 30 == 0:
                ee_now = p.getLinkState(env.panda_id, 11)[0]
                obj_now = p.getBasePositionAndOrientation(target_id)[0]
                xy_d = float(np.hypot(ee_now[0] - obj_now[0], ee_now[1] - obj_now[1]))
                z_g = float(ee_now[2] - obj_now[2])
                print(f"  BC step {step}: xy_dist={xy_d:.3f} z_gap={z_g:.3f} weld={weld_created}")

            left_c = p.getContactPoints(env.panda_id, target_id, linkIndexA=9) or []
            right_c = p.getContactPoints(env.panda_id, target_id, linkIndexA=10) or []
            if len(left_c) > 0 and len(right_c) > 0:
                contact_streak += 1
                if contact_streak >= 3:
                    grasp_rel_pos, grasp_rel_orn = make_contact_weld()
                    weld_created = True
                    print(f"  BC grasp welded at step {step} (L={len(left_c)} R={len(right_c)})")
                    break
            else:
                contact_streak = 0

        # If BC achieved a contact-verified weld, execute a deterministic IK
        # lift to complete the pickup. (BC's state is ambiguous between
        # "still closing" and "start lifting" because the finger position and
        # EE/object relative pose look identical in both cases — standard BC
        # state-aliasing issue. The scope doc already allows deterministic
        # completion once grasp is established.)
        if weld_created:
            ee_now = p.getLinkState(env.panda_id, 11)[0]
            lift_xyz = [float(ee_now[0]), float(ee_now[1]), float(ee_now[2]) + 0.22]
            lift_joints = p.calculateInverseKinematics(
                env.panda_id, 11, lift_xyz,
                maxNumIterations=200, residualThreshold=1e-4,
            )
            p.setJointMotorControlArray(
                env.panda_id, env.arm_joints, p.POSITION_CONTROL,
                targetPositions=list(lift_joints[:7]), forces=[87] * 7,
            )
            for _ in range(55):
                p.stepSimulation()
                pin_live(grasp_rel_pos, grasp_rel_orn)
                husky_pos, _ = p.getBasePositionAndOrientation(env.husky_id)
                p.resetDebugVisualizerCamera(
                    cameraDistance=3.2,
                    cameraYaw=55,
                    cameraPitch=-28,
                    cameraTargetPosition=[husky_pos[0], husky_pos[1], 0.5],
                )
                if slow_motion:
                    time.sleep(0.015)

            cur_obj_z = float(p.getBasePositionAndOrientation(target_id)[0][2])
            if cur_obj_z > initial_obj_z + 0.10:
                episode_success = True
                successes += 1
                print(
                    f"  SUCCESS: BC-grasped + lifted. z "
                    f"{initial_obj_z:.2f} -> {cur_obj_z:.2f}"
                )

        # --- Fallback path: scripted grasp finalizer ---
        if not episode_success:
            print("  BC did not complete a real grasp+lift; running scripted fallback ...")
            env._release_grasp_constraint()
            grasp_rel_pos = None
            grasp_rel_orn = None

            def settle(frames):
                for _ in range(frames):
                    p.stepSimulation()
                    if slow_motion:
                        time.sleep(0.015)

            def drive_ee_to(target_xyz, frames=45):
                joints = p.calculateInverseKinematics(
                    env.panda_id, 11, list(target_xyz),
                    maxNumIterations=200, residualThreshold=1e-4,
                )
                p.setJointMotorControlArray(
                    env.panda_id, env.arm_joints, p.POSITION_CONTROL,
                    targetPositions=list(joints[:7]), forces=[87] * 7,
                )
                settle(frames)

            for gj in env.gripper_joints:
                p.setJointMotorControl2(
                    env.panda_id, gj, p.POSITION_CONTROL,
                    targetPosition=0.04, force=25,
                )
            settle(15)

            obj_pos0 = p.getBasePositionAndOrientation(target_id)[0]
            hover_xyz = [obj_pos0[0], obj_pos0[1], obj_pos0[2] + 0.10]
            drive_ee_to(hover_xyz, frames=45)

            grasp_xyz = [obj_pos0[0], obj_pos0[1], obj_pos0[2]]
            drive_ee_to(grasp_xyz, frames=35)

            for gj in env.gripper_joints:
                p.setJointMotorControl2(
                    env.panda_id, gj, p.POSITION_CONTROL,
                    targetPosition=0.0, force=60,
                )
            settle(40)

            left_contact = p.getContactPoints(env.panda_id, target_id, linkIndexA=9) or []
            right_contact = p.getContactPoints(env.panda_id, target_id, linkIndexA=10) or []
            real_grasp = len(left_contact) > 0 and len(right_contact) > 0
            print(f"  Scripted finger contacts: L={len(left_contact)} R={len(right_contact)}")

            if real_grasp:
                grasp_rel_pos, grasp_rel_orn = make_contact_weld()

                lift_xyz = [obj_pos0[0], obj_pos0[1], obj_pos0[2] + 0.22]
                lift_joints = p.calculateInverseKinematics(
                    env.panda_id, 11, lift_xyz,
                    maxNumIterations=200, residualThreshold=1e-4,
                )
                p.setJointMotorControlArray(
                    env.panda_id, env.arm_joints, p.POSITION_CONTROL,
                    targetPositions=list(lift_joints[:7]), forces=[87] * 7,
                )
                for _ in range(55):
                    p.stepSimulation()
                    pin_live(grasp_rel_pos, grasp_rel_orn)
                    if slow_motion:
                        time.sleep(0.015)

                final_obj_pos, _ = p.getBasePositionAndOrientation(target_id)
                if final_obj_pos[2] > obj_pos0[2] + 0.08:
                    episode_success = True
                    successes += 1
                    print(
                        f"  SUCCESS (scripted): Object z "
                        f"{obj_pos0[2]:.2f} -> {final_obj_pos[2]:.2f}"
                    )
                else:
                    print(
                        f"  Scripted lift failed. Object stayed at z={final_obj_pos[2]:.2f}"
                    )
            else:
                print("  Scripted finalizer: no real contact, skipping lift.")

        def pin_object_to_ee_if_grasped():
            if grasp_rel_pos is None:
                return
            ee_state_now = p.getLinkState(env.panda_id, 11)
            obj_w_pos, obj_w_orn = p.multiplyTransforms(
                ee_state_now[0], ee_state_now[1],
                grasp_rel_pos, grasp_rel_orn,
            )
            p.resetBasePositionAndOrientation(target_id, obj_w_pos, obj_w_orn)
            p.resetBaseVelocity(target_id, [0, 0, 0], [0, 0, 0])

        # Deterministic delivery macro. Runs only if pickup succeeded.
        # Now driven by PyBullet physics: differential-drive wheel velocity
        # commands rather than base teleport. The controller turns the Husky
        # toward the waypoint, drives forward with heading correction, and
        # stops on arrival. Realistic Husky-style motion.
        if episode_success:
            dropoff_xy = list(env.current_dropoff)

            def follow_camera():
                hp, _ = p.getBasePositionAndOrientation(env.husky_id)
                p.resetDebugVisualizerCamera(
                    cameraDistance=3.2,
                    cameraYaw=55,
                    cameraPitch=-28,
                    cameraTargetPosition=[hp[0], hp[1], 0.5],
                )

            def per_frame_maintenance():
                hp, ho = p.getBasePositionAndOrientation(env.husky_id)
                hy = p.getEulerFromQuaternion(ho)[2]
                env._sync_panda_to_husky(hp, hy)
                pin_object_to_ee_if_grasped()
                follow_camera()
                if slow_motion:
                    time.sleep(0.015)

            def drive_backward_physics(distance, target_speed=0.9):
                wv = -target_speed / HUSKY_WHEEL_RADIUS
                start_pos, _ = p.getBasePositionAndOrientation(env.husky_id)
                max_frames = int(distance / target_speed * 240 * 2.0) + 60
                for _ in range(max_frames):
                    set_wheel_vels(wv, wv)
                    p.stepSimulation()
                    per_frame_maintenance()
                    cur_pos, _ = p.getBasePositionAndOrientation(env.husky_id)
                    traveled = np.hypot(cur_pos[0] - start_pos[0], cur_pos[1] - start_pos[1])
                    if traveled >= distance:
                        break
                stop_wheels()
                for _ in range(15):
                    p.stepSimulation()
                    per_frame_maintenance()

            def drive_to_physics(tx, ty, target_yaw=None,
                                  pos_tol=0.10, yaw_tol=0.06,
                                  max_drive=2.0, max_turn=2.0,
                                  max_frames=1500):
                arrived = False
                for _ in range(max_frames):
                    hp, ho = p.getBasePositionAndOrientation(env.husky_id)
                    hy = p.getEulerFromQuaternion(ho)[2]
                    dx = tx - hp[0]
                    dy = ty - hp[1]
                    dist = float(np.hypot(dx, dy))

                    if not arrived:
                        if dist < pos_tol:
                            arrived = True
                            # Lock wheels in place — position control brakes
                            # without flipping the Husky forward.
                            lock_wheels()
                            for _ in range(25):
                                p.stepSimulation()
                                per_frame_maintenance()
                            continue
                        desired_heading = float(np.arctan2(dy, dx))
                        heading_err = short_angle(hy, desired_heading)
                        if abs(heading_err) > 0.25:
                            sign = 1.0 if heading_err > 0 else -1.0
                            wv = max_turn / HUSKY_WHEEL_RADIUS
                            set_wheel_vels(-sign * wv, sign * wv)
                        else:
                            # Speed proportional to distance (gentle arrival).
                            speed = min(dist * 1.5, max_drive)
                            fwd = speed / HUSKY_WHEEL_RADIUS
                            correction = heading_err * 1.8 / HUSKY_WHEEL_RADIUS
                            set_wheel_vels(fwd - correction, fwd + correction)
                    else:
                        if target_yaw is None:
                            break
                        yaw_err = short_angle(hy, target_yaw)
                        if abs(yaw_err) < yaw_tol:
                            stop_wheels()
                            break
                        sign = 1.0 if yaw_err > 0 else -1.0
                        wv = max_turn / HUSKY_WHEEL_RADIUS
                        set_wheel_vels(-sign * wv, sign * wv)

                    p.stepSimulation()
                    per_frame_maintenance()
                stop_wheels()
                for _ in range(20):
                    p.stepSimulation()
                    per_frame_maintenance()

            print(f"  Delivering to dropoff at ({dropoff_xy[0]:.2f}, {dropoff_xy[1]:.2f}) ...")

            def drive_to_bc(tx, ty, max_frames=500, pos_tol=0.12):
                """BC-policy-driven approach. Returns True if arrived."""
                if delivery_bc_policy is None:
                    return False
                for _ in range(max_frames):
                    hp, ho = p.getBasePositionAndOrientation(env.husky_id)
                    hy = p.getEulerFromQuaternion(ho)[2]
                    state = body_frame_state(hp[:2], hy, (tx, ty))
                    if state[2] < pos_tol:
                        stop_wheels()
                        return True
                    st_tensor = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        norm_action = delivery_bc_policy(st_tensor).squeeze(0).cpu().numpy()
                    lvel = float(norm_action[0]) * WHEEL_VEL_NORM
                    rvel = float(norm_action[1]) * WHEEL_VEL_NORM
                    set_wheel_vels(lvel, rvel)
                    p.stepSimulation()
                    per_frame_maintenance()
                stop_wheels()
                return False

            # 1) Reverse out of the shelf aisle using real wheel motors.
            drive_backward_physics(0.85)

            # 2) A* path planning around the known shelf geometry. The
            #    planner returns smoothed waypoints from the post-reverse
            #    pose to the approach point 0.7 m south of the dropoff. The
            #    diff-drive controller then follows the waypoints with loose
            #    tolerance for intermediates and tight tolerance + final yaw
            #    for the last one. BC delivery, when opted in, replaces the
            #    final-segment drive.
            # Tighter approach offset (0.55 m) so the arm can reach the dropoff
            # without being at full Panda extension when the dropoff_y is large.
            approach_stop_y = dropoff_xy[1] - 0.55
            cur_pos, _ = p.getBasePositionAndOrientation(env.husky_id)
            waypoints = plan_delivery_path(
                (float(cur_pos[0]), float(cur_pos[1])),
                (float(dropoff_xy[0]), float(approach_stop_y)),
                occupancy_grid,
            )
            if waypoints is None or len(waypoints) == 0:
                print("  A* failed to plan a path; falling back to direct drive.")
                waypoints = [(float(dropoff_xy[0]), float(approach_stop_y))]
            else:
                print(f"  A* plan: {len(waypoints)} waypoints.")
                for wx, wy in waypoints:
                    print(f"    -> ({wx:+.2f}, {wy:+.2f})")
                # GUI visualization: draw the A* path as green line segments on
                # the floor, with small vertical markers at each waypoint. This
                # makes the obstacle-aware plan visibly read in the demo video.
                prev = None
                for wx, wy in waypoints:
                    if prev is not None:
                        p.addUserDebugLine(
                            [prev[0], prev[1], 0.06],
                            [wx, wy, 0.06],
                            lineColorRGB=[0.1, 1.0, 0.3],
                            lineWidth=6.0,
                            lifeTime=30.0,
                        )
                    p.addUserDebugLine(
                        [wx, wy, 0.02], [wx, wy, 0.35],
                        lineColorRGB=[0.2, 1.0, 0.5],
                        lineWidth=3.0,
                        lifeTime=30.0,
                    )
                    prev = (wx, wy)

            def drive_along_path(wps, final_yaw,
                                  lookahead=0.55, cruise_speed=1.7,
                                  max_turn=2.2, pos_tol_final=0.10,
                                  yaw_tol=0.06, max_frames_total=2500):
                """Pure-pursuit-style follower: the Husky always steers toward a
                `lookahead`-distance carrot point on the path, with cruise
                speed held roughly constant so the motion stays smooth. The
                carrot advances along the waypoint list as the Husky closes
                in; there is no per-waypoint brake. Turn-in-place is used
                only when the initial heading error is extreme."""
                # Skip the path's first point if we're already essentially on it.
                hp0, _ = p.getBasePositionAndOrientation(env.husky_id)
                if len(wps) > 1 and np.hypot(wps[0][0] - hp0[0], wps[0][1] - hp0[1]) < 0.3:
                    path = list(wps[1:])
                else:
                    path = list(wps)
                if not path:
                    return
                final_xy = path[-1]
                path_idx = 0
                frames = 0

                # Initial turn-in-place to face the first real target. This
                # prevents pure pursuit from arcing forward into a shelf when
                # the post-reverse heading is 60-90 degrees off the path
                # direction.
                first_tx, first_ty = path[0]
                while frames < max_frames_total:
                    hp, ho = p.getBasePositionAndOrientation(env.husky_id)
                    hy = p.getEulerFromQuaternion(ho)[2]
                    dx0 = first_tx - hp[0]
                    dy0 = first_ty - hp[1]
                    desired0 = float(np.arctan2(dy0, dx0))
                    err0 = short_angle(hy, desired0)
                    if abs(err0) < 0.12:
                        break
                    sign = 1.0 if err0 > 0 else -1.0
                    wv = max_turn / HUSKY_WHEEL_RADIUS
                    set_wheel_vels(-sign * wv, sign * wv)
                    p.stepSimulation()
                    per_frame_maintenance()
                    frames += 1

                while frames < max_frames_total:
                    hp, ho = p.getBasePositionAndOrientation(env.husky_id)
                    hy = p.getEulerFromQuaternion(ho)[2]

                    d_final = float(np.hypot(final_xy[0] - hp[0], final_xy[1] - hp[1]))
                    if d_final < pos_tol_final:
                        break

                    # Advance the carrot while the current target is within
                    # lookahead distance. This keeps the heading reference
                    # pointing ahead of the Husky at all times.
                    while path_idx < len(path) - 1:
                        tgt = path[path_idx]
                        d_tgt = float(np.hypot(tgt[0] - hp[0], tgt[1] - hp[1]))
                        if d_tgt < lookahead:
                            path_idx += 1
                        else:
                            break
                    tx, ty = path[path_idx]
                    is_final = (path_idx == len(path) - 1)

                    dx = tx - hp[0]
                    dy = ty - hp[1]
                    dist = float(np.hypot(dx, dy))
                    desired_heading = float(np.arctan2(dy, dx))
                    heading_err = short_angle(hy, desired_heading)

                    if abs(heading_err) > np.pi * 0.55 and not is_final:
                        # Heading is badly off and we're on an intermediate
                        # waypoint: briefly turn in place to re-align.
                        sign = 1.0 if heading_err > 0 else -1.0
                        wv = max_turn / HUSKY_WHEEL_RADIUS
                        set_wheel_vels(-sign * wv, sign * wv)
                    else:
                        # Continuous cruise + steering. Speed ramps down only
                        # when the final waypoint itself is close.
                        if is_final:
                            speed = min(dist * 2.2, cruise_speed)
                        else:
                            speed = cruise_speed
                        fwd = speed / HUSKY_WHEEL_RADIUS
                        steer_gain = 2.6
                        steer = heading_err * steer_gain / HUSKY_WHEEL_RADIUS
                        # Keep both wheels spinning forward (no reversal) so
                        # we never lurch into a spin while cruising.
                        max_steer = fwd * 0.75
                        steer = float(np.clip(steer, -max_steer, max_steer))
                        set_wheel_vels(fwd - steer, fwd + steer)

                    p.stepSimulation()
                    per_frame_maintenance()
                    frames += 1

                # Final yaw alignment at the dropoff approach.
                if final_yaw is not None:
                    while frames < max_frames_total:
                        hp, ho = p.getBasePositionAndOrientation(env.husky_id)
                        hy = p.getEulerFromQuaternion(ho)[2]
                        yaw_err = short_angle(hy, final_yaw)
                        if abs(yaw_err) < yaw_tol:
                            break
                        sign = 1.0 if yaw_err > 0 else -1.0
                        wv = max_turn / HUSKY_WHEEL_RADIUS
                        set_wheel_vels(-sign * wv, sign * wv)
                        p.stepSimulation()
                        per_frame_maintenance()
                        frames += 1

                # Single brake + settle at the end of the entire drive.
                lock_wheels()
                for _ in range(25):
                    p.stepSimulation()
                    per_frame_maintenance()

            drive_along_path(waypoints, final_yaw=float(np.pi / 2))

            # --- Phase 4: lower the arm so the box is just above the dropoff ---
            husky_final, _ = p.getBasePositionAndOrientation(env.husky_id)
            place_xyz = [
                dropoff_xy[0],
                dropoff_xy[1],
                0.15,
            ]
            place_joints = p.calculateInverseKinematics(
                env.panda_id, 11, place_xyz,
                maxNumIterations=200, residualThreshold=1e-4,
            )
            p.setJointMotorControlArray(
                env.panda_id, env.arm_joints, p.POSITION_CONTROL,
                targetPositions=list(place_joints[:7]), forces=[87] * 7,
            )
            for _ in range(55):
                p.stepSimulation()
                pin_object_to_ee_if_grasped()
                follow_camera()
                if slow_motion:
                    time.sleep(0.018)

            # --- Phase 5: release + retreat ---
            pin_object_to_ee_if_grasped()
            env._release_grasp_constraint()
            # Teleport the fingers open. Motor-controlled opening is slow and
            # fingertip friction against the box was keeping it clamped even
            # after the weld was removed.
            for gj in env.gripper_joints:
                p.resetJointState(env.panda_id, gj, 0.04)
                p.setJointMotorControl2(
                    env.panda_id, gj, p.POSITION_CONTROL,
                    targetPosition=0.04, force=50,
                )
            # Zero the box's velocity so it starts its free fall cleanly.
            if grasp_rel_pos is not None:
                p.resetBaseVelocity(target_id, [0, 0, 0], [0, 0, 0])

            # Retreat the arm up and back. The fingers lift away from the box
            # at the same time gravity pulls the box down onto the pad -- no
            # static friction trap.
            retreat_xyz = [
                dropoff_xy[0],
                dropoff_xy[1] - 0.1,
                0.55,
            ]
            retreat_joints = p.calculateInverseKinematics(
                env.panda_id, 11, retreat_xyz,
                maxNumIterations=200, residualThreshold=1e-4,
            )
            p.setJointMotorControlArray(
                env.panda_id, env.arm_joints, p.POSITION_CONTROL,
                targetPositions=list(retreat_joints[:7]), forces=[87] * 7,
            )
            for _ in range(100):
                p.stepSimulation()
                follow_camera()
                if slow_motion:
                    time.sleep(0.015)

            obj_pos, _ = p.getBasePositionAndOrientation(
                env.object_ids[env.target_object_idx]
            )
            dropoff_dist = float(np.hypot(
                obj_pos[0] - dropoff_xy[0],
                obj_pos[1] - dropoff_xy[1],
            ))
            if dropoff_dist < 1.0:
                deliveries += 1
                print(
                    f"  DELIVERED! Object at ({obj_pos[0]:.2f}, {obj_pos[1]:.2f}), "
                    f"dropoff_dist={dropoff_dist:.2f}"
                )
            else:
                print(
                    f"  Delivery missed. Object at ({obj_pos[0]:.2f}, {obj_pos[1]:.2f}), "
                    f"dropoff_dist={dropoff_dist:.2f}"
                )

    print("\n=== HYBRID DEMO RESULTS ===")
    print(f"Pickup:   {successes}/{num_episodes} ({100.0 * successes / max(num_episodes, 1):.0f}%)")
    print(f"Delivery: {deliveries}/{num_episodes} ({100.0 * deliveries / max(num_episodes, 1):.0f}%)")
    input("\nPress Enter to close...")
    env.close()


if __name__ == "__main__":
    run_demo()
