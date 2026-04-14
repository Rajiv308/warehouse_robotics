"""
Phase 2 Demo — Watch the mobile robot navigate, grasp, and deliver in PyBullet GUI.
Run: python3 src/eval/demo_phase2.py
"""
import sys, os, torch, numpy as np, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pybullet as p
import pybullet_data
from src.training.train_state_rl_phase2 import MobileStatePolicy, get_state, step_env, compute_reward


def run_demo(num_episodes=5, slow_motion=True):
    print("Loading Phase 2 trained policy...", flush=True)
    policy = MobileStatePolicy(state_dim=25, action_dim=10)
    ckpt_path = "checkpoints/phase2_state_policy_wheels.pth"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: {ckpt_path} not found!")
        print("Copy it from cloud: scp -P 10123 root@ssh8.vast.ai:/workspace/warehouse_robotics/checkpoints/phase2_state_policy_wheels.pth checkpoints/")
        return
    policy.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    policy.eval()
    print("Policy loaded!", flush=True)

    # Setup environment with GUI
    cid = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(0.01)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    # Nice camera
    p.resetDebugVisualizerCamera(
        cameraDistance=4.0,
        cameraYaw=60,
        cameraPitch=-30,
        cameraTargetPosition=[1.0, 0.5, 0.3]
    )

    husky = p.loadURDF("husky/husky.urdf", [0, 0, 0.15], useFixedBase=False)
    panda = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.65], useFixedBase=False)
    constraint = p.createConstraint(husky, -1, panda, -1, p.JOINT_FIXED,
                                     [0,0,0], [0,0,0.5], [0,0,0])
    p.changeConstraint(constraint, maxForce=10000)

    wheel_joints = []
    for j in range(p.getNumJoints(husky)):
        info = p.getJointInfo(husky, j)
        if 'wheel' in info[1].decode('utf-8').lower():
            wheel_joints.append(j)

    # Shelf
    shelf_pos = [2.5, 0.0]
    sc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.3, 0.02])
    sv = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.3, 0.02],
                              rgbaColor=[0.5, 0.35, 0.1, 1])
    p.createMultiBody(0, sc, sv, [shelf_pos[0], shelf_pos[1], 0.5])
    for lx, ly in [(-0.5,-0.25),(0.5,-0.25),(-0.5,0.25),(0.5,0.25)]:
        lc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03,0.03,0.25])
        lv = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03,0.03,0.25],
                                  rgbaColor=[0.3,0.3,0.3,1])
        p.createMultiBody(0, lc, lv, [shelf_pos[0]+lx, shelf_pos[1]+ly, 0.25])

    # Box
    bc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.04,0.04,0.04])
    bv = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.04,0.04,0.04],
                              rgbaColor=[1,0,0,1])
    box_id = p.createMultiBody(0.1, bc, bv, [shelf_pos[0], shelf_pos[1], 0.58])

    # Dropoff
    dc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3,0.3,0.01])
    dv = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3,0.3,0.01],
                              rgbaColor=[0,0.8,0,0.5])
    dropoff_pos = [0.0, 2.0]
    p.createMultiBody(0, dc, dv, [dropoff_pos[0], dropoff_pos[1], 0.01])

    # Add text labels
    p.addUserDebugText("SHELF", [shelf_pos[0], shelf_pos[1], 0.8],
                        textColorRGB=[1,1,1], textSize=1.5)
    p.addUserDebugText("DROPOFF", [dropoff_pos[0], dropoff_pos[1], 0.3],
                        textColorRGB=[0,1,0], textSize=1.5)

    for ep in range(num_episodes):
        print(f"\n--- Episode {ep+1}/{num_episodes} ---")

        # Reset
        yaw = np.random.uniform(-0.3, 0.3)
        p.resetBasePositionAndOrientation(husky, [0,0,0.15],
                                           p.getQuaternionFromEuler([0,0,yaw]))
        p.resetBaseVelocity(husky, [0,0,0], [0,0,0])
        p.resetBasePositionAndOrientation(panda, [0,0,0.65],
                                           p.getQuaternionFromEuler([0,0,yaw]))
        home = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        for i, pos in enumerate(home):
            p.resetJointState(panda, i, pos)
        noise = np.random.uniform(-0.1, 0.1, 2)
        p.resetBasePositionAndOrientation(box_id,
            [shelf_pos[0]+noise[0], shelf_pos[1]+noise[1], 0.58], [0,0,0,1])
        for _ in range(10):
            p.stepSimulation()

        state = get_state(husky, panda, box_id, shelf_pos, dropoff_pos)

        for step in range(500):
            st = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = policy(st)
            a_np = action.squeeze().numpy()

            step_env(a_np, husky, panda, wheel_joints)
            reward, ds, do, dd = compute_reward(husky, panda, box_id,
                                                 shelf_pos, dropoff_pos)
            state = get_state(husky, panda, box_id, shelf_pos, dropoff_pos)

            # Follow robot with camera
            base_pos, _ = p.getBasePositionAndOrientation(husky)
            p.resetDebugVisualizerCamera(
                cameraDistance=3.0,
                cameraYaw=60,
                cameraPitch=-25,
                cameraTargetPosition=[base_pos[0], base_pos[1], 0.5]
            )

            if slow_motion:
                time.sleep(0.02)

            if step % 100 == 0:
                print(f"  Step {step}: d_shelf={ds:.2f} d_obj={do:.2f} d_drop={dd:.2f}")

        obj_pos = np.array(p.getBasePositionAndOrientation(box_id)[0])
        print(f"  Final: d_shelf={ds:.2f} d_obj={do:.2f} d_drop={dd:.2f} obj_z={obj_pos[2]:.2f}")

    input("\nPress Enter to close...")
    p.disconnect()


if __name__ == "__main__":
    run_demo(num_episodes=5, slow_motion=True)
