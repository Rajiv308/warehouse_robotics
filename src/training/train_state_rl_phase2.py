"""
Phase 2 State-based RL with REAL WHEEL PHYSICS.
Full mobile manipulation: navigate to shelf → reach object → grasp → deliver to dropoff.
No image rendering — trains in minutes, not hours.
"""
import sys, os, torch, torch.nn as nn, numpy as np, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pybullet as p
import pybullet_data
from collections import deque


class MobileStatePolicy(nn.Module):
    def __init__(self, state_dim=25, action_dim=10):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action=None):
        mean = self.actor_mean(state)
        std = self.actor_log_std.exp().clamp(min=0.05)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        return (action, dist.log_prob(action).sum(-1),
                dist.entropy().sum(-1), self.critic(state).squeeze(-1))


def setup_env():
    """Build the full warehouse environment with wheel physics."""
    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(0.01)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

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

    # Target box
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

    return husky, panda, wheel_joints, box_id, shelf_pos, dropoff_pos


def get_state(husky, panda, box_id, shelf_pos, dropoff_pos):
    base_pos, base_orn = p.getBasePositionAndOrientation(husky)
    yaw = p.getEulerFromQuaternion(base_orn)[2]
    base_vel, base_ang = p.getBaseVelocity(husky)
    gripper_pos = np.array(p.getLinkState(panda, 11)[0])
    obj_pos = np.array(p.getBasePositionAndOrientation(box_id)[0])
    arm_j = [p.getJointState(panda, j)[0] for j in range(6)]
    robot_xy = np.array([base_pos[0], base_pos[1]])

    return np.array([
        base_pos[0], base_pos[1], yaw,
        base_vel[0], base_vel[1], base_ang[2],
        *arm_j,
        gripper_pos[0], gripper_pos[1], gripper_pos[2],
        obj_pos[0], obj_pos[1], obj_pos[2],
        shelf_pos[0], shelf_pos[1],
        dropoff_pos[0], dropoff_pos[1],
        np.linalg.norm(robot_xy - np.array(shelf_pos)),
        np.linalg.norm(gripper_pos - obj_pos),
        np.linalg.norm(obj_pos[:2] - np.array(dropoff_pos)),
    ], dtype=np.float32)


def reset_env(husky, panda, box_id, shelf_pos):
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


def step_env(action, husky, panda, wheel_joints):
    vx = float(action[0]) * 5.0
    wz = float(action[2]) * 3.0
    wheel_radius, wheel_base = 0.165, 0.555
    left_vel = (vx - wz * wheel_base / 2) / wheel_radius
    right_vel = (vx + wz * wheel_base / 2) / wheel_radius

    for i, wj in enumerate(wheel_joints):
        vel = left_vel if i % 2 == 0 else right_vel
        p.setJointMotorControl2(husky, wj, p.VELOCITY_CONTROL,
                                 targetVelocity=vel, force=500)

    p.setJointMotorControlArray(panda, list(range(6)), p.POSITION_CONTROL,
                                 targetPositions=action[3:9], forces=[87]*6)
    gpos = 0.04 if float(action[9]) > 0.5 else 0.0
    for gj in [9, 10]:
        p.setJointMotorControl2(panda, gj, p.POSITION_CONTROL,
                                 targetPosition=gpos, force=10)
    p.stepSimulation()


def compute_reward(husky, panda, box_id, shelf_pos, dropoff_pos):
    base_pos, _ = p.getBasePositionAndOrientation(husky)
    robot_xy = np.array(base_pos[:2])
    gripper_pos = np.array(p.getLinkState(panda, 11)[0])
    obj_pos = np.array(p.getBasePositionAndOrientation(box_id)[0])

    ds = np.linalg.norm(robot_xy - np.array(shelf_pos))
    do = np.linalg.norm(gripper_pos - obj_pos)
    dd = np.linalg.norm(obj_pos[:2] - np.array(dropoff_pos))

    reward = -0.3 * ds - 0.5 * do - 0.2 * dd
    if ds < 1.0: reward += 1.0
    if do < 0.2: reward += 2.0
    if do < 0.08: reward += 3.0
    if obj_pos[2] > 0.65: reward += 5.0
    if dd < 0.5 and obj_pos[2] > 0.1: reward += 10.0

    return reward, ds, do, dd


if __name__ == "__main__":
    NUM_EPISODES = 50000  # Train to perfection
    MAX_STEPS = 500

    print("Phase 2 State RL — WHEEL PHYSICS — 50K episodes", flush=True)
    husky, panda, wheel_joints, box_id, shelf_pos, dropoff_pos = setup_env()
    print(f"Wheel joints: {wheel_joints}", flush=True)

    policy = MobileStatePolicy(state_dim=25, action_dim=10)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    print(f"Params: {sum(pp.numel() for pp in policy.parameters()):,}", flush=True)

    episode_rewards = deque(maxlen=200)
    best_reward = -9999
    start = time.time()

    os.makedirs("checkpoints", exist_ok=True)

    for episode in range(NUM_EPISODES):
        reset_env(husky, panda, box_id, shelf_pos)
        state = get_state(husky, panda, box_id, shelf_pos, dropoff_pos)
        states_b, actions_b, lps_b, rewards_b, values_b = [], [], [], [], []
        ep_reward = 0

        for step in range(MAX_STEPS):
            st = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, lp, _, val = policy(st)
            a_np = action.squeeze().numpy()
            step_env(a_np, husky, panda, wheel_joints)
            reward, ds, do, dd = compute_reward(husky, panda, box_id,
                                                 shelf_pos, dropoff_pos)

            states_b.append(st.squeeze())
            actions_b.append(action.squeeze())
            lps_b.append(lp.squeeze())
            rewards_b.append(reward)
            values_b.append(val.squeeze())
            ep_reward += reward
            state = get_state(husky, panda, box_id, shelf_pos, dropoff_pos)

        episode_rewards.append(ep_reward)

        # PPO update
        R = torch.FloatTensor(rewards_b)
        V = torch.stack(values_b)
        adv = torch.zeros_like(R)
        lg = 0
        for t in reversed(range(len(R))):
            nv = V[t+1] if t < len(R)-1 else torch.tensor(0.0)
            d = R[t] + 0.99 * nv - V[t]
            lg = d + 0.99 * 0.95 * lg
            adv[t] = lg
        ret = adv + V.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        S = torch.stack(states_b)
        A = torch.stack(actions_b)
        olp = torch.stack(lps_b).detach()

        for _ in range(4):
            _, nlp, ent, nv = policy(S, A)
            ratio = (nlp - olp).exp()
            pl = -torch.min(adv * ratio, adv * ratio.clamp(0.8, 1.2)).mean()
            vl = nn.MSELoss()(nv, ret)
            loss = pl + 0.5 * vl - 0.02 * ent.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        mr = np.mean(episode_rewards)
        if mr > best_reward:
            best_reward = mr
            torch.save(policy.state_dict(),
                       "checkpoints/phase2_state_policy_wheels.pth")

        if episode % 500 == 0:
            eps = (episode + 1) / (time.time() - start)
            _, ds, do, dd = compute_reward(husky, panda, box_id,
                                            shelf_pos, dropoff_pos)
            print(f"Ep {episode:6d} | R: {mr:8.1f} | Best: {best_reward:.1f} | "
                  f"d_shelf: {ds:.2f} | d_obj: {do:.2f} | d_drop: {dd:.2f} | "
                  f"{eps:.1f} ep/s", flush=True)

    p.disconnect()
    elapsed = (time.time() - start) / 60
    print(f"\nDONE in {elapsed:.1f} min | Best reward: {best_reward:.1f}",
          flush=True)
