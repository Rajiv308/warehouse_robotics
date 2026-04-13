"""
Phase 1 State-based RL — Fixed Panda arm pick-and-lift.
No image rendering — trains in minutes.
"""
import sys, os, torch, torch.nn as nn, numpy as np, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pybullet as p
from src.env.warehouse_env import WarehouseEnv
from collections import deque


class StatePolicy(nn.Module):
    def __init__(self, state_dim=19, action_dim=7):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def get_state(self, env):
        gripper = np.array(p.getLinkState(env.robot_id, 11)[0])
        objs = []
        for oid in env.object_ids:
            pos, _ = p.getBasePositionAndOrientation(oid)
            objs.extend(pos)
        joints = [p.getJointState(env.robot_id, j)[0] for j in range(7)]
        return np.concatenate([gripper, objs, joints]).astype(np.float32)

    def forward(self, state, action=None):
        mean = self.actor_mean(state)
        std = self.actor_log_std.exp().clamp(min=0.05)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        return (action, dist.log_prob(action).sum(-1),
                dist.entropy().sum(-1), self.critic(state).squeeze(-1))


if __name__ == "__main__":
    NUM_EPISODES = 50000
    MAX_STEPS = 300

    print(f"Phase 1 State RL — {NUM_EPISODES} episodes", flush=True)

    policy = StatePolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    print(f"Params: {sum(pp.numel() for pp in policy.parameters()):,}", flush=True)

    env = WarehouseEnv(render=False)
    env.initialize()

    episode_rewards = deque(maxlen=200)
    successes = deque(maxlen=200)
    best_success_rate = 0
    total_successes = 0
    start = time.time()

    os.makedirs("checkpoints", exist_ok=True)

    for episode in range(NUM_EPISODES):
        # Reset without camera
        env.step_count = 0
        home = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        for i, pos in enumerate(home):
            p.resetJointState(env.robot_id, i, pos)
        for i, obj_id in enumerate(env.object_ids):
            noise = np.random.uniform(-0.05, 0.05, 2)
            p.resetBasePositionAndOrientation(
                obj_id, [0.5+noise[0], (i-1)*0.3+noise[1], 0.05], [0,0,0,1])
        env._near_object = False
        env._grasped = False
        env._lift_count = 0

        state = policy.get_state(env)
        states_b, actions_b, lps_b, rewards_b, values_b = [], [], [], [], []
        ep_reward = 0

        for step in range(MAX_STEPS):
            st = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, lp, _, val = policy(st)
            a_np = action.squeeze().numpy()

            env.apply_action(a_np)
            p.stepSimulation()
            env.step_count += 1
            reward = env.compute_reward()
            if env._grasped:
                env._lift_count += 1
            else:
                env._lift_count = 0
            success = env._lift_count >= 5
            done = success or env.step_count >= MAX_STEPS
            if success:
                reward += 50.0

            states_b.append(st.squeeze())
            actions_b.append(action.squeeze())
            lps_b.append(lp.squeeze())
            rewards_b.append(reward)
            values_b.append(val.squeeze())
            ep_reward += reward
            state = policy.get_state(env)
            if done:
                break

        if success:
            total_successes += 1
        episode_rewards.append(ep_reward)
        successes.append(1 if success else 0)

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
            loss = pl + 0.5 * vl - 0.01 * ent.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        sr = np.mean(successes) * 100
        if sr > best_success_rate:
            best_success_rate = sr
            torch.save(policy.state_dict(),
                       "checkpoints/phase1_state_policy.pth")

        if episode % 500 == 0:
            mr = np.mean(episode_rewards)
            eps = (episode + 1) / (time.time() - start)
            print(f"Ep {episode:6d} | R: {mr:7.1f} | Success: {sr:5.1f}% | "
                  f"Best: {best_success_rate:.1f}% | Total: {total_successes} | "
                  f"{eps:.1f} ep/s", flush=True)

    env.close()
    elapsed = (time.time() - start) / 60
    print(f"\nDONE in {elapsed:.1f} min | Best success: {best_success_rate:.1f}% | "
          f"Total: {total_successes}/{NUM_EPISODES}", flush=True)
