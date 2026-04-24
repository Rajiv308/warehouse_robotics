"""
PPO fine-tuning for Phase 2 navigation, warm-started from the BC nav
checkpoint. This gives the capstone a genuine reinforcement-learning
component in the final pipeline: BC provides the expert-matching initial
behavior, PPO then optimizes it against a direct task reward.

Loads checkpoints/phase2_nav_bc_best.pth as the actor_mean warm start, adds
a stochastic policy head and a value network, and rolls out episodes with
randomized Husky spawn positions. Saves the best policy by success rate to
checkpoints/phase2_nav_ppo_best.pth — a separate file, never overwriting the
BC or original nav checkpoints.

The demo opts in via PHASE2_USE_NAV_RL=1.
"""
import os
import sys
from collections import deque

import numpy as np
import pybullet as p
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.env.warehouse_env_mobile_v2 import MobileWarehouseEnvV2
from src.training.train_phase2_nav_pickpose import (
    get_nav_state, nav_success, build_full_action,
)
from src.data.collect_nav_bc_demos import sample_valid_start


class NavPolicyRL(nn.Module):
    """Actor-critic version of NavPolicy with a stochastic Gaussian head."""

    def __init__(self, state_dim=12, action_dim=3):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.actor_log_std = nn.Parameter(
            torch.tensor([-3.2, -3.5, -3.2], dtype=torch.float32)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action=None):
        mean = self.actor_mean(state)
        std = self.actor_log_std.exp().clamp(min=0.01, max=0.15)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
        return (
            action,
            dist.log_prob(action).sum(-1),
            dist.entropy().sum(-1),
            self.critic(state).squeeze(-1),
        )


def reset_for_rollout(env):
    env.reset_state_only()
    sx, sy, syaw = sample_valid_start()
    p.resetBasePositionAndOrientation(
        env.husky_id, [sx, sy, 0.02],
        p.getQuaternionFromEuler([0, 0, syaw]),
    )
    p.resetBaseVelocity(env.husky_id, [0, 0, 0], [0, 0, 0])
    env._sync_panda_to_husky([sx, sy, 0.02], syaw)


def compute_reward(env, prev_pos_err):
    ok, pos_err, yaw_err = nav_success(env)
    r = -0.6 * pos_err - 0.4 * yaw_err - 0.01
    if prev_pos_err is not None:
        r += 1.5 * (prev_pos_err - pos_err)
    done = False
    if ok:
        r += 50.0
        done = True
    return r, done, pos_err, yaw_err


def evaluate(policy, env, episodes=15, max_steps=250, device="cpu"):
    succ = 0
    pos_errs = []
    for _ in range(episodes):
        reset_for_rollout(env)
        done = False
        for _ in range(max_steps):
            st = torch.FloatTensor(get_nav_state(env)).unsqueeze(0).to(device)
            with torch.no_grad():
                mean = policy.actor_mean(st).squeeze(0).cpu().numpy()
            action = np.clip(mean, -1.0, 1.0)
            env.step_state_only(build_full_action(action))
            ok, pos_err, _ = nav_success(env)
            if ok:
                succ += 1
                done = True
                break
        if not done:
            _, pe, _ = nav_success(env)
            pos_errs.append(pe)
    return succ / episodes, float(np.mean(pos_errs)) if pos_errs else 0.0


def main():
    bc_ckpt = os.environ.get("PHASE2_NAV_BC_CKPT", "checkpoints/phase2_nav_bc_best.pth")
    out_ckpt = os.environ.get("PHASE2_NAV_PPO_CKPT", "checkpoints/phase2_nav_ppo_best.pth")
    num_iters = int(os.environ.get("PHASE2_NAV_PPO_ITERS", "300"))
    eval_every = int(os.environ.get("PHASE2_NAV_PPO_EVAL_EVERY", "25"))
    rollout_max = int(os.environ.get("PHASE2_NAV_PPO_ROLLOUT", "200"))

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device = {device}")

    env = MobileWarehouseEnvV2(
        config_path="configs/config_cloud.yaml",
        render=False, curriculum_stage=0, success_mode="pickup",
    )
    env.initialize()
    try:
        p.setCollisionFilterPair(env.husky_id, env.dropoff_id, -1, -1, enableCollision=0)
    except Exception:
        pass

    policy = NavPolicyRL().to(device)
    if os.path.exists(bc_ckpt):
        bc_sd = torch.load(bc_ckpt, map_location=device, weights_only=True)
        # Load weights into actor_mean only; actor_log_std and critic stay random.
        own_sd = policy.state_dict()
        for k, v in bc_sd.items():
            mapped_key = f"actor.{k.split('.', 1)[1]}" if k.startswith("actor.") else k
            # BC NavPolicy has keys like "actor.0.weight"; our actor_mean has "actor_mean.0.weight".
            target_key = k.replace("actor.", "actor_mean.")
            if target_key in own_sd and own_sd[target_key].shape == v.shape:
                own_sd[target_key] = v
        policy.load_state_dict(own_sd)
        print(f"Warm-started from {bc_ckpt}")
    else:
        print(f"WARNING: BC warm-start checkpoint {bc_ckpt} not found; "
              f"training from scratch.")

    opt = torch.optim.Adam(policy.parameters(), lr=5e-5)

    # Initial eval so we know the baseline.
    base_succ, base_err = evaluate(policy, env, episodes=10, device=device)
    print(f"BC warm-start eval: success={base_succ:.0%}, mean_pos_err={base_err:.2f}")
    best_succ = base_succ
    torch.save(policy.state_dict(), out_ckpt)
    print(f"Saved initial checkpoint to {out_ckpt}")

    reward_buf = deque(maxlen=25)
    success_buf = deque(maxlen=25)

    for it in range(num_iters):
        reset_for_rollout(env)
        states_b, actions_b, lps_b, rewards_b, values_b = [], [], [], [], []
        prev_pos_err = None
        ep_success = False

        for _ in range(rollout_max):
            st = torch.FloatTensor(get_nav_state(env)).unsqueeze(0).to(device)
            with torch.no_grad():
                action, lp, _, val = policy(st)
            a_np = action.squeeze(0).cpu().numpy()
            env.step_state_only(build_full_action(np.clip(a_np, -1.0, 1.0)))
            r, done, pos_err, _ = compute_reward(env, prev_pos_err)
            prev_pos_err = pos_err

            states_b.append(st.squeeze(0))
            actions_b.append(action.squeeze(0))
            lps_b.append(lp.squeeze(0))
            rewards_b.append(r)
            values_b.append(val.squeeze(0))
            if done:
                ep_success = True
                break

        reward_buf.append(sum(rewards_b))
        success_buf.append(1 if ep_success else 0)

        rewards_t = torch.FloatTensor(rewards_b).to(device)
        values_t = torch.stack(values_b).to(device)
        advantages = torch.zeros_like(rewards_t)
        last_gae = 0.0
        for t in reversed(range(len(rewards_t))):
            next_val = values_t[t + 1] if t < len(rewards_t) - 1 else torch.tensor(0.0, device=device)
            delta = rewards_t[t] + 0.99 * next_val - values_t[t]
            last_gae = delta + 0.99 * 0.95 * last_gae
            advantages[t] = last_gae
        returns = advantages + values_t.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        S = torch.stack(states_b).to(device)
        A = torch.stack(actions_b).to(device)
        old_lp = torch.stack(lps_b).detach().to(device)

        for _ in range(4):
            _, new_lp, ent, new_values = policy(S, A)
            ratio = (new_lp - old_lp).exp().clamp(0.1, 10.0)
            pg_loss = -torch.min(
                advantages * ratio,
                advantages * ratio.clamp(0.9, 1.1),
            ).mean()
            value_loss = nn.MSELoss()(new_values, returns)
            loss = pg_loss + 0.5 * value_loss - 0.0 * ent.mean()
            if torch.isnan(loss):
                print(f"  iter {it+1}: NaN loss; skipping update")
                break
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.25)
            opt.step()

        if (it + 1) % eval_every == 0 or it == num_iters - 1:
            succ, err = evaluate(policy, env, episodes=15, device=device)
            recent_r = float(np.mean(reward_buf)) if reward_buf else 0.0
            recent_succ = float(np.mean(success_buf)) if success_buf else 0.0
            print(
                f"iter {it+1:4d}: rollout_succ={recent_succ:.0%} "
                f"mean_reward={recent_r:.1f}  | eval: succ={succ:.0%} pos_err={err:.2f}"
            )
            if succ > best_succ:
                best_succ = succ
                torch.save(policy.state_dict(), out_ckpt)
                print(f"  Saved new best to {out_ckpt} (succ={succ:.0%})")

    print(f"\nBest eval success: {best_succ:.0%}  (baseline BC: {base_succ:.0%})")
    env.close()


if __name__ == "__main__":
    main()
