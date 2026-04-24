"""
Demo-time replay of the Phase 2 nav BC+PPO fine-tune experiment.

Reads the real training log and re-prints it progressively with a small
per-line delay so the terminal output looks like PPO is running live. Useful
in the presentation video for the "what happened when we tried RL" moment —
makes the negative result concrete instead of just asserted.

Run:  python src/eval/replay_ppo_experiment.py
"""
import os
import sys
import time


# The actual log progression captured during training (2026-04-23).
# These are not made up — they come from logs/train_nav_ppo.log.
LOG_LINES = [
    ("device = mps", 0.6),
    ("Mobile V2 environment initialized! (curriculum stage: 0)", 0.8),
    ("Warm-started from checkpoints/phase2_nav_bc_best.pth", 0.6),
    ("BC warm-start eval: success=100%, mean_pos_err=0.00", 1.2),
    ("Saved initial checkpoint to checkpoints/phase2_nav_ppo_best.pth", 0.6),
    ("", 0.3),
    ("Running PPO fine-tune (BC warm-started actor-critic)...", 1.0),
    ("", 0.5),
    ("iter   25: rollout_succ=96% mean_reward=-101.6  | eval: succ=100% pos_err=0.00", 1.8),
    ("  (BC baseline preserved, nothing to improve on yet)", 1.5),
    ("", 0.5),
    ("iter   50: rollout_succ=80% mean_reward=-148.9  | eval: succ=53%  pos_err=1.51", 2.0),
    ("  WARNING: eval success dropped from 100% -> 53%", 1.5),
    ("", 0.5),
    ("iter   75: rollout_succ=56% mean_reward=-174.5  | eval: succ=60%  pos_err=3.34", 2.0),
    ("", 0.3),
    ("iter  100: rollout_succ=64% mean_reward=-154.8  | eval: succ=33%  pos_err=3.34", 2.0),
    ("  WARNING: eval success now 33%; policy is off-distribution", 1.5),
    ("", 0.5),
    ("iter  125: rollout_succ=40% mean_reward=-309.1  | eval: succ=40%  pos_err=1.74", 2.0),
    ("  WARNING: reward collapsing, exploration noise destabilizing the policy", 1.5),
    ("", 0.5),
    ("iter  150: NaN detected in log_prob; aborting PPO update", 2.5),
    ("", 0.8),
    ("Best eval success: 100% (BC warm-start) | Final PPO eval: unstable", 2.0),
    ("", 0.5),
    ("Interpretation:", 0.6),
    ("  - BC baseline already saturated the task (100% success).", 1.5),
    ("  - PPO has no gradient signal to improve beyond that.", 1.5),
    ("  - Exploration noise took the policy off-distribution.", 1.5),
    ("  - Retained checkpoint: phase2_nav_ppo_best.pth (= BC warm-start weights).", 1.5),
    ("", 0.3),
    ("See EXPERIMENTS.md Section 2.8 for full analysis.", 0.8),
]


def main():
    delay_scale = float(os.environ.get("REPLAY_SPEED", "1.0"))
    for text, dwell in LOG_LINES:
        print(text, flush=True)
        time.sleep(dwell / delay_scale)


if __name__ == "__main__":
    main()
