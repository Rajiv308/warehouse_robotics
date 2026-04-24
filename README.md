# Warehouse Robotics — Instruction-Conditioned Mobile Manipulation

CS 5100 (Foundations of AI) capstone, Northeastern University — Rajiv Menon, 2026.

A hybrid learned + classical pipeline in PyBullet. A Husky-mounted Franka Panda takes a natural-language instruction ("pick up the red box and deliver it"), drives to the correct shelf, picks up the target, plans an obstacle-aware path around the shelves, and places the box at a randomized dropoff.

**Final demo result:** `Pickup 5/5, Delivery 5/5` with object placement within 0–6 cm of the randomized drop zone center, from a center-of-aisle random spawn.

---

## Quick start

### Prerequisites
- Python 3.11
- Dependencies: `pybullet`, `torch`, `numpy` (MPS supported on Apple Silicon)

### Run the main demo
```bash
PHASE2_USE_NAV_BROAD=1 python src/eval/demo_phase2_hybrid.py
```

You'll see:
- Husky spawning at the center of the aisle
- Language parser extracting the target color from the instruction
- Vision confirming the target color in the camera image
- Learned BC navigation driving the Husky to the correct shelf
- Learned BC pickup closing on real two-finger contact
- A\* path drawn on the floor, Husky following it with real wheel physics
- Box released on the randomized dropoff pad

### Other demos
```bash
# Phase 1 hybrid (fixed-base Panda, 3 colored boxes on table):
python src/eval/demo_phase1_hybrid.py

# Replay the BC+PPO nav ablation experiment (prints the training log
# progressively so it looks live):
python src/eval/replay_ppo_experiment.py

# Opt in to the BC delivery policy instead of A* (ablation):
PHASE2_USE_NAV_BROAD=1 PHASE2_DELIVERY_USE_BC=1 python src/eval/demo_phase2_hybrid.py

# Use the PPO-finetuned nav checkpoint (functionally identical to BC since
# PPO destabilized — see EXPERIMENTS.md §2.8):
PHASE2_USE_NAV_RL=1 python src/eval/demo_phase2_hybrid.py

# Standalone headless eval of the BC delivery policy:
python src/eval/eval_delivery_bc.py
```

---

## Architecture (final pipeline)

| Stage | Kind | Implementation |
|---|---|---|
| Instruction → target color | **classical language grounding** | `src/perception/instruction_parser.py` |
| Vision confirmation | **classical CV** | `src/perception/color_detector.py` |
| Navigation (random spawn → shelf-front) | **learned (BC)** | `src/training/train_bc_nav.py` → `checkpoints/phase2_nav_bc_best.pth` |
| Arm bridge to pickup-ready | scripted | joint interpolation |
| Pickup (approach + descend + close) | **learned (BC)** | `src/training/train_bc_pickup.py` → `checkpoints/phase2_pick_bc_best.pth` |
| Contact-gated weld | scripted | fires on real `getContactPoints` from both fingers |
| Lift | scripted | IK |
| Delivery path planning | **classical AI** | `src/planning/astar_delivery.py` (A\* + line-of-sight smoothing) |
| Delivery path following | scripted | pure-pursuit diff-drive with real PyBullet wheel motors |
| Arm lower + release | scripted | IK + finger teleport |

Two load-bearing neural networks (BC nav + BC pickup), one classical planner (A\*), and a classical perception+language layer.

### Key design decisions
- **BC distillation from scripted experts** replaced end-to-end RL after PPO attempts either reward-hacked simulator bugs or destabilized a saturated BC baseline.
- **A\* path planning** replaced a behavior-cloned delivery policy because the BC expert was geometry-agnostic (straight-line driver), so the BC inherited no obstacle awareness. A\* has it natively.
- **Contact-gated weld** (`env.auto_weld = False` + `getContactPoints` check on both finger links) forces the pickup policy to produce *real* two-finger grasps, not just proximity triggers.

See `EXPERIMENTS.md` for the full record of what was tried (VLA, joint-space PPO, Cartesian BC, BC+PPO, etc.) and what worked vs didn't.

---

## Repository layout

```
.
├── README.md                         (this file)
├── EXPERIMENTS.md                    full experimental record / ablation
├── FINAL_PROJECT_SCOPE.md            locked scope and claims
├── FINAL_REPORT_DRAFT.md             skeleton for the final report
├── PRESENTATION_OUTLINE.md           slide-by-slide outline
├── PROJECT_STATUS_HANDOFF.md         engineering snapshot
├── OVERNIGHT_HANDOFF.md              session handoff notes
├── configs/config_cloud.yaml         env config (shelves, dropoff, workspace)
├── demos/                            BC training datasets (.npz)
│   ├── nav_bc.npz
│   ├── pickup_bc.npz
│   └── delivery_bc.npz
├── checkpoints/                      (gitignored; shared separately)
└── src/
    ├── env/
    │   ├── warehouse_env.py          Phase 1 env
    │   └── warehouse_env_mobile_v2.py  Phase 2 env (Husky + Panda, repaired)
    ├── perception/
    │   ├── instruction_parser.py     keyword parser
    │   └── color_detector.py         RGB color segmentation
    ├── planning/
    │   └── astar_delivery.py         A* + line-of-sight smoothing
    ├── data/
    │   ├── collect_nav_bc_demos.py
    │   ├── collect_pickup_bc_demos.py
    │   └── collect_delivery_bc_demos.py
    ├── training/
    │   ├── train_bc_nav.py           BC nav trainer (broad spawn)
    │   ├── train_bc_pickup.py        BC pickup trainer
    │   ├── train_bc_delivery.py      BC delivery trainer (ablation)
    │   ├── train_nav_ppo.py          BC+PPO nav fine-tune (ablation)
    │   ├── train_phase2_nav_pickpose.py   original BC nav (narrow)
    │   └── train_phase2_pick_cartesian.py original PPO+BC pickup (ablation)
    └── eval/
        ├── demo_phase2_hybrid.py     final Phase 2 demo
        ├── demo_phase1_hybrid.py     final Phase 1 demo
        ├── eval_delivery_bc.py       standalone BC delivery eval
        └── replay_ppo_experiment.py  replays the BC+PPO ablation log
```

---

## Reproducing the learned modules

All collectors are headless and deterministic given a seed. Typical run times are minutes.

```bash
# Collect + train BC nav (broad spawn)
PHASE2_NAV_BC_NUM_EPISODES=500 python src/data/collect_nav_bc_demos.py
PHASE2_NAV_BC_EPOCHS=80        python src/training/train_bc_nav.py

# Collect + train BC pickup
PHASE2_BC_NUM_EPISODES=200     python src/data/collect_pickup_bc_demos.py
PHASE2_BC_EPOCHS=150           python src/training/train_bc_pickup.py

# Collect + train BC delivery (ablation)
PHASE2_DELIVERY_BC_NUM_EPISODES=400 python src/data/collect_delivery_bc_demos.py
PHASE2_DELIVERY_BC_EPOCHS=80        python src/training/train_bc_delivery.py

# BC+PPO fine-tune ablation (expect destabilization; see EXPERIMENTS.md §2.8)
PHASE2_NAV_PPO_ITERS=150       python src/training/train_nav_ppo.py
```

---

## Checkpoints

Not stored in git (2.2 GB). For the final demo you need at minimum:
- `checkpoints/phase2_nav_bc_best.pth`
- `checkpoints/phase2_pick_bc_best.pth`

Recommended to also retain:
- `checkpoints/phase2_delivery_bc_best.pth` (ablation)
- `checkpoints/phase2_nav_ppo_best.pth` (ablation — retains BC warm-start weights)
- `checkpoints/phase2_nav_pickpose_best.pth` (original narrow-start BC, safe fallback)

---

## Results

### Phase 2 (main showcase)
- Pickup: **5/5 (100%)** with real two-finger contact verified
- Delivery: **5/5 (100%)** with drop accuracy 0–6 cm from the randomized dropoff center
- Randomized per episode: target color, dropoff position `x ∈ [-1, 1], y ∈ [1.5, 2.7]`
- Husky spawns at the center of the aisle and the learned BC nav drives it to the correct shelf in ~100 steps

### Phase 1 hybrid (supporting)
- Pickup: **5/5 (100%)** with real two-finger contact
- Three colored boxes visible, target selected by instruction parse + vision confirm

### Negative results (documented, see `EXPERIMENTS.md`)
- End-to-end VLA (ResNet-18 + DistilBERT fusion): did not converge
- Joint-space PPO for Phase 1: "drugged-hand" behavior, no reliable grasp
- BC+PPO pickup on Phase 2: reward-hacked the simulator's proximity weld
- BC+PPO nav on Phase 2: BC already saturated the task, PPO destabilized it
- BC delivery standalone: 60% success due to inherited expert blindness to obstacles

---

## Citation / context

Inspired by RT-2 and OpenVLA's vision-language-action formulations, but the project's empirical finding is that distillation from well-designed scripted experts plus classical planning consistently outperformed end-to-end neural methods under our academic compute and time budget. The final system is a legitimate hybrid: learned where learning is tractable (subskill-level imitation), classical where classical is strictly better (obstacle-aware path planning).
