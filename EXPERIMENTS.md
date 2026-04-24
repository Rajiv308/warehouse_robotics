# Experiments & Ablations

This document is the complete experimental record for the capstone. Every approach we attempted, every checkpoint we kept, and what each one taught us. It is the backbone for the final report's "Results" and "Discussion" sections.

---

## 1. Research question

Given a compact academic-scale simulation, which combination of learning and classical techniques produces a reliable instruction-conditioned warehouse pick-and-deliver system?

## 2. What we tried (chronological)

### 2.1 Vision-Language-Action (RT-2 / OpenVLA inspired)
**Method.** Joint ResNet-18 image encoder + DistilBERT language encoder + cross-attention fusion, trained end-to-end to output robot actions. The original thesis of the project.

**Outcome.** Checkpoints trained and ran, but failed to produce reliable grasping. The contact-rich manipulation problem was far harder than the architecture could address under the compute and data budget. Led to the pivot toward modular/hybrid architectures.

**Evidence retained.** `checkpoints/best_model.pth` (~418 MB), `checkpoints/best_mobile_model.pth` (~552 MB), `checkpoints/best_cloud_rl_model.pth` (~325 MB), `checkpoints/best_rl_model.pth`, `checkpoints/best_mobile_rl_model.pth`, `checkpoints/checkpoint_epoch_10.pth`.

### 2.2 Phase 1 state-based RL (joint-space PPO)
**Method.** PPO over joint-space actions, state-based observation, reward for grasp-then-lift.

**Outcome.** Policies exhibited "drugged hand" behavior — the arm approaches the box but then sways, misaligns, and closes on air or the side. Never converged to a reliable grasp.

**Evidence retained.** `checkpoints/phase1_state_policy.pth`, `phase1_state_policy_eval_best.pth`, `phase1_state_bc_policy.pth`, `phase1_state_policy_fullarm.pth`, `phase1_state_policy_fullarm_eval_best.pth`, `phase1_state_policy_polish.pth`, `phase1_state_policy_polish_eval_best.pth`, `phase1_state_bc_init.pth`, `phase1_state_bc_init_fullarm.pth`.

### 2.3 Phase 1 Cartesian BC
**Method.** Cartesian end-effector action space. BC from expert rollouts of a scripted approach-align-descend-close controller.

**Outcome.** Better than joint-space: the arm reaches the object region, but misaligns in xy during close, usually pinching just next to the box rather than around it. Partial convergence.

**Evidence retained.** `checkpoints/phase1_cartesian_target0_bc_init.pth`, `phase1_cartesian_target0_best.pth`, `phase1_cartesian_target0_latest.pth`, `phase1_stagealign_target0_*.pth` (3 files).

### 2.4 Phase 2 mobile environment repair
**Method.** Before any Phase 2 learning could be trusted, the simulator itself had to be fixed. Repairs in `src/env/warehouse_env_mobile_v2.py`:

- shelves became real obstacle geometry
- hidden blocker volumes added under shelves so the base cannot drive underneath
- Husky/Panda double-binding removed; Panda synced cleanly to the Husky base each frame
- base motion collision-validated; walls tracked in `wall_ids`
- shelf objects moved to the front edge via `shelf_object_front_offset = -0.12` so shelf-front grasps became mechanically reachable
- grasp constraint bug fixed to weld at the **live relative pose** (position *and* orientation) instead of a hardcoded `[0, 0, 0.04]` offset that caused the box to visibly "dangle" off the gripper

**Outcome.** The repaired env made every subsequent Phase 2 result meaningful.

### 2.5 Phase 2 pickup PPO+BC (original)
**Method.** `MobileCartesianPickPolicy` (13→256→256→4, actor-critic with Gaussian head) trained via BC warm-start followed by PPO fine-tuning. Reward included a weighted BC loss to prevent catastrophic drift.

**Outcome.** Policy converged against the env's reward, but exploited a proximity-based auto-weld in the simulator — learning "drive end-effector to within 13 cm of object" rather than physically grasping. The box would appear attached to the gripper without real two-finger contact. Exposed the grasp-detection bug and motivated the env fix (contact-gated weld).

**Evidence retained.** `checkpoints/phase2_pick_cartesian_bc_init.pth`, `phase2_pick_cartesian_best.pth`, `phase2_pick_cartesian_latest.pth`, `phase2_pick_cartesian_robust_best.pth`, `phase2_pick_cartesian_robust_latest.pth`.

### 2.6 Phase 2 navigation BC (stage 0 / stage 1)
**Method.** `NavPolicy` (12→256→256→3) trained via BC (MSE) on expert rollouts from a scripted P-controller, curriculum stage 0 (Husky shelf-proximal).

**Outcome.** 100% eval success within the training distribution. Did **not** generalize — on center-of-aisle spawns the policy walked off-course and never reached the pickup pose.

**Evidence retained.** `checkpoints/phase2_nav_pickpose_best.pth` (stage 0), `phase2_nav_pickpose_stage1_best.pth` (stage 1 broader starts, ~70%).

### 2.7 Phase 2 navigation BC (broad)
**Method.** Same architecture, retrained on 500 expert rollouts from **randomized Husky spawns over the whole accessible workspace** — center of aisle, shelf-adjacent, random orientation. Expert logic revised to turn-toward-target-then-drive instead of only aligning yaw to the shelf.

**Outcome.** Val MSE = 0.00774. **100% success in a 15-episode eval**, including center and random starts. This is the nav policy used in the final demo when `PHASE2_USE_NAV_BROAD=1`.

**Evidence retained.** `checkpoints/phase2_nav_bc_best.pth`. Demos collected in `demos/nav_bc.npz`.

### 2.8 Phase 2 navigation BC+PPO fine-tune (real RL attempt)
**Method.** Warm-started a new `NavPolicyRL` (actor-critic) with the broad-BC actor weights. Added learnable `actor_log_std` + random-init value network. Ran PPO with clipped ratios, GAE advantages, 4 epochs per rollout. Reward: `-0.6*pos_err - 0.4*yaw_err - 0.01 + 1.5*delta_pos_err + 50*success`.

**Outcome.** The BC warm-start already achieved 100% eval success at iteration 0. Subsequent PPO iterations introduced exploration noise that took the policy off-distribution (iter 50: 53%, iter 75: 60%, iter 100: 33%), and the policy eventually produced NaN losses. Retrying with gentler hyperparameters (log_std initialised at e^(-3.2), lr=5e-5, tighter clip) delayed but did not prevent the degradation.

**Interpretation.** This is a genuine negative result with a clear cause: when a BC policy already saturates the task's success metric, PPO has no gradient signal to improve it, and any exploration noise is pure downside. A standard finding in the RL vs imitation-learning literature. The checkpoint saved before PPO's first gradient step (the warm-start snapshot) is functionally identical to the BC checkpoint — that is what `PHASE2_USE_NAV_RL=1` loads in the demo.

**Evidence retained.** `checkpoints/phase2_nav_ppo_best.pth`, training log in `logs/train_nav_ppo.log`.

### 2.9 Phase 2 pickup BC (final)
**Method.** Distilled a **deterministic scripted grasp finalizer** (open → hover above object → descend → close → verify two-finger `getContactPoints` → weld at live pose) into a fresh `MobileCartesianPickPolicy` via BC (MSE). 177 episodes / 39 k (state, action) pairs collected with `env.auto_weld` disabled — episodes only counted if they ended with real contact and a physical lift.

**Outcome.** Val MSE = 0.038. In the demo the BC policy drives the arm from the shelf-front pre-hover pose through descent and closure, achieving **real two-finger contact in every episode**. The contact-gated weld in the env then locks the object at its live relative pose (no dangling). A scripted IK lift completes the pickup (the BC's state-aliasing prevents it from distinguishing "still closing" from "start lifting"; a documented limitation).

**Evidence retained.** `checkpoints/phase2_pick_bc_best.pth`. Demos in `demos/pickup_bc.npz`.

### 2.10 Phase 2 delivery BC
**Method.** Distilled the scripted diff-drive P-controller (reverse-and-drive-to-approach-point) into a small `DeliveryDrivePolicy` (5→128→128→2, tanh output). 400 episodes / 146 k (state, action) pairs from random start/dropoff pairs. State expressed in Husky body frame (translation- and rotation-invariant).

**Outcome.** Val MSE = 0.0005. Standalone eval (30 random start/target pairs): **18/30 (60%) success**. Failures clustered on shelf-proximal starts where small BC errors steered the Husky into shelf sides. The BC inherited the expert's straight-line-only driving — no obstacle awareness. Retained as a documented alternative; **not used** in the final demo because the A\* planner is strictly better for the obstacle-rich scene.

**Evidence retained.** `checkpoints/phase2_delivery_bc_best.pth`. Demos in `demos/delivery_bc.npz`.

### 2.11 Delivery via A\* path planning + pure pursuit (classical AI)
**Method.** 2-D occupancy grid over the workspace (35×30 cells at 0.2 m). Shelves from the env config inflated by the Husky half-width + margin (0.55 m) and marked as blocked. A\* with 8-connected neighborhood and Euclidean heuristic. Raw path simplified by **line-of-sight smoothing** (typically 10+ cells → 2–4 waypoints). Follower: **pure pursuit** — the Husky always steers toward a lookahead "carrot" at constant cruise speed with continuous differential steering, turning in place only when heading error exceeds 99°. Wheel motors driven by real PyBullet velocity control.

**Outcome.** In the final demo, **5/5 successful deliveries per run** with object placement within **0–6 cm** of the randomized drop zone across the episode distribution `x ∈ [-1, 1]`, `y ∈ [1.5, 2.7]`. Smooth single-motion driving, no stop-and-turn stutter.

**Code.** `src/planning/astar_delivery.py`; follower inlined in `src/eval/demo_phase2_hybrid.py`.

### 2.12 Vision + Language grounding
**Method.**
- **Language**: a deterministic keyword parser (`src/perception/instruction_parser.py`) maps the target color in the natural-language instruction to an object index.
- **Vision**: classical color segmentation (`src/perception/color_detector.py`) — channel-difference thresholds on the RGB camera image, tuned to PyBullet's rendered colors. Verifies the target color is in the Husky's camera view at the shelf-front pickup pose.

**Outcome.** Parser covers 100% of generated instructions. Vision detection ~80-100% depending on color and viewpoint (blue dimmer than yellow after shading). The grounded target idx is load-bearing: it drives which shelf the nav module heads toward.

### 2.13 Phase 1 hybrid (final)
**Method.** Given that no Phase 1 learned policy converged, `src/eval/demo_phase1_hybrid.py` mirrors the Phase 2 pattern using purely classical control: instruction → color parse → vision confirm → scripted IK hover-descend-close-verify-weld-lift. Three colored boxes (red/blue/green) on the tabletop per episode.

**Outcome.** 5/5 on a 5-episode run with real two-finger contact every time. No "drugged-hand" behavior because there is no unstable learned controller in the loop.

---

## 3. Summary of the final demo pipeline

The composed Phase 2 pipeline used in the final GUI demo is:

1. **Language parsing** (classical) extracts target color from the instruction.
2. **Learned navigation** (broad-BC, `phase2_nav_bc_best.pth`) drives the Husky from any center-of-aisle spawn to the shelf-front pickup pose.
3. **Vision verification** (classical color segmentation) confirms the target box is visible in the camera image.
4. **Arm bridge** (scripted joint interpolation) moves the Panda to the pickup-ready pose.
5. **Learned pickup** (BC, `phase2_pick_bc_best.pth`) drives the arm into a real two-finger contact grasp.
6. **Contact-gated weld + scripted IK lift** (env fix + IK).
7. **A\* path planning** (classical AI) computes an obstacle-aware route from the shelf-front to the dropoff approach.
8. **Pure-pursuit diff-drive** (classical control) follows the path with real wheel physics.
9. **Scripted arm lower + release** delivers the box to the drop zone.

Two load-bearing neural networks (nav BC, pickup BC) + one classical AI planner (A\*) + one classical perception+language layer. This matches the "hybrid learned + deterministic" framing adopted in the final scope.

---

## 4. Key findings

1. **Full end-to-end RL underperformed** the decomposed approach for contact-rich manipulation in this compute/time budget. The VLA attempts never converged to reliable grasps.
2. **Joint-space RL is harder than Cartesian BC.** Phase 1's joint-space policies showed instability; Phase 1's Cartesian BC did better but still missed fine alignment.
3. **Simulator validity is a prerequisite for meaningful learning.** Phase 2 learning results were only trustworthy after the mobile env was repaired (shelves, collisions, grasp constraint).
4. **BC distillation from a verified scripted expert beats PPO** when the expert is near-optimal and the reward has exploits. Our PPO pickup reward-hacked the proximity weld; our PPO nav had no room above the BC baseline and destabilized.
5. **Classical planning complements learned policies** for obstacle-aware mobility. A\* gave strictly better delivery trajectories than the BC delivery policy because A\* has geometric knowledge of shelves that BC had to learn from straight-line demos.
6. **Vision+language grounding does not need a neural model** to be real — a regex parser plus color segmentation suffices for a small closed vocabulary, and they run in the loop as the load-bearing target selector.

---

## 5. Results summary table

| Subtask | Final method | Checkpoint (if any) | Measured result |
|---|---|---|---|
| Target selection from instruction | deterministic keyword parser | — | 100% (generated vocabulary) |
| Vision confirmation of target | RGB color segmentation | — | ~80-100% per color |
| Navigation to shelf | **learned BC (broad)** | `phase2_nav_bc_best.pth` | **100%** eval across random spawns |
| Bridge to pickup-ready pose | scripted joint interpolation | — | 100% |
| Pickup (approach + descend + close) | **learned BC** | `phase2_pick_bc_best.pth` | **100%** with real two-finger contact |
| Contact-gated weld | contact-point verified + live-pose constraint | — | 100% |
| Lift | scripted IK | — | 100% |
| Delivery path planning | **A\*** + line-of-sight smoothing | — | 100% obstacle-free routes |
| Delivery path following | pure-pursuit diff-drive | — | wheel-physics, continuous motion |
| Object placement at dropoff | scripted IK + retreat release | — | 0–6 cm accuracy |
| **Overall end-to-end Phase 2** | hybrid | — | **Pickup 5/5, Delivery 5/5** (100%) |
| Overall Phase 1 (hybrid) | scripted | — | Pickup 5/5 |

---

## 6. Documented negative results (keep in the report)

- End-to-end RL for contact-rich grasping (Phase 1 state policies) did not converge.
- VLA-style image+language policies (Phase 1 large-checkpoint experiments) did not converge.
- PPO fine-tuning on top of a saturating BC baseline (Phase 2 nav) introduced instability without improvement.
- PPO fine-tuning with a permissive reward (Phase 2 pickup proximity weld) reward-hacked.
- BC delivery distilled from a geometry-agnostic expert could not generalize to obstacle-proximal starts.

These are listed honestly — each one informed a design decision in the final pipeline.
