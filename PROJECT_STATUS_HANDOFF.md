# Project Status Handoff

Last updated: 2026-04-23 (final push)

This file is the current ground truth for the repo state, working artifacts, and remaining work. Deadline: **2026-04-23 23:30 ET**.

The full experimental record (every approach tried + every retained checkpoint) lives in `EXPERIMENTS.md`. This file is the packaging-focused snapshot of the FINAL system.

## 1. Final Project Direction (locked)

**Instruction-conditioned warehouse manipulation in PyBullet using a staged hybrid learned + deterministic control pipeline.**

Phase 2 is the main showcase. Phase 1 is supporting context. End-to-end RL for contact-rich grasping was ruled out; the working system is decomposed into learned navigation and pickup modules composed through deterministic interface logic.

## 2. Current System — What Actually Runs in the Demo

The full demo path in `src/eval/demo_phase2_hybrid.py`:

1. **Instruction parsing** — env assigns a target object color from a natural-language instruction (randomly sampled per episode).
2. **Learned navigation** (`phase2_nav_pickpose_best.pth`) — drives the Husky base from a curriculum start to a shelf-front pickup pose.
3. **Arm bridge to pickup-ready pose** — scripted joint interpolation puts the Panda arm in a standard shelf-front approach configuration.
4. **Pickup approach** — the learned pickup module (`phase2_pick_cartesian_best.pth`) runs for up to 35 steps to drive the arm toward the target object.
5. **Deterministic grasp finalizer** — open gripper → IK hover above object → IK descend to object center → close gripper → **verify real two-finger contact** via `p.getContactPoints` on both finger links → weld at live relative pose (no hardcoded offset).
6. **Lift with rigid pin** — IK lift 22 cm, with the object rigidly pinned to the EE each frame to eliminate constraint lag under teleport-based base motion.
7. **Structured delivery path** — reverse out of the aisle (differential-drive wheel animation) → rotate in place to face the aisle center → drive forward to the center → rotate again to face the dropoff → drive forward to stop 0.7 m south of the dropoff pad (no overlap).
8. **Place and retreat** — IK lower the arm to 0.15 m over the dropoff → release the constraint → teleport fingers open → retreat the arm up and back so the fingers clear the box before static friction can trap it.

## 3. Current Results (latest GUI runs, 2026-04-23)

**Phase 2 hybrid demo** (main showcase) with `PHASE2_USE_NAV_BROAD=1 python src/eval/demo_phase2_hybrid.py`:
```
Pickup:   5/5 (100%)   real two-finger contact verified
Delivery: 5/5 (100%)   box lands on dropoff pad, dropoff_dist 0.00-0.18 m
```
Husky spawns at the center of the aisle (0, -2) facing north. Learned BC nav drives it to the correct shelf in ~100 steps. Vision confirms the target color at the shelf. Learned BC pickup produces a real two-finger grasp. Contact-gated weld + scripted IK lift completes the pickup. A\* path planner routes around shelves to the randomized dropoff. Pure-pursuit diff-drive with real wheel physics follows the path. Scripted arm lower + retreat release places the box. Randomized per episode: target color, dropoff position `x ∈ [-1, 1]`, `y ∈ [1.5, 2.7]`.

**Phase 1 hybrid demo** with `python src/eval/demo_phase1_hybrid.py`:
```
Pickup: 5/5 (100%)   real two-finger contact, 3 colored boxes visible
```
Language parser extracts the target color, vision confirms it from the camera view, scripted grasp finalizer (hover/descend/close/verify/weld/lift) executes the pick. Replaces the previously non-converging Phase 1 learned policies.

## 4. Learned vs Classical Breakdown (honest)

| Component | Kind | Method | Checkpoint / file |
|---|---|---|---|
| Instruction → target color | **classical language grounding** | deterministic keyword parser | `src/perception/instruction_parser.py` |
| Vision confirmation of target | **classical CV** | RGB channel-difference thresholds | `src/perception/color_detector.py` |
| Navigation (arbitrary spawn → shelf-front) | **learned (BC)** | MSE on 500 expert rollouts, random spawns | `phase2_nav_bc_best.pth` |
| BC + PPO fine-tune attempt on nav | **learned (RL ablation)** | BC warm-start + PPO; destabilized | `phase2_nav_ppo_best.pth` (retains BC warm-start) |
| Bridge to pickup-ready arm pose | scripted | joint interpolation to pre-hover | inlined |
| Pickup approach + descend + close | **learned (BC)** | MSE on 177 contact-verified rollouts | `phase2_pick_bc_best.pth` |
| Contact-gated weld | scripted | fires only after real `getContactPoints` on both fingers | `warehouse_env_mobile_v2.py` |
| Lift (post-weld 22 cm up) | scripted | IK | inlined |
| Object carry during base motion | scripted | rigid pin to EE each frame | inlined |
| Delivery path planning | **classical AI** | A\* over occupancy grid + line-of-sight smoothing | `src/planning/astar_delivery.py` |
| Delivery path following | scripted | pure-pursuit diff-drive with wheel-physics motor control | inlined |
| Arm lower over dropoff | scripted | IK | inlined |
| Release + retreat | scripted | finger teleport + IK retreat | inlined |

The final pipeline therefore has **two load-bearing learned components** (BC nav, BC pickup) plus **one classical AI component** (A\* path planning) plus **classical vision + language**. The delivery BC (`phase2_delivery_bc_best.pth`, 60% standalone) was trained as an alternative but inherits the scripted expert's lack of obstacle awareness; it is kept as an opt-in ablation (`PHASE2_DELIVERY_USE_BC=1`) and a documented result, but A\* is the default. See `EXPERIMENTS.md` for the full record of approaches tried.

## 5. Critical Environment Fixes (already landed)

All in `src/env/warehouse_env_mobile_v2.py`:

- Shelves are real obstacle geometry; hidden blocker volumes prevent the base from driving under shelves.
- Husky/Panda double-binding removed; Panda synced cleanly to Husky.
- Base motion is collision-validated; walls tracked in `wall_ids`.
- Objects spawn on the real shelf geometry at the front edge via `shelf_object_front_offset = -0.12`, making pickup mechanically reachable.
- New pickup primitives: `apply_cartesian_action`, `reset_pickup_task`, `step_pickup_cartesian`, `set_pickup_ready_pose`, `animate_pickup_ready_pose`, `servo_pickup_ready_pose`.
- **Grasp-attach logic repaired** (2026-04-22): the constraint is now welded at the object's **live relative pose** to the EE (not a hardcoded `[0, 0, 0.04]` offset) so the box never snaps to a dangling position. Auto-weld is gated by `env.auto_weld` flag so the demo's grasp finalizer can disable the old proximity-based weld and run a contact-verified weld instead.

## 6. Checkpoints in Use

**Navigation:**
- `checkpoints/phase2_nav_pickpose_best.pth` (Stage 0, ~100%) — used by the demo
- `checkpoints/phase2_nav_pickpose_stage1_best.pth` (Stage 1, ~70% broader starts) — available

**Pickup:**
- `checkpoints/phase2_pick_bc_best.pth` — **primary pickup policy** in the demo. BC-distilled from a scripted expert using the same `MobileCartesianPickPolicy` architecture. Produces a genuine two-finger contact grasp that the contact-gated weld accepts.
- `checkpoints/phase2_pick_cartesian_best.pth` — original PPO-trained policy, used as fallback if BC isn't available.
- `checkpoints/phase2_pick_cartesian_robust_best.pth` — alternate PPO checkpoint with stronger handoff jitter.

**Delivery navigation:**
- `checkpoints/phase2_delivery_bc_best.pth` — BC policy distilled from the scripted diff-drive controller. Small MLP 5 → 128 → 128 → 2 with `tanh` output. 400 expert episodes / 146 k (state, action) pairs. Val MSE ~ 0.0005. Standalone eval: **~60 %** success reaching random targets from random starts; used in the demo as the **primary driver with the scripted controller as a fallback**, so the demo does not regress.

## 7. BC Distillation Pipelines (both landed)

### Pickup BC (landed)

Scripted grasp finalizer distilled into `MobileCartesianPickPolicy`:
1. **Collection** (`src/data/collect_pickup_bc_demos.py`): 177 successful episodes, 39 k (state, action) pairs from a scripted hover → descend → close expert with `env.auto_weld = False` and a real two-finger contact gate.
2. **Training** (`src/training/train_bc_pickup.py`): MSE regression, val loss ~ 0.038, 150 epochs on MPS.
3. **Integration**: primary pickup in `demo_phase2_hybrid.py`. BC drives approach + descend + close; once contact is verified, a scripted IK lift completes the pickup. Scripted finalizer remains as fallback if BC fails to achieve contact.

### Delivery BC (trained, not used in the demo)

Scripted diff-drive controller distilled into a small `DeliveryDrivePolicy`:
1. **Collection** (`src/data/collect_delivery_bc_demos.py`): 400 successful episodes, 146 k (state, action) pairs. State = target position in Husky body frame + heading sin/cos (5 dims). Action = (left wheel vel, right wheel vel), normalized by 15 rad/s.
2. **Training** (`src/training/train_bc_delivery.py`): MSE regression, val loss ~ 0.0005, 80 epochs on MPS.
3. **Standalone eval** (`src/eval/eval_delivery_bc.py`): 60 % success reaching random targets from random starts in 30 eval episodes.
4. **Outcome**: BC policy learned the expert's straight-line driving but not obstacle avoidance (since the scripted expert itself has none). When used as the primary delivery driver it occasionally drove the Husky into shelf sides. Kept as an opt-in alternative via `PHASE2_DELIVERY_USE_BC=1`, not the default.

### A* Path Planner (the primary delivery system)

After observing that straight-line driving (scripted or BC-distilled) was fragile for random dropoffs, the delivery leg was upgraded to classical obstacle-aware planning:
1. **Planner** (`src/planning/astar_delivery.py`): 2-D occupancy grid over the workspace (35 × 30 cells at 0.2 m). Shelves from the env config are inflated by the Husky half-width + margin and marked as blocked. A\* with 8-connected neighbors + Euclidean heuristic.
2. **Smoothing**: line-of-sight smoothing reduces a staircase path (typically 10+ cells) to 2-4 meaningful waypoints along the safe aisle.
3. **Follower** (inside `demo_phase2_hybrid.py`): pure-pursuit-style diff-drive controller. Skips the path's first cell if the Husky is already there, does a brief turn-in-place to face the first real target (prevents arcing into shelves), then cruises continuously toward a lookahead "carrot" on the path at constant speed with continuous steering correction. Only the final waypoint triggers a full brake + yaw alignment for the arm lower phase.

Result: smooth, single-motion delivery to any dropoff in the randomized `x ∈ [-1, 1]`, `y ∈ [1.5, 2.7]` range, with observed dropoff distances of 0.00-0.06 m in a 5-episode GUI run.

## 8. Files That Matter Right Now

- `src/env/warehouse_env_mobile_v2.py` — repaired simulator, pickup primitives, auto_weld flag
- `src/training/train_phase2_nav_pickpose.py` — PPO nav trainer, `NavPolicy`, state/action helpers
- `src/training/train_phase2_pick_cartesian.py` — original PPO pickup trainer, `MobileCartesianPickPolicy`, `MobilePickupExpert`
- `src/training/train_bc_pickup.py` — BC trainer for pickup (distills the scripted grasp finalizer)
- `src/training/train_bc_delivery.py` — BC trainer for delivery nav (distills the scripted diff-drive controller)
- `src/data/collect_pickup_bc_demos.py` — headless pickup demo collector
- `src/data/collect_delivery_bc_demos.py` — headless delivery demo collector
- `src/eval/demo_phase2_hybrid.py` — final demo (learned nav + BC pickup + contact weld + scripted lift + BC delivery + scripted place/release)
- `src/eval/eval_delivery_bc.py` — standalone headless eval for the delivery BC policy
- `FINAL_PROJECT_SCOPE.md`, `FINAL_REPORT_DRAFT.md`, `PRESENTATION_OUTLINE.md` — packaging docs

## 9. Commands Worth Keeping

**Run the hybrid demo:**
```bash
python src/eval/demo_phase2_hybrid.py
```

**Collect pickup BC demos (headless):**
```bash
PHASE2_BC_NUM_EPISODES=300 python src/data/collect_pickup_bc_demos.py
```

**Train pickup BC:**
```bash
python src/training/train_bc_pickup.py
```

**Collect delivery BC demos (headless):**
```bash
PHASE2_DELIVERY_BC_NUM_EPISODES=400 python src/data/collect_delivery_bc_demos.py
```

**Train delivery BC:**
```bash
PHASE2_DELIVERY_BC_EPOCHS=80 python src/training/train_bc_delivery.py
```

**Eval delivery BC (headless, 30 episodes):**
```bash
PHASE2_DELIVERY_BC_EVAL_EPISODES=30 python src/eval/eval_delivery_bc.py
```

**Run the demo (defaults to BC pickup + BC delivery with scripted fallbacks):**
```bash
python src/eval/demo_phase2_hybrid.py
```

**Force scripted-only delivery (disable BC):**
```bash
PHASE2_DELIVERY_BC_CKPT=/does/not/exist python src/eval/demo_phase2_hybrid.py
```

## 10. Remaining Work (in priority order)

1. Run BC collection + training in background.
2. Integrate BC pickup into demo with scripted fallback.
3. Final GUI run with BC pickup; if successful, this becomes the recorded artifact.
4. Record demo video + screenshots.
5. Update `.pptx` with current results wording + recorded visuals.
6. Finalize report from `FINAL_REPORT_DRAFT.md`.

## 11. Honest Limitations (must remain in submission)

- The final demo is a hybrid pipeline, not end-to-end RL.
- The pickup module was distilled from a scripted expert (BC), not learned from scratch via RL in a single stage.
- Object carry uses teleport-based rigid pinning for visual stability (equivalent in spirit to a perfectly rigid grasp; the weld is physical but supplemented to eliminate teleport-induced constraint lag).
- Delivery path is a scripted 5-phase waypoint sequence, not learned.
- The release + retreat is scripted.

These limitations are consistent with the scope doc's "hybrid learned + deterministic" framing.

## 12. Bottom Line

The demo reliably shows instruction-conditioned mobile pickup and delivery at 5/5 success. The repaired simulator, learned nav, and (post-BC) learned pickup are all genuine artifacts. The remaining scripted components are the standard glue layer in modular robotics pipelines — not hidden hacks.
