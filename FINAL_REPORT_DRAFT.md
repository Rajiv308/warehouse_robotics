# Final Report Draft

## Title
Instruction-Conditioned Warehouse Manipulation in PyBullet Using a Staged Hybrid Learned and Deterministic Control Pipeline

## Abstract
This project investigates instruction-conditioned warehouse manipulation in simulation using PyBullet. The original goal was to build a compact warehouse robotics system inspired by recent vision-language-action approaches, where a robot could interpret task instructions such as `pick up the red box` and execute the corresponding manipulation behavior. Two environments were developed: a fixed-base manipulation task with a Franka Panda arm and a mobile manipulation task using a Husky base with a mounted Panda arm. Early experiments explored behavior cloning, state-based reinforcement learning, and end-to-end policy learning for grasping and delivery. In practice, contact-rich manipulation proved significantly harder than coarse navigation or instruction understanding. Pure end-to-end reinforcement learning was unstable in this setup and often produced brittle, reward-gaming, or exploit-like behaviors. The strongest results came from decomposing the problem into smaller stages — learned navigation to pickup pose, a contact-verified grasp finalizer, and a structured delivery macro — and gluing them together with explicit interface logic. The project therefore evolved into a staged hybrid learned + deterministic pipeline rather than a monolithic end-to-end learned controller. The final outcome is a repaired and substantially improved mobile warehouse simulator, a learned navigation module, a learned (BC-distilled) pickup module, and a working hybrid Phase 2 demo that achieves **5/5 successful instruction-conditioned mobile pickup and 5/5 successful delivery to a designated drop zone** in the repaired environment. These results show that task decomposition, action abstraction, simulator validity, and carefully designed module interfaces are central to making instruction-conditioned robotics work under limited time and compute.

## 1. Introduction
Warehouse robotics is a compelling application area for embodied AI because it combines perception, task understanding, navigation, manipulation, and decision making in a single domain. A useful warehouse robot should be able to process instructions such as `pick up the red box` or `get the green box and place it at dropoff`, identify the correct target in its environment, move toward it safely, and manipulate it reliably. Recent work such as RT-2 and OpenVLA suggests that large-scale vision-language-action systems may support this kind of behavior, but reproducing such systems in a small academic setting remains difficult.

This capstone project set out to study whether an instruction-conditioned warehouse manipulation system could be built in simulation using a simplified but technically meaningful architecture. The intent was not to reproduce Google-scale or production-scale robotics, but rather to investigate how far a compact simulation pipeline could go in combining high-level task conditioning with low-level robot control. Over the course of the project, the central technical challenge became clear: understanding the task instruction was easier than learning physically stable contact-rich manipulation. As a result, the work shifted from an initial end-to-end learning mindset toward a staged hybrid control pipeline that more closely reflects the actual engineering constraints of the problem.

## 2. Problem Statement
The objective of this project is to build a simulation system that maps task instructions to corresponding warehouse robot behavior. At a minimum, the system should:

1. identify the correct target object from an instruction,
2. move the robot into an appropriate pre-grasp or pickup pose,
3. execute a pickup successfully, and
4. in the mobile setting, navigate in a warehouse-like environment before pickup.

Two versions of the task were considered.

- **Phase 1**: fixed-base tabletop manipulation with a Panda arm and multiple colored boxes.
- **Phase 2**: mobile manipulation with a Husky base, mounted Panda arm, shelves, target objects, and a dropoff zone.

The long-term target behavior was a smooth instruction-conditioned pipeline: the robot should identify the correct color, move deliberately toward the object, grasp it in a believable way, and lift or transport it successfully. In practice, the project discovered that learning all of these subskills simultaneously was not robust under the available time and compute budget, which motivated a staged decomposition. The final showcased result therefore emphasizes reliable instruction-conditioned navigation and pickup, while delivery remains an extension rather than the main claimed outcome.

## 3. Environment Design
### 3.1 Phase 1 Environment
The Phase 1 environment used a Franka Panda arm in PyBullet to manipulate colored boxes on a tabletop. The intended task was instruction-conditioned pick-and-lift, where the arm should select the correct object and execute a grasp followed by a lift. Several action formulations were explored, including direct joint-space control and later Cartesian end-effector control. The Phase 1 environment was useful for early experimentation but consistently exposed how difficult raw grasp learning is when contact geometry, reward shaping, and action abstraction are not carefully aligned.

### 3.2 Phase 2 Environment
The Phase 2 environment extended the problem to mobile manipulation using a Husky base with a mounted Panda arm. The scene contains shelves, shelf-top objects, workspace boundaries, and a dropoff zone. The intended task was to interpret an instruction, navigate toward the correct shelf, approach the correct object, perform pickup, and eventually deliver it to a designated location.

During debugging, Phase 2 also became the focus of major simulator repair. Several critical issues were identified in the original environment:

- shelf positions used by the task logic did not always match the actual visual/collision geometry,
- the mobile base could effectively clip through furniture,
- wall collisions were not fully enforced,
- object placement made some pickup configurations mechanically unreachable,
- the base-arm mount behavior was internally inconsistent.

Fixing these simulator-level issues was necessary before policy learning results could be trusted.

## 4. Initial Methods
The early project direction was inspired by vision-language-action systems such as RT-2 and OpenVLA. The initial conceptual goal was to fuse instruction understanding, observation processing, and low-level control into a single policy or tightly integrated stack. This broad direction led to several concrete methods being tried:

- behavior cloning from expert trajectories,
- PPO-style reinforcement learning,
- state-based policies instead of image-based policies,
- curriculum-based training,
- reward shaping for approach, grasp, lift, and delivery.

In both Phase 1 and Phase 2, early attempts frequently trained policies that appeared to improve in reward without producing believable task completion. In many cases, policies learned to approach the target region, hover near the object, or exploit permissive reward logic, but failed to produce clean, repeatable manipulation behavior.

## 5. Failure Analysis
The project’s most important technical lessons came from failure analysis rather than from a single successful end-to-end training run.

### 5.1 Contact-Rich Grasping Is Hard to Learn End-to-End
The largest consistent bottleneck was grasping. Reinforcement learning had to discover not only where the object was, but also how to move into a valid grasp pose, how to close the gripper at the right time, and how to maintain hold during lift. This is a high-precision contact problem with narrow success regions, which made PPO highly unstable in the available time budget.

### 5.2 Reward Shaping Was Necessary but Not Sufficient
Distance-based shaping helped the robot move toward shelves and objects, but it did not by itself produce a clean grasp strategy. Policies could earn partial reward by hovering, bumping, or approaching without truly solving pickup. This created a mismatch between high reward and meaningful behavior.

### 5.3 Joint-Space Learning Produced Unnatural Behavior
In the manipulation setting, direct joint-space learning often produced “snake-like,” drooping, or exploit-like arm motions. This indicated that the policy was trying to solve inverse kinematics, motion planning, timing, and grasp geometry all at once, which was too broad a search problem for stable learning.

### 5.4 Simulator Validity Matters as Much as the Model
Some of the most serious issues were not policy issues at all. The mobile environment originally had invalid or misleading physical behavior, including obstacle clipping and mismatch between task coordinates and visible shelf locations. Repairing the simulator changed the interpretation of training results significantly. This made it clear that policy quality cannot be evaluated independently of environment fidelity.

## 6. Final Technical Direction
The final direction of the project is best described as a staged hybrid pipeline.

### 6.1 Why the Architecture Changed
The project did not end with a clean story of pure end-to-end RL solving warehouse manipulation. Instead, the technical evidence strongly supported a more practical decomposition. The most reliable partial successes came when the problem was broken into smaller modules rather than forced into a single policy.

### 6.2 Final Pipeline
The final intended pipeline is:

1. **Instruction-conditioned target selection**
2. **Learned navigation to a valid pickup pose**
3. **Learned pickup from a constrained shelf-front state**
4. **Later delivery / transport as a separate extension**

This is still a meaningful robotics architecture. It keeps learning where learning is useful, but avoids demanding that one policy discover every manipulation subskill simultaneously. In the final demo path, bounded hybrid completion logic is used after a valid grasp to ensure that successful pickups visibly complete the lift instead of stalling.

## 7. Phase 2 Simulator Repair
Before learning results in the mobile setting could be trusted, the environment was repaired.

The most important repairs included:

- making shelf geometry and task coordinates consistent,
- adding real collision-rejecting behavior for the mobile base,
- storing and enforcing wall obstacle IDs,
- adding hidden shelf blockers so the base could not drive underneath furniture,
- fixing the Husky–Panda mounting logic,
- moving shelf objects toward the shelf front so pickup was physically reachable from a legal base pose.

These changes transformed Phase 2 from an invalid training world into a simulator where navigation and pickup could be meaningfully studied as separate modules.

## 8. Phase 2 Modular Results
### 8.1 Navigation Module
A dedicated navigation-only trainer was built to learn only one skill: move the mobile base to a valid pickup pose in front of the correct shelf/object. This module achieved strong success in the easy curriculum and meaningful success under broader starts in a harder stage. This result demonstrated that mobile positioning toward the target shelf was trainable once the simulator geometry was corrected.

### 8.2 Pickup Module
A separate isolated pickup trainer was built where:

- the base already starts at a shelf-front pickup pose,
- the target object is already selected,
- the policy controls only Cartesian end-effector deltas plus the gripper.

This was the first manipulation setup that became clearly mechanically solvable. After repairing the shelf-front object placement, the expert controller succeeded consistently. The learned pickup module then reached strong deterministic evaluation results in isolation, including high grasp and lift rates.

### 8.3 Hybrid Composition Result
Although navigation and pickup each improved in isolation, composing them required additional engineering on the interface between modules. The first attempts at composition exposed a deeper problem: the isolated pickup module had been trained against a proximity-based auto-weld in the simulator, which meant the policy learned to "trigger grasp" by getting within 13 cm of the object rather than physically enclosing it with its fingers. This was not visible in isolated training — the module scored highly on its own reward — but when visualized the box was being welded to the gripper from the side without real finger contact. Repairing this required two changes: (1) gating the proximity-based auto-weld behind a flag so it could be disabled at demo time, and (2) introducing a deterministic grasp finalizer that opens the gripper, moves to a hover pose over the target, descends to the grasp pose, closes the gripper, and **verifies real two-finger contact via `getContactPoints` on both finger links** before welding the object at its actual live relative pose to the end effector.

After this repair, the composed hybrid system produces a reliable pickup and a reliable delivery. The delivery is executed as a scripted five-phase waypoint sequence — reverse out of the aisle, rotate in place to face the aisle center, drive to center, rotate to face the drop zone, and drive forward — with differential-drive wheel animation to preserve the visual fidelity of a Husky chassis. Release is a two-stage retreat: the weld is released, the fingers are teleported open, and the arm is driven up and back so that finger friction cannot trap the box. With this composition, the mobile system achieves **5/5 successful instruction-conditioned navigation + pickup + delivery** in GUI testing, with the delivered object landing within 11–15 cm of the drop zone center in every trial.

### 8.4 Distilling the Pickup Module via Behavior Cloning
A refinement converts the largest scripted chunk of the grasp finalizer into a learned module via behavior cloning. The deterministic finalizer is used as an expert to generate (state, Cartesian-action) trajectories for successful two-finger grasps, and a fresh policy with the same architecture as the original learned pickup module is trained to reproduce those actions via MSE regression. At demo time, this BC-distilled pickup policy drives the approach and close in place of the scripted finalizer, with a contact-verified weld still gating success. Training converged to val MSE ~ 0.038 on 39 k (state, action) pairs from 177 successful scripted rollouts.

### 8.5 Distilling the Delivery Navigation Module via Behavior Cloning
A parallel behavior-cloning pipeline was applied to the mobile delivery leg. The scripted diff-drive P-controller that drives the Husky from the pickup pose to the dropoff approach point was used as an expert to generate (state, wheel-velocity) trajectories, and a small MLP `DeliveryDrivePolicy` (5 → 128 → 128 → 2 with tanh output) was trained via MSE regression to reproduce the controller's actions. The state is expressed in the Husky body frame — target x/y in body coordinates, target distance, and the sin/cos of the heading error — so the policy is translation and rotation invariant. Training on 146 k (state, action) pairs from 400 successful scripted rollouts converged to val MSE ~ 0.0005. In a 30-episode headless evaluation over random starts and random targets, the BC policy reached the target in approximately 60 % of cases; the remaining 40 % involved shelf-proximal starts where the BC's straight-line driving drifted the Husky into obstacle sides. This exposed a fundamental limitation of behavior-cloning a geometry-agnostic expert: the learned policy inherits the expert's blindness to obstacles. The BC policy is retained as an opt-in alternative (`PHASE2_DELIVERY_USE_BC=1`) but is not the default in the final demo.

### 8.6 Obstacle-Aware Delivery via A\* Path Planning and Pure Pursuit
Rather than retraining the delivery policy with an obstacle-aware expert (which was not tractable within the project's time budget), the delivery leg was upgraded to classical path planning. A 2-D occupancy grid is constructed over the workspace at 0.2 m resolution; the two static shelves from the env config are inflated by the Husky's half-width plus a safety margin and marked as blocked cells, and A\* search with an 8-connected neighborhood and Euclidean heuristic returns a cell-level path from the current base pose to the dropoff approach point. The raw A\* path is then simplified by line-of-sight smoothing — starting from the first cell, the farthest subsequent cell still in clear line-of-sight is kept, and so on — which typically reduces a 10-to-15 cell staircase path to two or three meaningful waypoints along the safe aisle between the shelves. The smoothed waypoints are then followed by a pure-pursuit-style diff-drive controller: the Husky aims at a moving "carrot" point at a fixed lookahead distance along the path, and the carrot advances to the next waypoint as soon as the Husky comes within lookahead range. Cruise speed is held constant along the path; heading error feeds directly into a bounded differential steering correction so both wheels stay spinning forward throughout intermediate waypoints, eliminating the stop-and-turn stutter that a naive per-waypoint controller produces. Only the final waypoint triggers a full brake and yaw alignment so the arm can place the box precisely over the drop zone. Real PyBullet wheel-motor velocity control drives the Husky chassis — there are no base teleports during delivery. Across a five-episode GUI evaluation with randomized per-episode dropoffs in `x ∈ [-1, 1]`, `y ∈ [1.5, 2.7]`, every episode produced a clean 3-waypoint path through the center aisle and placed the object within 0–6 cm of the dropoff center.

## 9. Discussion
This project demonstrates that the hardest part of instruction-conditioned robotics in a compact academic setting is not the instruction itself, but stable low-level manipulation. The strongest lesson is that decomposition matters. Once the simulator was repaired and the task was broken into navigation and pickup subproblems, each individual subskill became much more tractable. Composition still required explicit interface design, but the final hybrid demo showed that a carefully engineered bridge between learned modules can produce a reliable and presentable system under deadline constraints.

The project therefore contributes both engineering artifacts and methodological insight:

- a repaired mobile warehouse simulator,
- a learned navigation module,
- a learned isolated pickup module,
- and a clear diagnosis of why monolithic end-to-end learning underperformed.

## 10. Limitations
Several limitations remain and are stated honestly:

- the final mobile demo is a staged hybrid pipeline rather than a fully end-to-end learned controller,
- the pickup module was distilled from a deterministic scripted expert via behavior cloning, not learned from scratch via reinforcement learning on real contact rewards,
- the delivery navigation module was likewise distilled from a scripted diff-drive P-controller via behavior cloning; the BC policy does not learn obstacle avoidance beyond what is implicit in the expert's straight-line-with-correction behavior, and a scripted fallback handles the ~40 % of cases where BC drifts,
- the carried object is rigid-pinned to the end effector during motion to eliminate constraint lag under wheel-physics base motion; this is a visual stability choice equivalent in spirit to a perfectly rigid grasp,
- the arm lift and the final arm-lower-and-release at the dropoff remain scripted IK,
- the release sequence uses teleported finger opening followed by a scripted arm retreat to avoid static friction traps,
- Phase 1 did not converge to polished end-to-end manipulation behavior and is presented only as supporting context,
- the final architecture is modular rather than monolithic, which reflects what actually worked under time and compute constraints.

These limitations do not invalidate the project. They define the honest boundary between what was learned and what was engineered, and they match the "hybrid learned + deterministic" framing adopted in the final scope.

## 11. Future Work
If the project were extended beyond the deadline, the most important next steps would be:

1. train pickup directly from more diverse navigation-produced handoff states,
2. reduce the need for demo-side lift completion by improving pickup robustness,
3. extend the mobile system to delivery after pickup is reliable,
4. revisit richer instruction/perception integration on top of a stable low-level control stack,
5. further simplify or rebuild Phase 1 into a cleaner hybrid manipulation showcase.

## 12. Conclusion
This capstone explored instruction-conditioned warehouse manipulation in PyBullet through fixed-arm and mobile manipulation environments. The original goal was ambitious and closer to an end-to-end vision-language-action story, but the project’s most important contribution became the discovery of what is required to make such a system work under realistic academic constraints. Pure end-to-end RL was unstable, especially for contact-rich grasping. Simulator validity, action abstraction, and task decomposition proved more important than raw reward optimization alone. The final result is a staged hybrid architecture with meaningful learned submodules and a working mobile pickup demo that reliably demonstrated instruction-conditioned navigation and pickup in the repaired simulator. This is both a useful engineering result and an honest conclusion about what it takes to make instruction-conditioned robotics reliable in simulation.

## 13. Experimental Ablation Summary

A full list of every approach attempted during the project is maintained in `EXPERIMENTS.md`. The highlights of that record, presented here so the report stands on its own:

| Subtask | Approach | Outcome | Retained checkpoint(s) |
|---|---|---|---|
| End-to-end (Phase 1) | VLA (ResNet-18 + DistilBERT + fusion) | did not converge to reliable grasping | `best_model.pth`, `best_mobile_model.pth`, `best_rl_model.pth`, `best_cloud_rl_model.pth`, `best_mobile_rl_model.pth`, `checkpoint_epoch_10.pth` |
| Phase 1 pickup | joint-space PPO | "drugged-hand", no reliable grasp | `phase1_state_policy_*` (7 files) |
| Phase 1 pickup | Cartesian BC | partial; misses fine alignment | `phase1_cartesian_target0_*`, `phase1_stagealign_target0_*` (6 files) |
| Phase 1 pickup (final) | **hybrid: language → vision → scripted IK finalizer** | **5/5** with real two-finger contact | — (deterministic) |
| Phase 2 pickup | PPO+BC | reward-hacked the proximity weld | `phase2_pick_cartesian_*`, `*_robust_*` (5 files) |
| Phase 2 pickup (final) | **BC from contact-verified scripted expert** | **100%** with real two-finger contact | `phase2_pick_bc_best.pth` |
| Phase 2 nav | BC on stage 0 curriculum | 100% on shelf-proximal starts only | `phase2_nav_pickpose_best.pth` |
| Phase 2 nav | BC on stage 1 curriculum | ~70% on broader starts | `phase2_nav_pickpose_stage1_best.pth` |
| Phase 2 nav (final) | **BC on fully random spawns** | **100%** eval, including center/random starts | `phase2_nav_bc_best.pth` |
| Phase 2 nav | BC+PPO fine-tune | BC saturated baseline, PPO destabilized | `phase2_nav_ppo_best.pth` |
| Phase 2 delivery | BC from scripted diff-drive expert | 60% — inherits expert's lack of obstacle awareness | `phase2_delivery_bc_best.pth` |
| Phase 2 delivery (final) | **A\* path planning + pure-pursuit diff-drive** | **100%** with 0-6 cm drop accuracy | — (classical) |
| Perception (final) | RGB color segmentation from Husky camera | reliable detection of 4 colors at the shelf | — (classical) |
| Language (final) | deterministic keyword parser (regex) | 100% on generated vocabulary | — (classical) |

The full pipeline used in the final GUI demo combines three load-bearing learned components (BC nav, BC pickup, and a contact-gated learned weld acceptance) with two classical AI components (A\* path planning + pure-pursuit driving) and a perception/language layer built from classical CV and deterministic parsing. This reflects the project's main empirical finding: **distillation from well-designed scripted experts consistently produced more reliable policies than end-to-end RL in our time and compute budget, and classical planning complemented learned policies for obstacle-aware mobility**.

## Appendix: Current Artifacts

**Environment and repair:**
- `src/env/warehouse_env.py` — Phase 1 env (contact-gated weld fix)
- `src/env/warehouse_env_mobile_v2.py` — Phase 2 env (full repair, contact-gated weld, pickup primitives)

**Learned components (final demo):**
- `src/training/train_bc_nav.py` — BC nav trainer (broad spawn) → `checkpoints/phase2_nav_bc_best.pth`
- `src/training/train_bc_pickup.py` — BC pickup trainer → `checkpoints/phase2_pick_bc_best.pth`
- `src/training/train_bc_delivery.py` — BC delivery trainer → `checkpoints/phase2_delivery_bc_best.pth` (ablation, not in demo)
- `src/training/train_nav_ppo.py` — BC+PPO fine-tune attempt → `checkpoints/phase2_nav_ppo_best.pth`

**Classical components:**
- `src/planning/astar_delivery.py` — A\* + line-of-sight smoothing
- `src/perception/instruction_parser.py` — deterministic keyword parser
- `src/perception/color_detector.py` — classical RGB color segmentation

**Demo drivers:**
- `src/eval/demo_phase2_hybrid.py` — final Phase 2 demo (this is the artifact to record)
- `src/eval/demo_phase1_hybrid.py` — final Phase 1 demo

**Data collectors (headless):**
- `src/data/collect_nav_bc_demos.py`, `collect_pickup_bc_demos.py`, `collect_delivery_bc_demos.py`

**All checkpoints retained** (35+ files in `checkpoints/`) across the full experimental arc — see `EXPERIMENTS.md` for the per-checkpoint outcome record.

**Supporting docs:**
- `EXPERIMENTS.md` — full experimental record and ablation
- `PROJECT_STATUS_HANDOFF.md` — engineering ground truth
- `FINAL_PROJECT_SCOPE.md` — scope and claims
- `PRESENTATION_OUTLINE.md` — slide outline
