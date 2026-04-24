# Final Project Scope

## Title
Instruction-Conditioned Warehouse Manipulation in PyBullet Using a Hybrid Learned + Deterministic Control Pipeline

## Final Goal
Build a simulation system that can take an instruction such as `pick up the red box`, identify the correct target, move the robot smoothly into a valid pre-grasp pose, and execute a stable pick-and-lift routine.

This project is presented in two phases:

- Phase 1: fixed-base Panda arm tabletop manipulation
- Phase 2: mobile Husky + Panda manipulation with navigation before pickup

## Final Technical Position
The final system is **not** framed as pure end-to-end RL solving contact-rich grasping from scratch.

The final system is framed as a **hybrid pipeline**:

- Learned components:
  - mobile navigation to pickup pose
  - pickup from constrained shelf-front states
- Deterministic components:
  - exact module handoff / transition logic
  - bounded lift completion once grasp is already established

This reflects the actual engineering lesson from the experiments: full end-to-end RL was unstable in this setup, while staged control was more reliable and interpretable.

## What We Will Claim
- We implemented instruction-conditioned warehouse manipulation in simulation.
- We explored BC, PPO, state-only RL, and hybrid control approaches.
- We found that decomposition into learned navigation, learned pickup, and hybrid completion logic produced the strongest results.
- Phase 2 is the main final showcase.
- The strongest final result is a working hybrid mobile pickup demo with instruction-conditioned shelf approach and pickup.
- Delivery remains future work unless additional stable evidence is produced.

## What We Will Not Claim
- Robust fully learned grasping from scratch.
- Stable end-to-end mobile pick-and-deliver unless final evidence supports it.
- Full RT-2 or OpenVLA reproduction.

## Final Architecture

### Phase 1 (supporting showcase)
1. Parse instruction for target color (deterministic keyword parser).
2. Vision confirms target in camera view (classical color segmentation).
3. Scripted grasp finalizer: hover → descend → close → verify real two-finger contact → weld at live relative pose → lift.

### Phase 2 (main showcase)
1. Parse instruction for target color (language grounding).
2. **Learned navigation** (`phase2_nav_bc_best.pth`, BC on randomized spawns) drives the Husky from any center-of-aisle spawn to the shelf-front pickup pose.
3. Vision confirms target at the shelf (camera + color segmentation).
4. Scripted joint interpolation bridges to the pickup-ready arm pose.
5. **Learned pickup** (`phase2_pick_bc_best.pth`, BC distilled from a contact-verified scripted expert) drives the arm into a real two-finger contact grasp.
6. Contact-gated weld at the live relative pose + scripted IK lift.
7. **A\* path planning** over a Husky-clearance-inflated occupancy grid + line-of-sight smoothing computes an obstacle-aware route.
8. Pure-pursuit diff-drive with real PyBullet wheel physics follows the path smoothly to the dropoff approach.
9. Scripted arm lower + retreat release deposits the box at the randomized dropoff.

Two load-bearing neural networks (nav BC, pickup BC) + one classical AI planner (A\*) + classical vision and language. Real end-to-end learning was attempted (VLA, PPO on both subtasks) and is documented as negative results in `EXPERIMENTS.md`.

## Strongest Deliverables
- A Phase 2 demo showing instruction-conditioned navigation to the correct shelf, contact-verified pickup, and obstacle-aware delivery to a randomized drop zone, scoring **Pickup 5/5, Delivery 5/5** with 0-6 cm drop accuracy.
- A Phase 1 hybrid demo with the same perception/language/grasp pattern scoring **Pickup 5/5**.
- A repaired mobile simulator with valid shelves, walls, collision-aware base motion, and a correct contact-gated grasp weld.
- A full ablation / negative-result catalog (`EXPERIMENTS.md`): VLA attempts, joint-space PPO, Cartesian BC, BC+PPO at two subtasks, BC delivery — explaining why the final hybrid won.

## Submission Strategy
- Use Phase 2 as the primary showcase.
- Use Phase 1 as supporting context / exploratory work.
- Present the project as an engineering and research investigation into what works for instruction-conditioned manipulation under limited time and compute.

## Remaining Build Priorities
1. Freeze the current working Phase 2 hybrid demo candidate.
2. Record clean demo videos and screenshots immediately.
3. Align the report and presentation with the final Phase 2 hybrid result.
4. Treat delivery only as a stretch demo extension, not a core claim.
