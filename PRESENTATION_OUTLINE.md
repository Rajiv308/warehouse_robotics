# Presentation Outline

## Slide 1: Title
Instruction-Conditioned Warehouse Manipulation in PyBullet  
Rajiv Menon  
Capstone Project

## Slide 2: Motivation
- Warehouse robots need to act on task instructions such as `pick up the red box`.
- Vision-language-action systems are promising, but reliable control remains difficult.
- This project studies how to make instruction-conditioned manipulation work in simulation.

## Slide 3: Project Goal
- Build a simulation pipeline that maps language-like instructions to robot behavior.
- Phase 1: fixed-arm tabletop pickup.
- Phase 2: mobile manipulation with navigation to the correct shelf and reliable pickup of the correct object.

## Slide 4: Initial Approach
- Inspired by RT-2 / OpenVLA style ideas.
- Explored behavior cloning and PPO-based RL.
- Attempted end-to-end learned control for grasping and manipulation.

## Slide 5: Core Challenge
- Contact-rich grasping was unstable under pure RL in this setup.
- Reward shaping alone was not enough.
- Joint-space policies produced unnatural, exploit-like behaviors.

## Slide 6: Key Insight
- Full end-to-end learning was too hard for this environment and deadline.
- A hybrid decomposition worked better:
  - learn navigation and pickup subskills
  - use bounded deterministic assistance only where needed for reliable lift completion

## Slide 7: Final System
### Phase 1
- target selection
- learned pre-grasp alignment
- deterministic descend-close-lift

### Phase 2 (main showcase)
- learned navigation to pickup pose (PPO policy)
- BC-distilled pickup from shelf-front states (expert: a contact-verified grasp finalizer)
- contact-verified weld: attach only when both finger links register real `getContactPoints`
- **A\* path planning over a Husky-clearance-inflated occupancy grid** for obstacle-aware delivery routing
- line-of-sight path smoothing + **pure-pursuit diff-drive follower** for smooth single-motion driving
- wheel-physics driving: real Husky motor torques, not teleport
- randomized dropoff per episode (x ∈ [-1, 1], y ∈ [1.5, 2.7]); observed dropoff distance 0–6 cm
- retreat-based release to avoid static-friction finger trap

## Slide 8: Phase 1 Environment
- Panda arm in PyBullet
- three colored boxes
- instruction-conditioned target selection
- focus on correct-object pick-and-lift

## Slide 9: Phase 2 Environment
- Husky + Panda in PyBullet
- shelves, target objects, dropoff zone
- staged curriculum from easy pickup to harder mobile starts

## Slide 10: Experimental Findings
- Pure RL from scratch underperformed for contact-rich grasping and out-of-distribution navigation.
- Simulator validity mattered before learning results could be trusted (grasp-weld bug, shelf collisions, object placement).
- Pickup became trainable only after decomposition into navigation and isolated pickup modules.
- Deterministic evaluation (contact verification) mattered more than raw reward.
- Action abstraction strongly affected behavior quality (joint-space policies unstable, Cartesian easier).
- BC distillation from a verified scripted expert consistently beat PPO when the expert was near-optimal.

## Slide 11: Ablation / Experimental Record
- **Attempted**: end-to-end VLA (ResNet-18 + DistilBERT fusion), joint-space PPO, Cartesian BC, BC+PPO fine-tune, BC-only, A\* planning, classical vision, keyword parser.
- **VLA** (~300–550 MB checkpoints): no reliable grasp. Documented.
- **Joint-space PPO** (Phase 1): drugged-hand behavior. Documented.
- **BC+PPO pickup** (Phase 2): reward-hacked the env's proximity weld. Motivated the env fix.
- **BC+PPO nav** (Phase 2): BC already saturated the task; PPO destabilized. Classic RL-vs-imitation finding.
- **BC delivery**: inherited expert's lack of obstacle awareness. Shows BC's limitation when the expert is incomplete.
- **Final pipeline**: 2 learned BC policies + A\* classical planner + contact-gated weld + classical vision/language.
- All 35+ checkpoints retained for `EXPERIMENTS.md` comparison.

## Slide 12: Results
- Phase 1 hybrid demo: **Pickup 5/5** with real two-finger contact every episode.
- Phase 2 end-to-end demo: **Pickup 5/5, Delivery 5/5** across randomized dropoffs.
- Delivery placement: **0–6 cm** from the drop zone center in every trial.
- Husky spawns center-of-aisle and the learned BC nav drives it to the correct shelf in ~100 steps, then BC pickup closes on real contact, then A\* plans around the shelves, then pure-pursuit diff-drive reaches the dropoff approach, then scripted IK places.
- Best results came from decomposition into learned navigation/pickup + classical geometric planning for mobility, not from monolithic end-to-end RL.

## Slide 13: Lessons Learned
- Simulator validity is a prerequisite for meaningful learning: bugs in grasp-detection or collision masks silently corrupt any RL reward signal.
- BC distillation from a verified scripted expert is often a shorter path than PPO, especially when the expert is near-optimal.
- When a BC baseline already saturates the task metric, PPO fine-tune has no gradient signal and can only destabilize the policy.
- Classical planning (A\*) beats a BC-learned planner when the BC expert is geometry-agnostic — geometry knowledge has to come from somewhere.
- Small closed-vocabulary language grounding does not need an LM; a keyword parser suffices and is easier to verify.
- Joint-space learning is harder than Cartesian learning for contact-rich pick-and-place.
- Staging and curriculum are critical for embodied AI tasks.

## Slide 14: Future Work
- Reintroduce a neural VLA layer (e.g. CLIP for zero-shot visual grounding) on top of the stable low-level control stack.
- Train a proper RL nav with obstacle penalties in the reward so the learned policy competes with A\*.
- Train pickup on nav-produced handoff states to bridge the imitation-to-execution distribution gap.
- Extend to multi-step task chaining ("pick red, deliver, pick blue, deliver").
- Randomized obstacles so A\* replans dynamically.

## Slide 15: Conclusion
- Implemented an instruction-conditioned warehouse manipulation system in simulation from a natural-language instruction through delivery.
- Identified the limits of pure end-to-end RL in this setup, with specific negative results for VLA, joint-space PPO, and BC+PPO fine-tuning over a saturated baseline.
- Repaired the mobile simulator (contact-gated weld, collision-aware base, reachable object placement) and developed a hybrid mobile pipeline with two load-bearing BC policies (navigation + pickup), one classical AI planner (A\*), and a classical perception+language layer.
- Final end-to-end Phase 2 demo: **Pickup 5/5, Delivery 5/5** with 0-6 cm drop accuracy across randomized dropoffs. Phase 1 hybrid: **Pickup 5/5**.

## Short Speaker Notes
- Emphasize the engineering lesson, not just the final number.
- Be explicit that the final project is about making the task work reliably, not forcing a pure RL story.
- Show the working Phase 2 hybrid pickup result early if possible.
