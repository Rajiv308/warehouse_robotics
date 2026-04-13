# VLA Warehouse Robotics — Claude Code Context Document

**Project:** CS5100 Foundations of AI Capstone, Northeastern University, Spring 2026
**Student:** Rajiv Premnath Menon (solo)
**Instructor:** Jonathan Mwaura
**Deadline:** ~April 20, 2026 (final presentation + report). **We have ~2 days to produce a demonstrable result.**
**Repo:** `Rajiv308/warehouse_robotics`
**Local machine:** MacBook M2 Pro
**Cloud:** Vast.ai (used previously; may use again if needed)
**Project root:** `~/Documents/Projects/masters_neu/CS_5100_FAI/Capstone/warehouse_robotics`

---

## 1. What This Project Is

A scaled-down implementation of Google's RT-2 Vision-Language-Action model in a PyBullet warehouse simulation. A robot receives a natural language instruction (e.g., "pick up the red box"), observes the scene through a camera, and must execute manipulation tasks.

**Architecture:**
- **Vision encoder:** ResNet-18 (pretrained ImageNet, fine-tuned)
- **Language encoder:** DistilBERT (pretrained, frozen during BC)
- **Fusion:** Cross-attention module (language attends to vision features)
- **Training pipeline:** Behavioral Cloning (BC) from expert demos → PPO reinforcement learning fine-tuning
- **Total params:** ~79M

---

## 2. CRITICAL: NEITHER PHASE IS CURRENTLY WORKING AS A DEMO

This is the honest truth. Both phases have trained models that show learning in metrics (loss reduction, reward improvement) but neither produces a visually convincing demo of task completion.

### Phase 1: Fixed Panda Arm (Tabletop Pick-and-Place)

**What exists:**
- Robot: Franka Panda arm, fixed base
- Task: Pick up a colored box from the table
- Action space: 7-dim (joint velocities)
- Environment: `src/env/warehouse_env.py`
- Training script: `src/training/train_rl.py` (classes: `VLAPPOPolicy`, `preprocess_image`)
- BC checkpoint: `checkpoints/best_model.pth` (val loss 0.115, 67% reduction over 30 epochs)
- RL checkpoint: `checkpoints/best_rl_model.pth` (75.8% reward improvement over 160 episodes)
- 600 expert demonstrations in `data/demos/demonstrations.pkl`

**What's actually broken — confirmed by diagnostic tests:**

1. **BC model alone:** Robot gets stuck. Reward starts at -0.47 (reasonably close to an object) then worsens to -0.69 and stays locked there for all 300 steps. The arm learned an initial pose from BC but outputs repetitive actions that don't progress. Classic BC compounding error.

2. **RL model (best_rl_model.pth):** Tested with 5 evaluation episodes — **0/5 success**, never triggers `done`, negative rewards throughout (-39 to -51 per episode). The arm IS moving toward objects (distance drops from 0.573 → 0.337) but oscillates around the target without committing to a grasp.

3. **Root cause identified:** The reward function during RL training was just `return -min_dist` — no grasp detection, no success condition, no terminal bonus. `done` only triggers at max_episode_steps timeout. The arm had zero incentive to actually close the gripper and lift. It learned "get vaguely close" and stopped.

4. **A dense reward patch was written** (approach bonus, proximity bonus at <0.05m, grasp detection when object z>0.1, lift reward, time penalty, +50 terminal bonus for sustained lift). The step function was also patched to set `done=True` on success. **BUT: It is UNKNOWN whether this patch is currently applied to the file on disk, or whether a retrain was ever completed with it.**

**⚠️ FIRST ACTION: Read `src/env/warehouse_env.py` and check what `compute_reward` currently returns. If it's still just `-min_dist`, the dense reward needs to be applied and RL needs to be retrained from the BC checkpoint.**

### Phase 2: Mobile Manipulation (Husky + Panda)

**What exists:**
- Robot: Husky mobile base + Panda arm mounted on top
- Task: Navigate to shelf → reach object → grasp → deliver to dropoff zone
- Action space: 10-dim (3 navigation: vx, vy, wz + 6 arm joints + 1 gripper)
- Environment: `src/env/warehouse_env_mobile_v2.py`
- Training script: `src/training/train_rl_cloud.py` (classes: `CloudVLAPPOPolicy`, `get_robot_state`, `preprocess`)
- BC checkpoint: `checkpoints/best_mobile_model.pth` (527MB, val loss 0.905)
- RL checkpoint: `checkpoints/best_cloud_rl_model.pth` (311MB)
- Demo data: `data/demos_mobile/demonstrations.pkl` (8.9GB, 300 demos)

**What's actually broken — confirmed by diagnostic tests:**

1. **Navigation goes wrong direction:** dist_shelf INCREASES from 4.38 → 5.71 over 80 steps despite large forward actions (action[0] = 1.42, 2.55). Robot moves confidently the wrong way.

2. **Root cause: BC→RL distribution shift.** Oracle demos used `resetBasePositionAndOrientation` (direct position updates) for navigation, always starting yaw-aligned toward the target shelf. RL uses wheel physics with random initial yaw. BC learned "action[0] > 0 = toward shelf" but in RL the robot faces a random direction.

3. **Wheel physics instability.** Top-heavy Husky+Panda stalls even with force fixes (maxForce=10000, wheel force=500).

4. **Oracle expert works perfectly:** 5/5 trials, 100% on all phases. But it uses direct position updates, not wheel physics. The expert is correct; the RL transfer is broken.

5. **BC loss plateau at 0.905.** Very high — the model barely learned the 10-dim action mapping.

---

## 3. THE KEY DECISION: Remove Wheel Dynamics for Phase 2

**Use `resetBasePositionAndOrientation` for navigation in BOTH expert demos AND RL.** This means:
- Navigation uses position updates — robot moves smoothly A to B each step
- Arm manipulation still uses real physics (IK-based joint control)
- Grasping still uses real physics
- Eliminates distribution shift entirely
- BC and RL see identical dynamics

**This is standard practice** in robotics simulation. Real robots abstract navigation via motion planners. The VLA model still needs to parse language, locate objects visually, and output correct actions.

**In `step()` this means:**
```python
vx = float(action[0]) * 0.05  # meters per step
wz = float(action[2]) * 0.05  # radians per step
pos, orn = p.getBasePositionAndOrientation(self.husky_id)
yaw = p.getEulerFromQuaternion(orn)[2]
new_yaw = yaw + wz
new_x = pos[0] + vx * np.cos(new_yaw)
new_y = pos[1] + vx * np.sin(new_yaw)
p.resetBasePositionAndOrientation(self.husky_id, [new_x, new_y, pos[2]],
                                    p.getQuaternionFromEuler([0, 0, new_yaw]))
p.resetBasePositionAndOrientation(self.panda_id, [new_x, new_y, pos[2] + 0.5],
                                    p.getQuaternionFromEuler([0, 0, new_yaw]))
```

---

## 4. TWO-DAY STRATEGY — Diagnose First, Then Fix

### STEP 0: DIAGNOSE BEFORE FIXING (Do This First!)

Before writing ANY new code, Claude Code must:

1. **Read `src/env/warehouse_env.py`** — check if compute_reward has the dense reward or is still just `-min_dist`
2. **Read `src/env/warehouse_env_mobile_v2.py`** — check if step() uses wheel physics or position updates
3. **Read `src/training/train_rl.py`** — understand Phase 1 training loop and policy class
4. **Read `configs/config.yaml`** — check Phase 1 hyperparameters
5. **Run Phase 1 eval** — load best_rl_model.pth, run 5 episodes headless, measure:
   - Distance from end-effector to nearest object at steps 0, 50, 100, 150, 200
   - Whether any object's z-position ever exceeds 0.10 (indicating a lift)
   - What actions the policy outputs (are they diverse or repetitive?)
6. **Report findings** before proposing any changes

### STEP 1: Fix Phase 1 (Safety Net — ~4 hours)

Phase 1 is simpler and closest to working. The arm already gets within 0.337m.

a) **Apply dense reward** to warehouse_env.py (if not already there):
   - Approach: `-distance * 2.0`
   - Proximity bonus: `+5.0` when distance < 0.05m
   - Grasp detection: `+10.0` when any object z > 0.10
   - Lift reward: `object_z * 20.0` when grasped
   - Time penalty: `-0.01` per step
   - Terminal success: `+50.0` when object held lifted for 5+ steps
   - `done = True` on success

b) **Retrain RL from BC checkpoint** (NOT from scratch):
   - Load `best_model.pth` (BC weights) as initialization
   - PPO with dense reward, ~200-300 episodes
   - On M2 this may take 2-4 hours
   - Monitor distance-to-object trend and grasp occurrence

c) **Evaluate:** Run 10-20 episodes, count grasps. Even 30% is a valid demo.

d) **Also consider:** Are the Phase 1 expert demos themselves good? Does the expert controller actually grasp objects successfully? If the BC demos were poor quality, the BC prior might be wrong. Check `data/demos/demonstrations.pkl` — does the expert achieve grasps in those demos?

### STEP 2: Fix Phase 2 with Position-Update Navigation (~4-6 hours)

Only after Phase 1 is working.

a) **Modify `warehouse_env_mobile_v2.py`** — replace wheel motor commands with position updates in step(). Keep arm physics intact.

b) **Verify expert controller** uses same position-update dynamics (it already does).

c) **Initialize robot facing target shelf** in reset():
   ```python
   shelf_pos = self.current_shelf_positions[self.target_object_idx // 2]
   start_yaw = np.arctan2(shelf_pos[1] - start_y, shelf_pos[0] - start_x)
   ```

d) **Regenerate demos** (100-200, fast with position updates)

e) **BC retrain** (10-20 epochs)

f) **RL retrain** with dense multi-stage reward

### STEP 3: Fallback — Scripted Demo + Honest Analysis

If training doesn't converge in time:
- Show oracle controller executing perfect demos (environment works)
- Show BC training curves (model learns from data)
- Show RL reward curves improving (RL improves over BC)
- Show Phase 1 arm attempting grasps (VLA architecture responds to inputs)
- Write honest analysis of failures and what would fix them

---

## 5. Key File Paths

```
Environments:
  src/env/warehouse_env.py              # Phase 1 — CHECK REWARD FUNCTION FIRST
  src/env/warehouse_env_mobile_v2.py    # Phase 2 — CHECK step() NAVIGATION METHOD

Training:
  src/training/train_rl.py              # Phase 1 RL (VLAPPOPolicy, preprocess_image)
  src/training/train_rl_cloud.py        # Phase 2 RL (CloudVLAPPOPolicy, get_robot_state, preprocess)
  src/training/train_bc_mobile.py       # Phase 2 BC

Data:
  src/data/collect_demos_cloud.py       # Phase 2 oracle expert
  src/data/collect_demos_mobile_v2.py   # Earlier demo collection (ImprovedExpert)

Configs:
  configs/config.yaml                   # Phase 1
  configs/config_mobile.yaml            # Phase 2 local
  configs/config_cloud.yaml             # Phase 2 cloud

Checkpoints (status):
  checkpoints/best_model.pth            # Phase 1 BC — val loss 0.115 ✅ (good BC prior)
  checkpoints/best_rl_model.pth         # Phase 1 RL — ❌ 0/5 success, hovers near objects
  checkpoints/best_mobile_model.pth     # Phase 2 BC — val loss 0.905 ⚠️ (barely learned)
  checkpoints/best_cloud_rl_model.pth   # Phase 2 RL — ❌ navigates wrong direction
```

---

## 6. Architecture Reference

```python
# Phase 1 — src/training/train_rl.py
class VLAPPOPolicy(nn.Module):
    # ResNet-18 vision + DistilBERT language + cross-attention
    # Input: 224x224 RGB image + text instruction
    # Output: 7-dim action (joint velocities)
    # Note: No robot state input in Phase 1 — vision only

# Phase 2 — src/training/train_rl_cloud.py  
class CloudVLAPPOPolicy(nn.Module):
    # Same base architecture + robot state input
    # Input: 224x224 RGB image + text instruction + robot state vector
    # Output: 10-dim action (3 nav + 6 arm + 1 gripper)
    # Helpers: get_robot_state(env), preprocess(obs, robot_state, device)
```

---

## 7. Lessons Learned (Principles for Claude Code to Follow)

- **Reward function design is the #1 failure mode.** Missing grasp detection caused Phase 1 to hover. Missing navigation shaping caused Phase 2 to wander. Every reward must encode the FULL task structure.
- **BC→RL distribution shift kills transfer.** If demo collection uses different dynamics than RL, the policy breaks. Match them EXACTLY.
- **Diagnose before fixing.** Run diagnostic scripts to measure specific quantities (dist_to_object, action magnitudes, object z-position) rather than guessing.
- **Oracle demos before BC.** Guarantee 100% demo accuracy analytically — don't collect noisy demos and hope BC generalizes.
- **Don't patch over a broken foundation.** If the reward function is wrong, no amount of hyperparameter tuning will fix it. Fix the root cause.
- **Time is extremely limited.** Every change must be justified by a specific diagnosed problem. No speculative "improvements."

---

## 8. What the Examiner Cares About

1. **Architecture understanding** — Why ResNet + DistilBERT + cross-attention? (Language guides visual attention)
2. **Training pipeline** — BC provides behavioral prior, RL corrects compounding errors
3. **Empirical comparison** — BC alone vs BC+RL, with training curves
4. **Honest failure analysis** — Distribution shift, reward design, what more time/compute would fix
5. **Some working demo** — Even partial (arm approaching objects, navigation toward correct shelf) is fine

---

## 9. IMMEDIATE INSTRUCTIONS FOR CLAUDE CODE

```
1. Read ALL files in src/env/ and src/training/ and configs/
2. Report what you find — especially:
   a) What does compute_reward() return in warehouse_env.py?
   b) What does step() do for navigation in warehouse_env_mobile_v2.py?
   c) Is there a working gripper close mechanism in Phase 1?
   d) What are the expert demos actually doing? (check collection scripts)
3. DO NOT make changes until you've reported diagnostic findings
4. Then propose a specific fix plan — Phase 1 first, Phase 2 second
5. Phase 1 goal: arm grasps and lifts an object in >30% of eval episodes
6. Phase 2 goal: robot navigates to correct shelf based on language instruction
7. Time constraint: ~2 days total — be ruthlessly practical
8. If something looks unfixable in the time remaining, say so immediately
```
