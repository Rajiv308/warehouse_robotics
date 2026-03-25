"""
Quick script to add noise to existing demonstrations
instead of re-collecting from scratch - saves time
"""
import pickle
import numpy as np

print("Loading demonstrations...")
with open("data/demos/demonstrations.pkl", 'rb') as f:
    demos = pickle.load(f)

print(f"Adding action noise to {len(demos)} demonstrations...")
noisy_demos = []
for demo in demos:
    # Create 2 noisy copies of each demo
    for _ in range(2):
        noisy_demo = {
            'instruction': demo['instruction'],
            'observations': demo['observations'],
            'rewards': demo['rewards'],
            'done': demo['done'],
            'actions': []
        }
        for action in demo['actions']:
            # Add small gaussian noise to arm joints, less to gripper
            noise = np.random.normal(0, 0.05, 7)
            noise[6] *= 0.1  # much less noise on gripper
            noisy_action = np.clip(action + noise, -1.0, 1.0)
            noisy_demo['actions'].append(noisy_action)
        noisy_demos.append(noisy_demo)

# Combine original + noisy
all_demos = demos + noisy_demos
print(f"Total demonstrations: {len(all_demos)} (was {len(demos)})")

# Save
with open("data/demos/demonstrations.pkl", 'wb') as f:
    pickle.dump(all_demos, f)

# Rebuild train/val split
all_samples = []
for demo in all_demos:
    for i in range(len(demo['actions'])):
        all_samples.append({
            'image': demo['observations'][i],
            'action': demo['actions'][i],
            'instruction': demo['instruction']
        })

np.random.shuffle(all_samples)
split = int(0.8 * len(all_samples))
train_data = all_samples[:split]
val_data = all_samples[split:]

with open("data/demos/train_data.pkl", 'wb') as f:
    pickle.dump(train_data, f)
with open("data/demos/val_data.pkl", 'wb') as f:
    pickle.dump(val_data, f)

print(f"Total samples: {len(all_samples)}")
print(f"Train: {len(train_data)}, Val: {len(val_data)}")
print("Done!")
