"""
Behavior-cloning trainer for the Phase 2 pickup module.

Reads (state, action) pairs from demos/pickup_bc.npz (produced by
src/data/collect_pickup_bc_demos.py) and trains a MobileCartesianPickPolicy
via MSE regression. Saves the best checkpoint by validation loss to
checkpoints/phase2_pick_bc_best.pth. This checkpoint is consumed by
src/eval/demo_phase2_hybrid.py via the PHASE2_PICK_CKPT environment
variable, which replaces the scripted grasp finalizer at demo time.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.training.train_phase2_pick_cartesian import MobileCartesianPickPolicy


def main():
    demo_path = os.environ.get("PHASE2_BC_DEMO_PATH", "demos/pickup_bc.npz")
    ckpt_path = os.environ.get("PHASE2_BC_CKPT", "checkpoints/phase2_pick_bc_best.pth")
    epochs = int(os.environ.get("PHASE2_BC_EPOCHS", "120"))
    batch_size = int(os.environ.get("PHASE2_BC_BATCH", "256"))
    lr = float(os.environ.get("PHASE2_BC_LR", "3e-4"))

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    data = np.load(demo_path)
    states = torch.tensor(data["states"], dtype=torch.float32)
    actions = torch.tensor(data["actions"], dtype=torch.float32)
    print(f"Loaded {states.shape[0]} (state, action) pairs from {demo_path}")
    print(f"  state dim = {states.shape[1]}, action dim = {actions.shape[1]}")

    # Use MPS on Apple Silicon if available, otherwise CUDA, otherwise CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  device = {device}")

    dataset = TensorDataset(states, actions)
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    policy = MobileCartesianPickPolicy().to(device)
    opt = torch.optim.Adam(policy.actor_mean.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(epochs):
        policy.train()
        train_loss = 0.0
        for s, a in train_loader:
            s, a = s.to(device), a.to(device)
            pred = policy.actor_mean(s)
            loss = loss_fn(pred, a)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * s.size(0)
        train_loss /= len(train_ds)

        policy.eval()
        val_loss = 0.0
        with torch.no_grad():
            for s, a in val_loader:
                s, a = s.to(device), a.to(device)
                pred = policy.actor_mean(s)
                val_loss += loss_fn(pred, a).item() * s.size(0)
        val_loss /= len(val_ds)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(policy.state_dict(), ckpt_path)
            tag = " (saved)"
        else:
            tag = ""

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: train={train_loss:.5f}  val={val_loss:.5f}{tag}")

    print(f"\nBest val loss: {best_val:.5f}")
    print(f"Saved best checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
