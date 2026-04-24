"""
Behavior-cloning trainer for the Phase 2 delivery navigation module.

Reads (state, wheel_velocity) pairs from demos/delivery_bc.npz (produced by
src/data/collect_delivery_bc_demos.py) and trains a small MLP via MSE
regression. Saves best checkpoint by validation loss to
checkpoints/phase2_delivery_bc_best.pth.

Actions are stored normalized by WHEEL_VEL_NORM (15 rad/s); at inference
time the BC output is de-normalized before being applied to the wheel motors.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class DeliveryDrivePolicy(nn.Module):
    def __init__(self, state_dim=5, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),  # bounded in [-1, 1] matching normalized target actions
        )

    def forward(self, x):
        return self.net(x)


def main():
    demo_path = os.environ.get("PHASE2_DELIVERY_BC_DEMO_PATH", "demos/delivery_bc.npz")
    ckpt_path = os.environ.get("PHASE2_DELIVERY_BC_CKPT", "checkpoints/phase2_delivery_bc_best.pth")
    epochs = int(os.environ.get("PHASE2_DELIVERY_BC_EPOCHS", "80"))
    batch_size = int(os.environ.get("PHASE2_DELIVERY_BC_BATCH", "256"))
    lr = float(os.environ.get("PHASE2_DELIVERY_BC_LR", "3e-4"))

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    data = np.load(demo_path)
    states = torch.tensor(data["states"], dtype=torch.float32)
    actions = torch.tensor(data["actions"], dtype=torch.float32)
    # Clip outliers to [-1, 1] in case the scripted controller ever produced
    # a magnitude larger than WHEEL_VEL_NORM.
    actions = torch.clamp(actions, -1.0, 1.0)
    print(f"Loaded {states.shape[0]} (state, action) pairs from {demo_path}")
    print(f"  state dim = {states.shape[1]}, action dim = {actions.shape[1]}")

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

    policy = DeliveryDrivePolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(epochs):
        policy.train()
        train_loss = 0.0
        for s, a in train_loader:
            s, a = s.to(device), a.to(device)
            pred = policy(s)
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
                pred = policy(s)
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
