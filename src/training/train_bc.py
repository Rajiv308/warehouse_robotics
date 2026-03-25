import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import os
import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.vla_model import VLAModel, freeze_language_encoder
from src.training.dataset import get_dataloaders

def train_behavioral_cloning(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    tc = cfg['training']
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")

    model = VLAModel(config_path).to(device)

    # Freeze DistilBERT - only train vision + fusion + action head
    freeze_language_encoder(model)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tc['learning_rate']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc['bc_epochs'], eta_min=1e-6
    )
    criterion = nn.MSELoss()

    train_loader, val_loader = get_dataloaders(cfg, batch_size=tc['batch_size'])
    writer = SummaryWriter(log_dir=cfg['logging']['log_dir'])
    os.makedirs("checkpoints", exist_ok=True)

    best_val_loss = float('inf')
    global_step = 0

    print(f"Starting training for {tc['bc_epochs']} epochs...")

    for epoch in range(tc['bc_epochs']):
        # Training
        model.train()
        train_losses = []

        for imgs, instructions, actions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{tc['bc_epochs']}"):
            imgs = imgs.to(device)
            actions = actions.to(device)

            optimizer.zero_grad()
            predicted_actions = model(imgs, list(instructions))
            loss = criterion(predicted_actions, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            global_step += 1

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, instructions, actions in val_loader:
                imgs = imgs.to(device)
                actions = actions.to(device)
                predicted_actions = model(imgs, list(instructions))
                loss = criterion(predicted_actions, actions)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)

        writer.add_scalar('Loss/train_epoch', avg_train, epoch)
        writer.add_scalar('Loss/val_epoch', avg_val, epoch)

        print(f"Epoch {epoch+1:3d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, "checkpoints/best_model.pth")
            print(f"  ✓ Best model saved (val={best_val_loss:.4f})")

        if (epoch + 1) % tc['checkpoint_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val,
            }, f"checkpoints/checkpoint_epoch_{epoch+1}.pth")

    writer.close()
    print(f"\nDone! Best val loss: {best_val_loss:.4f}")
    return model

if __name__ == "__main__":
    train_behavioral_cloning()
