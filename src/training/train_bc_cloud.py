"""
Cloud BC training with:
- CUDA GPU
- Real robot state vectors (not zeros)
- Larger batch size
- More epochs
- Unfrozen last 2 layers of DistilBERT for fine-tuning
"""
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
from src.models.vla_model_mobile import MobileVLAModel, freeze_language_encoder
from src.training.dataset_mobile import get_mobile_dataloaders


def train_cloud_bc(config_path="configs/config_cloud.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tc     = cfg['training']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cloud BC Training on: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = MobileVLAModel(config_path).to(device)

    # Freeze most of BERT but unfreeze last 2 transformer layers
    freeze_language_encoder(model)
    for i in [4, 5]:  # last 2 layers of DistilBERT
        for param in model.language_encoder.bert.transformer.layer[i].parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tc['learning_rate'],
        weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc['bc_epochs'], eta_min=1e-7
    )
    criterion = nn.MSELoss()

    train_loader, val_loader = get_mobile_dataloaders(cfg, batch_size=tc['batch_size'])
    writer = SummaryWriter(log_dir="logs/cloud_bc")
    os.makedirs("checkpoints", exist_ok=True)

    best_val_loss = float('inf')
    global_step   = 0

    print(f"Starting Cloud BC for {tc['bc_epochs']} epochs...")
    print(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}\n")

    for epoch in range(tc['bc_epochs']):
        model.train()
        train_losses = []

        for imgs, instructions, actions, states in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{tc['bc_epochs']}"
        ):
            imgs    = imgs.to(device)
            actions = actions.to(device)
            states  = states.to(device)

            optimizer.zero_grad()
            predicted = model(imgs, list(instructions), states)
            loss = criterion(predicted, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            global_step += 1

        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, instructions, actions, states in val_loader:
                imgs    = imgs.to(device)
                actions = actions.to(device)
                states  = states.to(device)
                loss = criterion(model(imgs, list(instructions), states), actions)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val   = np.mean(val_losses)

        writer.add_scalar('CloudBC/train', avg_train, epoch)
        writer.add_scalar('CloudBC/val',   avg_val,   epoch)

        print(f"Epoch {epoch+1:3d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':             best_val_loss,
            }, "checkpoints/best_mobile_model.pth")
            print(f"  ✓ Best model saved (val={best_val_loss:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val,
            }, f"checkpoints/cloud_bc_epoch_{epoch+1}.pth")

    writer.close()
    print(f"\nCloud BC complete! Best val loss: {best_val_loss:.4f}")
    return model


if __name__ == "__main__":
    train_cloud_bc()
