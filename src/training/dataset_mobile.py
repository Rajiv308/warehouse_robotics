import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import pybullet as p
from torchvision import transforms


class MobileDemoDataset(Dataset):
    def __init__(self, data_path, augment=False):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img    = self.transform(sample['image'])
        action = torch.FloatTensor(sample['action'])

        # Robot state: zeros as placeholder
        # (real state comes from env during RL, BC just needs the shape)
        state  = torch.zeros(9)

        return img, sample['instruction'], action, state


def get_mobile_dataloaders(cfg, batch_size=32):
    train_ds = MobileDemoDataset("data/demos_mobile/train_data.pkl", augment=True)
    val_ds   = MobileDemoDataset("data/demos_mobile/val_data.pkl",   augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader
