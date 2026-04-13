import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import json
from torchvision import transforms


class MobileDemoDataset(Dataset):
    def __init__(self, data_path, augment=False,
                 stats_path="data/demos_mobile/action_stats.json"):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        # Load action normalization stats
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        self.action_min = np.array(stats['action_min'], dtype=np.float32)
        self.action_max = np.array(stats['action_max'], dtype=np.float32)
        self.action_range = np.maximum(self.action_max - self.action_min, 1e-6)

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

        # Raw actions — no normalization needed since Tanh was removed.
        action = torch.FloatTensor(np.array(sample['action'], dtype=np.float32))

        # Robot state: use saved state if available, otherwise zeros
        # (zeros are better than random noise — at least consistent)
        if 'state' in sample and sample['state'] is not None:
            state = torch.FloatTensor(sample['state'])
        else:
            state = torch.zeros(9)

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
