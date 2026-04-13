import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import json
from torchvision import transforms

class DemoDataset(Dataset):
    """
    PyTorch Dataset that wraps our collected demonstrations.
    Each sample is (image, instruction, action).
    Actions are normalized to [-1, 1] so the Tanh output model can represent them.
    """
    def __init__(self, data_path, augment=False, stats_path="data/demos/action_stats.json"):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        # Load action normalization stats
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        self.action_min = np.array(stats['action_min'], dtype=np.float32)
        self.action_max = np.array(stats['action_max'], dtype=np.float32)
        # Avoid division by zero for constant dims
        self.action_range = np.maximum(self.action_max - self.action_min, 1e-6)

        # Image normalization - same values ResNet was pretrained with
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if augment:
            # Training: add random augmentations to improve generalization
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                normalize
            ])
        else:
            # Validation: just normalize, no augmentation
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                normalize
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Process image
        img = self.transform(sample['image'])  # (3, 224, 224)

        # Normalize action to [-1, 1] range (so Tanh model can represent it)
        raw_action = np.array(sample['action'], dtype=np.float32)
        norm_action = 2.0 * (raw_action - self.action_min) / self.action_range - 1.0
        action = torch.FloatTensor(norm_action)

        # Instruction stays as string (DistilBERT tokenizes it internally)
        instruction = sample['instruction']

        return img, instruction, action


def get_dataloaders(cfg, batch_size=32):
    train_dataset = DemoDataset("data/demos/train_data.pkl", augment=True)
    val_dataset = DemoDataset("data/demos/val_data.pkl", augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,    # 0 = load on main process (safer on Mac)
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader
