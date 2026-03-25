import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from torchvision import transforms

class DemoDataset(Dataset):
    """
    PyTorch Dataset that wraps our collected demonstrations.
    Each sample is (image, instruction, action).
    """
    def __init__(self, data_path, augment=False):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

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

        # Action as float tensor
        action = torch.FloatTensor(sample['action'])

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
