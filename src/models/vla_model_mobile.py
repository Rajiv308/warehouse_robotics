import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertModel, DistilBertTokenizer
import numpy as np
import yaml


class VisionEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        features = self.backbone(x)
        features = features.flatten(1)
        return self.relu(self.projection(features))


class LanguageEncoder(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.projection = nn.Linear(768, output_dim)
        self.relu = nn.ReLU()

    def forward(self, instructions):
        tokens = self.tokenizer(
            instructions,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=32
        ).to(next(self.bert.parameters()).device)
        outputs = self.bert(**tokens)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.relu(self.projection(cls))


class RobotStateEncoder(nn.Module):
    """
    New in Phase 2: encodes the robot's proprioceptive state.
    Takes base position (x,y,yaw) + arm joints (6) = 9 numbers
    and projects to a feature vector the fusion can use.
    """
    def __init__(self, state_dim=9, output_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.ReLU()
        )

    def forward(self, state):
        return self.network(state)


class CrossAttentionFusion(nn.Module):
    def __init__(self, vision_dim=512, language_dim=768, state_dim=64, fusion_dim=256):
        super().__init__()
        self.vision_proj   = nn.Linear(vision_dim, fusion_dim)
        self.language_proj = nn.Linear(language_dim, fusion_dim)
        self.state_proj    = nn.Linear(state_dim, fusion_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(fusion_dim)
        self.dropout    = nn.Dropout(0.1)

        # Final projection after concatenating vision + state context
        self.output_proj = nn.Linear(fusion_dim * 2, fusion_dim)

    def forward(self, vision_features, language_features, state_features):
        v = self.vision_proj(vision_features).unsqueeze(1)    # (B, 1, fusion_dim)
        l = self.language_proj(language_features).unsqueeze(1) # (B, 1, fusion_dim)
        s = self.state_proj(state_features)                    # (B, fusion_dim)

        # Vision attends to language
        attended, _ = self.attention(query=v, key=l, value=l)
        attended = attended.squeeze(1)

        # Residual connection
        v_flat = self.vision_proj(vision_features)
        fused_vl = self.layer_norm(v_flat + self.dropout(attended))

        # Concatenate with state features and project
        fused = torch.cat([fused_vl, s], dim=-1)  # (B, fusion_dim*2)
        fused = self.output_proj(fused)            # (B, fusion_dim)
        return fused


class MobileActionHead(nn.Module):
    """
    Outputs 10-dim action:
    [vx, vy, wz, j0, j1, j2, j3, j4, j5, gripper]
    """
    def __init__(self, input_dim=256, action_dim=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)


class MobileVLAModel(nn.Module):
    def __init__(self, config_path="configs/config_mobile.yaml"):
        super().__init__()
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        mc = cfg['model']

        self.vision_encoder   = VisionEncoder(output_dim=mc['vision_features'])
        self.language_encoder = LanguageEncoder(output_dim=mc['language_features'])
        self.state_encoder    = RobotStateEncoder(state_dim=9, output_dim=64)
        self.fusion = CrossAttentionFusion(
            vision_dim=mc['vision_features'],
            language_dim=mc['language_features'],
            state_dim=64,
            fusion_dim=mc['fusion_dim']
        )
        self.action_head = MobileActionHead(
            input_dim=mc['fusion_dim'],
            action_dim=mc['action_dim']
        )

    def forward(self, images, instructions, robot_state):
        v = self.vision_encoder(images)
        l = self.language_encoder(instructions)
        s = self.state_encoder(robot_state)
        fused = self.fusion(v, l, s)
        return self.action_head(fused)

    def predict(self, image_np, instruction, robot_state_np, device='cpu'):
        """Single inference step"""
        self.eval()
        with torch.no_grad():
            img = torch.FloatTensor(image_np).permute(2,0,1).unsqueeze(0) / 255.0
            img = img.to(device)
            state = torch.FloatTensor(robot_state_np).unsqueeze(0).to(device)
            action = self.forward(img, [instruction], state)
        return action.squeeze().cpu().numpy()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def freeze_language_encoder(model):
    for param in model.language_encoder.bert.parameters():
        param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after freezing BERT: {trainable:,}")
