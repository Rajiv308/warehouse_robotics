import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertModel, DistilBertTokenizer
import numpy as np
import yaml

class VisionEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super(VisionEncoder, self).__init__()
        # Load pretrained ResNet-18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # Project to our desired output dimension
        self.projection = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, 3, 224, 224)
        features = self.backbone(x)           # (batch, 512, 1, 1)
        features = features.flatten(1)        # (batch, 512)
        features = self.relu(self.projection(features))  # (batch, output_dim)
        return features


class LanguageEncoder(nn.Module):
    def __init__(self, output_dim=768):
        super(LanguageEncoder, self).__init__()
        # Load pretrained DistilBERT
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.projection = nn.Linear(768, output_dim)
        self.relu = nn.ReLU()

    def forward(self, instructions):
        # Tokenize the text instructions
        tokens = self.tokenizer(
            instructions,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=32
        ).to(next(self.bert.parameters()).device)

        # Pass through DistilBERT
        outputs = self.bert(**tokens)
        # Use [CLS] token as sentence representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        features = self.relu(self.projection(cls_output)) # (batch, output_dim)
        return features


class CrossAttentionFusion(nn.Module):
    def __init__(self, vision_dim=512, language_dim=768, fusion_dim=256):
        super(CrossAttentionFusion, self).__init__()
        # Project both modalities to same dimension
        self.vision_proj = nn.Linear(vision_dim, fusion_dim)
        self.language_proj = nn.Linear(language_dim, fusion_dim)

        # Multi-head attention: vision attends to language
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, vision_features, language_features):
        # Project to fusion_dim
        v = self.vision_proj(vision_features).unsqueeze(1)    # (batch, 1, fusion_dim)
        l = self.language_proj(language_features).unsqueeze(1) # (batch, 1, fusion_dim)

        # Cross attention: vision queries, language is key and value
        attended, _ = self.attention(query=v, key=l, value=l)  # (batch, 1, fusion_dim)
        attended = attended.squeeze(1)                          # (batch, fusion_dim)

        # Residual connection + layer norm
        vision_proj_flat = self.vision_proj(vision_features)
        fused = self.layer_norm(vision_proj_flat + self.dropout(attended))
        return fused


class ActionHead(nn.Module):
    def __init__(self, input_dim=256, action_dim=7):
        super(ActionHead, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # bound actions to [-1, 1]
        )

    def forward(self, x):
        return self.network(x)


class VLAModel(nn.Module):
    def __init__(self, config_path="configs/config.yaml"):
        super(VLAModel, self).__init__()
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        mc = cfg['model']

        self.vision_encoder = VisionEncoder(output_dim=mc['vision_features'])
        self.language_encoder = LanguageEncoder(output_dim=mc['language_features'])
        self.fusion = CrossAttentionFusion(
            vision_dim=mc['vision_features'],
            language_dim=mc['language_features'],
            fusion_dim=mc['fusion_dim']
        )
        self.action_head = ActionHead(
            input_dim=mc['fusion_dim'],
            action_dim=mc['action_dim']
        )

    def forward(self, images, instructions):
        vision_features = self.vision_encoder(images)
        language_features = self.language_encoder(instructions)
        fused = self.fusion(vision_features, language_features)
        actions = self.action_head(fused)
        return actions

    def predict(self, image_np, instruction, device='cpu'):
        """Single inference: numpy image + string instruction -> action array"""
        self.eval()
        with torch.no_grad():
            # Preprocess image
            img = torch.FloatTensor(image_np).permute(2, 0, 1).unsqueeze(0) / 255.0
            img = img.to(device)
            action = self.forward(img, [instruction])
        return action.squeeze().cpu().numpy()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def freeze_language_encoder(model):
    """Freeze DistilBERT weights - use pretrained features as-is"""
    for param in model.language_encoder.bert.parameters():
        param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after freezing BERT: {trainable:,}")
