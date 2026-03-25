import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Hyperfusion.hyper_base import (
    HyperPreactivResBlock_TTT,
    Conv3DLayer
)

# -------------------------------
# Hyper Image Encoder
# -------------------------------
class HyperImageEncoder(nn.Module):
    def __init__(
        self,
        tabular_embedding_model,   # FTTransformer (or wrapper)
        tabular_dim,
        in_channels=1,
        init_features=8,
        n_outputs=2
    ):
        super().__init__()

        # First conv (hyper-conditioned)
        self.conv1 = Conv3DLayer(
            in_channels=in_channels,
            out_channels=init_features,
            kernel_size=3,
            stride=1,
            padding=1,
            embedding_model=tabular_embedding_model,
            embedding_output_size=tabular_dim
        )

        self.pool = nn.MaxPool3d(2)

        # Hyper ResBlocks (TTT = strongest)
        self.block1 = HyperPreactivResBlock_TTT(
            init_features,
            2 * init_features,
            embedding_model=tabular_embedding_model,
            embedding_output_size=tabular_dim
        )

        self.block2 = HyperPreactivResBlock_TTT(
            2 * init_features,
            4 * init_features,
            stride=2,
            embedding_model=tabular_embedding_model,
            embedding_output_size=tabular_dim
        )

        self.block3 = HyperPreactivResBlock_TTT(
            4 * init_features,
            8 * init_features,
            stride=2,
            embedding_model=tabular_embedding_model,
            embedding_output_size=tabular_dim
        )

        self.pool_out = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Linear(8 * init_features, n_outputs)

    def forward(self, x, tab_features):
        # x: [B, C, D, H, W]
        # tab_features: [B, tab_dim]

        x = self.conv1((x, tab_features))
        x = F.relu(x)
        x = self.pool(x)

        x = self.block1((x, tab_features))
        x = self.block2((x, tab_features))
        x = self.block3((x, tab_features))

        x = self.pool_out(x)
        x = x.view(x.size(0), -1)

       
        return 

class HyperFiLM(nn.Module):
    def __init__(self, tabular_dim, image_dim):
        super().__init__()
        self.gamma = nn.Linear(tabular_dim, image_dim)
        self.beta = nn.Linear(tabular_dim, image_dim)

    def forward(self, img_feat, tab_feat):
        gamma = self.gamma(tab_feat)
        beta = self.beta(tab_feat)

        return gamma * img_feat + beta
    

class HyperMedViTEncoder(nn.Module):
    def __init__(self, medvit_model, tabular_dim):
        super().__init__()

        self.medvit = medvit_model
        self.feature_dim = medvit_model.feature_dim

        # Hyper modulation
        self.hyper = HyperFiLM(tabular_dim, self.feature_dim)

        # Optional classifier (Phase 1)
        self.classifier = nn.Linear(self.feature_dim, 2)

    def forward(self, images, tab_features):
        # ---- MedViT features ----
        img_features, _ = self.medvit(images, return_feature=True)

        # ---- Hyper modulation ----
        img_features = self.hyper(img_features, tab_features)

        logits = self.classifier(img_features)
        return logits, img_features
    

class HyperMultimodalModel(nn.Module):
    def __init__(self, fttransformer, tabular_dim):
        super().__init__()

        self.fttransformer = fttransformer

        self.image_encoder = HyperImageEncoder(
            tabular_embedding_model=self.fttransformer,
            tabular_dim=tabular_dim,
            in_channels=1,
            init_features=8,
            n_outputs=2
        )

    def forward(self, batch):
        # ---- TABULAR ----
        tab_features, _ = self.fttransformer(batch, return_feature=True)
        tab_features = F.layer_norm(tab_features, tab_features.shape[1:])

        # ---- IMAGE ----
        images = batch["images"]  # [B, C, D, H, W]

        logits = self.image_encoder(images, tab_features)

        return logits
    

class HyperMultimodalMedViT(nn.Module):
    def __init__(self, fttransformer, medvit_model):
        super().__init__()

        self.fttransformer = fttransformer

        self.image_encoder = HyperMedViTEncoder(
            medvit_model=medvit_model,
            tabular_dim=fttransformer.head[1].in_features  # or hidden_dim
        )

    def forward(self, inputs):
        images, tabular = inputs
        num, cat = tabular

        tab_features, _ = self.fttransformer(
            (num, cat),
            return_feature=True
        )

        tab_features = F.layer_norm(tab_features, tab_features.shape[1:])

        logits, _ = self.image_encoder(images, tab_features)

        return logits
