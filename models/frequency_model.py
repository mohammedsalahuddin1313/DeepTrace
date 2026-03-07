from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


class FrequencyBranch(nn.Module):
    """
    Frequency-domain feature extractor.
    Expects a 3-channel magnitude spectrum image (same size as spatial input).
    Uses a ResNet-18 backbone with final classifier removed.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        modules = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.out_dim = backbone.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        return feats

