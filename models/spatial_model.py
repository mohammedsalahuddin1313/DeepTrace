from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models

from config import cfg


class SpatialBranch(nn.Module):
    """
    Spatial-domain feature extractor based on a pretrained ResNet-50.
    The final classification layer is removed; the output is a feature vector.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        modules = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.out_dim = backbone.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        return feats

