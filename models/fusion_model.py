from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from config import cfg
from models.spatial_model import SpatialBranch
from models.frequency_model import FrequencyBranch


class FusionModel(nn.Module):
    """
    Dual-branch spatial + frequency network with late feature fusion.
    Outputs a single logit for binary classification.
    """

    def __init__(self, pretrained_backbones: bool = True):
        super().__init__()
        self.spatial_branch = SpatialBranch(pretrained=pretrained_backbones)
        self.frequency_branch = FrequencyBranch(pretrained=pretrained_backbones)

        fusion_dim = self.spatial_branch.out_dim + self.frequency_branch.out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, cfg.FUSION_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg.DROPOUT),
            nn.Linear(cfg.FUSION_HIDDEN_DIM, 1),
        )

    def forward(
        self, spatial: torch.Tensor, freq: torch.Tensor, return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        spatial_feats = self.spatial_branch(spatial)
        freq_feats = self.frequency_branch(freq)
        fused = torch.cat([spatial_feats, freq_feats], dim=1)
        logits = self.fusion(fused).squeeze(1)

        if return_features:
            return logits, {
                "spatial": spatial_feats,
                "frequency": freq_feats,
                "fused": fused,
            }
        return logits, None

