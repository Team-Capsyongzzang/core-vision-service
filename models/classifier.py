"""
models/classifier.py — 추론 전용 재난 분류기
"""

from __future__ import annotations
import torch
import torch.nn as nn
from config.config import cfg, DISASTER_CLASSES
from models.backbone import build_backbone


class DisasterClassifier(nn.Module):
    def __init__(self, backbone_name: str | None = None):
        super().__init__()
        name = backbone_name or cfg.model.backbone
        self.backbone, feat_dim = build_backbone(name)
        mid_dim = max(feat_dim // 4, 128)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(feat_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(mid_dim, cfg.model.num_classes),
        )
        self.class_names = DISASTER_CLASSES

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
