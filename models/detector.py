"""
models/detector.py — 추론 전용 재난 탐지기
"""

from __future__ import annotations
import torch
import torch.nn as nn
from config.config import cfg
from models.backbone import build_backbone


class DisasterDetector(nn.Module):
    def __init__(self, backbone_name: str | None = None):
        super().__init__()
        name = backbone_name or cfg.detector.backbone
        self.backbone, feat_dim = build_backbone(name)
        mid_dim = max(feat_dim // 8, 64)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(cfg.detector.dropout),
            nn.Linear(feat_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.detector.dropout),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
