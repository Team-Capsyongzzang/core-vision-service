"""
models/backbone.py — 추론 전용 백본 팩토리 (9개)
"""

from __future__ import annotations
import torch.nn as nn
import torchvision.models as tvm

BACKBONE_INFO = {
    "resnet50":           {"feat_dim": 2048},
    "resnet101":          {"feat_dim": 2048},
    "efficientnet_b0":    {"feat_dim": 1280},
    "efficientnet_b3":    {"feat_dim": 1536},
    "mobilenet_v3_large": {"feat_dim": 960},
    "mobilenet_v3_small": {"feat_dim": 576},
    "convnext_tiny":      {"feat_dim": 768},
    "efficientnet_v2_s":  {"feat_dim": 1280},
    "swin_t":             {"feat_dim": 768},
}


def _resnet50(pt):
    m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1 if pt else None)
    return nn.Sequential(*list(m.children())[:-1])

def _resnet101(pt):
    m = tvm.resnet101(weights=tvm.ResNet101_Weights.IMAGENET1K_V1 if pt else None)
    return nn.Sequential(*list(m.children())[:-1])

def _effb0(pt):
    m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pt else None)
    return nn.Sequential(m.features, m.avgpool)

def _effb3(pt):
    m = tvm.efficientnet_b3(weights=tvm.EfficientNet_B3_Weights.IMAGENET1K_V1 if pt else None)
    return nn.Sequential(m.features, m.avgpool)

def _mv3l(pt):
    m = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pt else None)
    return nn.Sequential(m.features, m.avgpool)

def _mv3s(pt):
    m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pt else None)
    return nn.Sequential(m.features, m.avgpool)

def _convnext_tiny(pt):
    m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pt else None)
    return nn.Sequential(m.features, m.avgpool)

def _effv2s(pt):
    m = tvm.efficientnet_v2_s(weights=tvm.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pt else None)
    return nn.Sequential(m.features, m.avgpool)

def _swin_t(pt):
    m = tvm.swin_t(weights=tvm.Swin_T_Weights.IMAGENET1K_V1 if pt else None)
    class SwinWrapper(nn.Module):
        def __init__(self, features):
            super().__init__()
            self.features = features
            self.avgpool  = nn.AdaptiveAvgPool2d(1)
        def forward(self, x):
            x = self.features(x)
            x = x.permute(0, 3, 1, 2)
            return self.avgpool(x)
    return SwinWrapper(m.features)

_LOADERS = {
    "resnet50":           _resnet50,
    "resnet101":          _resnet101,
    "efficientnet_b0":    _effb0,
    "efficientnet_b3":    _effb3,
    "mobilenet_v3_large": _mv3l,
    "mobilenet_v3_small": _mv3s,
    "convnext_tiny":      _convnext_tiny,
    "efficientnet_v2_s":  _effv2s,
    "swin_t":             _swin_t,
}


def build_backbone(name: str, pretrained: bool = False) -> tuple[nn.Module, int]:
    if name not in _LOADERS:
        raise ValueError(f"Unknown backbone: {name}. Available: {list(_LOADERS)}")
    return _LOADERS[name](pretrained), BACKBONE_INFO[name]["feat_dim"]
