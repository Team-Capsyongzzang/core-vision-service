"""
config/config.py
================
추론 전용 설정. 모든 경로는 환경변수로 주입.
"""

import os
from dataclasses import dataclass, field

DISASTER_CLASSES = [
    'no_disaster', 'earthquake', 'flood',
    'hurricane',   'tornado',   'tsunami', 'wildfire',
]
IDX_TO_CLASS = {i: c for i, c in enumerate(DISASTER_CLASSES)}
CLASS_TO_IDX = {c: i for i, c in enumerate(DISASTER_CLASSES)}


@dataclass
class ModelConfig:
    backbone:    str   = "resnet101"
    num_classes: int   = len(DISASTER_CLASSES)
    dropout:     float = 0.3
    image_size:  int   = 224


@dataclass
class DetectorConfig:
    backbone:   str   = "resnet101"
    dropout:    float = 0.3
    threshold:  float = 0.4
    image_size: int   = 224


@dataclass
class InferenceConfig:
    detector_ckpt:   str = os.getenv("DETECTOR_CKPT",   "./checkpoints/detector/best_ft.pth")
    classifier_ckpt: str = os.getenv("CLASSIFIER_CKPT", "./checkpoints/classifier/best_ft.pth")
    device:          str = os.getenv("DEVICE",           "cuda")
    # 결과를 전송할 대시보드 서버 주소
    dashboard_url:   str = os.getenv("DASHBOARD_URL",   "http://dashboard-service:8001/api/ingest")


@dataclass
class Config:
    model:     ModelConfig     = field(default_factory=ModelConfig)
    detector:  DetectorConfig  = field(default_factory=DetectorConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


cfg = Config()
