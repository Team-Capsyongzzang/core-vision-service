"""
core/pipeline.py
================
탐지기 + 분류기 추론 파이프라인.
서버 시작 시 한 번 로드, 이후 싱글턴으로 사용.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from config.config import cfg, DISASTER_CLASSES
from models.detector import DisasterDetector
from models.classifier import DisasterClassifier

# ── 전처리 ──────────────────────────────────────────────
IMG_TF = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

DIFF_TF = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def _compute_diff(pre: Image.Image, post: Image.Image) -> torch.Tensor:
    pre_arr  = np.array(pre .resize((224, 224)), dtype=np.float32) / 255.0
    post_arr = np.array(post.resize((224, 224)), dtype=np.float32) / 255.0
    diff     = np.abs(post_arr - pre_arr)
    return DIFF_TF(Image.fromarray((diff * 255).astype(np.uint8))).unsqueeze(0)


class DisasterPipeline:
    def __init__(self):
        ic = cfg.inference
        self.device = torch.device(
            ic.device if torch.cuda.is_available() else "cpu"
        )
        print(f"[Pipeline] Device: {self.device}")

        # 탐지기
        self.detector = DisasterDetector().to(self.device)
        ckpt = torch.load(ic.detector_ckpt, map_location=self.device)
        self.detector.load_state_dict(ckpt["model_state_dict"])
        self.detector.eval()
        print(f"[Pipeline] Detector  loaded: {ic.detector_ckpt}")

        # 분류기
        self.classifier = DisasterClassifier().to(self.device)
        ckpt = torch.load(ic.classifier_ckpt, map_location=self.device)
        self.classifier.load_state_dict(ckpt["model_state_dict"])
        self.classifier.eval()
        print(f"[Pipeline] Classifier loaded: {ic.classifier_ckpt}")

        self.threshold  = cfg.detector.threshold
        self.no_dis_idx = DISASTER_CLASSES.index("no_disaster")

    @torch.no_grad()
    def predict(
        self,
        pre_image:  Image.Image,
        post_image: Image.Image,
        tile_id:    str   | None = None,
        priority:   str   | None = None,
        score:      float | None = None,
        lat:        float | None = None,
        lng:        float | None = None,
    ) -> dict:
        # 1단계: 탐지기
        diff_t   = _compute_diff(pre_image, post_image).to(self.device)
        t0       = time.perf_counter()
        det_prob = torch.sigmoid(self.detector(diff_t).view(-1)).item()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        det_ms = round((time.perf_counter() - t0) * 1000, 1)

        has_disaster = det_prob >= self.threshold

        # 2단계: 분류기 (탐지된 경우만)
        if has_disaster:
            post_t     = IMG_TF(post_image).unsqueeze(0).to(self.device)
            t1         = time.perf_counter()
            logits     = self.classifier(post_t)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            cls_ms     = round((time.perf_counter() - t1) * 1000, 1)
            probs      = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx   = int(probs.argmax())
            confidence = float(probs[pred_idx])
            disaster   = DISASTER_CLASSES[pred_idx]
        else:
            cls_ms     = 0.0
            disaster   = "no_disaster"
            confidence = round(1.0 - det_prob, 4)

        return {
            "id":           str(uuid.uuid4())[:8],
            "tile_id":      tile_id,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "disaster":     disaster,
            "confidence":   round(confidence, 4),
            "has_disaster": int(has_disaster),
            "det_prob":     round(det_prob, 4),
            "det_ms":       det_ms,
            "cls_ms":       cls_ms,
            "priority":     priority,
            "score":        score,
            "lat":          lat,
            "lng":          lng,
        }


# 싱글턴
pipeline: DisasterPipeline | None = None
