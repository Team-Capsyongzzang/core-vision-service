"""
core/worker.py
==============
백그라운드 워커.

1. 큐에서 QueueItem 꺼냄
2. S3 URI에서 이미지 다운로드
3. 파이프라인 추론
4. 결과를 대시보드 서버로 POST
"""

from __future__ import annotations

import asyncio
import io

import boto3
import httpx
from PIL import Image

from config.config import cfg
from core.queue_manager import queue_manager
import core.pipeline as pipeline_module

# S3 클라이언트 (IAM Role로 자동 인증 — 별도 키 불필요)
s3 = boto3.client("s3")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """s3://bucket/key → (bucket, key)"""
    parts = uri.replace("s3://", "").split("/", 1)
    return parts[0], parts[1]


def _load_from_s3(uri: str) -> Image.Image:
    """S3 URI에서 이미지를 직접 메모리로 읽어옵니다."""
    bucket, key = _parse_s3_uri(uri)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")


async def process_worker():
    print("[Worker] Started")
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            item = await queue_manager.dequeue()

            if item is None:
                await asyncio.sleep(0.1)
                continue

            if pipeline_module.pipeline is None:
                await queue_manager.fail(item.image_id)
                print(f"[Worker] ✗ Pipeline not ready: {item.image_id}")
                continue

            try:
                # S3에서 이미지 다운로드
                pre_img  = await asyncio.get_event_loop().run_in_executor(
                    None, _load_from_s3, item.pre_image_uri
                )
                post_img = await asyncio.get_event_loop().run_in_executor(
                    None, _load_from_s3, item.post_image_uri
                )

                # 추론
                result = pipeline_module.pipeline.predict(
                    pre_image  = pre_img,
                    post_image = post_img,
                    tile_id    = item.image_id,
                    priority   = item.priority_label,
                    lat        = None,
                    lng        = None,
                )

                await queue_manager.complete(item.image_id)

                print(
                    f"[Worker] ✓ {item.image_id:<28} "
                    f"{result['disaster']:<15} "
                    f"conf={result['confidence']:.2f} "
                    f"({result['det_ms']+result['cls_ms']:.0f}ms) "
                    f"priority={item.priority}({item.priority_label})"
                )

                # 대시보드로 결과 전송
                try:
                    await client.post(cfg.inference.dashboard_url, json=result)
                except Exception as e:
                    print(f"[Worker] ⚠ Dashboard notify failed: {e}")

            except Exception as e:
                await queue_manager.fail(item.image_id)
                print(f"[Worker] ✗ {item.image_id}: {e}")
