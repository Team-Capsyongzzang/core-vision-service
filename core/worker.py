"""
core/worker.py
==============
백그라운드 워커.

1. 큐에서 타일 꺼냄
2. 파이프라인 추론
3. 결과를 대시보드 서버로 POST
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
from PIL import Image

from config.config import cfg
from core.queue_manager import queue_manager
import core.pipeline as pipeline_module


async def process_worker():
    """큐를 지속적으로 처리하는 백그라운드 워커."""
    print("[Worker] Started")
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            item = await queue_manager.dequeue()

            if item is None:
                await asyncio.sleep(0.1)
                continue

            if pipeline_module.pipeline is None:
                await queue_manager.fail(item.tile_id)
                print(f"[Worker] ✗ Pipeline not ready: {item.tile_id}")
                continue

            try:
                # 이미지 로드
                if not Path(item.pre_path).exists():
                    raise FileNotFoundError(f"pre_path not found: {item.pre_path}")
                if not Path(item.post_path).exists():
                    raise FileNotFoundError(f"post_path not found: {item.post_path}")

                pre_img  = Image.open(item.pre_path ).convert("RGB")
                post_img = Image.open(item.post_path).convert("RGB")

                # 추론
                result = pipeline_module.pipeline.predict(
                    pre_image  = pre_img,
                    post_image = post_img,
                    tile_id    = item.tile_id,
                    priority   = item.priority,
                    score      = item.score,
                    lat        = item.lat,
                    lng        = item.lng,
                )

                await queue_manager.complete(item.tile_id)

                print(
                    f"[Worker] ✓ {item.tile_id:<28} "
                    f"{result['disaster']:<15} "
                    f"conf={result['confidence']:.2f} "
                    f"({result['det_ms']+result['cls_ms']:.0f}ms) "
                    f"priority={item.priority}"
                )

                # 대시보드 서버로 결과 전송
                dashboard_url = cfg.inference.dashboard_url
                try:
                    await client.post(dashboard_url, json=result)
                except Exception as e:
                    print(f"[Worker] ⚠ Dashboard notify failed: {e}")

            except Exception as e:
                await queue_manager.fail(item.tile_id)
                print(f"[Worker] ✗ {item.tile_id}: {e}")
