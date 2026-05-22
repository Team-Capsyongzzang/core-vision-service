"""
main.py
=======
CORE Vision Service — 추론 서버 진입점.

역할:
  1. 민규 스크리닝 결과 수신 (POST /api/queue)
  2. 우선순위 큐 처리 (워커)
  3. 추론 결과를 대시보드 서버로 전송

환경변수:
  DETECTOR_CKPT    탐지기 체크포인트 경로
  CLASSIFIER_CKPT  분류기 체크포인트 경로
  DASHBOARD_URL    결과 전송할 대시보드 주소
  DEVICE           cuda / cpu (기본 cuda)

실행:
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.queue  import router as queue_router
from core.worker import process_worker
import core.pipeline as pipeline_module


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 시작 ────────────────────────────────────────────
    print("[Startup] CORE Vision Service starting...")

    # 파이프라인 로드
    try:
        from core.pipeline import DisasterPipeline
        pipeline_module.pipeline = DisasterPipeline()
        print("[Startup] Pipeline ready ✓")
    except Exception as e:
        print(f"[Startup] Pipeline load failed: {e}")
        print("[Startup] Queue will reject requests until pipeline is ready")

    # 백그라운드 워커 시작
    worker_task = asyncio.create_task(process_worker())
    print("[Startup] Worker ready ✓")
    print("[Startup] Server ready")

    yield

    # ── 종료 ────────────────────────────────────────────
    worker_task.cancel()
    print("[Shutdown] Done")


app = FastAPI(
    title       = "CORE Vision Service",
    description = "위성 이미지 재난 탐지 및 분류 추론 서버",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(queue_router)


@app.get("/api/health")
def health():
    from core.queue_manager import queue_manager
    return {
        "status":   "ok",
        "pipeline": pipeline_module.pipeline is not None,
        "queue":    queue_manager.get_status(),
    }
