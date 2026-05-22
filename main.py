"""
main.py
=======
CORE Vision Service 진입점.

실행:
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.jobs   import router as jobs_router
from core.worker import process_worker
import core.pipeline as pipeline_module


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] CORE Vision Service starting...")

    try:
        from core.pipeline import DisasterPipeline
        pipeline_module.pipeline = DisasterPipeline()
        print("[Startup] Pipeline ready ✓")
    except Exception as e:
        print(f"[Startup] Pipeline load failed: {e}")

    worker_task = asyncio.create_task(process_worker())
    print("[Startup] Worker ready ✓")
    print("[Startup] Server ready")

    yield

    worker_task.cancel()
    print("[Shutdown] Done")


app = FastAPI(
    title   = "CORE Vision Service",
    version = "1.0.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_router)


@app.get("/api/health")
def health():
    from core.queue_manager import queue_manager
    return {
        "status":   "ok",
        "pipeline": pipeline_module.pipeline is not None,
        "queue":    queue_manager.get_status(),
    }
