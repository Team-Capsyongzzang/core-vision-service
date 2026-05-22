"""
api/queue.py
============
POST /api/queue — 민규 스크리닝 결과 수신.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.queue_manager import queue_manager, QueueItem

router = APIRouter(prefix="/api", tags=["queue"])


class TileItem(BaseModel):
    tile_id:   str
    pre_path:  str
    post_path: str
    score:     float
    priority:  str        = "medium"
    lat:       float | None = None
    lng:       float | None = None


@router.post("/queue")
async def enqueue_tiles(tiles: list[TileItem]):
    """
    민규 스크리닝 결과를 큐에 등록합니다.
    score 기준 내림차순 정렬 후 순차 처리됩니다.
    """
    if not tiles:
        raise HTTPException(400, "타일 목록이 비어있습니다")

    items = [
        QueueItem(
            tile_id   = t.tile_id,
            pre_path  = t.pre_path,
            post_path = t.post_path,
            score     = t.score,
            priority  = t.priority,
            lat       = t.lat,
            lng       = t.lng,
        )
        for t in tiles
    ]
    await queue_manager.enqueue_batch(items)
    return {"queued": len(items), "message": "ok"}


@router.get("/queue/status")
def queue_status():
    """큐 현황 확인."""
    return queue_manager.get_status()
