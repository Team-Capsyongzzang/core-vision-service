"""
core/queue_manager.py
=====================
우선순위 큐 — 민규 스크리닝 결과를 score 기준 정렬 후 처리.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Status(str, Enum):
    WAITING    = "waiting"
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"


@dataclass
class QueueItem:
    tile_id:   str
    pre_path:  str
    post_path: str
    score:     float
    priority:  str        = "medium"
    lat:       float | None = None
    lng:       float | None = None
    status:    Status     = Status.WAITING
    timestamp: str        = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class QueueManager:
    def __init__(self):
        self._waiting:    list[QueueItem]      = []
        self._processing: dict[str, QueueItem] = {}
        self._done_count: int = 0
        self._fail_count: int = 0
        self._lock = asyncio.Lock()

    async def enqueue_batch(self, items: list[QueueItem]):
        """score 내림차순 정렬 후 큐 추가."""
        async with self._lock:
            self._waiting.extend(items)
            self._waiting.sort(key=lambda x: x.score, reverse=True)

    async def dequeue(self) -> QueueItem | None:
        async with self._lock:
            if not self._waiting:
                return None
            item        = self._waiting.pop(0)
            item.status = Status.PROCESSING
            self._processing[item.tile_id] = item
        return item

    async def complete(self, tile_id: str):
        async with self._lock:
            self._processing.pop(tile_id, None)
            self._done_count += 1

    async def fail(self, tile_id: str):
        async with self._lock:
            self._processing.pop(tile_id, None)
            self._fail_count += 1

    def get_status(self) -> dict:
        return {
            "waiting":    len(self._waiting),
            "processing": len(self._processing),
            "completed":  self._done_count,
            "failed":     self._fail_count,
        }


queue_manager = QueueManager()
