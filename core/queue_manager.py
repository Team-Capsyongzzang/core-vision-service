"""
core/queue_manager.py
=====================
우선순위 큐.

정렬 기준:
  1. priority 내림차순 (3 → 2 → 1 → 0)
  2. 같은 priority면 created_at 오름차순 (먼저 생성된 것부터)
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
    image_id:       str
    pre_image_uri:  str
    post_image_uri: str
    priority:       int              # 3=high, 2=medium, 1=low, 0=no_building
    priority_label: str              # "high" / "medium" / "low" / "no_building"
    created_at:     datetime
    status:         Status = Status.WAITING

    def __lt__(self, other: "QueueItem") -> bool:
        """
        정렬 기준:
          priority 높은 것 먼저 (내림차순)
          같은 priority면 created_at 빠른 것 먼저 (오름차순)
        """
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at


class QueueManager:
    def __init__(self):
        self._waiting:    list[QueueItem]      = []
        self._processing: dict[str, QueueItem] = {}
        self._done_count: int = 0
        self._fail_count: int = 0
        self._lock = asyncio.Lock()

    async def enqueue(self, item: QueueItem):
        """큐에 추가 후 정렬."""
        async with self._lock:
            self._waiting.append(item)
            self._waiting.sort()   # QueueItem.__lt__ 기준 정렬

    async def dequeue(self) -> QueueItem | None:
        async with self._lock:
            if not self._waiting:
                return None
            item        = self._waiting.pop(0)
            item.status = Status.PROCESSING
            self._processing[item.image_id] = item
        return item

    async def complete(self, image_id: str):
        async with self._lock:
            self._processing.pop(image_id, None)
            self._done_count += 1

    async def fail(self, image_id: str):
        async with self._lock:
            self._processing.pop(image_id, None)
            self._fail_count += 1

    def get_status(self) -> dict:
        return {
            "waiting":    len(self._waiting),
            "processing": len(self._processing),
            "completed":  self._done_count,
            "failed":     self._fail_count,
        }


queue_manager = QueueManager()
