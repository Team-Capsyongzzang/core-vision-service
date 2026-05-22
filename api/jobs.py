"""
api/jobs.py
===========
POST /jobs — 민규 스크리닝 결과 수신.

민규 스펙:
  - image_id       : 이미지 쌍 ID
  - pre_image_uri  : S3 URI (재난 전)
  - post_image_uri : S3 URI (재난 후)
  - priority       : 3=high / 2=medium / 1=low / 0=no_building
  - created_at     : job 생성 시간 (ISO 8601)

처리 순서:
  priority 내림차순 → 같은 priority면 created_at 오름차순
"""

from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.queue_manager import queue_manager, QueueItem

router = APIRouter(tags=["jobs"])

PRIORITY_MAP = {3: "high", 2: "medium", 1: "low", 0: "no_building"}


class JobRequest(BaseModel):
    image_id:       str
    pre_image_uri:  str
    post_image_uri: str
    priority:       int
    created_at:     str


@router.post("/jobs", status_code=200)
async def create_job(job: JobRequest):
    """
    민규 스크리닝 결과를 받아 우선순위 큐에 등록합니다.
    """
    # 유효성 검사
    if not job.pre_image_uri:
        raise HTTPException(400, detail="missing pre_image_uri")
    if not job.post_image_uri:
        raise HTTPException(400, detail="missing post_image_uri")
    if job.priority not in PRIORITY_MAP:
        raise HTTPException(400, detail=f"invalid priority: {job.priority}. Must be 0/1/2/3")

    # created_at 파싱
    try:
        created_at_dt = datetime.fromisoformat(job.created_at)
    except ValueError:
        raise HTTPException(400, detail="invalid created_at format. Use ISO 8601")

    item = QueueItem(
        image_id       = job.image_id,
        pre_image_uri  = job.pre_image_uri,
        post_image_uri = job.post_image_uri,
        priority       = job.priority,
        priority_label = PRIORITY_MAP[job.priority],
        created_at     = created_at_dt,
    )

    await queue_manager.enqueue(item)

    return {
        "status":   "accepted",
        "image_id": job.image_id,
    }
