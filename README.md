# CORE Vision Service

위성 이미지 재난 탐지 및 분류 추론 서버.  
대시보드(`core-dashboard`)와 분리된 별도 서비스입니다.

---

## 구조

```
core-vision-service/
├── main.py              ← FastAPI 진입점
├── api/
│   └── queue.py         ← POST /api/queue  (민규 수신)
│                           GET  /api/queue/status
├── core/
│   ├── pipeline.py      ← 탐지기 + 분류기 추론
│   ├── queue_manager.py ← 우선순위 큐 (score 기준 정렬)
│   └── worker.py        ← 큐 처리 → 대시보드로 결과 전송
├── models/
│   ├── backbone.py      ← 9개 백본 팩토리
│   ├── detector.py      ← 재난 탐지기 (이진 분류)
│   └── classifier.py    ← 재난 분류기 (7 classes)
├── config/
│   └── config.py        ← 환경변수 기반 설정
├── checkpoints/         ← .gitignore (EBS PVC로 마운트)
│   ├── detector/best_ft.pth
│   └── classifier/best_ft.pth
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 흐름

```
민규 스크리닝
    ↓ POST /api/queue
우선순위 큐 (score 내림차순)
    ↓ 워커가 순차 처리
탐지기 → 분류기
    ↓ POST DASHBOARD_URL
대시보드 서버 (/api/ingest)
```

---

## 로컬 실행

```bash
# 가상환경
python -m venv venv
source venv/bin/activate

# PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 환경변수
cp .env.example .env
# DETECTOR_CKPT, CLASSIFIER_CKPT 경로 수정

# 체크포인트 복사 (학습 리포에서)
cp ../xbd_disaster_classifier/checkpoints/detector/best_ft.pth  checkpoints/detector/
cp ../xbd_disaster_classifier/checkpoints/classifier/best_ft.pth checkpoints/classifier/

# 실행
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/api/health` | 서버 상태 + 큐 현황 |
| `POST` | `/api/queue` | 스크리닝 결과 수신 |
| `GET` | `/api/queue/status` | 큐 처리 현황 |

### POST /api/queue 스펙

```json
[
  {
    "tile_id":   "region_A_00000371",
    "pre_path":  "/data/pre/region_A_pre.png",
    "post_path": "/data/post/region_A_post.png",
    "score":     0.87,
    "priority":  "high",
    "lat":       34.05,
    "lng":       -118.24
  }
]
```

### 대시보드로 전송하는 결과

```json
{
  "id":           "a1b2c3d4",
  "tile_id":      "region_A_00000371",
  "timestamp":    "2026-05-08T12:34:56Z",
  "disaster":     "wildfire",
  "confidence":   0.87,
  "has_disaster": 1,
  "det_prob":     0.92,
  "det_ms":       6.8,
  "cls_ms":       7.1,
  "priority":     "high",
  "score":        0.87,
  "lat":          34.05,
  "lng":          -118.24
}
```

---

## 배포

```bash
# ECR 빌드 & 푸시
docker build -t core-vision-service .
docker tag core-vision-service:latest <ECR_REPO>/core-vision-service:latest
docker push <ECR_REPO>/core-vision-service:latest
```

민영이(인프라)에게 전달:
```
포트           : 8000
GPU 필요       : nvidia.com/gpu: 1 (p3.xlarge)
PVC 필요       : checkpoints/ (5Gi)
환경변수       : .env.example 참고
Service 이름   : vision-service:8000
```
