FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드 복사 (체크포인트 제외 → EBS PVC로 마운트)
COPY main.py .
COPY api/     ./api/
COPY core/    ./core/
COPY models/  ./models/
COPY config/  ./config/

# 체크포인트 마운트 포인트
RUN mkdir -p /app/checkpoints/detector /app/checkpoints/classifier

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
