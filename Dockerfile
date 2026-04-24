FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=120 \
    fastapi==0.111.0 \
    uvicorn==0.29.0 \
    python-multipart==0.0.9 \
    Pillow==10.3.0 \
    numpy==1.26.4 \
    matplotlib==3.8.4 \
    boto3==1.34.0 \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY fruit_ripeness_classifier/src ./fruit_ripeness_classifier/src
COPY app.py .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]