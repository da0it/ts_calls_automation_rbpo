FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WHISPERX_VENV_PYTHON=/opt/whisperx_venv/bin/python \
    FFMPEG_BIN=ffmpeg \
    FFPROBE_BIN=ffprobe

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    build-essential \
    pkg-config \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Запуск whisperx в отдельном виртуальном окружении (необходимо для избежания конфликтов зависимостей с библиотекой pytorch)
RUN python -m venv /opt/whisperx_venv \
    && /opt/whisperx_venv/bin/pip install --no-cache-dir --upgrade pip wheel setuptools \
    && /opt/whisperx_venv/bin/pip install --no-cache-dir whisperx grpcio protobuf python-dotenv

EXPOSE 50051

CMD ["/opt/whisperx_venv/bin/python", "/app/grpc_server.py"]
