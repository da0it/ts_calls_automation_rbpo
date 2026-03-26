# Docker Deployment

## Prerequisites

- Linux host with Docker Engine + Docker Compose plugin
- At least 16 GB RAM recommended (WhisperX on CPU)
- Enough disk for model caches

## 1. Configure env

Edit these files:

- `configs/transcription.env`
- `configs/routing.env`
- `configs/ticket.env`
- `configs/orchestrator.env`
- `configs/notification.env`
- `configs/entity.env`

Important settings:

- `HF_TOKEN` if your WhisperX model download path needs authentication
- `LLM_PROVIDER=ollama`
- `OLLAMA_MODEL=qwen2.5:7b`

## 2. Build

```bash
docker compose build
```

## 3. Start

```bash
docker compose up -d
```

If you want Docker to download the Ollama model for you once:

```bash
docker compose --profile ollama-pull up ollama-model
```

If the host must stay air-gapped, preload/import the model into `ollama` manually and skip the command above.

## 4. Verify

```bash
docker compose ps
curl http://localhost:8000/health
curl http://localhost:11434/api/tags
```

## 5. Test call processing

```bash
curl -s -X POST http://localhost:8000/api/v1/process-call \
  -F "audio=@services/transcription/dengi.mp3" | jq .
```

## 6. Logs

```bash
docker compose logs -f orchestrator
docker compose logs -f transcription
docker compose logs -f entity_extraction
```

## 7. Stop

```bash
docker compose down
```

With persistent postgres + model cache volumes:

```bash
docker compose down
docker volume ls | grep ts_calls_automation_submodule
```
