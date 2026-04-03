# Linux Deployment

## 1. Prepare machine (Ubuntu example)

```bash
sudo bash scripts/install_deps_ubuntu.sh
```

Install Docker separately if you want postgres from `docker-compose.yml`.

## 2. Clone project

```bash
sudo mkdir -p /opt
sudo chown -R "$USER":"$USER" /opt
cd /opt
git clone <YOUR_REPO_URL> ts_calls_automation_submodule
cd ts_calls_automation_submodule
```

## 3. Configure env files

Edit:

- `configs/transcription.env`
- `configs/routing.env`
- `configs/ticket.env`
- `configs/orchestrator.env`
- `configs/entity.env`

Important:

- Set real `HF_TOKEN` only if your WhisperX model download path requires it.
- Ensure `DATABASE_URL` points to postgres.
- If not using docker postgres, set `START_POSTGRES_WITH_DOCKER=0`.
- By default `ticket_creation` expects a local `ollama` at `http://localhost:11434`
  and uses `OLLAMA_MODEL=gemma3:4b`.

By default `./scripts/start_linux_stack.sh` will also try to start the `ollama` container from `docker-compose.yml`.
If you already run Ollama natively on the host, set `START_OLLAMA_WITH_DOCKER=0`.

If you run outside Docker, start Ollama separately before `./scripts/start_linux_stack.sh`:

```bash
ollama serve
```

If internet is allowed only for the initial model download, run once:

```bash
ollama pull gemma3:4b
```

If the machine must stay fully air-gapped, import/preload the model offline and keep `LLM_PROVIDER=ollama`.

## 4. Bootstrap runtimes

```bash
./scripts/bootstrap_linux.sh
```

This script creates:

- `.venv` for entity extraction
- `services/router/venv`
- `~/whisperx_venv` (WhisperX runtime)
- installs Python/Go dependencies

## 5. Smoke run

```bash
./scripts/start_linux_stack.sh
```

Health check:

```bash
curl http://localhost:8000/health
```

## 6. Run with systemd

1. Edit `deploy/linux/ts-calls.service`:
   - set `User`
   - set `WorkingDirectory`
   - set `HOME`
   - adjust `START_POSTGRES_WITH_DOCKER`

2. Install and enable:

```bash
sudo cp deploy/linux/ts-calls.service /etc/systemd/system/ts-calls.service
sudo systemctl daemon-reload
sudo systemctl enable --now ts-calls.service
```

3. Logs:

```bash
journalctl -u ts-calls.service -f
tail -f .run/logs/orchestrator.log
```

## 7. Update deployment

```bash
cd /opt/ts_calls_automation_submodule
git pull
./scripts/bootstrap_linux.sh
sudo systemctl restart ts-calls.service
```
