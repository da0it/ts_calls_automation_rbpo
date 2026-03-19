#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WHISPERX_VENV_DIR="${WHISPERX_VENV_DIR:-$HOME/whisperx_venv}"

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[ERROR] Missing command: $cmd"
    exit 1
  fi
}

ensure_venv() {
  local venv_path="$1"
  "$PYTHON_BIN" -m venv "$venv_path"
}

pip_install() {
  local pip_bin="$1"
  shift
  "$pip_bin" install --upgrade pip wheel setuptools >/dev/null
  "$pip_bin" install "$@"
}

main() {
  require_cmd "$PYTHON_BIN"
  require_cmd git
  require_cmd go
  require_cmd ffmpeg
  require_cmd ffprobe

  log "Preparing Python venv for entity_extraction..."
  ensure_venv "$ROOT_DIR/.venv"
  pip_install "$ROOT_DIR/.venv/bin/pip" -r "$ROOT_DIR/services/entity_extraction/requirements.txt"

  log "Preparing Python venv for router..."
  ensure_venv "$ROOT_DIR/services/router/venv"
  pip_install "$ROOT_DIR/services/router/venv/bin/pip" -r "$ROOT_DIR/services/router/requirements.txt"

  log "Preparing Python venv for transcription (WhisperX)..."
  ensure_venv "$WHISPERX_VENV_DIR"
  pip_install "$WHISPERX_VENV_DIR/bin/pip" whisperx grpcio protobuf python-dotenv

  log "Downloading Go modules..."
  (
    cd "$ROOT_DIR/services/orchestrator"
    go mod download
  )
  (
    cd "$ROOT_DIR/services/ticket_creation"
    go mod download
  )
  (
    cd "$ROOT_DIR/services/notification_sender"
    go mod download
  )

  log "Bootstrap complete."
  log "Next: set env values in $ROOT_DIR/configs/*.env and run ./scripts/start_linux_stack.sh"
}

main "$@"
