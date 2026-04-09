#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 /path/to/audio_folder [output_dir]"
  exit 1
fi

AUDIO_DIR="$1"
OUT_DIR="${2:-./load_test_results}"

if [ ! -d "$AUDIO_DIR" ]; then
  echo "Audio folder not found: $AUDIO_DIR"
  exit 1
fi

if ! command -v k6 >/dev/null 2>&1; then
  echo "k6 is not installed"
  exit 1
fi

BASE_URL="${BASE_URL:-http://localhost:8000}"
USERNAME="${USERNAME:-admin}"
PASSWORD="${PASSWORD:-admin123}"

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

mapfile -d '' FILES < <(find "$AUDIO_DIR" -type f \( -iname '*.wav' -o -iname '*.mp3' -o -iname '*.ogg' -o -iname '*.m4a' \) -print0 | sort -z)

if [ ${#FILES[@]} -eq 0 ]; then
  echo "No audio files found in: $AUDIO_DIR"
  exit 1
fi

AUDIO_FILES=""
for file in "${FILES[@]}"; do
  if [ -z "$AUDIO_FILES" ]; then
    AUDIO_FILES="$file"
  else
    AUDIO_FILES="$AUDIO_FILES,$file"
  fi
done

echo "Audio files: ${#FILES[@]}"
echo "Base URL: $BASE_URL"
echo "Output dir: $OUT_DIR"

AUDIO_LIST_FILE="$OUT_DIR/audio_files.txt"
printf '%s\n' "${FILES[@]}" > "$AUDIO_LIST_FILE"

BASE_URL="$BASE_URL" \
USERNAME="$USERNAME" \
PASSWORD="$PASSWORD" \
AUDIO_FILE_LIST="$AUDIO_LIST_FILE" \
SUMMARY_JSON="$OUT_DIR/summary.json" \
SUMMARY_TEXT="$OUT_DIR/summary.txt" \
k6 run tests/load_test_process_call_advanced.js | tee "$OUT_DIR/k6.log"
