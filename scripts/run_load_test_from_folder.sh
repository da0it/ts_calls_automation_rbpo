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
ENABLE_AUDIO_PRECHECK="${ENABLE_AUDIO_PRECHECK:-1}"
MIN_AUDIO_SECONDS="${MIN_AUDIO_SECONDS:-2}"

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

mapfile -d '' FILES < <(find "$AUDIO_DIR" -type f \( -iname '*.wav' -o -iname '*.mp3' -o -iname '*.ogg' -o -iname '*.m4a' \) -print0 | sort -z)

if [ ${#FILES[@]} -eq 0 ]; then
  echo "No audio files found in: $AUDIO_DIR"
  exit 1
fi

echo "Audio files: ${#FILES[@]}"
echo "Base URL: $BASE_URL"
echo "Output dir: $OUT_DIR"

AUDIO_LIST_FILE="$OUT_DIR/audio_files.txt"
VALID_AUDIO_LIST_FILE="$OUT_DIR/audio_files_valid.txt"
REJECTED_AUDIO_CSV="$OUT_DIR/audio_files_rejected.csv"
PRECHECK_SUMMARY_TXT="$OUT_DIR/precheck_summary.txt"
printf '%s\n' "${FILES[@]}" > "$AUDIO_LIST_FILE"

csv_escape() {
  local value="${1//\"/\"\"}"
  printf '"%s"' "$value"
}

VALID_FILES=("${FILES[@]}")

if [ "$ENABLE_AUDIO_PRECHECK" = "1" ]; then
  echo "Running audio precheck..."

  FFPROBE_AVAILABLE=0
  if command -v ffprobe >/dev/null 2>&1; then
    FFPROBE_AVAILABLE=1
  fi

  : > "$VALID_AUDIO_LIST_FILE"
  printf 'path,reason,duration_sec\n' > "$REJECTED_AUDIO_CSV"

  valid_count=0
  rejected_count=0
  empty_count=0
  too_short_count=0
  probe_failed_count=0
  VALID_FILES=()

  for file in "${FILES[@]}"; do
    reason=""
    duration="n/a"

    if [ ! -s "$file" ]; then
      reason="empty_file"
      empty_count=$((empty_count + 1))
    fi

    if [ -z "$reason" ] && [ "$FFPROBE_AVAILABLE" -eq 1 ]; then
      duration="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file" 2>/dev/null | tail -n 1 | tr -d '\r')"
      if [ -z "$duration" ]; then
        reason="ffprobe_failed"
        duration="n/a"
        probe_failed_count=$((probe_failed_count + 1))
      elif awk -v d="$duration" -v m="$MIN_AUDIO_SECONDS" 'BEGIN { exit !((d + 0.0) < (m + 0.0)) }'; then
        reason="too_short"
        too_short_count=$((too_short_count + 1))
      fi
    fi

    if [ -n "$reason" ]; then
      {
        csv_escape "$file"
        printf ','
        csv_escape "$reason"
        printf ','
        csv_escape "$duration"
        printf '\n'
      } >> "$REJECTED_AUDIO_CSV"
      rejected_count=$((rejected_count + 1))
      continue
    fi

    VALID_FILES+=("$file")
    printf '%s\n' "$file" >> "$VALID_AUDIO_LIST_FILE"
    valid_count=$((valid_count + 1))
  done

  {
    echo "Audio precheck summary"
    echo
    echo "input_total: ${#FILES[@]}"
    echo "valid_total: $valid_count"
    echo "rejected_total: $rejected_count"
    echo "ffprobe_available: $FFPROBE_AVAILABLE"
    echo "min_audio_seconds: $MIN_AUDIO_SECONDS"
    echo "rejected_empty_file: $empty_count"
    echo "rejected_too_short: $too_short_count"
    echo "rejected_ffprobe_failed: $probe_failed_count"
    echo "valid_list: $VALID_AUDIO_LIST_FILE"
    echo "rejected_csv: $REJECTED_AUDIO_CSV"
  } > "$PRECHECK_SUMMARY_TXT"

  echo "Precheck valid files: $valid_count"
  echo "Precheck rejected files: $rejected_count"
  echo "Precheck summary: $PRECHECK_SUMMARY_TXT"

  if [ ${#VALID_FILES[@]} -eq 0 ]; then
    echo "No audio files passed precheck."
    exit 1
  fi
else
  printf '%s\n' "${FILES[@]}" > "$VALID_AUDIO_LIST_FILE"
fi

BASE_URL="$BASE_URL" \
USERNAME="$USERNAME" \
PASSWORD="$PASSWORD" \
AUDIO_FILE_LIST="$VALID_AUDIO_LIST_FILE" \
SUMMARY_JSON="$OUT_DIR/summary.json" \
SUMMARY_TEXT="$OUT_DIR/summary.txt" \
k6 run tests/load_test_process_call_advanced.js | tee "$OUT_DIR/k6.log"
