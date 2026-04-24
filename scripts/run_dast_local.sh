#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORTS_DIR="${ROOT_DIR}/reports/dast"
FIXTURE_PORT="${FIXTURE_PORT:-8013}"
LOCAL_TARGET_URL="${LOCAL_TARGET_URL:-http://127.0.0.1:${FIXTURE_PORT}}"
ZAP_TARGET_URL="${ZAP_TARGET_URL:-http://host.docker.internal:${FIXTURE_PORT}}"

mkdir -p "${REPORTS_DIR}"

pushd "${ROOT_DIR}/services/orchestrator" >/dev/null
go run ./cmd/dast_fixture >"${REPORTS_DIR}/dast-fixture.log" 2>&1 &
FIXTURE_PID=$!
popd >/dev/null

cleanup() {
  kill "${FIXTURE_PID}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 30); do
  if curl -fsS "${LOCAL_TARGET_URL}/health" >/dev/null; then
    break
  fi
  sleep 1
done

curl -fsS "${LOCAL_TARGET_URL}/api/info" >"${REPORTS_DIR}/api-info.json"

docker run --rm \
  -v "${REPORTS_DIR}:/zap/wrk:rw" \
  ghcr.io/zaproxy/zaproxy:stable \
  zap-baseline.py \
  -t "${ZAP_TARGET_URL}" \
  -j \
  -m 2 \
  -T 5 \
  -I \
  -J zap-report.json \
  -r zap-report.html \
  -x zap-report.xml

python3 "${ROOT_DIR}/security_gate.py" \
  --threshold medium \
  --zap "${REPORTS_DIR}/zap-report.json"

python3 "${ROOT_DIR}/security_gate.py" \
  --threshold high \
  --policy "${ROOT_DIR}/dast_policy.yaml" \
  --zap "${REPORTS_DIR}/zap-report.json"
