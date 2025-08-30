#!/usr/bin/env bash
set -euo pipefail

# Timed rehearsal against a running backend at $BASE_URL.
# Assumes dataset is already mounted inside the container at /app/data/api_datasets/$DATASET_ID/images
# Usage:
#   bash scripts/utilities/rehearsal.sh EVAL_2025 http://localhost:8000

DATASET_ID=${1:-EVAL_2025}
BASE_URL=${2:-http://localhost:8000}

echo "Rehearsal starting for dataset_id=${DATASET_ID} at ${BASE_URL}"

echo "Checking health..."
curl -sf "${BASE_URL}/health" >/dev/null
echo "Health OK"

echo "Starting timed /process-all..."
START_TS=$(date +%s)
RESP=$(curl -s -X POST "${BASE_URL}/process-all" \
  -F "dataset_id=${DATASET_ID}" \
  -F "parallel_processing=true" \
  -F "max_workers=8" \
  -F "gpu_acceleration=true" \
  -F "batch_size=50" \
  -F "optimization_level=speed")
END_TS=$(date +%s)
ELAPSED=$((END_TS-START_TS))

echo "process-all elapsed seconds: ${ELAPSED}"
echo "Raw response: ${RESP}" | sed 's/\n/ /g'

# Extract service-reported processing_time if present
SERVICE_TIME=$(python - <<'PY'
import json,sys
try:
    data=json.loads(sys.stdin.read())
    print(int(data.get('processing_time',0)))
except Exception:
    print(0)
PY
<<<"${RESP}")

echo "service processing_time (s): ${SERVICE_TIME}"

echo "Rehearsal completed."

