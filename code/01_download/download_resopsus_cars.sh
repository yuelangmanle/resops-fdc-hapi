#!/usr/bin/env bash
# Download the ResOpsUS+CARS v1.0 dataset.
# Zenodo DOI: 10.5281/zenodo.15978041
# Output: ../../data/raw/ResOpsUS+CARS_v10.zip
set -euo pipefail

cd "$(dirname "$0")/../.."

URL="https://zenodo.org/api/records/15978041/files/ResOpsUS%2BCARS_v10.zip/content"
OUT="data/raw/ResOpsUS+CARS_v10.zip"
CHECKSUM_MD5="47af90a5f5a32e2466ffd5cf721816ab"

mkdir -p data/raw

echo "[download] $URL"
echo "[download] output: $OUT"

if [[ -f "$OUT" ]] && command -v md5 >/dev/null 2>&1; then
  actual=$(md5 -q "$OUT" 2>/dev/null || true)
  if [[ "$actual" == "$CHECKSUM_MD5" ]]; then
    echo "[download] already complete and checksum ok"
    exit 0
  fi
fi

max_attempts=50
attempt=0

while true; do
  attempt=$((attempt + 1))
  echo "[download] attempt $attempt"

  if [[ -f "$OUT" ]] && command -v md5 >/dev/null 2>&1; then
    actual=$(md5 -q "$OUT" 2>/dev/null || true)
    if [[ "$actual" == "$CHECKSUM_MD5" ]]; then
      echo "[download] complete and checksum ok"
      break
    fi
  fi

  # Limit each request, resume partial transfers, and retry failures.
  curl -L --retry 5 --retry-all-errors --retry-delay 3 -C - --max-time 600 -o "$OUT" "$URL" \
    && echo "[download] curl returned success" \
    || echo "[download] curl interrupted, will resume"

  if [[ $attempt -ge $max_attempts ]]; then
    echo "[download] ERROR: too many attempts" >&2
    exit 1
  fi

  sleep 5
done

echo "[download] finished, verifying md5..."
if command -v md5 >/dev/null 2>&1; then
  actual=$(md5 -q "$OUT")
  echo "actual:   $actual"
  echo "expected: $CHECKSUM_MD5"
  if [[ "$actual" != "$CHECKSUM_MD5" ]]; then
    echo "[download] ERROR: checksum mismatch" >&2
    exit 1
  fi
fi
echo "[download] OK"
