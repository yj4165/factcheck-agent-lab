#!/usr/bin/env bash
# 데이터 다운로드: dev.json + knowledge store(zip 약 11GB, 풀면 약 34GB)를 datasets/ 에 받는다
# 인자를 덧붙이면 아래 값을 덮어쓴다. 예: scripts/download_data.sh --num-claims 100
# 이미 받은 파일은 건너뛰므로 끊겼을 때 다시 실행하면 이어서 진행한다.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python src/download_data.py \
  --data-dir datasets \
  --start-claim 0 \
  --num-claims 500 \
  "$@"
