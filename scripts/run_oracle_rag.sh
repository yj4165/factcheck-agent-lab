#!/usr/bin/env bash
# Oracle RAG: 검색 대신 데이터셋의 정답 QA를 근거로 넣고 판정 (검색 상한선 확인용)
# 인자를 덧붙이면 아래 값을 덮어쓴다. 예: scripts/run_oracle_rag.sh --num-samples 10
# 결과 폴더 이름은 인자에서 자동 생성된다 (직접 정하려면 --run-name).
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python oracle_rag.py \
  --model gpt-5-nano \
  --prompt base \
  --data-path datasets/dev.json \
  --num-samples 500 \
  --concurrency 10 \
  "$@"
