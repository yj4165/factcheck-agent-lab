#!/usr/bin/env bash
# Simple RAG: claim으로 근거를 한 번 검색해 프롬프트에 넣고 판정
# 인자를 덧붙이면 아래 값을 덮어쓴다. 예: scripts/run_simple_rag.sh --num-samples 10
# 결과 폴더 이름은 인자에서 자동 생성된다 (직접 정하려면 --run-name).
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python simple_rag.py \
  --model gpt-5-nano \
  --prompt base \
  --data-path datasets/dev.json \
  --num-samples 500 \
  --concurrency 10 \
  --top-k 10 \
  --db-path datasets/chroma_db_head50 \
  --embedding-model BAAI/bge-small-en-v1.5 \
  "$@"
