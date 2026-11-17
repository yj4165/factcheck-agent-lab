#!/usr/bin/env bash
# DB 구축: knowledge store를 임베딩해 클레임별 Chroma 컬렉션으로 저장 (scripts/download_data.sh 이후 실행)
# 인자를 덧붙이면 아래 값을 덮어쓴다. 예: scripts/build_db.sh --num-claims 100
# 이미 구축된 클레임은 건너뛴다 (다시 만들려면 --overwrite).
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python build_db.py \
  --knowledge-store datasets/knowledge_store \
  --db-path datasets/chroma_db_head50 \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --start-claim 0 \
  --num-claims 500 \
  --max-docs 50 \
  --max-sents 200 \
  --batch-size 256 \
  "$@"
