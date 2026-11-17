#!/usr/bin/env bash
# 결과 평가 (accuracy / F1). 인자가 없으면 results/ 아래 모든 실행 폴더를 비교한다.
# 예: scripts/eval.sh results/norag_base_n500 results/rag_base_k10_n500
#     scripts/eval.sh results/norag_base_n500 --show
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python eval.py "$@"
