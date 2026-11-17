# 분석

메인 [README](../../README.md)의 실험 결과를 더 자세히 들여다본 분석 모음입니다.
각 문서는 질문, 사용한 실행, 결과, 해석 순서로 정리되어 있고, 숫자는 [analysis/](../../analysis/)의 스크립트로 다시 만들 수 있습니다.

| 문서 | 질문 | 한 줄 요약 |
|---|---|---|
| [agent_steps.md](agent_steps.md) | 에이전트가 검색을 오래 할수록 결과가 어떻게 되나? | 검색 1회 후 답하면 accuracy 0.68, 최대 스텝(10)까지 가면 0.21. 최대 스텝 도달은 대부분 Not Enough Evidence로 끝난다 |
| [confidence.md](confidence.md) | 모델이 낸 확신도는 정답 여부와 관련이 있나? | 전체로는 강하게 관련(AUROC 0.84–0.90)되지만 대부분 예측 라벨(특히 Not Enough Evidence)에서 오는 신호. 같은 라벨 안에서는 약하고, Conflicting에서는 없다 |

## 분석 스크립트 실행

`results/`의 실행 폴더가 있어야 합니다 (메인 README의 재현 방법 참고).

```bash
uv run python analysis/agent_steps.py    # 기본: results/agent_base_k10_n500
uv run python analysis/confidence.py     # 기본: No RAG, Simple RAG, Agent, Oracle RAG (n=500)
uv run python analysis/confidence.py results/<run-name> ...   # 실행 지정
```
