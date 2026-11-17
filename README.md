# factcheck-agent-lab

[English](README.en.md) | **한국어**

LLM으로 실세계 클레임(claim)을 팩트체크하는 실험 저장소입니다.
[AVeriTeC](https://huggingface.co/chenxwh/AVeriTeC) dev 셋(500개 클레임)에서 클레임마다 근거를 찾아
LLM에 넣고, 아래 4개 라벨 중 하나로 판정하게 한 뒤 정답과 비교합니다.

| 라벨 | dev 셋 개수 |
|---|---:|
| Supported | 122 |
| Refuted | 305 |
| Not Enough Evidence | 35 |
| Conflicting Evidence/Cherrypicking | 38 |

현재 공개된 방법은 세 가지입니다.

- **No RAG** (`no_rag.py`): 근거 없이 클레임만 넣고, 모델이 가진 지식만으로 판정하게 합니다.
- **Simple RAG** (`simple_rag.py`): 클레임 문장 그대로 클레임별 문서 풀(Chroma)을 한 번 검색해 상위 k개 문장을 근거로 넣습니다.
- **Oracle RAG** (`oracle_rag.py`): 검색 대신 데이터셋에 사람이 달아 둔 정답 QA(질문·답변)를 근거로 넣습니다.
  "검색이 완벽하다면 어디까지 가능한가"를 보는 상한선 실험입니다. 판정이 그대로 적힌 `justification` 필드는 넣지 않습니다.

세 방법은 같은 시스템 프롬프트([prompts/base.txt](prompts/base.txt))를 쓰고, 사용자 메시지에 넣는 **근거만** 다릅니다
(없음 → 검색한 문장 → 정답 QA).
LLM은 `reasoning`, `verdict`, `confidence`를 구조화된 출력(JSON)으로 돌려줍니다.

## 결과

`gpt-5-nano`, `prompts/base.txt`, dev 셋 500개 전체, 실행 1회 기준입니다.

| 방법 | 근거 | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| No RAG | 없음 | 0.434 | 0.364 | 0.510 |
| Simple RAG | 검색 상위 10문장 | 0.482 | 0.413 | 0.550 |
| Oracle RAG | 정답 QA | **0.766** | **0.625** | **0.786** |

라벨별 F1과 예측 개수(괄호):

| 라벨 (정답 개수) | No RAG | Simple RAG | Oracle RAG |
|---|---:|---:|---:|
| Supported (122) | 0.497 (47) | 0.635 (89) | 0.808 (91) |
| Refuted (305) | 0.594 (170) | 0.597 (174) | 0.866 (293) |
| Not Enough Evidence (35) | 0.194 (274) | 0.185 (181) | 0.385 (95) |
| Conflicting Evidence/Cherrypicking (38) | 0.170 (9) | 0.234 (56) | 0.441 (21) |

결과에서 보이는 점:

- **검색이 가장 큰 병목입니다.** 검색 근거는 근거 없음 대비 accuracy를 4.8%p(0.434 → 0.482) 올리는 데 그치지만,
  근거를 정답 QA로 바꾸면 33.2%p(→ 0.766) 오릅니다.
- **모델은 근거가 부족하면 Not Enough Evidence로 판단을 보류합니다.** 정답은 35개인데
  No RAG는 274개, Simple RAG는 181개를 이 라벨로 예측했습니다. 근거가 좋아질수록 이 과잉 예측이 줄고(Oracle 95개),
  그만큼 Refuted recall이 오릅니다(0.46 → 0.47 → 0.85).
- **근거가 완벽해도 Not Enough Evidence와 Conflicting은 어렵습니다.** Oracle에서도 두 라벨의 F1은 0.4 안팎입니다.
  이 부분은 검색보다 프롬프트·라벨 정의 쪽 문제로 보입니다.

참고 사항:

- 실행 1회 결과라 LLM 출력의 무작위성에 따른 편차가 있습니다. 100개 샘플로 돌렸을 때 Simple RAG(k=10) accuracy는 0.50이었습니다.
- Simple RAG의 검색 대상은 AVeriTeC가 제공하는 클레임별 knowledge store입니다. 이 파일은 type 순(gold → question → ...)으로
  정렬돼 있고, 기본 설정은 앞 50개 문서만 쓰므로 정답 출처 문서가 포함되기 쉬운, 실제 웹 검색보다 유리한 조건입니다.
- 출력 파싱에 실패하면 `Not Enough Evidence`로 처리합니다 (위 세 실행은 실패 0건).

## 재현 방법

### 1. 환경 설정

Python 3.13과 [uv](https://docs.astral.sh/uv/)가 필요합니다.

```bash
uv sync
cp .env.example .env   # OPENAI_API_KEY를 채운다
```

LLM 호출은 [LiteLLM](https://docs.litellm.ai/)을 거치므로 `--model`로 다른 모델을 지정할 수 있습니다
(해당 공급자의 API 키를 `.env`에 추가).

### 2. 데이터 다운로드 (모든 방법 공통)

```bash
scripts/download_data.sh
```

`datasets/dev.json`(클레임 + 정답)과 knowledge store를 받습니다.
knowledge store는 zip이 약 11GB, 압축을 풀면 약 34GB입니다. 끊기면 다시 실행해 이어 받을 수 있습니다.
No RAG·Oracle RAG만 돌릴 거라면 `scripts/download_data.sh --skip-knowledge-store`로 `dev.json`만 받으면 됩니다.

### 3. 검색 DB 구축 (Simple RAG만 필요)

```bash
scripts/build_db.sh
```

클레임마다 문서 앞 50개 × 문서당 앞 200문장을 `BAAI/bge-small-en-v1.5`로 임베딩해
`datasets/chroma_db_head50`에 클레임별 컬렉션(`claim_<id>`)으로 저장합니다 (약 7GB).
Apple Silicon이면 MPS, 아니면 CPU를 씁니다. 이미 구축된 클레임은 건너뜁니다.

### 4. 실험 실행

```bash
scripts/run_no_rag.sh       # → results/norag_base_n500/
scripts/run_simple_rag.sh   # → results/rag_base_k10_n500/
scripts/run_oracle_rag.sh   # → results/oracle_base_n500/
```

인자를 덧붙이면 기본값을 덮어씁니다. 예: `scripts/run_simple_rag.sh --num-samples 10 --top-k 3`.
전체 인자는 `uv run python simple_rag.py --help`로 볼 수 있습니다.
동시 요청 10개 기준으로 500개 실행에 No RAG·Simple RAG 각각 약 11분, Oracle RAG 약 8분이 걸렸습니다.

실행 하나는 `results/<run-name>/` 폴더 하나로 저장됩니다.

| 파일 | 내용 |
|---|---|
| `config.json` | 실행 인자, 시스템 프롬프트 전문, 사용자 메시지 틀, git 커밋, 소요 시간·토큰 사용량, 지표 |
| `results.pkl` | LLM 원본 응답 (LiteLLM `ModelResponse` 리스트) |
| `evidence.json` | 클레임별로 모델에 넣은 근거 (Simple RAG·Oracle RAG만) |

폴더 이름은 인자에서 자동으로 만들어집니다 (`<방법>_<프롬프트>_k<top-k>_n<샘플수>`, 이미 있으면 `_r2`, `_r3` ...).

### 5. 평가

실행이 끝나면 지표가 자동으로 출력되고 `config.json`에 기록됩니다. 여러 실행을 다시 비교하려면:

```bash
scripts/eval.sh                                              # results/ 아래 전체 비교
scripts/eval.sh results/norag_base_n500 results/rag_base_k10_n500 results/oracle_base_n500
scripts/eval.sh results/rag_base_k10_n500 --show             # 정답/예측 쌍까지 출력
```

## 파일 구성

```
├── no_rag.py            # No RAG (근거 없음)
├── simple_rag.py        # Simple RAG
├── oracle_rag.py        # Oracle RAG
├── common.py            # CLI 인자, 프롬프트 로딩, 출력 스키마, 실행 기록 저장
├── eval.py              # accuracy / F1 계산 및 실행 비교
├── download_data.py     # dev.json + knowledge store 다운로드
├── build_db.py          # knowledge store → 클레임별 Chroma 컬렉션
├── prompts/
│   └── base.txt         # 시스템 프롬프트
└── scripts/             # 위 스크립트들의 기본 인자를 담은 실행 스크립트
```

`datasets/`와 `results/`는 용량 문제로 저장소에 포함하지 않습니다.
