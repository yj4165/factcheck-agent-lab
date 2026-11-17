# factcheck-agent-lab

**English** | [한국어](README.md)

Experiments on fact-checking real-world claims with LLMs.
On the [AVeriTeC](https://huggingface.co/chenxwh/AVeriTeC) dev set (500 claims), each claim is given to an LLM
together with some evidence, the LLM classifies it into one of the four labels below, and the prediction is compared with the gold label.

| Label | Count in dev set |
|---|---:|
| Supported | 122 |
| Refuted | 305 |
| Not Enough Evidence | 35 |
| Conflicting Evidence/Cherrypicking | 38 |

Three methods are currently available.

- **No RAG** (`no_rag.py`): the claim is given with no evidence, so the model must rely on its own knowledge.
- **Simple RAG** (`simple_rag.py`): the claim text is used as a query to search the claim's document pool (Chroma) once, and the top-k sentences are given as evidence.
- **Oracle RAG** (`oracle_rag.py`): instead of retrieval, the human-annotated gold QA pairs (questions and answers) from the dataset are given as evidence.
  This is an upper-bound experiment: how far can the model get if retrieval were perfect? The `justification` field, which states the verdict directly, is not included.

All three methods use the same system prompt ([prompts/base.txt](prompts/base.txt)) and differ **only in the evidence** placed in the user message
(none → retrieved sentences → gold QA).
The LLM returns `reasoning`, `verdict` and `confidence` as structured output (JSON).

## Results

`gpt-5-nano`, `prompts/base.txt`, all 500 dev claims, single run.

| Method | Evidence | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| No RAG | none | 0.434 | 0.364 | 0.510 |
| Simple RAG | top-10 retrieved sentences | 0.482 | 0.413 | 0.550 |
| Oracle RAG | gold QA | **0.766** | **0.625** | **0.786** |

Per-label F1, with the number of predictions in parentheses:

| Label (gold count) | No RAG | Simple RAG | Oracle RAG |
|---|---:|---:|---:|
| Supported (122) | 0.497 (47) | 0.635 (89) | 0.808 (91) |
| Refuted (305) | 0.594 (170) | 0.597 (174) | 0.866 (293) |
| Not Enough Evidence (35) | 0.194 (274) | 0.185 (181) | 0.385 (95) |
| Conflicting Evidence/Cherrypicking (38) | 0.170 (9) | 0.234 (56) | 0.441 (21) |

Observations:

- **Retrieval is the main bottleneck.** Retrieved evidence improves accuracy by only 4.8 points over no evidence (0.434 → 0.482),
  while gold QA evidence improves it by 33.2 points (→ 0.766).
- **When evidence is insufficient, the model falls back to Not Enough Evidence.** There are only 35 such claims, yet
  No RAG predicts this label 274 times and Simple RAG 181 times. Better evidence reduces this over-prediction (95 with Oracle),
  and Refuted recall rises accordingly (0.46 → 0.47 → 0.85).
- **Not Enough Evidence and Conflicting remain hard even with perfect evidence.** Even with Oracle RAG, F1 for both labels is around 0.4.
  This looks like a prompt / label-definition problem rather than a retrieval problem.

Notes:

- These are single-run results, so there is some variance from LLM sampling. On a 100-claim run, Simple RAG (k=10) reached 0.50 accuracy.
- Simple RAG searches the per-claim knowledge store provided by AVeriTeC. These files are sorted by type (gold → question → ...)
  and the default setting uses only the first 50 documents, so source documents of the gold answers are likely included.
  This is an easier setting than real web search.
- Outputs that fail to parse are counted as `Not Enough Evidence` (none of the three runs above had any failures).

## Reproduction

### 1. Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # fill in OPENAI_API_KEY
```

LLM calls go through [LiteLLM](https://docs.litellm.ai/), so other models can be used via `--model`
(add the provider's API key to `.env`).

### 2. Download data (all methods)

```bash
scripts/download_data.sh
```

Downloads `datasets/dev.json` (claims + gold labels) and the knowledge store.
The knowledge store is about 11GB zipped and about 34GB extracted. If the download is interrupted, rerun the script to resume.
If you only need No RAG and Oracle RAG, `scripts/download_data.sh --skip-knowledge-store` downloads just `dev.json`.

### 3. Build the retrieval DB (Simple RAG only)

```bash
scripts/build_db.sh
```

For each claim, embeds the first 200 sentences of each of the first 50 documents with `BAAI/bge-small-en-v1.5`
and stores them as a per-claim collection (`claim_<id>`) in `datasets/chroma_db_head50` (about 7GB).
Uses MPS on Apple Silicon and CPU otherwise. Claims that are already built are skipped.

### 4. Run experiments

```bash
scripts/run_no_rag.sh       # → results/norag_base_n500/
scripts/run_simple_rag.sh   # → results/rag_base_k10_n500/
scripts/run_oracle_rag.sh   # → results/oracle_base_n500/
```

Extra arguments override the defaults, e.g. `scripts/run_simple_rag.sh --num-samples 10 --top-k 3`.
See `uv run python src/simple_rag.py --help` for all arguments.
With 10 concurrent requests, a 500-claim run took about 11 minutes each for No RAG and Simple RAG, and about 8 minutes for Oracle RAG.

Each run is saved to its own `results/<run-name>/` folder.

| File | Contents |
|---|---|
| `config.json` | run arguments, full system prompt, user message template, git commit, elapsed time, token usage, metrics |
| `results.pkl` | raw LLM responses (list of LiteLLM `ModelResponse`) |
| `evidence.json` | evidence given to the model for each claim (Simple RAG and Oracle RAG only) |

Folder names are generated from the arguments (`<method>_<prompt>_k<top-k>_n<num-samples>`, with `_r2`, `_r3`, ... appended if the name is taken).

### 5. Evaluate

Metrics are printed when a run finishes and are also written to `config.json`. To compare runs again:

```bash
scripts/eval.sh                                              # compare all runs under results/
scripts/eval.sh results/norag_base_n500 results/rag_base_k10_n500 results/oracle_base_n500
scripts/eval.sh results/rag_base_k10_n500 --show             # also print gold/predicted pairs
```

## Repository layout

```
├── src/
│   ├── no_rag.py        # No RAG (no evidence)
│   ├── simple_rag.py    # Simple RAG
│   ├── oracle_rag.py    # Oracle RAG
│   ├── common.py        # CLI arguments, prompt loading, output schema, run logging
│   ├── eval.py          # accuracy / F1 and run comparison
│   ├── download_data.py # download dev.json + knowledge store
│   └── build_db.py      # knowledge store → per-claim Chroma collections
├── prompts/
│   └── base.txt         # system prompt
└── scripts/             # run scripts with default arguments for the above
```

`datasets/` and `results/` are not included in the repository because of their size.
