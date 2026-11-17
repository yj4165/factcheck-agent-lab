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

Four methods are currently available.

- **No RAG** (`no_rag.py`): the claim is given with no evidence, so the model must rely on its own knowledge.
- **Simple RAG** (`simple_rag.py`): the claim text is used as a query to search the claim's document pool (Chroma) once, and the top-k sentences are given as evidence.
- **Oracle RAG** (`oracle_rag.py`): instead of retrieval, the human-annotated gold QA pairs (questions and answers) from the dataset are given as evidence.
  This is an upper-bound experiment: how far can the model get if retrieval were perfect? The `justification` field, which states the verdict directly, is not included.
- **Agent** (`simple_agent.py`): the LLM writes its own queries and calls a `search` tool as many times as it wants, then gives its verdict through a `final_answer` tool.
  The search target (per-claim document pool) and the number of sentences per search (top 10) are the same as in Simple RAG;
  the **only difference is that the model decides what to search for and how many times**.

The first three methods use the same system prompt ([prompts/base.txt](prompts/base.txt)) and differ **only in the evidence** placed in the user message
(none → retrieved sentences → gold QA).
The LLM returns `reasoning`, `verdict` and `confidence` as structured output (JSON).
The Agent uses a separate prompt that explains the tools ([prompts/agent_base.txt](prompts/agent_base.txt)),
and its user message contains only the claim text. It returns the same three fields as the arguments of the `final_answer` tool.

## Experimental setup

All results below use this setup (No RAG, Simple RAG and Oracle RAG on 2026-11-16, Agent on 2026-11-18; one run per method).

| Item | Value |
|---|---|
| LLM | OpenAI `gpt-5-nano` (via [LiteLLM](https://docs.litellm.ai/); temperature, reasoning effort, etc. not set, so API defaults) |
| Output format | structured output (`reasoning`, `verdict`, `confidence`) |
| System prompt | [prompts/base.txt](prompts/base.txt) ([prompts/agent_base.txt](prompts/agent_base.txt) for Agent) |
| Data | all 500 claims of the AVeriTeC dev set |
| Embedding model (Simple RAG) | `BAAI/bge-small-en-v1.5` |
| Retrieval (Simple RAG) | one search with the claim text, top 10 sentences by cosine similarity |
| Retrieval (Agent) | as many searches as the model wants, with its own queries; top 10 sentences per search (same DB and embedding model as Simple RAG) |
| Max steps (Agent) | 10 (step = one LLM call + tool execution). A `final_answer` call is forced on the last step |

Usage per method (500 claims):

| Method | Input tokens | Output tokens | Time (10 concurrent requests) |
|---|---:|---:|---:|
| No RAG | 87,196 | 628,641 | ~11 min |
| Simple RAG | 290,690 | 611,724 | ~11 min |
| Oracle RAG | 189,979 | 542,183 | ~8 min |
| Agent | 4,041,280 | 1,698,187 | ~25 min |

`gpt-5-nano` is a reasoning model, so output tokens include reasoning tokens and far exceed input tokens.
The Agent makes 4.85 LLM calls per claim on average (2,426 in total) and resends earlier search results on every call, so it uses about 14× the input tokens of Simple RAG.

## Results

| Method | Evidence | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| No RAG | none | 0.434 | 0.364 | 0.510 |
| Simple RAG | top-10 retrieved sentences | 0.482 | 0.413 | 0.550 |
| Agent | sentences retrieved by the agent (top 10 per search) | 0.532 | 0.421 | 0.592 |
| Oracle RAG | gold QA | **0.766** | **0.625** | **0.786** |

Per-label F1, with the number of predictions in parentheses:

| Label (gold count) | No RAG | Simple RAG | Agent | Oracle RAG |
|---|---:|---:|---:|---:|
| Supported (122) | 0.497 (47) | 0.635 (89) | 0.645 (92) | 0.808 (91) |
| Refuted (305) | 0.594 (170) | 0.597 (174) | 0.668 (216) | 0.866 (293) |
| Not Enough Evidence (35) | 0.194 (274) | 0.185 (181) | 0.134 (129) | 0.385 (95) |
| Conflicting Evidence/Cherrypicking (38) | 0.170 (9) | 0.234 (56) | 0.238 (63) | 0.441 (21) |

Observations:

- **Retrieval is the main bottleneck.** Retrieved evidence improves accuracy by only 4.8 points over no evidence (0.434 → 0.482),
  while gold QA evidence improves it by 33.2 points (→ 0.766).
- **When evidence is insufficient, the model falls back to Not Enough Evidence.** There are only 35 such claims, yet
  No RAG predicts this label 274 times and Simple RAG 181 times. Better evidence reduces this over-prediction (95 with Oracle),
  and Refuted recall rises accordingly (0.46 → 0.47 → 0.85).
- **Not Enough Evidence and Conflicting remain hard even with perfect evidence.** Even with Oracle RAG, F1 for both labels is around 0.4.
  This looks like a prompt / label-definition problem rather than a retrieval problem.
- **Letting an agent choose its own queries improves accuracy by 5.0 points over Simple RAG** (0.482 → 0.532). The retrieval DB and the
  number of sentences per search are the same, so the gain comes from query choice and the number of searches. Most of it is on Refuted
  (F1 0.597 → 0.668, recall 0.47 → 0.57). Macro F1, however, barely moves (0.413 → 0.421), and the gap to Oracle RAG (0.766) is still large.
- **When the agent cannot find evidence, it keeps searching until the step limit and ends with Not Enough Evidence.**
  78 claims (15.6%) hit the 10-step limit; 50 of them were labeled Not Enough Evidence, and accuracy on these 78 was only 0.21 (16/78)
  (only 8 of the 78 are actually Not Enough Evidence). By contrast, the 142 claims answered right after a single search reached 0.68 (96/142).
  As a result, Not Enough Evidence F1 is even lower than with Simple RAG (0.185 → 0.134).

Agent steps and searches (500 claims):

| Steps | Claims | Accuracy |
|---|---:|---:|
| 2 (one search → verdict) | 142 | 0.676 |
| 3–9 | 280 | 0.550 |
| 10 (step limit reached) | 78 | 0.205 |

The agent makes 4.0 searches per claim on average (median 2). This differs from the step count because several searches can be called in a single step.

Notes:

- These are single-run results, so there is some variance from LLM sampling. On a 100-claim run, Simple RAG (k=10) reached 0.50 accuracy.
- Simple RAG searches the per-claim knowledge store provided by AVeriTeC. These files are sorted by type (gold → question → ...)
  and the default setting uses only the first 50 documents, so source documents of the gold answers are likely included.
  This is an easier setting than real web search.
- Outputs that fail to parse are counted as `Not Enough Evidence` (the No RAG, Simple RAG and Oracle RAG runs had no failures).
  For the Agent, 4 claims (gold: 2 Supported, 2 Refuted) reached the step limit without a `final_answer`; they were counted this way and are included in the metrics above.

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

### 3. Build the retrieval DB (Simple RAG and Agent only)

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
scripts/run_agent.sh        # → results/agent_base_k10_n500/
```

Extra arguments override the defaults, e.g. `scripts/run_simple_rag.sh --num-samples 10 --top-k 3`.
See `uv run python src/simple_rag.py --help` for all arguments.
With 10 concurrent requests, a 500-claim run took about 11 minutes each for No RAG and Simple RAG, about 8 minutes for Oracle RAG,
and about 25 minutes for the Agent (10 claims processed concurrently).
The Agent's tools and step limit can be changed with `--tools` (default `search`) and `--max-steps` (default 10).

Each run is saved to its own `results/<run-name>/` folder.

| File | Contents |
|---|---|
| `config.json` | run arguments, full system prompt, user message template (tool definitions for the Agent), git commit, elapsed time, token usage, metrics |
| `results.pkl` | raw LLM responses (list of LiteLLM `ModelResponse`). For the Agent: final output, step count and token usage per claim, plus the full trajectory (queries, search results, LLM responses) |
| `evidence.json` | evidence given to the model for each claim (Simple RAG and Oracle RAG only) |

Folder names are generated from the arguments (`<method>_<prompt>_k<top-k>_n<num-samples>`, with `_r2`, `_r3`, ... appended if the name is taken).

### 5. Evaluate

Metrics are printed when a run finishes and are also written to `config.json`. To compare runs again:

```bash
scripts/eval.sh                                              # compare all runs under results/
scripts/eval.sh results/norag_base_n500 results/rag_base_k10_n500 results/agent_base_k10_n500 results/oracle_base_n500
scripts/eval.sh results/rag_base_k10_n500 --show             # also print gold/predicted pairs
```

## Repository layout

```
├── src/
│   ├── no_rag.py        # No RAG (no evidence)
│   ├── simple_rag.py    # Simple RAG
│   ├── oracle_rag.py    # Oracle RAG
│   ├── simple_agent.py  # Agent (entry point)
│   ├── agent/           # agent implementation (agent loop, LLM client, search tool, ...)
│   ├── common.py        # CLI arguments, prompt loading, output schema, run logging
│   ├── eval.py          # accuracy / F1 and run comparison
│   ├── download_data.py # download dev.json + knowledge store
│   └── build_db.py      # knowledge store → per-claim Chroma collections
├── prompts/
│   ├── base.txt         # system prompt (No RAG, Simple RAG, Oracle RAG)
│   └── agent_base.txt   # system prompt (Agent)
└── scripts/             # run scripts with default arguments for the above
```

`datasets/` and `results/` are not included in the repository because of their size.
