"""Simple RAG baseline: claim별 Chroma 컬렉션에서 근거를 검색해 함께 넣고 판정한다.

실행 예시는 scripts/run_simple_rag.sh, 전체 인자는 `uv run python src/simple_rag.py --help` 참고.
"""

import argparse
import asyncio
import json
import re
import time
from urllib.parse import urlparse

import chromadb
from chromadb.utils import embedding_functions
from dotenv import find_dotenv, load_dotenv
from litellm import acompletion
from tqdm.asyncio import tqdm_asyncio

from common import (
    FactChecking,
    add_common_args,
    add_retrieval_args,
    finish_run,
    get_device,
    load_data,
    load_prompt,
    start_run,
    sum_usage,
)

load_dotenv(find_dotenv())

# LLM에 보내는 사용자 메시지 형식. 근거 하나는 EVIDENCE_TEMPLATE으로, 근거들은 빈 줄로 이어 붙인다.
USER_PROMPT_TEMPLATE = "Claim:\n{claim}\n\nEvidence:\n{evidence}"
EVIDENCE_TEMPLATE = "(source: {domain})\n{text}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(parser, default_prompt="base")
    add_retrieval_args(parser)
    return parser.parse_args()


def get_domain(url: str) -> str:
    m = re.match(r"https?://web\.archive\.org/web/\d+(?:[a-z_]+)?/(https?://.+)", url)
    if m:  # 아카이브면 원본 URL로 교체
        url = m.group(1)
    return urlparse(url).netloc.lower().removeprefix("www.")


def build_user_prompt(claim: str, evidence: list[dict]) -> str:
    ev = "\n\n".join(
        EVIDENCE_TEMPLATE.format(domain=get_domain(e["url"]), text=e["text"])
        for e in evidence
    )
    return USER_PROMPT_TEMPLATE.format(claim=claim, evidence=ev)


async def main():
    args = parse_args()
    system_prompt = load_prompt(args.prompt)
    data = load_data(args.data_path, args.num_samples)
    run_dir = start_run(args, system_prompt=system_prompt, extra={
        "user_prompt_template": USER_PROMPT_TEMPLATE,
        "evidence_template": EVIDENCE_TEMPLATE,
    })

    client = chromadb.PersistentClient(path=args.db_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=args.embedding_model,
        device=get_device(),
        normalize_embeddings=True,
    )

    def search(claim_id: int, query: str, k: int) -> list[dict]:
        col = client.get_or_create_collection(  # 클레임 번호로 풀 선택
            name=f"claim_{claim_id}",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        res = col.query(query_texts=[query], n_results=k)
        return [
            {"text": d, "url": m["url"], "score": 1 - dist}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    semaphore = asyncio.Semaphore(args.concurrency)
    # claim별 검색 근거 (data와 같은 순서). LLM 요청이 실패해도 근거는 남는다.
    evidences: list[list[dict] | None] = [None] * len(data)

    async def check_claim(i: int, d: dict):
        evidence = search(i, d["claim"], k=args.top_k)
        evidences[i] = evidence
        user_prompt = build_user_prompt(d["claim"], evidence)

        async with semaphore:
            return await acompletion(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=FactChecking,
                num_retries=3,
            )

    start = time.perf_counter()
    results = await tqdm_asyncio.gather(
        *[check_claim(i, d) for i, d in enumerate(data)], return_exceptions=True
    )
    elapsed = time.perf_counter() - start

    failed = [i for i, r in enumerate(results) if isinstance(r, Exception)]
    print(f"Successful: {len(results) - len(failed)}/{len(results)}, failed idx: {failed}")

    # results.pkl 형식(ModelResponse 리스트)은 그대로 두고, 근거는 별도 파일로 저장한다
    evidence_path = run_dir / "evidence.json"
    evidence_path.write_text(json.dumps(
        [{"claim_id": i, "claim": d["claim"], "evidence": ev}
         for i, (d, ev) in enumerate(zip(data, evidences))],
        indent=2, ensure_ascii=False,
    ) + "\n")

    finish_run(run_dir, results, {
        "total_elapsed_sec": elapsed,
        "num_samples": len(data),
        "failed_idx": failed,
        "usage": sum_usage(results),
    })
    print(f"저장: {evidence_path}")


if __name__ == "__main__":
    asyncio.run(main())
