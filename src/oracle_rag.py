"""Oracle RAG: 검색 대신 데이터셋의 정답 QA(사람이 단 질문·답변)를 근거로 넣고 판정한다.

검색이 완벽하다고 가정했을 때의 상한선을 보기 위한 실험이다. simple_rag.py와 같은
시스템 프롬프트·사용자 메시지 틀을 쓰고, 근거 부분만 정답 QA로 바꾼다.
판정이 그대로 적혀 있는 justification 필드는 넣지 않는다.

실행 예시는 scripts/run_oracle_rag.sh, 전체 인자는 `uv run python src/oracle_rag.py --help` 참고.
"""

import argparse
import asyncio
import json
import time

from dotenv import find_dotenv, load_dotenv
from litellm import acompletion
from tqdm.asyncio import tqdm_asyncio

from common import (
    FactChecking,
    add_common_args,
    finish_run,
    load_data,
    load_prompt,
    start_run,
    sum_usage,
)
from simple_rag import USER_PROMPT_TEMPLATE, get_domain

load_dotenv(find_dotenv())

# 근거 하나(= 질문에 달린 답변 하나)의 형식. 출처가 없는 답변은 source 줄을 뺀다.
EVIDENCE_TEMPLATE = "(source: {domain})\nQ: {question}\nA: {answer}"
EVIDENCE_TEMPLATE_NO_SOURCE = "Q: {question}\nA: {answer}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(parser, default_prompt="base")
    return parser.parse_args()


def gold_evidence(d: dict) -> list[dict]:
    """데이터셋 한 항목의 정답 QA를 (질문, 답변) 단위의 근거 목록으로 펼친다."""
    evidence = []
    for q in d["questions"]:
        for a in q["answers"]:
            answer = a["answer"]
            # Boolean 답변은 "Yes"/"No"뿐이라 설명을 덧붙인다
            if a.get("boolean_explanation"):
                answer = f"{answer}. {a['boolean_explanation']}"
            evidence.append({
                "question": q["question"],
                "answer": answer,
                "answer_type": a.get("answer_type"),
                "url": a.get("source_url") or "",
            })
    return evidence


def build_user_prompt(claim: str, evidence: list[dict]) -> str:
    ev = "\n\n".join(
        EVIDENCE_TEMPLATE.format(domain=get_domain(e["url"]), question=e["question"], answer=e["answer"])
        if e["url"]
        else EVIDENCE_TEMPLATE_NO_SOURCE.format(question=e["question"], answer=e["answer"])
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
        "evidence_source": "gold QA (questions[].answers[])",
    })

    evidences = [gold_evidence(d) for d in data]
    semaphore = asyncio.Semaphore(args.concurrency)

    async def check_claim(i: int, d: dict):
        user_prompt = build_user_prompt(d["claim"], evidences[i])

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

    # simple_rag.py와 같은 위치·구조로 근거를 저장한다 (근거 항목의 필드만 다름)
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
