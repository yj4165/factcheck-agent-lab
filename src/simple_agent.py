"""Fact-checking agent: 검색(및 선택적으로 계획/성찰) 도구를 쓰는 에이전트로 판정한다.

실행 예시는 scripts/run_agent.sh, scripts/run_agent_planning.sh,
전체 인자는 `uv run python src/simple_agent.py --help` 참고.
"""

import argparse
import asyncio
import sys
import time

from dotenv import find_dotenv, load_dotenv
from tqdm.asyncio import tqdm_asyncio

from agent.agent import Agent
from agent.llm import LlmClient
from agent.planning import create_tasks, reflection
from agent.tools.base import FunctionTool
from agent.tools.helpers import format_tool_definition
from agent.tools.search import Searcher
from common import (
    FactChecking,
    add_common_args,
    add_retrieval_args,
    finish_run,
    get_device,
    load_data,
    load_prompt,
    start_run,
)

load_dotenv(find_dotenv())

# --tools 로 고를 수 있는 도구. search는 claim마다 새로 만들기 때문에 여기서는 이름만 둔다.
STATIC_TOOLS = {"create_tasks": create_tasks, "reflection": reflection}
AVAILABLE_TOOLS = ["search", *STATIC_TOOLS]


class UsageTrackingClient(LlmClient):
    """호출 횟수와 토큰 사용량을 누적하는 LlmClient (claim마다 하나씩 만든다)."""

    def __init__(self, model: str, **config):
        super().__init__(model, **config)
        self.usage = {"llm_calls": 0, "llm_errors": 0, "input_tokens": 0, "output_tokens": 0}

    async def generate(self, request):
        response = await super().generate(request)
        self.usage["llm_calls"] += 1
        if response.error_message:
            self.usage["llm_errors"] += 1
        for key in ("input_tokens", "output_tokens"):
            self.usage[key] += response.usage_metadata.get(key) or 0
        return response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(parser, default_prompt="agent_refuted_rule")
    add_retrieval_args(parser)
    parser.add_argument("--tools", default="search",
                        help=f"쉼표로 구분한 도구 목록. 사용 가능: {', '.join(AVAILABLE_TOOLS)}")
    parser.add_argument("--max-steps", type=int, default=10, help="에이전트 최대 스텝 수")
    args = parser.parse_args()

    args.tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    unknown = [t for t in args.tools if t not in AVAILABLE_TOOLS]
    if unknown:
        sys.exit(f"알 수 없는 도구: {unknown}. 사용 가능: {AVAILABLE_TOOLS}")
    return args


def build_search_tool_definition(top_k: int) -> dict:
    """LLM에게 보이는 search 도구 설명. top-k 값이 문구에 반영된다."""
    description = (
        "Semantic search over a collection of web-page passages.\n"
        f"Returns the top {top_k} most similar sentences, each with its source domain."
    )
    return format_tool_definition(
        "search",
        description,
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural-language question or statement about fact to verify.",
                }
            },
            "required": ["query"],
        },
    )


async def main():
    args = parse_args()
    system_prompt = load_prompt(args.prompt)
    data = load_data(args.data_path, args.num_samples)
    search_definition = build_search_tool_definition(args.top_k)

    searcher = Searcher(
        db_path=args.db_path,
        embed_model=args.embedding_model,
        device=get_device(),
        k=args.top_k,
    )

    def build_agent(claim_id: int) -> Agent:
        # claim별로 검색 도구와 에이전트를 새로 만든다 (가벼운 객체)
        tools = []
        for name in args.tools:
            if name == "search":
                tools.append(FunctionTool(
                    searcher.make_search_fn(claim_id),
                    name="search",
                    tool_definition=search_definition,
                ))
            else:
                tools.append(STATIC_TOOLS[name])

        return Agent(
            model=UsageTrackingClient(model=args.model),
            tools=tools,
            instructions=system_prompt,
            output_type=FactChecking,
            max_steps=args.max_steps,
        )

    # LLM에 실제로 전달되는 도구 정의 전문 (final_answer 등 에이전트가 덧붙이는 도구 포함)
    tool_definitions = [t.tool_definition for t in build_agent(0).tools]
    run_dir = start_run(args, system_prompt=system_prompt, extra={"tool_definitions": tool_definitions})
    print("LLM에 전달되는 도구:", [t["function"]["name"] for t in tool_definitions])

    semaphore = asyncio.Semaphore(args.concurrency)

    async def run_one(claim_id: int, d: dict):
        agent = build_agent(claim_id)

        async with semaphore:
            start = time.perf_counter()
            result = await agent.run(d["claim"])
            elapsed = time.perf_counter() - start

        return {
            "claim_id": claim_id,
            "label": d["label"],
            "output": result.output.model_dump() if result.output else None,
            "steps": result.context.current_step,
            "elapsed_sec": elapsed,
            "usage": agent.model.usage,
            "events": [e.model_dump() for e in result.context.events],
        }

    total_start = time.perf_counter()
    outs = await tqdm_asyncio.gather(
        *[run_one(i, d) for i, d in enumerate(data)],
        return_exceptions=True,
    )
    total_elapsed = time.perf_counter() - total_start

    results = [r for r in outs if not isinstance(r, Exception)]
    failed = [i for i, r in enumerate(outs) if isinstance(r, Exception)]
    print(f"total: {total_elapsed:.1f}s | ok {len(results)}/{len(data)} | failed idx: {failed}")
    for i in failed[:3]:
        print(f"  [{i}] {outs[i]!r}")

    finish_run(run_dir, {"total_elapsed_sec": total_elapsed, "results": results}, {
        "total_elapsed_sec": total_elapsed,
        "num_samples": len(data),
        "failed_idx": failed,
        "avg_steps": sum(r["steps"] for r in results) / len(results) if results else None,
        "usage": {key: sum(r["usage"][key] for r in results)
                  for key in ("llm_calls", "llm_errors", "input_tokens", "output_tokens")},
    })


if __name__ == "__main__":
    asyncio.run(main())
