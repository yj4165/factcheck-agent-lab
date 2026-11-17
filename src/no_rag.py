"""No RAG baseline: 근거 없이 claim만 LLM에 넣고 판정한다.

실행 예시는 scripts/run_no_rag.sh, 전체 인자는 `uv run python src/no_rag.py --help` 참고.
"""

import argparse
import asyncio
import time

from dotenv import find_dotenv, load_dotenv
from litellm import acompletion
from tqdm.asyncio import tqdm_asyncio

from common import FactChecking, add_common_args, finish_run, load_data, load_prompt, start_run, sum_usage

load_dotenv(find_dotenv())

USER_PROMPT_TEMPLATE = "{claim}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(parser, default_prompt="base")
    return parser.parse_args()


async def main():
    args = parse_args()
    system_prompt = load_prompt(args.prompt)
    data = load_data(args.data_path, args.num_samples)
    run_dir = start_run(args, system_prompt=system_prompt,
                        extra={"user_prompt_template": USER_PROMPT_TEMPLATE})

    semaphore = asyncio.Semaphore(args.concurrency)

    async def check_claim(d: dict):
        async with semaphore:
            return await acompletion(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(claim=d["claim"])},
                ],
                response_format=FactChecking,
                num_retries=3,
            )

    start = time.perf_counter()
    results = await tqdm_asyncio.gather(*[check_claim(d) for d in data], return_exceptions=True)
    elapsed = time.perf_counter() - start

    failed = [i for i, r in enumerate(results) if isinstance(r, Exception)]
    print(f"Successful: {len(results) - len(failed)}/{len(results)}, failed idx: {failed}")

    finish_run(run_dir, results, {
        "total_elapsed_sec": elapsed,
        "num_samples": len(data),
        "failed_idx": failed,
        "usage": sum_usage(results),
    })


if __name__ == "__main__":
    asyncio.run(main())
