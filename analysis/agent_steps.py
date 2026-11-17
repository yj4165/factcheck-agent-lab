"""에이전트 실행의 스텝 수·검색 횟수와 정확도의 관계를 분석한다.

사용법:
    uv run python analysis/agent_steps.py                      # results/agent_base_k10_n500
    uv run python analysis/agent_steps.py results/<agent-run>

스텝 = LLM 호출 1회 + 도구 실행. 출력이 없는 클레임은 eval.py와 같이 Not Enough Evidence로 처리한다.
결과 해석은 docs/analysis/agent_steps.md 참고.
"""

import argparse
import json
from collections import Counter

import numpy as np

from load import FALLBACK, LABELS, ROOT, load_run


def analyze(run: str) -> None:
    records = load_run(run)
    if records[0]["steps"] is None:
        raise SystemExit(f"{run}: 에이전트 실행이 아닙니다 (스텝 정보 없음)")
    max_steps = json.loads((ROOT / run / "config.json").read_text())["args"]["max_steps"]

    steps = np.array([r["steps"] for r in records])
    searches = np.array([r["searches"] for r in records])
    pred = np.array([r["verdict"] or FALLBACK for r in records])
    gold = np.array([r["gold"] for r in records])
    correct = pred == gold
    missing = np.array([r["verdict"] is None for r in records])

    print(f"=== {run} (n={len(records)}, 최대 스텝 {max_steps}) ===")
    print(f"accuracy {correct.mean():.3f} | 평균 스텝 {steps.mean():.2f} | "
          f"검색 횟수 평균 {searches.mean():.2f}, 중앙값 {np.median(searches):.0f}")
    print(f"출력 없음 {missing.sum()}건 (정답: {dict(Counter(gold[missing].tolist()))}) → {FALLBACK}로 처리")

    print("\n스텝 수 구간별:")
    groups = [
        ("2 (검색 1회 → 판정)", steps == 2),
        (f"3-{max_steps - 1}", (steps > 2) & (steps < max_steps)),
        (f"{max_steps} (최대 스텝 도달)", steps >= max_steps),
    ]
    for name, m in groups:
        print(f"  {name:<22} n={m.sum():>3} ({m.mean():.1%})  accuracy {correct[m].mean():.3f} "
              f"({correct[m].sum()}/{m.sum()})")

    print("\n스텝 수별:")
    for s in sorted(set(steps)):
        m = steps == s
        print(f"  {s:>2}  n={m.sum():>3}  accuracy {correct[m].mean():.3f}")

    hit = steps >= max_steps
    print(f"\n최대 스텝 도달 {hit.sum()}건:")
    print(f"  {'라벨':<36}{'예측':>6}{'정답':>6}")
    for label in LABELS:
        print(f"  {label:<36}{(pred[hit] == label).sum():>6}{(gold[hit] == label).sum():>6}")
    print(f"  (예측 중 {(hit & missing).sum()}건은 출력 없음 → {FALLBACK})")

    print("\n검색 횟수 분포:")
    print("  " + ", ".join(f"{k}회 {v}" for k, v in sorted(Counter(searches.tolist()).items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="*", default=["results/agent_base_k10_n500"])
    for run in parser.parse_args().runs:
        analyze(run)


if __name__ == "__main__":
    main()
