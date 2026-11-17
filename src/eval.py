"""결과를 읽어 accuracy와 F1을 출력한다.

사용법:
    uv run python src/eval.py                                 # results/ 아래 모든 실행 폴더 비교
    uv run python src/eval.py results/norag_base_n500         # 실행 폴더 하나
    uv run python src/eval.py results/a results/b.pkl         # 여러 개 비교 (폴더와 pkl 혼용 가능)
    uv run python src/eval.py results/norag_base_n500 --show  # 정답/예측 쌍도 출력

입력:
    - 실행 폴더: results/<run-name>/ (results.pkl + config.json). 설정도 함께 보여준다.
    - 예전 방식의 단일 pickle 파일 (설정 정보 없음)

지원하는 결과 형식:
    - agent:        {"total_elapsed_sec": ..., "results": [{"label", "output", ...}]}
    - agent (구버전): [{"label", "output", ...}]
    - no_rag / rag: [ModelResponse, ...]  (정답은 데이터셋 앞에서부터 대응)
"""

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

DATA_PATH = "datasets/dev.json"
RESULTS_DIR = Path("results")

LABELS = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]
CANONICAL = {label.lower(): label for label in LABELS}
FALLBACK = "Not Enough Evidence"  # 출력이 없거나 실패한 경우의 기본 판정

# 비교표에 보여줄 설정 항목 (config.json의 args 키, 표 헤더)
CONFIG_COLUMNS = [("model", "model"), ("prompt", "prompt"), ("top_k", "k"), ("tools", "tools")]


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve(path) -> tuple[Path, dict | None]:
    """실행 폴더 또는 pickle 경로를 (pickle 경로, 설정)으로 바꾼다."""
    path = Path(path)
    pkl = path / "results.pkl" if path.is_dir() else path
    config_path = pkl.parent / "config.json"
    config = None
    if pkl.name == "results.pkl" and config_path.exists():
        config = json.loads(config_path.read_text())
    return pkl, config


def normalize(label: str) -> str:
    """대소문자 차이를 없애고 표준 라벨 표기로 맞춘다."""
    return CANONICAL.get(label.strip().lower(), label.strip())


def extract(raw, data_path: str = DATA_PATH) -> tuple[list[str], list[str], int]:
    """결과 객체에서 (정답, 예측, 출력 누락 수)를 뽑는다."""
    items = raw["results"] if isinstance(raw, dict) else raw

    if items and isinstance(items[0], dict):  # agent 형식: 정답이 결과에 들어 있음
        gold = [r["label"] for r in items]
        verdicts = [(r.get("output") or {}).get("verdict") for r in items]
    else:  # ModelResponse 리스트: 정답은 데이터셋에서 가져옴
        with open(data_path, "r") as f:
            gold = [d["label"] for d in json.load(f)[: len(items)]]
        verdicts = []
        for r in items:
            try:
                verdicts.append(json.loads(r.choices[0].message.content)["verdict"])
            except Exception:  # 요청 실패(Exception 객체)나 파싱 실패
                verdicts.append(None)

    missing = sum(v is None for v in verdicts)
    pred = [normalize(v or FALLBACK) for v in verdicts]
    return [normalize(g) for g in gold], pred, missing


def compute_metrics(gold: list[str], pred: list[str]) -> dict:
    """accuracy, 클래스별 precision/recall/F1, macro/weighted F1을 계산한다."""
    seen = set(gold) | set(pred)
    labels = [l for l in LABELS if l in seen] + sorted(seen - set(LABELS))

    precision, recall, f1, support = precision_recall_fscore_support(
        gold, pred, labels=labels, zero_division=0
    )
    per_class = {
        label: {"precision": precision[i], "recall": recall[i], "f1": f1[i], "support": int(support[i])}
        for i, label in enumerate(labels)
    }

    return {
        "n": len(gold),
        "correct": int(accuracy_score(gold, pred, normalize=False)) if gold else 0,
        "accuracy": accuracy_score(gold, pred) if gold else 0.0,
        "macro_f1": f1_score(gold, pred, labels=labels, average="macro", zero_division=0) if gold else 0.0,
        "weighted_f1": f1_score(gold, pred, labels=labels, average="weighted", zero_division=0) if gold else 0.0,
        "per_class": per_class,
    }


def format_config_value(value) -> str:
    if value is None:
        return "-"
    return ",".join(map(str, value)) if isinstance(value, list) else str(value)


def report(path, show: bool = False) -> dict:
    """한 실행의 지표를 출력하고 반환한다. path는 실행 폴더 또는 pickle 파일."""
    pkl, config = resolve(path)
    args = (config or {}).get("args", {})
    gold, pred, missing = extract(load_pickle(pkl), args.get("data_path", DATA_PATH))
    m = compute_metrics(gold, pred)
    m["config"] = args

    print(f"\n=== {path} ===")
    if config:
        skip = {"run_name", "results_dir"}
        print("설정: " + ", ".join(
            f"{k}={format_config_value(v)}" for k, v in args.items() if k not in skip
        ))
    print(f"Accuracy   : {m['accuracy']:.4f} ({m['correct']}/{m['n']})")
    print(f"Macro F1   : {m['macro_f1']:.4f}")
    print(f"Weighted F1: {m['weighted_f1']:.4f}")
    if missing:
        print(f"출력 누락 {missing}건은 '{FALLBACK}'로 처리")

    print(f"\n{'label':<36}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>9}{'predicted':>11}")
    pred_counts = Counter(pred)
    for label, c in m["per_class"].items():
        print(
            f"{label:<36}{c['precision']:>10.4f}{c['recall']:>10.4f}{c['f1']:>10.4f}"
            f"{c['support']:>9}{pred_counts[label]:>11}"
        )

    if show:
        print()
        for i, (g, p) in enumerate(zip(gold, pred)):
            print(f"{i:>3} {'O' if g == p else 'X'}  {g}\t{p}")

    return m


def print_summary(summary: list[tuple[str, dict]]) -> None:
    """여러 실행의 지표와 주요 설정을 한 표로 출력한다."""
    rows = [
        [path, str(m["n"]), f"{m['accuracy']:.4f}", f"{m['macro_f1']:.4f}", f"{m['weighted_f1']:.4f}"]
        + [format_config_value(m["config"].get(key)) for key, _ in CONFIG_COLUMNS]
        for path, m in summary
    ]
    header = ["run", "n", "acc", "macro_f1", "weighted_f1"] + [name for _, name in CONFIG_COLUMNS]
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]

    print()
    for row in [header, *rows]:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())


def main():
    parser = argparse.ArgumentParser(description="팩트체킹 결과의 accuracy / F1 평가")
    parser.add_argument("paths", nargs="*",
                        help="실행 폴더 또는 결과 pickle 경로 (생략하면 results/ 아래 모든 실행 폴더)")
    parser.add_argument("--show", action="store_true", help="샘플별 정답/예측 출력")
    args = parser.parse_args()

    paths = args.paths or sorted(str(p.parent) for p in RESULTS_DIR.glob("*/results.pkl"))
    if not paths:
        sys.exit(f"{RESULTS_DIR}/ 아래에 실행 폴더가 없습니다. 평가할 경로를 직접 지정하세요.")

    summary = [(path, report(path, args.show)) for path in paths]
    if len(summary) > 1:
        print_summary(summary)


if __name__ == "__main__":
    main()
