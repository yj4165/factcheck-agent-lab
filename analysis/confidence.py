"""모델이 낸 확신도(confidence)와 정답 여부의 관계를 분석한다.

사용법:
    uv run python analysis/confidence.py                       # 기본 4개 실행
    uv run python analysis/confidence.py results/<run> ...     # 실행 지정

출력이 없는 클레임(verdict 또는 confidence 누락)은 분석에서 제외한다.
결과 해석은 docs/analysis/confidence.md 참고.
"""

import argparse

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from load import LABELS, load_run

DEFAULT_RUNS = [
    "results/norag_base_n500",
    "results/rag_base_k10_n500",
    "results/agent_base_k10_n500",
    "results/oracle_base_n500",
]
BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 1.0)]
THRESHOLDS = [0.7, 0.75, 0.8]
NEE = "Not Enough Evidence"


def ece(correct: np.ndarray, conf: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (균등 폭 10구간)."""
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf >= lo) & ((conf < hi) if i < n_bins - 1 else (conf <= hi))
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total


def in_bin(conf: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))


def safe_auroc(correct: np.ndarray, score: np.ndarray) -> float:
    return roc_auc_score(correct, score) if 0 < correct.sum() < len(correct) else float("nan")


def analyze(run: str) -> None:
    records = load_run(run)
    valid = [r for r in records if r["verdict"] is not None and r["confidence"] is not None]
    gold = np.array([r["gold"] for r in valid])
    pred = np.array([r["verdict"] for r in valid])
    conf = np.array([float(r["confidence"]) for r in valid])
    correct = (gold == pred).astype(int)

    print(f"\n=== {run} (분석 {len(valid)}/{len(records)}건) ===")
    print(f"accuracy {correct.mean():.3f} | 확신도 평균 {conf.mean():.3f} "
          f"(정답 {conf[correct == 1].mean():.3f} / 오답 {conf[correct == 0].mean():.3f})")
    print(f"AUROC {safe_auroc(correct, conf):.3f} | ECE {ece(correct, conf):.3f} | "
          f"서로 다른 확신도 값 {len(set(conf))}개")

    print("\n확신도 구간별:")
    print(f"  {'구간':<12}{'n':>5}{'accuracy':>10}{'확신도 평균':>12}{'NEE 예측 비율':>14}")
    for lo, hi in BINS:
        m = in_bin(conf, lo, hi)
        if m.any():
            print(f"  [{lo:.1f}, {hi:.1f}{']' if hi == 1.0 else ')'}  {m.sum():>5}{correct[m].mean():>10.3f}"
                  f"{conf[m].mean():>12.3f}{(pred[m] == NEE).mean():>14.2f}")

    print("\n임계값 이상만 채택할 때:")
    for t in THRESHOLDS:
        m = conf >= t
        print(f"  conf >= {t:<5} 비율 {m.mean():.3f} (n={m.sum()})  accuracy {correct[m].mean():.3f}")

    print("\n예측 라벨별 (라벨 안에서 확신도로 정답/오답을 가르는 AUROC):")
    print(f"  {'라벨':<36}{'n':>5}{'accuracy':>10}{'정답 conf':>11}{'오답 conf':>11}{'AUROC':>8}")
    for label in LABELS:
        m = pred == label
        if m.sum() == 0:
            continue
        c_ok = conf[m][correct[m] == 1].mean() if correct[m].any() else float("nan")
        c_ng = conf[m][correct[m] == 0].mean() if (1 - correct[m]).any() else float("nan")
        print(f"  {label:<36}{m.sum():>5}{correct[m].mean():>10.3f}{c_ok:>11.3f}{c_ng:>11.3f}"
              f"{safe_auroc(correct[m], conf[m]):>8.3f}")

    # 예측 라벨만으로도 정답 여부를 얼마나 맞히는지와 비교 (로지스틱 회귀, 5-fold CV)
    x_label = np.array([[p == l for l in LABELS] for p in pred], dtype=float)
    features = {
        "예측 라벨만": x_label,
        "확신도만": conf[:, None],
        "예측 라벨 + 확신도": np.hstack([x_label, conf[:, None]]),
    }
    print("\n정답 여부 예측 (로지스틱 회귀, 5-fold CV AUROC):")
    for name, x in features.items():
        prob = cross_val_predict(LogisticRegression(max_iter=1000), x, correct,
                                 cv=5, method="predict_proba")[:, 1]
        print(f"  {name:<14} {roc_auc_score(correct, prob):.3f}")

    steps = [r["steps"] for r in valid]
    if steps[0] is not None:
        steps = np.array(steps)
        rho, p = spearmanr(steps, conf)
        print(f"\n[에이전트] 스텝 수와 확신도: Spearman rho {rho:.3f} (p={p:.1e})")
        for name, m in [("2", steps == 2), ("3-9", (steps > 2) & (steps < 10)), ("10", steps >= 10)]:
            print(f"  스텝 {name:<4} n={m.sum():>3}  accuracy {correct[m].mean():.3f}  "
                  f"확신도 평균 {conf[m].mean():.3f}  AUROC {safe_auroc(correct[m], conf[m]):.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="*", default=DEFAULT_RUNS, help="실행 폴더 (results/<run-name>)")
    for run in parser.parse_args().runs:
        analyze(run)


if __name__ == "__main__":
    main()
