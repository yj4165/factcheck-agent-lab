"""분석 스크립트가 공유하는 결과 로더.

실행 폴더(results/<run-name>/)의 results.pkl을 읽어 클레임별
정답·예측·확신도(·에이전트 스텝/검색 횟수)를 레코드 리스트로 돌려준다.
"""

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval import FALLBACK, LABELS, normalize  # noqa: E402

__all__ = ["FALLBACK", "LABELS", "ROOT", "load_run"]


def _count_searches(events: list[dict]) -> int:
    return sum(
        1
        for e in events
        for c in e["content"]
        if c.get("type") == "tool_call" and c.get("name") == "search"
    )


def load_run(run_dir: str | Path) -> list[dict]:
    """클레임별 레코드: gold, verdict(없으면 None), confidence(없으면 None), steps, searches."""
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    config = json.loads((run_dir / "config.json").read_text())
    with open(run_dir / "results.pkl", "rb") as f:
        raw = pickle.load(f)
    items = raw["results"] if isinstance(raw, dict) else raw

    records = []
    if items and isinstance(items[0], dict):  # agent 형식
        for r in items:
            out = r.get("output") or {}
            records.append({
                "gold": normalize(r["label"]),
                "verdict": normalize(out["verdict"]) if out.get("verdict") else None,
                "confidence": out.get("confidence"),
                "steps": r.get("steps"),
                "searches": _count_searches(r.get("events", [])),
            })
    else:  # ModelResponse 리스트: 정답은 데이터셋 앞에서부터 대응
        data_path = ROOT / config["args"].get("data_path", "datasets/dev.json")
        gold = [d["label"] for d in json.loads(data_path.read_text())[: len(items)]]
        for g, r in zip(gold, items):
            try:
                out = json.loads(r.choices[0].message.content)
            except Exception:
                out = {}
            records.append({
                "gold": normalize(g),
                "verdict": normalize(out["verdict"]) if out.get("verdict") else None,
                "confidence": out.get("confidence"),
                "steps": None,
                "searches": None,
            })
    return records
