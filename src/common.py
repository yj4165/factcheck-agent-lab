"""실험 스크립트 공통 코드: CLI 인자, 프롬프트 로딩, 설정/결과 저장.

실행 하나는 results/<run-name>/ 폴더 하나에 대응한다.
    config.json  : 실행에 쓰인 모든 변수 + 시스템 프롬프트 전문 + 실행 요약
    results.pkl  : 원본 결과
"""

import argparse
import json
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

PROMPT_DIR = Path(__file__).parent.parent / "prompts"


class FactChecking(BaseModel):
    reasoning: str
    verdict: Literal["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]
    confidence: float = Field(ge=0, le=1)


# ---------------------------------------------------------------- CLI 인자

def add_common_args(parser: argparse.ArgumentParser, *, default_prompt: str) -> None:
    """모든 실험 스크립트가 공유하는 인자."""
    parser.add_argument("--run-name", default=None,
                        help="결과 폴더 이름 (results/<run-name>/). 생략하면 인자에서 자동 생성 "
                             "(<방법>_<프롬프트>_k<top-k>_n<샘플수>, 이미 있으면 _r2 ...)")
    parser.add_argument("--model", default="gpt-5-nano", help="LiteLLM 모델 이름")
    parser.add_argument("--prompt", default=default_prompt,
                        help="시스템 프롬프트. prompts/<이름>.txt 의 이름 또는 파일 경로")
    parser.add_argument("--data-path", default="datasets/dev.json", help="평가 데이터셋 경로")
    parser.add_argument("--num-samples", type=int, default=100, help="앞에서부터 사용할 샘플 수")
    parser.add_argument("--concurrency", type=int, default=10, help="동시 실행 수 (1이면 순차 실행)")
    parser.add_argument("--results-dir", default="results", help="결과 저장 상위 폴더")
    parser.add_argument("--overwrite", action="store_true", help="같은 run-name의 기존 결과를 덮어쓴다")


def add_retrieval_args(parser: argparse.ArgumentParser) -> None:
    """검색을 쓰는 스크립트(simple_rag, agent)가 공유하는 인자."""
    parser.add_argument("--top-k", type=int, default=3, help="검색 1회당 가져올 근거 수")
    parser.add_argument("--db-path", default="datasets/chroma_db_head50", help="Chroma DB 경로")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5", help="임베딩 모델")


# ---------------------------------------------------------------- 입출력

def load_prompt(name: str) -> str:
    """prompts/<name>.txt 또는 주어진 경로에서 프롬프트를 읽는다."""
    path = Path(name)
    if not path.is_file():
        path = PROMPT_DIR / f"{name}.txt"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in PROMPT_DIR.glob("*.txt")))
        sys.exit(f"프롬프트 '{name}'을(를) 찾을 수 없습니다. 사용 가능: {available}")
    return path.read_text().strip()


def load_data(path: str, num_samples: int) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)[:num_samples]


def save_pickle(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # 폴더가 없으면 생성
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------- 실행 기록

# 자동 실행 이름에 쓰는 방법 약칭 (스크립트 이름 → 약칭)
METHOD_NAMES = {"no_rag": "norag", "simple_rag": "rag", "oracle_rag": "oracle",
                "simple_agent": "agent"}
# 이 값과 다를 때만 실행 이름에 덧붙이는 인자: (인자 이름, 기본값, 접두어)
NAME_EXTRAS = [("model", "gpt-5-nano", ""), ("max_steps", 10, "s")]


def auto_run_name(args: argparse.Namespace, method: str) -> str:
    """인자에서 실행 이름을 만든다: <방법>_<프롬프트>_k<top-k>_n<샘플수>[_<모델>][_s<max-steps>]"""
    prompt = Path(args.prompt).stem.removeprefix(f"{method}_").replace("_", "-")
    parts = [method, prompt]
    if hasattr(args, "top_k"):
        parts.append(f"k{args.top_k}")
    parts.append(f"n{args.num_samples}")
    for key, default, prefix in NAME_EXTRAS:
        value = getattr(args, key, default)
        if value != default:
            parts.append(f"{prefix}{value}".replace("/", "-"))
    return "_".join(parts)


def _git_state() -> dict:
    """현재 커밋과, 커밋되지 않은 코드 변경(추적 안 된 파일 포함)이 있는지 여부."""
    def git(*cmd):
        return subprocess.run(
            ["git", *cmd], capture_output=True, text=True, check=True, cwd=Path(__file__).parent.parent
        ).stdout.strip()

    try:
        # 결과·데이터 폴더는 코드가 아니므로 변경 여부 판단에서 뺀다
        dirty = bool(git("status", "--porcelain", "--", ".", ":!results", ":!datasets"))
        return {"git_commit": git("rev-parse", "--short", "HEAD"), "git_dirty": dirty}
    except Exception:
        return {"git_commit": None, "git_dirty": None}


def start_run(args: argparse.Namespace, *, system_prompt: str, extra: dict | None = None) -> Path:
    """실행 폴더를 만들고 설정을 출력·저장한다. API 호출 전에 부른다.

    extra: 인자로는 드러나지 않지만 결과에 영향을 주는 값 (예: 도구 정의, 프롬프트 템플릿).
    """
    script = Path(sys.argv[0]).stem
    results_dir = Path(args.results_dir)

    def taken(name: str) -> bool:
        return (results_dir / name / "results.pkl").exists()

    if args.run_name is None:
        # 이름을 인자에서 자동 생성하고, 이미 있으면 _r2, _r3 ... 을 붙인다
        base = auto_run_name(args, METHOD_NAMES.get(script, script))
        args.run_name, n = base, 1
        while taken(args.run_name) and not args.overwrite:
            n += 1
            args.run_name = f"{base}_r{n}"
    elif taken(args.run_name) and not args.overwrite:
        sys.exit(f"{results_dir / args.run_name} 에 이미 결과가 있습니다. "
                 "다른 --run-name을 쓰거나 --overwrite를 붙이세요.")

    run_dir = results_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "script": f"{script}.py",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        **_git_state(),
        "args": {k: v for k, v in vars(args).items() if k != "overwrite"},
        "system_prompt": system_prompt,
        **(extra or {}),
        "output_schema": FactChecking.model_json_schema(),
    }

    print(f"=== {config['script']} → {run_dir} ===")
    for key, value in config["args"].items():
        print(f"  {key:<16}: {value}")
    if config["git_dirty"]:
        print("  (주의) 커밋되지 않은 코드 변경이 있어 git_commit만으로는 이 실행의 코드를 복원할 수 없습니다.")
    print(f"--- system prompt ({args.prompt}) ---\n{system_prompt}\n---")

    _write_config(run_dir, config)
    return run_dir


def sum_usage(responses) -> dict:
    """LiteLLM 응답 목록의 토큰 사용량을 합산한다 (실패한 요청은 건너뜀)."""
    usages = [r.usage for r in responses if getattr(r, "usage", None) is not None]
    return {
        "llm_calls": len(usages),
        "input_tokens": sum(u.prompt_tokens for u in usages),
        "output_tokens": sum(u.completion_tokens for u in usages),
    }


def finish_run(run_dir: Path, results, summary: dict) -> None:
    """결과를 저장하고, 지표를 계산해 실행 요약과 함께 config.json에 덧붙인다."""
    from eval import report  # eval.py는 독립 실행도 되므로 여기서만 가져온다

    save_pickle(results, run_dir / "results.pkl")
    m = report(run_dir)
    summary["metrics"] = {
        "n": m["n"],
        "correct": m["correct"],
        "accuracy": float(m["accuracy"]),
        "macro_f1": float(m["macro_f1"]),
        "weighted_f1": float(m["weighted_f1"]),
        "per_class": {
            label: {k: float(v) if k != "support" else v for k, v in c.items()}
            for label, c in m["per_class"].items()
        },
    }

    config = json.loads((run_dir / "config.json").read_text())
    config["summary"] = summary
    _write_config(run_dir, config)
    print(f"\n저장: {run_dir}/results.pkl, {run_dir}/config.json")


def _write_config(run_dir: Path, config: dict) -> None:
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
