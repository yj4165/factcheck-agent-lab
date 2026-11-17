"""데이터 다운로드: AVeriTeC dev 셋(dev.json)과 knowledge store를 datasets/ 에 받는다.

    dev.json             : 평가용 클레임 + 정답
    knowledge_store.zip  : 클레임별 검색 대상 문서 묶음 (약 11GB)
    knowledge_store/     : zip에서 클레임별 파일(<claim_id>.json)만 풀어 둔 폴더

이미 받은 파일·이미 푼 파일은 건너뛰므로 중간에 끊겨도 다시 실행하면 이어서 진행한다.
DB 구축은 build_db.py가 따로 한다.

실행 예시는 scripts/download_data.sh, 전체 인자는 `uv run python src/download_data.py --help` 참고.
"""

import argparse
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from tqdm.auto import tqdm

BASE_URL = "https://huggingface.co/chenxwh/AVeriTeC/resolve/main"
DEV_URL = f"{BASE_URL}/data/dev.json"
KNOWLEDGE_STORE_URL = f"{BASE_URL}/data_store/knowledge_store/dev_knowledge_store.zip"
ZIP_MEMBER = "output_dev/{claim_id}.json"  # zip 안에서 클레임 하나의 경로


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="datasets", help="데이터를 저장할 폴더")
    parser.add_argument("--start-claim", type=int, default=0, help="압축을 풀 첫 클레임 번호")
    parser.add_argument("--num-claims", type=int, default=500,
                        help="압축을 풀 클레임 수 (dev 셋 전체는 500)")
    parser.add_argument("--skip-knowledge-store", action="store_true",
                        help="dev.json만 받고 knowledge store는 받지 않는다")
    return parser.parse_args()


def download(url: str, dest: Path) -> None:
    """url을 dest에 받는다. <dest>.part에 받다가 끝나면 이름을 바꾸므로 dest는 항상 완전한 파일이다."""
    if dest.exists():
        print(f"{dest} 이미 있음, 건너뜀")
        return

    part = dest.with_name(dest.name + ".part")
    done = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as e:
        if e.code != 416:  # 416: .part가 이미 전체 크기 → 받을 것이 없음
            raise
        part.rename(dest)
        return

    with response:
        if done and response.status != 206:  # 서버가 이어받기를 무시하면 처음부터 다시
            done = 0
        total = done + int(response.headers.get("Content-Length", 0))
        with open(part, "ab" if done else "wb") as f, tqdm(
            total=total or None, initial=done, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            while chunk := response.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    if total and part.stat().st_size != total:
        raise OSError(f"{dest.name}: 다운로드가 중간에 끊겼습니다. 다시 실행하면 이어받습니다.")
    part.rename(dest)


def extract(zip_path: Path, out_dir: Path, claim_ids: range) -> None:
    """zip에서 claim_ids의 파일만 out_dir/<claim_id>.json 으로 푼다 (이미 있으면 건너뜀)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in claim_ids if not (out_dir / f"{i}.json").exists()]
    print(f"압축 해제: {len(todo)}개 (이미 있음 {len(claim_ids) - len(todo)}개) → {out_dir}")
    if not todo:
        return

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        missing = [i for i in todo if ZIP_MEMBER.format(claim_id=i) not in names]
        if missing:
            print(f"(주의) zip에 없는 클레임 {len(missing)}개는 건너뜀: {missing[:10]}")
        for i in tqdm([i for i in todo if i not in set(missing)], desc="unzip"):
            dest = out_dir / f"{i}.json"
            part = dest.with_name(dest.name + ".part")
            with zf.open(ZIP_MEMBER.format(claim_id=i)) as src, open(part, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
            part.rename(dest)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    download(DEV_URL, data_dir / "dev.json")
    if args.skip_knowledge_store:
        return

    zip_path = data_dir / "knowledge_store.zip"
    download(KNOWLEDGE_STORE_URL, zip_path)
    extract(zip_path, data_dir / "knowledge_store",
            range(args.start_claim, args.start_claim + args.num_claims))


if __name__ == "__main__":
    main()
