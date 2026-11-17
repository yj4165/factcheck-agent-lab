"""DB 구축: knowledge store의 문장을 임베딩해 클레임별 Chroma 컬렉션(claim_<id>)으로 저장한다.

클레임마다 url2text가 비어 있지 않은 문서를 앞에서부터 --max-docs개 고르고(중복 URL 제외),
문서마다 앞 --max-sents 문장만 쓴다. 기본값(50, 200)이 실험에 쓰는 chroma_db_head50이다.
주의: knowledge store 파일은 type 순(gold → question → ...)이라 앞 50개는 gold/question 위주다.

이미 다 들어간 클레임은 건너뛰므로 중간에 끊겨도 다시 실행하면 이어서 진행한다.
데이터는 download_data.py로 먼저 받아 둔다.

실행 예시는 scripts/build_db.sh, 전체 인자는 `uv run python build_db.py --help` 참고.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from tqdm.auto import tqdm

from common import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--knowledge-store", default="datasets/knowledge_store",
                        help="클레임별 문서 파일(<claim_id>.json)이 있는 폴더")
    parser.add_argument("--db-path", default="datasets/chroma_db_head50", help="Chroma DB 경로")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5", help="임베딩 모델")
    parser.add_argument("--start-claim", type=int, default=0, help="첫 클레임 번호")
    parser.add_argument("--num-claims", type=int, default=500, help="구축할 클레임 수")
    parser.add_argument("--max-docs", type=int, default=50,
                        help="클레임당 사용할 문서 수 (0이면 전부)")
    parser.add_argument("--max-sents", type=int, default=200,
                        help="문서당 사용할 앞쪽 문장 수 (0이면 전부)")
    parser.add_argument("--batch-size", type=int, default=256, help="한 번에 임베딩·저장할 문장 수")
    parser.add_argument("--overwrite", action="store_true",
                        help="이미 있는 컬렉션을 지우고 다시 만든다")
    return parser.parse_args()


def select_sentences(path: Path, max_docs: int, max_sents: int) -> tuple[list, list, list]:
    """파일 하나에서 DB에 넣을 문장을 골라 (ids, 문장, 메타데이터)로 돌려준다."""
    ids, docs, metas = [], [], []
    seen, num_docs = set(), 0
    with open(path, "r") as f:
        for line_no, line in enumerate(f):
            row = json.loads(line)
            if row["url"] in seen:
                continue
            sents = row["url2text"][:max_sents] if max_sents else row["url2text"]
            sents = [(sent_idx, s.strip()) for sent_idx, s in enumerate(sents) if s.strip()]
            if not sents:  # 쓸 문장이 하나도 없는 문서는 문서 수에 세지 않는다
                continue
            seen.add(row["url"])
            for sent_idx, sent in sents:
                ids.append(f"{line_no}_{sent_idx}")
                docs.append(sent)
                metas.append({
                    "url": row["url"],
                    "type": row["type"],  # gold / 그 외
                    "line_no": line_no,
                    "sent_idx": sent_idx,
                })
            num_docs += 1
            if num_docs == max_docs:  # 다 채우면 나머지 줄은 읽지 않는다
                break
    return ids, docs, metas


def main():
    args = parse_args()
    store = Path(args.knowledge_store)
    if not store.is_dir():
        sys.exit(f"{store} 가 없습니다. 먼저 scripts/download_data.sh 를 실행하세요.")

    print(f"=== build_db.py → {args.db_path} ===")
    for key, value in vars(args).items():
        print(f"  {key:<16}: {value}")

    client = chromadb.PersistentClient(path=args.db_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=args.embedding_model,
        device=get_device(),
        normalize_embeddings=True,
    )

    start = time.perf_counter()
    built, skipped, missing = 0, 0, []
    for claim_id in tqdm(range(args.start_claim, args.start_claim + args.num_claims), desc="claims"):
        path = store / f"{claim_id}.json"
        if not path.exists():
            missing.append(claim_id)
            continue

        name = f"claim_{claim_id}"
        if args.overwrite:
            try:
                client.delete_collection(name)
            except Exception:  # 없는 컬렉션이면 지울 것도 없다
                pass
        # 검색 쪽(simple_rag.py, agent/tools/search.py)과 같은 설정이어야 한다
        col = client.get_or_create_collection(
            name=name, embedding_function=ef, metadata={"hnsw:space": "cosine"}
        )

        ids, docs, metas = select_sentences(path, args.max_docs, args.max_sents)
        if col.count() == len(ids):  # 이미 다 들어가 있음
            skipped += 1
            continue

        # leave=False: 클레임 하나가 끝나면 안쪽 막대는 지워지고 바깥(클레임) 막대만 남는다
        for s in tqdm(range(0, len(ids), args.batch_size), desc=name, leave=False):
            e = s + args.batch_size
            col.upsert(ids=ids[s:e], documents=docs[s:e], metadatas=metas[s:e])
        built += 1

    print(f"구축 {built}개, 건너뜀(이미 있음) {skipped}개, {time.perf_counter() - start:.0f}s")
    if missing:
        print(f"(주의) knowledge store에 파일이 없는 클레임 {len(missing)}개: {missing[:10]}")


if __name__ == "__main__":
    main()
