import chromadb
from chromadb.utils import embedding_functions
import re
from urllib.parse import urlparse

class Searcher:
    def __init__(self, db_path, embed_model="BAAI/bge-small-en-v1.5", device="cpu", start_claim_id=0, k=3):
        self.db_path = db_path
        self.embed_model = embed_model
        self.device =device
        self.k = k

        self.client = chromadb.PersistentClient(path=db_path)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embed_model,
            device=device,
            normalize_embeddings=True,
        )

        self.col = self.get_collection(start_claim_id)


    def set_col(self, claim_id: int):
        self.col = self.get_collection(claim_id)

    def get_collection(self, claim_id: int):
        return self.client.get_or_create_collection(
            name=f"claim_{claim_id}",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

    def get_domain(self, url: str) -> str:
        m = re.match(r"https?://web\.archive\.org/web/\d+(?:[a-z_]+)?/(https?://.+)", url)
        if m:                       # 아카이브면 원본 URL로 교체
            url = m.group(1)
        return urlparse(url).netloc.lower().removeprefix("www.")
    
        
    def search(self, query: str) -> str:
        """Semantic search over the current claim's evidence collection.

        The schema shown to the LLM is defined in simple_agent.py.
        """
        res = self.col.query(query_texts=[query], n_results=self.k)
        docs = zip(res["documents"][0], res["metadatas"][0])
        if not res["documents"][0]:
            return "No results found."
        return "\n\n".join(
            f"(source: {self.get_domain(m['url'])})\n{d}"
            for d, m in docs
        )

    def make_search_fn(self, claim_id: int):
        """claim_id의 컬렉션에 고정된 검색 함수를 반환한다 (동시 실행에 안전)."""
        col = self.get_collection(claim_id)

        def search(query: str, **_) -> str:
            res = col.query(query_texts=[query], n_results=self.k)
            if not res["documents"][0]:
                return "No results found."
            return "\n\n".join(
                f"(source: {self.get_domain(m['url'])})\n{d}"
                for d, m in zip(res["documents"][0], res["metadatas"][0])
            )

        return search