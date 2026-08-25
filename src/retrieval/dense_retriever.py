"""Dense retriever: Sentence-BERT embeddings + ChromaDB (screen 4.5, branch 2)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import PROJECT_ROOT
from src.retrieval.tfidf_retriever import RetrievalResult

DEFAULT_MODEL = "keepitreal/vietnamese-sbert"
CHROMA_DIR = PROJECT_ROOT / "chroma_store"


@dataclass
class DenseIndexInfo:
    collection_name: str
    model_name: str
    n_vectors: int
    dimension: int


class DenseRetriever:
    """Bi-encoder retriever backed by ChromaDB persistent store."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.model_name = model_name
        self._model = None
        self._device = device
        self._client = None
        self._collection = None

    # -- lazy loading ------------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    @property
    def client(self):
        if self._client is None:
            import chromadb

            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return self._client

    # -- indexing ----------------------------------------------------------
    def collection_name(self, preprocessing_id: str, chunking_id: str) -> str:
        safe_model = self.model_name.replace("/", "__")
        return f"dense_{safe_model}_p{preprocessing_id}_c{chunking_id}"

    def build_index(
        self,
        chunk_ids: list[str],
        texts: list[str],
        metas: list[dict],
        collection_name: str,
        batch_size: int = 64,
        progress_callback=None,
    ) -> DenseIndexInfo:
        """Embed texts and upsert into a NEW ChromaDB collection."""
        # get_or_create keeps an existing identical index reusable
        col = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        if col.count() == len(chunk_ids):
            self._collection = col
            return DenseIndexInfo(collection_name, self.model_name,
                                  col.count(), self.model.get_sentence_embedding_dimension())

        n = len(chunk_ids)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            embeddings = self.model.encode(
                texts[start:end],
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            col.upsert(
                ids=chunk_ids[start:end],
                embeddings=embeddings.tolist(),
                documents=texts[start:end],
                metadatas=[
                    {
                        "law_id": m.get("law_id", ""),
                        "article_id": m.get("article_id", ""),
                        **({"title": m["title"]} if m.get("title") else {}),
                    }
                    for m in metas[start:end]
                ],
            )
            if progress_callback:
                progress_callback(end / n)
        self._collection = col
        return DenseIndexInfo(
            collection_name,
            self.model_name,
            col.count(),
            self.model.get_sentence_embedding_dimension(),
        )

    def use_collection(self, collection_name: str):
        self._collection = self.client.get_collection(collection_name)

    def add_documents(
        self,
        chunk_ids: list[str],
        texts: list[str],
        metas: list[dict],
        collection_name: str,
        batch_size: int = 64,
        progress_callback=None,
    ) -> DenseIndexInfo:
        """Incrementally upsert NEW chunks into an EXISTING collection.

        Only the given chunks are embedded — existing vectors are kept
        (upsert with the same chunk_id overwrites safely).
        """
        col = self.client.get_collection(collection_name)  # must exist
        existing = set(col.get()["ids"]) if col.count() else set()
        # Tách chunks mới / chunks cần cập nhật
        pairs = [(i, t, m) for i, t, m in zip(chunk_ids, texts, metas)]
        to_upsert = [p for p in pairs if p[0] not in existing]
        n = len(to_upsert)
        for start in range(0, n, batch_size):
            batch = to_upsert[start : start + batch_size]
            ids = [b[0] for b in batch]
            texts_b = [b[1] for b in batch]
            metas_b = [b[2] for b in batch]
            embeddings = self.model.encode(
                texts_b,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            col.upsert(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts_b,
                metadatas=[
                    {
                        "law_id": m.get("law_id", ""),
                        "article_id": m.get("article_id", ""),
                        **({"title": m["title"]} if m.get("title") else {}),
                    }
                    for m in metas_b
                ],
            )
            if progress_callback:
                progress_callback((start + len(batch)) / max(n, 1))
        self._collection = col
        return DenseIndexInfo(
            collection_name,
            self.model_name,
            col.count(),
            self.model.get_sentence_embedding_dimension(),
        )

    def delete_documents(self, chunk_ids: list[str], collection_name: str) -> int:
        """Remove chunks (by id) from a collection. Returns remaining count."""
        col = self.client.get_collection(collection_name)
        col.delete(ids=chunk_ids)
        self._collection = col
        return col.count()

    def list_collections(self) -> list[str]:
        return [c.name for c in self.client.list_collections()]

    # -- search ------------------------------------------------------------
    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if self._collection is None:
            raise RuntimeError("Call use_collection() or build_index() first.")
        q_emb = self.model.encode(
            [query], normalize_embeddings=True
        ).tolist()
        res = self._collection.query(query_embeddings=q_emb, n_results=top_k)
        results = []
        for i, (cid, dist, doc, meta) in enumerate(
            zip(
                res["ids"][0],
                res["distances"][0],
                res["documents"][0],
                res["metadatas"][0],
            )
        ):
            # cosine distance -> cosine similarity
            results.append(
                RetrievalResult(
                    chunk_id=cid,
                    score=1.0 - float(dist),
                    text=doc,
                    law_id=meta.get("law_id", ""),
                    article_id=meta.get("article_id", ""),
                    rank=i + 1,
                )
            )
        return results
