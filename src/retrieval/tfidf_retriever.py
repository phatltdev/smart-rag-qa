"""TF-IDF + cosine baseline retriever (screen 4.5, branch 1)."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import PROJECT_ROOT, TFIDF_NGRAM_RANGE


@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    text: str
    law_id: str
    article_id: str
    rank: int = 0


class TfidfRetriever:
    """Sparse baseline retriever. Operates on preprocessed texts so that the
    segmentation experiment (RQ1) is controlled by the caller."""

    def __init__(self, ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            sublinear_tf=True,
            token_pattern=None,
            tokenizer=str.split,  # texts are already preprocessed/segmented
            lowercase=False,
            preprocessor=None,
        )
        self.matrix = None
        self.chunk_ids: list[str] = []
        self.texts: list[str] = []
        self.meta: list[dict] = []

    def fit(self, chunk_ids: list[str], texts: list[str], metas: list[dict]):
        self.chunk_ids = list(chunk_ids)
        self.texts = list(texts)
        self.meta = list(metas)
        self.matrix = self.vectorizer.fit_transform(texts)
        return self

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if self.matrix is None:
            raise RuntimeError("Call fit() before search().")
        qv = self.vectorizer.transform([query])
        scores = cosine_similarity(qv, self.matrix).ravel()
        order = np.argsort(-scores)[:top_k]
        results = []
        for rank, idx in enumerate(order, 1):
            if scores[idx] <= 0:
                break
            m = self.meta[idx]
            results.append(
                RetrievalResult(
                    chunk_id=self.chunk_ids[idx],
                    score=float(scores[idx]),
                    text=self.texts[idx],
                    law_id=m.get("law_id", ""),
                    article_id=m.get("article_id", ""),
                    rank=rank,
                )
            )
        return results

    # -- persistence ------------------------------------------------------
    def save(self, path: Path | None = None):
        path = path or (PROJECT_ROOT / "models" / "tfidf_retriever.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "matrix": self.matrix,
                    "chunk_ids": self.chunk_ids,
                    "texts": self.texts,
                    "meta": self.meta,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path | None = None) -> "TfidfRetriever":
        path = path or (PROJECT_ROOT / "models" / "tfidf_retriever.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.vectorizer = data["vectorizer"]
        obj.matrix = data["matrix"]
        obj.chunk_ids = data["chunk_ids"]
        obj.texts = data["texts"]
        obj.meta = data["meta"]
        return obj

    def matched_terms(self, query: str, chunk_index: int) -> list[str]:
        """Terms shared between query and chunk (for UI explanation)."""
        q_terms = {
            t for t in self.vectorizer.analyzer(query)
            if self.vectorizer.vocabulary_.get(t) is not None
        }
        row = self.matrix.getrow(chunk_index)
        c_terms = {self.vectorizer.feature_names_[j] for j in row.indices}
        return sorted(q_terms & c_terms)
