"""Ingestion: load and validate the Zalo AI 2021 Legal corpus.

Screens: 4.1 (overview), 4.2 (data management).
Unit of relevance: article = (law_id, article_id).
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path

from src.config import (
    CORPUS_FILENAME,
    PROCESSED_DIR,
    QNA_FILENAME,
    RANDOM_SEED,
    RAW_DIR,
    SPLIT_FILENAME,
    SPLIT_RATIOS,
    STOPWORDS_FILENAME,
)


@dataclass
class Document:
    document_id: str
    title: str
    source: str  # law_id
    raw_text: str
    dataset_version: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ArticleChunk:
    """One article of one law = default chunk (strategy='article')."""

    chunk_id: str
    document_id: str
    text: str
    title: str
    law_id: str
    article_id: str
    metadata: dict = field(default_factory=dict)


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:10]


def load_corpus(
    corpus_path: Path | None = None,
) -> tuple[list[Document], list[ArticleChunk], str, dict]:
    """Load legal_corpus.json into Documents + article-level Chunks.

    Returns:
        documents, chunks, dataset_version, validation_report
    """
    corpus_path = corpus_path or (RAW_DIR / CORPUS_FILENAME)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path} not found. See data/DATASET_GUIDE.md to download."
        )

    with open(corpus_path, encoding="utf-8") as f:
        data = json.load(f)

    dataset_version = f"zalo2021-{_file_sha(corpus_path)}"

    documents: list[Document] = []
    chunks: list[ArticleChunk] = []
    report = {
        "total_laws": 0,
        "valid_laws": 0,
        "total_articles": 0,
        "valid_articles": 0,
        "empty_articles": 0,
        "duplicate_articles": 0,
        "errors": [],
    }

    seen_texts: set[str] = set()

    for law in data:
        report["total_laws"] += 1
        law_id = (law.get("law_id") or "").strip()
        articles = law.get("articles") or []
        if not law_id or not articles:
            report["errors"].append(f"Invalid law record skipped: {law_id!r}")
            continue
        report["valid_laws"] += 1

        doc_id = f"law::{law_id}"
        full_text = "\n".join(
            f"{(a.get('title') or '').strip()}\n{(a.get('text') or '').strip()}"
            for a in articles
        )
        documents.append(
            Document(
                document_id=doc_id,
                title=law_id,
                source=law_id,
                raw_text=full_text,
                dataset_version=dataset_version,
                metadata={"n_articles": len(articles)},
            )
        )

        for art in articles:
            report["total_articles"] += 1
            article_id = (art.get("article_id") or "").strip()
            text = (art.get("text") or "").strip()
            title = (art.get("title") or "").strip()
            if not text:
                report["empty_articles"] += 1
                continue
            key = hashlib.md5(text.encode("utf-8")).hexdigest()
            if key in seen_texts:
                report["duplicate_articles"] += 1
                continue
            seen_texts.add(key)
            report["valid_articles"] += 1
            chunks.append(
                ArticleChunk(
                    chunk_id=f"{law_id}::{article_id}",
                    document_id=doc_id,
                    text=f"{title}\n{text}" if title else text,
                    title=title,
                    law_id=law_id,
                    article_id=article_id,
                    metadata={"text_length": len(text)},
                )
            )

    return documents, chunks, dataset_version, report


def load_questions(
    qna_path: Path | None = None,
) -> list[dict]:
    """Load train_question_answer.json (relevance judgments)."""
    qna_path = qna_path or (RAW_DIR / QNA_FILENAME)
    with open(qna_path, encoding="utf-8") as f:
        data = json.load(f)
    # The Kaggle file wraps the list under an "items" key
    # ({"_name_": ..., "_count_": ..., "items": [...]}).
    if isinstance(data, dict):
        data = data.get("items", [])
    questions = []
    for item in data:
        relevant = {
            f"{r['law_id']}::{r['article_id']}"
            for r in item.get("relevant_articles", [])
            if r.get("law_id") and r.get("article_id")
        }
        questions.append(
            {
                "question_id": item.get("question_id"),
                "text": (item.get("question") or "").strip(),
                "relevant_chunk_ids": sorted(relevant),
            }
        )
    return questions


def split_questions(
    questions: list[dict],
    ratios: dict | None = None,
    seed: int = RANDOM_SEED,
    save: bool = True,
) -> dict[str, list[str]]:
    """Split question ids into train/dev/test; persist to processed/.

    Stratified lightly by number of relevant articles (rounded).
    """
    ratios = ratios or SPLIT_RATIOS
    rng = random.Random(seed)

    buckets: dict[int, list[dict]] = {}
    for q in questions:
        buckets.setdefault(min(len(q["relevant_chunk_ids"]), 4), []).append(q)

    split_ids: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for group in buckets.values():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * ratios["train"])
        n_dev = int(n * ratios["dev"])
        for i, q in enumerate(group):
            if i < n_train:
                split_ids["train"].append(q["question_id"])
            elif i < n_train + n_dev:
                split_ids["dev"].append(q["question_id"])
            else:
                split_ids["test"].append(q["question_id"])

    if save:
        out = PROCESSED_DIR / SPLIT_FILENAME
        with open(out, "w", encoding="utf-8") as f:
            json.dump(
                {"seed": seed, "ratios": ratios, "split_ids": split_ids},
                f,
                ensure_ascii=False,
                indent=2,
            )
    return split_ids


def load_stopwords(path: Path | None = None) -> list[str]:
    path = path or (RAW_DIR / STOPWORDS_FILENAME)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
