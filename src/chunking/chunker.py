"""Chunking strategies (screen 4.4 — Chunking Lab).

Strategies:
- "article": one legal article = one chunk (default for Zalo corpus).
- "fixed":   fixed-size chunks in tokens with optional overlap, split on
             sentence boundaries when possible (never mid-sentence unless a
             single sentence exceeds chunk_size).
- "sentence": group consecutive sentences up to chunk_size tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.config import ChunkingConfig
from src.ingestion.zalo_loader import ArticleChunk
from src.preprocessing.pipeline import sentence_segment


def _token_len(text: str) -> int:
    return len(text.split())


@dataclass
class ChunkStats:
    n_chunks: int = 0
    length_min: int = 0
    length_max: int = 0
    length_mean: float = 0.0
    n_over_limit: int = 0
    length_distribution: list[int] = field(default_factory=list)


def chunk_article_fixed(
    text: str,
    law_id: str,
    article_id: str,
    chunk_size: int = 256,
    overlap: int = 0,
    max_limit: int = 256,
) -> list[ArticleChunk]:
    """Split one article into fixed-size sentence-aligned chunks."""
    sentences = sentence_segment(text)
    chunks: list[ArticleChunk] = []
    current: list[str] = []
    current_len = 0

    def flush():
        nonlocal current, current_len
        if current:
            chunks.append(
                ArticleChunk(
                    chunk_id=f"{law_id}::{article_id}::{len(chunks)}",
                    document_id=f"law::{law_id}",
                    text=" ".join(current),
                    title="",
                    law_id=law_id,
                    article_id=article_id,
                    metadata={"part": len(chunks)},
                )
            )
            # keep overlap sentences
            if overlap > 0:
                kept, kept_len = [], 0
                for sent in reversed(current):
                    if kept_len + _token_len(sent) > overlap:
                        break
                    kept.insert(0, sent)
                    kept_len += _token_len(sent)
                current, current_len = kept, kept_len
            else:
                current, current_len = [], 0

    for sent in sentences:
        n = _token_len(sent)
        if n > chunk_size:
            # single over-long sentence: hard split
            words = sent.split()
            for i in range(0, len(words), chunk_size):
                piece = " ".join(words[i : i + chunk_size])
                chunks.append(
                    ArticleChunk(
                        chunk_id=f"{law_id}::{article_id}::{len(chunks)}",
                        document_id=f"law::{law_id}",
                        text=piece,
                        title="",
                        law_id=law_id,
                        article_id=article_id,
                        metadata={"part": len(chunks), "hard_split": True},
                    )
                )
            continue
        if current_len + n > chunk_size:
            flush()
        current.append(sent)
        current_len += n
    flush()
    return chunks


def build_chunks(
    articles: list[ArticleChunk],
    config: ChunkingConfig,
    max_token_limit: int = 256,
) -> tuple[list[ArticleChunk], ChunkStats]:
    """Build the chunk set from article-level chunks according to config."""
    result: list[ArticleChunk] = []
    if config.strategy == "article":
        result = [
            ArticleChunk(
                chunk_id=a.chunk_id,
                document_id=a.document_id,
                text=a.text,
                title=a.title,
                law_id=a.law_id,
                article_id=a.article_id,
                metadata=dict(a.metadata),
            )
            for a in articles
        ]
    else:
        for a in articles:
            result.extend(
                chunk_article_fixed(
                    a.text,
                    a.law_id,
                    a.article_id,
                    chunk_size=config.chunk_size,
                    overlap=config.chunk_overlap,
                    max_limit=max_token_limit,
                )
            )

    lengths = [_token_len(c.text) for c in result]
    stats = ChunkStats(
        n_chunks=len(result),
        length_min=min(lengths) if lengths else 0,
        length_max=max(lengths) if lengths else 0,
        length_mean=round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        n_over_limit=sum(1 for l in lengths if l > max_token_limit),
        length_distribution=lengths,
    )
    return result, stats
