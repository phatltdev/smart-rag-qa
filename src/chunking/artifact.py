"""Persist reproducible chunking artifacts and their manifests."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

from src.chunking.chunker import build_chunks
from src.config import PROCESSED_DIR, RANDOM_SEED, ChunkingConfig
from src.ingestion.zalo_loader import ArticleChunk, load_corpus
from src.preprocessing.dataset_processor import (
    list_manifests as list_preprocessing_manifests,
    load_processed_records_by_id,
)

ProgressCallback = Callable[[int, int], None]


def _artifact_id(source_id: str, config: ChunkingConfig) -> str:
    return f"{source_id}_{config.config_id()}"


def chunk_artifact_paths(
    source_id: str,
    config: ChunkingConfig,
    output_dir: Path = PROCESSED_DIR,
) -> tuple[Path, Path]:
    artifact_id = _artifact_id(source_id, config)
    return (
        output_dir / f"chunks_{artifact_id}.jsonl",
        output_dir / "manifests" / f"chunking_{artifact_id}.json",
    )


def load_chunk_manifest(
    source_id: str,
    config: ChunkingConfig,
    output_dir: Path = PROCESSED_DIR,
) -> dict | None:
    data_path, manifest_path = chunk_artifact_paths(source_id, config, output_dir)
    if not data_path.exists() or not manifest_path.exists():
        return None
    with open(manifest_path, encoding="utf-8") as file:
        return json.load(file)


def chunk_artifact_is_current(
    source_id: str,
    config: ChunkingConfig,
    dataset_version: str,
    output_dir: Path = PROCESSED_DIR,
) -> bool:
    manifest = load_chunk_manifest(source_id, config, output_dir)
    return bool(
        manifest
        and manifest.get("status") == "completed"
        and manifest.get("dataset_version") == dataset_version
        and manifest.get("source_id") == source_id
        and manifest.get("chunking_config_id") == config.config_id()
    )


def _load_source_articles(
    source_id: str,
    corpus_path: Path | None,
    output_dir: Path,
) -> tuple[list[ArticleChunk], str, dict | None]:
    _, raw_articles, dataset_version, _ = load_corpus(corpus_path)
    if source_id == "raw":
        return raw_articles, dataset_version, None

    records = load_processed_records_by_id(source_id, output_dir)
    articles = [
        ArticleChunk(
            chunk_id=record["chunk_id"],
            document_id=record["document_id"],
            text=record["text"],
            title=record["title"],
            law_id=record["law_id"],
            article_id=record["article_id"],
            metadata=record.get("metadata", {}),
        )
        for record in records
    ]
    preprocessing_manifest = next(
        (
            manifest
            for manifest in list_preprocessing_manifests(output_dir)
            if manifest.get("config_id") == source_id
        ),
        None,
    )
    if not preprocessing_manifest:
        raise FileNotFoundError(f"Preprocessing manifest not found: {source_id}")
    if preprocessing_manifest.get("dataset_version") != dataset_version:
        raise ValueError("Preprocessing artifact does not match the current dataset")
    return articles, dataset_version, preprocessing_manifest


def generate_and_save_chunks(
    source_id: str,
    config: ChunkingConfig,
    *,
    corpus_path: Path | None = None,
    output_dir: Path = PROCESSED_DIR,
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Generate all chunks and atomically save JSONL plus a manifest."""
    articles, dataset_version, preprocessing_manifest = _load_source_articles(
        source_id, corpus_path, output_dir
    )
    data_path, manifest_path = chunk_artifact_paths(source_id, config, output_dir)
    if not force and chunk_artifact_is_current(
        source_id, config, dataset_version, output_dir
    ):
        manifest = load_chunk_manifest(source_id, config, output_dir)
        assert manifest is not None
        return {**manifest, "reused": True}

    chunks, stats = build_chunks(articles, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    data_tmp = data_path.with_name(f".{data_path.name}.{run_id}.tmp")
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.{run_id}.tmp")

    try:
        with open(data_tmp, "w", encoding="utf-8") as output_file:
            for index, chunk in enumerate(chunks, start=1):
                output_file.write(
                    json.dumps(
                        {
                            "chunk_id": chunk.chunk_id,
                            "document_id": chunk.document_id,
                            "law_id": chunk.law_id,
                            "article_id": chunk.article_id,
                            "title": chunk.title,
                            "text": chunk.text,
                            "metadata": chunk.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if progress_callback:
                    progress_callback(index, len(chunks))

        artifact_id = _artifact_id(source_id, config)
        manifest = {
            "status": "completed",
            "reused": False,
            "artifact_id": artifact_id,
            "dataset": "Zalo AI 2021 — Legal Text Retrieval",
            "dataset_version": dataset_version,
            "source_id": source_id,
            "source_type": "raw" if source_id == "raw" else "preprocessing_artifact",
            "preprocessing_config_id": (
                None if source_id == "raw" else source_id
            ),
            "preprocessing_config": (
                preprocessing_manifest.get("preprocessing_config", {})
                if preprocessing_manifest
                else {}
            ),
            "chunking_config_id": config.config_id(),
            "chunking_config": asdict(config),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "random_seed": RANDOM_SEED,
            "input_articles": len(articles),
            "output_chunks": stats.n_chunks,
            "length_min": stats.length_min,
            "length_max": stats.length_max,
            "length_mean": stats.length_mean,
            "n_over_limit": stats.n_over_limit,
            "artifact_path": str(data_path),
        }
        with open(manifest_tmp, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
        data_tmp.replace(data_path)
        manifest_tmp.replace(manifest_path)
        return manifest
    except Exception:
        data_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
        raise


def list_chunk_manifests(output_dir: Path = PROCESSED_DIR) -> list[dict]:
    manifests_dir = output_dir / "manifests"
    if not manifests_dir.exists():
        return []
    manifests: list[dict] = []
    for path in manifests_dir.glob("chunking_*.json"):
        try:
            with open(path, encoding="utf-8") as file:
                manifest = json.load(file)
            data_path = output_dir / f"chunks_{manifest['artifact_id']}.jsonl"
            if manifest.get("status") == "completed" and data_path.exists():
                manifests.append(manifest)
        except (KeyError, OSError, json.JSONDecodeError):
            continue
    return sorted(manifests, key=lambda item: item.get("generated_at", ""), reverse=True)


def load_chunk_records(
    artifact_id: str,
    output_dir: Path = PROCESSED_DIR,
) -> list[dict]:
    data_path = output_dir / f"chunks_{artifact_id}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Chunk artifact not found: {data_path}")
    with open(data_path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
