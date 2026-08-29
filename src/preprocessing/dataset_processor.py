"""Materialize reproducible preprocessing artifacts for the legal corpus."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

from src.config import (
    CORPUS_FILENAME,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DIR,
    PreprocessingConfig,
)
from src.ingestion.zalo_loader import load_corpus, load_stopwords
from src.preprocessing.pipeline import run_pipeline

ProgressCallback = Callable[[int, int], None]


def artifact_paths(
    config: PreprocessingConfig,
    output_dir: Path = PROCESSED_DIR,
) -> tuple[Path, Path]:
    """Return data and manifest paths for a preprocessing configuration."""
    config_id = config.config_id()
    return (
        output_dir / f"corpus_{config_id}.jsonl",
        output_dir / "manifests" / f"preprocessing_{config_id}.json",
    )


def load_manifest(
    config: PreprocessingConfig,
    output_dir: Path = PROCESSED_DIR,
) -> dict | None:
    """Load a manifest only when both manifest and artifact exist."""
    artifact_path, manifest_path = artifact_paths(config, output_dir)
    if not artifact_path.exists() or not manifest_path.exists():
        return None
    with open(manifest_path, encoding="utf-8") as file:
        return json.load(file)


def artifact_is_current(
    config: PreprocessingConfig,
    dataset_version: str,
    output_dir: Path = PROCESSED_DIR,
) -> bool:
    """Check whether a reusable artifact matches dataset and configuration."""
    manifest = load_manifest(config, output_dir)
    return bool(
        manifest
        and manifest.get("dataset_version") == dataset_version
        and manifest.get("config_id") == config.config_id()
        and manifest.get("status") == "completed"
    )


def process_dataset(
    config: PreprocessingConfig,
    *,
    corpus_path: Path | None = None,
    output_dir: Path = PROCESSED_DIR,
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Preprocess every valid article and save an atomic JSONL artifact.

    Raw data is read-only. Existing, current artifacts are reused unless
    ``force`` is true. A failed run leaves any previous valid artifact intact.
    """
    corpus_path = corpus_path or (RAW_DIR / CORPUS_FILENAME)
    _, chunks, dataset_version, validation_report = load_corpus(corpus_path)
    artifact_path, manifest_path = artifact_paths(config, output_dir)

    if not force and artifact_is_current(config, dataset_version, output_dir):
        manifest = load_manifest(config, output_dir)
        assert manifest is not None
        return {**manifest, "reused": True}

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    artifact_tmp = artifact_path.with_name(f".{artifact_path.name}.{run_id}.tmp")
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.{run_id}.tmp")
    stopwords = set(load_stopwords()) if config.remove_stopwords else set()

    chars_before = 0
    chars_after = 0
    tokens_before = 0
    tokens_after = 0
    changed_articles = 0

    try:
        with open(artifact_tmp, "w", encoding="utf-8") as output_file:
            for index, chunk in enumerate(chunks, start=1):
                result = run_pipeline(chunk.text, config, stopwords)
                processed_text = result.text
                chars_before += len(chunk.text)
                chars_after += len(processed_text)
                tokens_before += len(chunk.text.split())
                tokens_after += len(processed_text.split())
                changed_articles += int(chunk.text != processed_text)
                output_file.write(
                    json.dumps(
                        {
                            "chunk_id": chunk.chunk_id,
                            "document_id": chunk.document_id,
                            "law_id": chunk.law_id,
                            "article_id": chunk.article_id,
                            "title": chunk.title,
                            "text": processed_text,
                            "metadata": chunk.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if progress_callback:
                    progress_callback(index, len(chunks))

        manifest = {
            "status": "completed",
            "reused": False,
            "dataset": "Zalo AI 2021 — Legal Text Retrieval",
            "dataset_version": dataset_version,
            "source_file": str(corpus_path),
            "config_id": config.config_id(),
            "preprocessing_config": asdict(config),
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "random_seed": RANDOM_SEED,
            "input_articles": len(chunks),
            "output_articles": len(chunks),
            "changed_articles": changed_articles,
            "chars_before": chars_before,
            "chars_after": chars_after,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "validation_report": validation_report,
            "artifact_path": str(artifact_path),
        }
        with open(manifest_tmp, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)

        artifact_tmp.replace(artifact_path)
        manifest_tmp.replace(manifest_path)
        return manifest
    except Exception:
        artifact_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
        raise


def load_processed_records(
    config: PreprocessingConfig,
    output_dir: Path = PROCESSED_DIR,
) -> list[dict]:
    """Load all records from a materialized preprocessing artifact."""
    artifact_path, _ = artifact_paths(config, output_dir)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Preprocessing artifact not found: {artifact_path}")
    with open(artifact_path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def list_manifests(output_dir: Path = PROCESSED_DIR) -> list[dict]:
    """List completed artifacts whose JSONL data file is still available."""
    manifests_dir = output_dir / "manifests"
    if not manifests_dir.exists():
        return []
    manifests: list[dict] = []
    for manifest_path in manifests_dir.glob("preprocessing_*.json"):
        try:
            with open(manifest_path, encoding="utf-8") as file:
                manifest = json.load(file)
            artifact_path = Path(manifest.get("artifact_path", ""))
            if manifest.get("status") == "completed" and artifact_path.exists():
                manifests.append(manifest)
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(manifests, key=lambda item: item.get("processed_at", ""), reverse=True)


def load_processed_records_by_id(
    config_id: str,
    output_dir: Path = PROCESSED_DIR,
) -> list[dict]:
    """Load an artifact by stable configuration identifier."""
    artifact_path = output_dir / f"corpus_{config_id}.jsonl"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Preprocessing artifact not found: {artifact_path}")
    with open(artifact_path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
