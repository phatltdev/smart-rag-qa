"""Retrieval evaluation experiment runner (Week 4: RQ1, RQ2).

Research questions
------------------
RQ1: Word segmentation impact — none vs underthesea vs pyvi
     (retriever, chunking, dataset, top_k fixed).
RQ2: TF-IDF baseline vs dense SBERT retrieval.

Usage (CLI):
    python -m experiments.evaluate_retrieval --rq1 --retriever tfidf --split dev
    python -m experiments.evaluate_retrieval --rq2 --split dev

Results are saved to results/metrics/<experiment_id>.json
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from src.config import (
    PROCESSED_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    SPLIT_FILENAME,
    TOP_K_CHOICES,
    ChunkingConfig,
    PreprocessingConfig,
)
from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.ingestion.zalo_loader import load_corpus, load_questions
from src.preprocessing.pipeline import run_pipeline
from src.retrieval.tfidf_retriever import TfidfRetriever

MAX_K = max(TOP_K_CHOICES)
SEGMENTATIONS = ["none", "underthesea", "pyvi"]


def load_split() -> dict[str, list[str]]:
    path = PROCESSED_DIR / SPLIT_FILENAME
    with open(path, encoding="utf-8") as f:
        return json.load(f)["split_ids"]


def run_eval(
    retriever_kind: str,
    segmentation: str,
    split: str = "dev",
    max_questions: int | None = None,
) -> dict:
    """Run retrieval evaluation for one configuration.

    Returns a result dict ready to be dumped as JSON.
    """
    from src.chunking.chunker import build_chunks

    _, raw_chunks, version, _ = load_corpus()
    questions = load_questions()
    split_ids = set(load_split()[split])

    eval_qs = [q for q in questions if q["question_id"] in split_ids]
    if max_questions:
        eval_qs = eval_qs[:max_questions]

    pcfg = PreprocessingConfig(word_segmentation=segmentation)
    ccfg = ChunkingConfig(strategy="article")
    exp_id = f"retrieval_{retriever_kind}_seg-{segmentation}_{split}"

    # -- build/load index + search -----------------------------------------
    ranked_lists: list[list[str]] = []
    gold_sets: list[set[str]] = []
    t0 = time.time()

    if retriever_kind == "tfidf":
        texts, ids = [], []
        for c in build_chunks(raw_chunks, ccfg)[0]:
            texts.append(run_pipeline(c.text, pcfg).text)
            ids.append(c.chunk_id)
        retriever = TfidfRetriever().fit(ids, texts, [{"law_id": c.law_id, "article_id": c.article_id} for c in build_chunks(raw_chunks, ccfg)[0]])
        search = lambda q: retriever.search(q, top_k=MAX_K)  # noqa: E731
        index_info = {"type": "tfidf", "ngram": (1, 2)}
    else:
        from src.retrieval.dense_retriever import DEFAULT_MODEL, DenseRetriever

        dr = DenseRetriever(model_name=DEFAULT_MODEL)
        col_name = dr.collection_name(pcfg.config_id(), ccfg.config_id())
        existing = dr.list_collections()
        if col_name not in existing:
            raise RuntimeError(
                f"Dense collection '{col_name}' not found. Build it in screen 4.5 first."
            )
        dr.use_collection(col_name)
        search = lambda q: dr.search(q, top_k=MAX_K)  # noqa: E731
        index_info = {"type": "dense", "model": DEFAULT_MODEL, "collection": col_name}

    for q in eval_qs:
        qtext = run_pipeline(q["text"], pcfg).text
        results = search(qtext)
        ranked_lists.append([r.chunk_id for r in results])
        gold_sets.append(set(q["relevant_chunk_ids"]))

    metrics = evaluate_retrieval(ranked_lists, gold_sets, ks=TOP_K_CHOICES)
    elapsed = round(time.time() - t0, 1)

    out = {
        "experiment_id": exp_id,
        "date": datetime.now().isoformat(timespec="seconds"),
        "research_question": "RQ1" if retriever_kind == "tfidf" else "RQ2",
        "dataset_version": version,
        "split": split,
        "n_questions": len(eval_qs),
        "preprocessing": pcfg.__dict__,
        "chunking": ccfg.__dict__,
        "index": index_info,
        "random_seed": RANDOM_SEED,
        "elapsed_seconds": elapsed,
        "metrics": metrics.as_dict(),
    }
    return out


def save_result(result: dict) -> Path:
    metrics_dir = RESULTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / f"{result['experiment_id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rq1", action="store_true", help="Run RQ1 (segmentation comparison)")
    ap.add_argument("--rq2", action="store_true", help="Run RQ2 (tfidf vs dense)")
    ap.add_argument("--retriever", choices=["tfidf", "dense"], default="tfidf")
    ap.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    ap.add_argument("--max-questions", type=int, default=None)
    args = ap.parse_args()

    if args.rq1:
        for seg in SEGMENTATIONS:
            st = time.time()
            res = run_eval(args.retriever, seg, args.split, args.max_questions)
            path = save_result(res)
            print(f"[RQ1] seg={seg:<12} MRR={res['metrics']['mrr']:.4f} "
                  f"Hit@5={res['metrics']['hit@k']['5']:.4f} "
                  f"({res['elapsed_seconds']}s) -> {path.name}")
    elif args.rq2:
        for kind in ["tfidf", "dense"]:
            res = run_eval(kind, "none", args.split, args.max_questions)
            path = save_result(res)
            print(f"[RQ2] {kind:<6} MRR={res['metrics']['mrr']:.4f} "
                  f"Hit@5={res['metrics']['hit@k']['5']:.4f} "
                  f"({res['elapsed_seconds']}s) -> {path.name}")
    else:
        res = run_eval(args.retriever, "none", args.split, args.max_questions)
        path = save_result(res)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
