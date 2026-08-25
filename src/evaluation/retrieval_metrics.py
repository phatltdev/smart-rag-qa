"""Retrieval evaluation metrics (Week 4: RQ1, RQ2).

Metrics over ranked lists of chunk ids against gold relevant sets:
- Hit@K (a.k.a. Recall@K in Zalo AI 2021 terminology)
- Precision@K
- Recall@K (fraction of gold items found in top-K)
- MRR (Mean Reciprocal Rank)

Conventions
-----------
For each question we have a gold set of relevant chunk ids
(``law_id::article_id``) and a ranked prediction list. All metrics are
computed per question then averaged (macro average over questions).
"""
from __future__ import annotations

from dataclasses import dataclass, field


def hit_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """1.0 if at least one gold item appears in the top-k, else 0.0."""
    if not gold:
        return 0.0
    return 1.0 if set(ranked[:k]) & gold else 0.0


def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """|relevant in top-k| / k."""
    if k <= 0:
        return 0.0
    return len(set(ranked[:k]) & gold) / k


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """|relevant in top-k| / |gold|."""
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    """1/rank of the first relevant item (0 if none in list)."""
    for i, cid in enumerate(ranked, start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


@dataclass
class RetrievalMetrics:
    """Aggregated metrics for one run over many questions."""

    n_questions: int = 0
    hit: dict[int, float] = field(default_factory=dict)      # hit@k
    precision: dict[int, float] = field(default_factory=dict)  # p@k
    recall: dict[int, float] = field(default_factory=dict)   # recall@k
    mrr: float = 0.0

    def as_dict(self) -> dict:
        return {
            "n_questions": self.n_questions,
            "hit@k": {str(k): v for k, v in self.hit.items()},
            "precision@k": {str(k): v for k, v in self.precision.items()},
            "recall@k": {str(k): v for k, v in self.recall.items()},
            "mrr": self.mrr,
        }


def evaluate_retrieval(
    ranked_lists: list[list[str]],
    gold_sets: list[set[str]],
    ks: list[int] | None = None,
) -> RetrievalMetrics:
    """Compute aggregated metrics.

    Parameters
    ----------
    ranked_lists : per-question ranked chunk ids (best first)
    gold_sets : per-question set of relevant chunk ids
    ks : list of K values for the @k metrics
    """
    assert len(ranked_lists) == len(gold_sets), "ranked/gold length mismatch"
    ks = ks or [1, 3, 5, 10]
    m = RetrievalMetrics(n_questions=len(ranked_lists))
    if not ranked_lists:
        return m

    for k in ks:
        m.hit[k] = sum(hit_at_k(r, g, k) for r, g in zip(ranked_lists, gold_sets)) / len(ranked_lists)
        m.precision[k] = sum(precision_at_k(r, g, k) for r, g in zip(ranked_lists, gold_sets)) / len(ranked_lists)
        m.recall[k] = sum(recall_at_k(r, g, k) for r, g in zip(ranked_lists, gold_sets)) / len(ranked_lists)
    m.mrr = sum(reciprocal_rank(r, g) for r, g in zip(ranked_lists, gold_sets)) / len(ranked_lists)
    return m
