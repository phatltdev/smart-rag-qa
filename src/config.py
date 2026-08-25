"""Global configuration for Smart RAG QA.

All experiment-relevant parameters live here so runs are reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib
import json

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"

for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Dataset (Zalo AI 2021 Legal Text Retrieval)
# ---------------------------------------------------------------------------
CORPUS_FILENAME = "legal_corpus.json"
QNA_FILENAME = "train_question_answer.json"
TEST_QUESTION_FILENAME = "public_test_question.json"
STOPWORDS_FILENAME = "stopwords.txt"

SPLIT_FILENAME = "question_split.json"
SPLIT_RATIOS = {"train": 0.7, "dev": 0.1, "test": 0.2}


# ---------------------------------------------------------------------------
# Preprocessing config
# ---------------------------------------------------------------------------
@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline (screen 4.3)."""

    unicode_normalization: str = "NFC"  # "NFC" | "none"
    whitespace_normalization: bool = True
    remove_noise_chars: bool = True
    lowercase: bool = False  # dense models: keep case until verified
    word_segmentation: str = "none"  # "none" | "underthesea" | "pyvi"
    remove_stopwords: bool = False  # applied for TF-IDF only by default

    def config_id(self) -> str:
        """Stable hash identifying this configuration (for caching/versioning)."""
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.md5(payload).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Chunking config (screen 4.4)
# ---------------------------------------------------------------------------
@dataclass
class ChunkingConfig:
    strategy: str = "article"  # "article" | "fixed" | "sentence"
    chunk_size: int = 256  # tokens (used by "fixed")
    chunk_overlap: int = 0

    def config_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.md5(payload).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Retrieval / generation defaults (screens 4.5-4.7)
# ---------------------------------------------------------------------------
TOP_K_DEFAULT = 10
TOP_K_CHOICES = [1, 3, 5, 10]
TFIDF_NGRAM_RANGE = (1, 2)

GENERATION_TEMPERATURE = 0.2
GENERATION_MAX_TOKENS = 512
