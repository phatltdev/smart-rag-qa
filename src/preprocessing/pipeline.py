"""Preprocessing pipeline for Vietnamese legal text (screen 4.3).

Each step is a pure function: text in -> (text out, diff report).
This makes every step observable on the UI as required by design
principle #1 (transparent pipeline).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from src.config import PreprocessingConfig

# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
# Safe noise regex: control chars + uncommon symbols (keeps Vietnamese diacritics,
# punctuation used in legal texts, digits, parentheses, slashes, percent).
_NOISE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_unicode(text: str, form: str = "NFC") -> str:
    return unicodedata.normalize(form, text) if form != "none" else text


def normalize_whitespace(text: str) -> str:
    text = _WS_RE.sub(" ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return _MULTI_NEWLINE_RE.sub("\n\n", text).strip()


def remove_noise_chars(text: str) -> str:
    return _NOISE_RE.sub("", text)


def lowercase(text: str) -> str:
    return text.lower()


def segment_words(text: str, method: str) -> str:
    """Vietnamese word segmentation joining syllables with '_'."""
    if method == "none":
        return text
    if method == "underthesea":
        from underthesea import word_tokenize

        return word_tokenize(text, format="text")
    if method == "pyvi":
        from pyvi import ViTokenizer

        return ViTokenizer.tokenize(text)
    raise ValueError(f"Unknown segmentation method: {method}")


def remove_stopwords(text: str, stopwords: set[str]) -> str:
    if "_" in text:  # already segmented -> tokens separated by space, may contain '_'
        tokens = text.split(" ")
        kept = [t for t in tokens if t.lower().strip("_") not in stopwords]
    else:
        tokens = text.split(" ")
        kept = [t for t in tokens if t.lower() not in stopwords]
    return " ".join(kept)


def sentence_segment(text: str) -> list[str]:
    """Simple sentence splitter tuned for Vietnamese legal text.

    Splits on . ! ? followed by whitespace + uppercase/digit, avoiding
    common abbreviations (Điều, khoản, điểm, Art., v.v...).
    """
    protected = text
    abbrevs = ["Điều", "khoản", "điểm", "Mục", "Chương", "ttp", "TW", "ĐH", "QH"]
    for i, ab in enumerate(abbrevs):
        protected = protected.replace(f"{ab}.", f"{ab}\u0001{i}\u0001")
    parts = re.split(r"(?<=[.!?])\s+", protected)
    sentences = []
    for p in parts:
        for i, ab in enumerate(abbrevs):
            p = p.replace(f"{ab}\u0001{i}\u0001", f"{ab}.")
        p = p.strip()
        if p:
            sentences.append(p)
    return sentences


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class StepReport:
    step: str
    chars_before: int = 0
    chars_after: int = 0
    changed: bool = False
    details: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    text: str
    steps: list[StepReport] = field(default_factory=list)

    @property
    def n_changed_steps(self) -> int:
        return sum(1 for s in self.steps if s.changed)


def run_pipeline(
    text: str,
    config: PreprocessingConfig,
    stopwords: set[str] | None = None,
) -> PipelineResult:
    """Run configured preprocessing steps sequentially, recording each step."""
    result = PipelineResult(text=text)

    def apply(step: str, fn, detail: dict | None = None) -> None:
        before = result.text
        after = fn(before)
        result.steps.append(
            StepReport(
                step=step,
                chars_before=len(before),
                chars_after=len(after),
                changed=before != after,
                details=detail or {},
            )
        )
        result.text = after

    if config.unicode_normalization != "none":
        apply("unicode_normalization", lambda t: normalize_unicode(t, config.unicode_normalization))
    if config.whitespace_normalization:
        apply("whitespace_normalization", normalize_whitespace)
    if config.remove_noise_chars:
        removed = len(_NOISE_RE.findall(text))
        apply("remove_noise_chars", remove_noise_chars, {"removed_count": removed})
    if config.lowercase:
        apply("lowercase", lowercase)
    if config.word_segmentation != "none":
        apply(
            "word_segmentation",
            lambda t: segment_words(t, config.word_segmentation),
            {"method": config.word_segmentation},
        )
    if config.remove_stopwords and stopwords:
        before_tokens = len(result.text.split())
        apply("stopword_removal", lambda t: remove_stopwords(t, stopwords),
              {"tokens_before": before_tokens})
    return result


def preprocess_query(text: str, config: PreprocessingConfig) -> str:
    """Runtime query preprocessing: same steps as corpus (no segmentation-only
    steps skipped) so query and corpus live in the same space."""
    return run_pipeline(text, config).text


def compute_stats(text: str) -> dict:
    sentences = sentence_segment(text)
    return {
        "n_chars": len(text),
        "n_sentences": len(sentences),
        "n_tokens": len(text.split()),
    }
