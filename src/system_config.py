"""Persisted RAG & LLM runtime configuration, managed on the admin screen.

The admin "Cấu hình" screen (4.10) is the only writer; every chat surface
(admin playground 4.7 and the public portal assistant 5.1) reads the active
configuration through :func:`get_active_config` at query time. End users
never supply retrieval or generation parameters, so the persisted values
stay authoritative for the whole pipeline.

Storage follows the project's JSON-artifact convention (manifests under
``data/processed``): a small JSON file under ``configs/`` written atomically,
plus an append-only audit trail (timestamp, actor, old/new config, result).
The file is re-read on every call — cheap for a tiny document and it
guarantees a configuration saved by the admin is picked up by the very next
user request without any cache invalidation.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

from src.config import CONFIGS_DIR
from src.generation.ollama_client import list_ollama_models, ollama_available

# Option domains enforced by the backend (the admin UI mirrors these ranges).
SUPPORTED_RETRIEVERS: tuple[str, ...] = ("dense", "tfidf")
SUPPORTED_SEGMENTATIONS: tuple[str, ...] = ("none", "underthesea", "pyvi")
SUPPORTED_PROVIDERS: tuple[str, ...] = ("ollama",)
TOP_K_RANGE = (1, 20)
TEMPERATURE_RANGE = (0.0, 1.0)
MAX_TOKENS_RANGE = (128, 4096)

CONFIG_FILENAME = "rag_llm_config.json"
AUDIT_FILENAME = "rag_llm_config_audit.jsonl"
CONFIG_PATH = CONFIGS_DIR / CONFIG_FILENAME
AUDIT_PATH = CONFIGS_DIR / AUDIT_FILENAME

# Streamlit serves sessions on separate threads; serialize file writes.
_WRITE_LOCK = threading.Lock()


class ConfigValidationError(ValueError):
    """A candidate configuration failed backend validation."""


@dataclass
class RagLlmConfig:
    """Active RAG retrieval + LLM generation settings."""

    retriever: str = "dense"          # dense (SBERT) | tfidf (baseline)
    top_k: int = 5                    # chunks fed into the prompt
    word_segmentation: str = "none"   # none | underthesea | pyvi

    llm_provider: str = "ollama"      # ollama (extensible to other providers)
    llm_model: str = "qwen2.5:7b"
    temperature: float = 0.2
    max_tokens: int = 512

    # Bookkeeping — set by save_config, never edited from the form.
    updated_at: str = ""
    updated_by: str = ""

    def public_dict(self) -> dict:
        """Editable fields only (admin form state, audit old/new diff)."""
        return {
            "retriever": self.retriever,
            "top_k": self.top_k,
            "word_segmentation": self.word_segmentation,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RagLlmConfig":
        """Build a config from stored JSON, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def validate_config(config: RagLlmConfig, *, check_model: bool = False) -> None:
    """Backend validation of every persisted field.

    ``check_model`` additionally verifies the model exists in the Ollama
    registry when the server is reachable. Offline saves are allowed so a
    down LLM host never blocks configuration work.
    """
    problems: list[str] = []
    if config.retriever not in SUPPORTED_RETRIEVERS:
        problems.append(
            f"Retriever phải là một trong {list(SUPPORTED_RETRIEVERS)}, "
            f"nhận được '{config.retriever}'."
        )
    if not isinstance(config.top_k, int) or isinstance(config.top_k, bool):
        problems.append("top_k phải là số nguyên.")
    elif not TOP_K_RANGE[0] <= config.top_k <= TOP_K_RANGE[1]:
        problems.append(f"top_k phải nằm trong khoảng {TOP_K_RANGE[0]}–{TOP_K_RANGE[1]}.")
    if config.word_segmentation not in SUPPORTED_SEGMENTATIONS:
        problems.append(
            f"Tách từ phải là một trong {list(SUPPORTED_SEGMENTATIONS)}, "
            f"nhận được '{config.word_segmentation}'."
        )
    if config.llm_provider not in SUPPORTED_PROVIDERS:
        problems.append(
            f"Provider phải là một trong {list(SUPPORTED_PROVIDERS)}, "
            f"nhận được '{config.llm_provider}'."
        )
    if (
        not isinstance(config.temperature, (int, float))
        or isinstance(config.temperature, bool)
    ):
        problems.append("temperature phải là số.")
    elif not TEMPERATURE_RANGE[0] <= config.temperature <= TEMPERATURE_RANGE[1]:
        problems.append(
            f"temperature phải nằm trong khoảng "
            f"{TEMPERATURE_RANGE[0]}–{TEMPERATURE_RANGE[1]}."
        )
    if not isinstance(config.max_tokens, int) or isinstance(config.max_tokens, bool):
        problems.append("max_tokens phải là số nguyên.")
    elif not MAX_TOKENS_RANGE[0] <= config.max_tokens <= MAX_TOKENS_RANGE[1]:
        problems.append(
            f"max_tokens phải nằm trong khoảng "
            f"{MAX_TOKENS_RANGE[0]}–{MAX_TOKENS_RANGE[1]}."
        )
    model = (config.llm_model or "").strip()
    if not model:
        problems.append("Tên model LLM không được để trống.")
    if check_model and model and config.llm_provider == "ollama" and ollama_available():
        available = list_ollama_models()
        if available and model not in available:
            preview = ", ".join(available[:5]) + (", ..." if len(available) > 5 else "")
            problems.append(
                f"Model '{model}' không tồn tại trong Ollama (có sẵn: {preview})."
            )
    if problems:
        raise ConfigValidationError(" ".join(problems))


def load_config(config_path: Path = CONFIG_PATH) -> tuple[RagLlmConfig, str | None]:
    """Read the active configuration.

    Returns ``(config, error)``. A missing file yields the documented
    defaults; a corrupted file also falls back to defaults, with a
    descriptive error so the chat pipeline keeps working and the admin
    screen can explain why.
    """
    if not config_path.exists():
        return RagLlmConfig(), None
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("nội dung phải là một JSON object")
        config = RagLlmConfig.from_dict(raw)
        validate_config(config)  # display-time check: never requires Ollama
        return config, None
    except (OSError, ValueError) as error:
        return RagLlmConfig(), f"Không đọc được {config_path.name}: {error}"


def get_active_config(config_path: Path = CONFIG_PATH) -> RagLlmConfig:
    """Convenience accessor for the chat pipeline (defaults on any error)."""
    config, _ = load_config(config_path)
    return config


def save_config(
    config: RagLlmConfig,
    updated_by: str = "Quản trị viên",
    config_path: Path = CONFIG_PATH,
    audit_path: Path = AUDIT_PATH,
) -> RagLlmConfig:
    """Validate and persist the new active configuration (atomic write).

    Raises :class:`ConfigValidationError` when the candidate config is
    invalid; the previous file is left untouched in that case.
    """
    config.llm_model = (config.llm_model or "").strip()
    validate_config(config, check_model=True)
    had_previous = config_path.exists()
    previous, _ = load_config(config_path)

    config.updated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    config.updated_by = updated_by

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(asdict(config), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, config_path)
        _append_audit(
            {
                "timestamp": config.updated_at,
                "updated_by": updated_by,
                "old": previous.public_dict() if had_previous else None,
                "new": config.public_dict(),
                "result": "success",
            },
            audit_path,
        )
    return config


def _append_audit(entry: dict, audit_path: Path) -> None:
    """Best-effort append-only trail; a logging failure must not undo the save."""
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def read_audit_entries(limit: int = 20, audit_path: Path = AUDIT_PATH) -> list[dict]:
    """Most recent audit entries (newest first) for the system log screen."""
    if not audit_path.exists():
        return []
    try:
        lines = audit_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries: list[dict] = []
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return list(reversed(entries))[:limit]
