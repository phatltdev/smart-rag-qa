"""Unit tests for the persisted RAG & LLM configuration service.

Covers: defaults when no file exists, save/load round-trip persistence,
backend validation of every field, audit trail, corrupt-file fallback and
model-existence checks against Ollama.

Run: python -m pytest tests/test_system_config.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.system_config import (  # noqa: E402
    ConfigValidationError,
    RagLlmConfig,
    load_config,
    read_audit_entries,
    save_config,
    validate_config,
)


@pytest.fixture()
def paths(tmp_path):
    """Isolated config + audit files per test (never touch configs/)."""
    return tmp_path / "rag_llm_config.json", tmp_path / "audit.jsonl"


def test_defaults_when_no_file(paths):
    config_path, _ = paths
    config, error = load_config(config_path)
    assert error is None
    assert config.retriever == "dense"
    assert config.top_k == 5
    assert config.word_segmentation == "none"
    assert config.llm_provider == "ollama"
    assert config.llm_model == "qwen2.5:7b"
    assert config.temperature == pytest.approx(0.2)
    assert config.max_tokens == 512


def test_save_and_reload_roundtrip(paths):
    config_path, audit_path = paths
    saved = save_config(
        RagLlmConfig(
            retriever="tfidf", top_k=10, word_segmentation="underthesea",
            llm_provider="ollama", llm_model="qwen2.5:7b",
            temperature=0.7, max_tokens=1024,
        ),
        updated_by="Quản trị viên",
        config_path=config_path,
        audit_path=audit_path,
    )
    assert saved.updated_at != "" and saved.updated_by == "Quản trị viên"

    # Reload from disk — persistence across "restarts".
    reloaded, error = load_config(config_path)
    assert error is None
    assert reloaded.public_dict() == saved.public_dict()
    assert reloaded.retriever == "tfidf"
    assert reloaded.top_k == 10
    assert reloaded.temperature == pytest.approx(0.7)
    assert reloaded.max_tokens == 1024


def test_second_save_overwrites_active_config(paths):
    config_path, audit_path = paths
    save_config(RagLlmConfig(top_k=5), config_path=config_path, audit_path=audit_path)
    save_config(RagLlmConfig(top_k=10), config_path=config_path, audit_path=audit_path)
    active, _ = load_config(config_path)
    assert active.top_k == 10  # active config = last save


def test_audit_trail_records_old_and_new(paths):
    config_path, audit_path = paths
    save_config(RagLlmConfig(top_k=5), updated_by="Admin A",
                config_path=config_path, audit_path=audit_path)
    save_config(RagLlmConfig(top_k=10), updated_by="Admin B",
                config_path=config_path, audit_path=audit_path)

    entries = read_audit_entries(10, audit_path=audit_path)
    assert len(entries) == 2
    # Newest first
    assert entries[0]["updated_by"] == "Admin B"
    assert entries[0]["old"]["top_k"] == 5
    assert entries[0]["new"]["top_k"] == 10
    assert entries[0]["result"] == "success"
    # First save had no previous config
    assert entries[1]["old"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("retriever", "hybrid"),            # unsupported retriever
        ("retriever", ""),
        ("word_segmentation", "vncorenlp"),  # unsupported segmentation
        ("llm_provider", "openai"),          # unsupported provider
        ("llm_model", "  "),                 # blank model name
        ("temperature", -0.1),
        ("temperature", 1.5),
        ("max_tokens", 0),
        ("max_tokens", 8192),
    ],
)
def test_validation_rejects_out_of_domain_values(field, value):
    with pytest.raises(ConfigValidationError):
        validate_config(RagLlmConfig(**{field: value}))


@pytest.mark.parametrize("top_k", [0, 21, -3])
def test_validation_rejects_bad_top_k(top_k):
    with pytest.raises(ConfigValidationError):
        validate_config(RagLlmConfig(top_k=top_k))


def test_validation_accepts_full_supported_range():
    validate_config(RagLlmConfig(retriever="tfidf", top_k=1, temperature=0.0,
                                 max_tokens=4096))
    validate_config(RagLlmConfig(retriever="dense", top_k=20, temperature=1.0,
                                 max_tokens=128))


def test_failed_save_leaves_previous_config_untouched(paths):
    config_path, audit_path = paths
    save_config(RagLlmConfig(top_k=5), config_path=config_path, audit_path=audit_path)
    before = config_path.read_text(encoding="utf-8")

    with pytest.raises(ConfigValidationError):
        save_config(RagLlmConfig(top_k=99), config_path=config_path,
                    audit_path=audit_path)

    assert config_path.read_text(encoding="utf-8") == before
    active, _ = load_config(config_path)
    assert active.top_k == 5


def test_corrupt_file_falls_back_to_defaults(paths):
    config_path, _ = paths
    config_path.write_text("{ not valid json", encoding="utf-8")
    config, error = load_config(config_path)
    assert error is not None
    assert config == RagLlmConfig()  # chat pipeline keeps working


def test_saved_file_must_validate_on_load(paths):
    config_path, _ = paths
    config_path.write_text('{"retriever": "mystery"}', encoding="utf-8")
    config, error = load_config(config_path)
    assert error is not None
    assert config.retriever == "dense"  # defaults, not the injected value


def test_model_check_only_when_ollama_online(paths, monkeypatch):
    import src.system_config as sc

    config_path, audit_path = paths
    monkeypatch.setattr(sc, "ollama_available", lambda: True)
    monkeypatch.setattr(sc, "list_ollama_models", lambda: ["qwen2.5:7b", "llama3"])

    with pytest.raises(ConfigValidationError, match="không tồn tại trong Ollama"):
        save_config(RagLlmConfig(llm_model="gpt-imaginary"),
                    config_path=config_path, audit_path=audit_path)

    # Offline Ollama: save is allowed (cannot verify, must not block).
    monkeypatch.setattr(sc, "ollama_available", lambda: False)
    saved = save_config(RagLlmConfig(llm_model="qwen2.5:7b"),
                        config_path=config_path, audit_path=audit_path)
    assert saved.llm_model == "qwen2.5:7b"
