"""LLM generation via Ollama (grounded RAG answers).

Uses the local Ollama REST API (default http://localhost:11434).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

import requests

from src.config import (
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
)

OLLAMA_BASE_URL = "http://localhost:11434"
GENERATE_TIMEOUT = 300  # seconds; local inference of 7B models can be slow

SYSTEM_PROMPT = """\
Bạn là trợ lý pháp lý tiếng Việt. Chỉ trả lời dựa trên các ĐIỀU LUẬT được cung cấp.
Quy tắc:
1. Trích dẫn điều luật bạn dùng dưới dạng [Điều <số>, <law_id>] ngay sau thông tin.
2. Nếu các điều luật cung cấp KHÔNG đủ để trả lời, hãy trả lời đúng câu:
   "Các điều luật được cung cấp không đủ thông tin để trả lời câu hỏi này."
3. Không bịa bổ sung kiến thức ngoài các điều luật được cung cấp.
4. Trả lời ngắn gọn, bằng tiếng Việt.
"""


@dataclass
class CitedAnswer:
    """LLM answer with parsed citations and raw metadata."""

    answer: str
    citations: list[str] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    eval_tokens: int = 0
    eval_duration_ms: float = 0.0
    insufficient: bool = False


def format_context(results: list) -> str:
    """Format retrieved chunks into a numbered context block.

    Each retrieval result needs .rank, .chunk_id, .text.
    """
    parts = []
    for r in results:
        parts.append(f"[Điều nguồn {r.rank}] (id: {r.chunk_id})\n{r.text}")
    return "\n\n".join(parts)


def build_prompt(question: str, context: str) -> str:
    """Grounded prompt: context first, then question, strict rules."""
    return (
        f"Các điều luật được cung cấp:\n"
        f"{context}\n\n"
        f"Câu hỏi: {question}\n\n"
        f"Dựa trên các điều luật trên, hãy trả lời câu hỏi và trích dẫn nguồn."
    )


def ollama_available(base_url: str = OLLAMA_BASE_URL) -> bool:
    """Check whether the Ollama server is reachable."""
    try:
        return requests.get(f"{base_url}/api/tags", timeout=3).ok
    except requests.RequestException:
        return False


def list_ollama_models(base_url: str = OLLAMA_BASE_URL) -> list[str]:
    """List locally available Ollama model names."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except requests.RequestException:
        return []


CITATION_RE = re.compile(r"\[Điều nguồn (\d+)\]")


def _parse_citations(answer: str, results: list) -> list[str]:
    """Map [Điều nguồn n] markers back to chunk ids."""
    ids = []
    for m in CITATION_RE.finditer(answer):
        n = int(m.group(1))
        if 0 < n <= len(results):
            cid = results[n - 1].chunk_id
            if cid not in ids:
                ids.append(cid)
    return ids


INSUFFICIENT_PATTERNS = (
    "không đủ thông tin để trả lời",
)


def generate_answer_stream(
    question: str,
    results: list,
    model: str = "qwen2.5:7b",
    temperature: float = GENERATION_TEMPERATURE,
    max_tokens: int = GENERATION_MAX_TOKENS,
    base_url: str = OLLAMA_BASE_URL,
    on_token: Callable[[str, str], None] | None = None,
) -> "CitedAnswer":
    """Streamed generation: call ``on_token(text_so_far, new_token)`` per token.

    Otherwise identical to :func:`generate_answer` (returns the final
    CitedAnswer with metadata parsed from the last streamed chunk).
    """
    context = format_context(results)
    prompt = build_prompt(question, context)

    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "seed": 42,
        },
    }

    collected: list[str] = []
    final_meta: dict = {}
    with requests.post(
        f"{base_url}/api/generate", json=payload, timeout=GENERATE_TIMEOUT,
        stream=True,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("response", "")
            if token:
                collected.append(token)
                if on_token:
                    on_token("".join(collected), token)
            if chunk.get("done"):
                final_meta = chunk
                break

    answer = "".join(collected).strip()
    citations = _parse_citations(answer, results)
    insufficient = any(p in answer.lower() for p in INSUFFICIENT_PATTERNS)

    return CitedAnswer(
        answer=answer,
        citations=citations,
        model=model,
        prompt_tokens=final_meta.get("prompt_eval_count", 0) or 0,
        eval_tokens=final_meta.get("eval_count", 0) or 0,
        eval_duration_ms=(final_meta.get("eval_duration", 0) or 0) / 1e6,
        insufficient=insufficient,
    )


def generate_answer(
    question: str,
    results: list,
    model: str = "qwen2.5:7b",
    temperature: float = GENERATION_TEMPERATURE,
    max_tokens: int = GENERATION_MAX_TOKENS,
    base_url: str = OLLAMA_BASE_URL,
    progress_callback: Callable[[str], None] | None = None,
) -> CitedAnswer:
    """Generate a grounded answer via Ollama's /api/generate."""
    context = format_context(results)
    prompt = build_prompt(question, context)

    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "seed": 42,
        },
    }
    if progress_callback:
        progress_callback(f"Đang gọi Ollama ({model})...")

    resp = requests.post(
        f"{base_url}/api/generate", json=payload, timeout=GENERATE_TIMEOUT
    )
    resp.raise_for_status()
    data = resp.json()

    answer = (data.get("response") or "").strip()
    citations = _parse_citations(answer, results)
    insufficient = any(p in answer.lower() for p in INSUFFICIENT_PATTERNS)

    return CitedAnswer(
        answer=answer,
        citations=citations,
        model=model,
        prompt_tokens=data.get("prompt_eval_count", 0) or 0,
        eval_tokens=data.get("eval_count", 0) or 0,
        eval_duration_ms=(data.get("eval_duration", 0) or 0) / 1e6,
        insufficient=insufficient,
    )
