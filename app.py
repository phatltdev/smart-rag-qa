"""Streamlit app — Smart RAG QA (Zalo AI 2021 Legal Text Retrieval).

Week 1 scope: screens 4.1 (overview), 4.2 (data management),
4.3 (preprocessing lab).

Run: streamlit run app.py
"""
from __future__ import annotations

import html
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    CORPUS_FILENAME,
    QNA_FILENAME,
    RANDOM_SEED,
    RAW_DIR,
    SPLIT_FILENAME,
    TOP_K_CHOICES,
    TOP_K_DEFAULT,
    ChunkingConfig,
    PreprocessingConfig,
)
from src.ingestion.zalo_loader import (  # noqa: E402
    ArticleChunk,
    load_corpus,
    load_questions,
    load_stopwords,
    split_questions,
)
from src.preprocessing.pipeline import (  # noqa: E402
    compute_stats,
    run_pipeline,
    sentence_segment,
    segment_words,
)
from src.preprocessing.dataset_processor import (  # noqa: E402
    artifact_is_current,
    artifact_paths,
    load_manifest,
    load_processed_records,
    load_processed_records_by_id,
    list_manifests,
    process_dataset,
)
from src.chunking.chunker import build_chunks  # noqa: E402
from src.chunking.artifact import (  # noqa: E402
    chunk_artifact_is_current,
    chunk_artifact_paths,
    generate_and_save_chunks,
    list_chunk_manifests,
    load_chunk_manifest,
    load_chunk_records,
)
from src.retrieval.tfidf_retriever import TfidfRetriever  # noqa: E402
from src.retrieval.dense_retriever import DEFAULT_MODEL, DenseRetriever  # noqa: E402
from src.generation.ollama_client import (  # noqa: E402
    generate_answer_stream,
    list_ollama_models,
    ollama_available,
)
from src.evaluation.retrieval_metrics import evaluate_retrieval  # noqa: E402
from src.system_config import (  # noqa: E402
    ConfigValidationError,
    RagLlmConfig,
    get_active_config,
    load_config,
    read_audit_entries,
    save_config,
)
from src.ui.legal_portal import render_legal_portal  # noqa: E402
from src.ui.theme import admin_page_header, apply_admin_theme  # noqa: E402

st.set_page_config(
    page_title="RAG Management & QA Platform",
    page_icon=":material/hub:",
    layout="wide",
)

# Shared design system: light "legal portal" visual language for the admin
# surface. The public portal keeps its own stylesheet (legal_portal.py).
if st.session_state.get("app_role", "Quản trị viên") == "Quản trị viên":
    apply_admin_theme(compact=st.session_state.get("sidebar_compact", False))

# Modern chat UI styling
st.markdown(
    """
    <style>
    /* Chat container */
    [data-testid="stChatMessage"] {
        background: transparent;
        border: none;
        animation: fadeSlideIn 0.25s ease;
    }
    /* Consistent Material Symbols sizing across the application. */
    span[role="img"][aria-label$=" icon"] {
        font-size: 1.3em;
        vertical-align: -0.18em;
    }
    /* Make the assistant identity visible at a glance. */
    [data-testid="stChatMessage"] [data-testid*="Avatar"] {
        width: 2.75rem;
        height: 2.75rem;
    }
    /* One visual group per assistant message.  Style the message content
       wrapper instead of every nested Markdown block (status/expanders also
       contain Markdown and must not become separate bubbles).
       Palette matches the user portal: slate bubble + gold left accent. */
    [data-testid="stChatMessage"] > [data-testid="stChatMessageContent"] {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #B88732;
        border-radius: 4px 16px 16px 16px;
        padding: 10px 16px;
    }
    /* Custom user bubble (right-aligned) — navy tint like the portal */
    .user-msg-row {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        justify-content: flex-end;
        width: 100%;
        margin: 0.6rem 0;
        animation: fadeSlideIn 0.25s ease;
    }
    .user-msg-bubble {
        background: rgba(15, 42, 68, 0.06);
        border: 1px solid rgba(15, 42, 68, 0.14);
        border-right: 3px solid rgba(15, 42, 68, 0.35);
        border-radius: 16px 4px 16px 16px;
        padding: 10px 16px;
        width: fit-content;
        max-width: 75%;
        color: #0F172A;
        font-size: 0.95rem;
        line-height: 1.5;
        overflow-wrap: anywhere;
        word-break: normal;
        white-space: pre-wrap;
    }
    .user-msg-label {
        font-size: 0.7rem;
        opacity: 0.55;
        text-align: right;
        margin-bottom: 2px;
    }
    .assistant-msg-time {
        font-size: 0.7rem;
        opacity: 0.55;
        text-align: left;
        margin-bottom: 2px;
    }
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def current_message_time() -> str:
    """Return the local display time for a newly sent chat message."""
    return datetime.now().astimezone().strftime("%H:%M")


def render_user_message(text: str, sent_at: str | None = None):
    """Right-aligned user bubble (custom HTML — st.chat_message can't do this)."""
    label = f"Bạn · {html.escape(sent_at)}" if sent_at else "Bạn"
    st.markdown(
        f'<div class="user-msg-row"><div class="user-msg-label">{label}</div>'
        f'<div class="user-msg-bubble">{html.escape(text)}</div></div>',
        unsafe_allow_html=True,
    )


def render_assistant_time(sent_at: str | None):
    """Render the assistant name and timestamp above its message."""
    if sent_at:
        st.markdown(
            f'<div class="assistant-msg-time">Trợ lý · {html.escape(sent_at)}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Đang nạp corpus...")
def get_corpus():
    docs, chunks, version, report = load_corpus()
    return docs, chunks, version, report


@st.cache_resource(show_spinner="Đang náp câu hỏi...")
def get_questions():
    return load_questions()


@st.cache_resource
def get_stopwords() -> set[str]:
    return set(load_stopwords())


def dataset_ready() -> bool:
    return (RAW_DIR / CORPUS_FILENAME).exists() and (RAW_DIR / QNA_FILENAME).exists()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _set_user_role() -> None:
    """Leave the admin dashboard for the public legal portal."""
    st.session_state.app_role = "Người dùng"
    st.session_state.user_portal_page = "home"


def _toggle_sidebar_compact() -> None:
    """Switch the admin sidebar between expanded and icon-rail modes."""
    st.session_state["sidebar_compact"] = not st.session_state.get(
        "sidebar_compact", False
    )


with st.sidebar:
    _brand_col, _collapse_col = st.columns(
        [0.74, 0.26], vertical_alignment="center", gap="small"
    )
    with _brand_col:
        st.markdown(
            '<div class="admin-brand"><span class="admin-brand-mark">&#9878;</span>'
            "<span class='admin-brand-text'><span class='admin-brand-name'>"
            "Legal RAG</span><div class='admin-brand-sub'>Quản trị hệ thống"
            "</div></span></div>",
            unsafe_allow_html=True,
        )
    with _collapse_col:
        _collapse_col.button(
            "",
            icon=":material/keyboard_double_arrow_left:"
            if not st.session_state.get("sidebar_compact", False)
            else ":material/keyboard_double_arrow_right:",
            key="sidebar_compact_toggle",
            on_click=_toggle_sidebar_compact,
            help="Thu gọn thanh điều hướng"
            if not st.session_state.get("sidebar_compact", False)
            else "Mở rộng thanh điều hướng",
            width="stretch",
        )

if not dataset_ready():
    st.warning("Chưa có dataset trong `data/raw/`. Xem hướng dẫn `data/DATASET_GUIDE.md` để tải.")
    st.stop()

# -- Role-based navigation -------------------------------------------------
# Authentication is outside the current project scope. The role stored in
# session state emulates the identity returned by an auth provider and
# strictly hides admin controls. It flips via the sidebar footer action
# ("Giao diện người dùng") and the portal's "Quản trị viên" entry — no
# segmented control is rendered in the sidebar.
st.session_state.setdefault("app_role", "Quản trị viên")
role = st.session_state["app_role"]
is_admin = role == "Quản trị viên"

# Order defines the sidebar groups styled by the shared theme
# (1 = TỔNG QUAN, 2..4 = RAG, 5..7 = HỆ THỐNG). "Đánh giá" is NOT here on
# purpose: it is level-2 navigation, the last step of RAG Pipeline tabs.
_ADMIN_PAGES = {
    ":material/space_dashboard: Tổng quan": "4.1 Dashboard",
    ":material/forum: Thử nghiệm Chat": "4.7 Chat với LLM (RAG)",
    ":material/description: Tài liệu": "4.2 Quản lý dữ liệu",
    ":material/account_tree: Quy trình RAG": "4.3 Phòng tiền xử lý",
    ":material/psychology: Mô hình": "4.9 Models",
    ":material/settings: Cấu hình": "4.10 Settings",
    ":material/receipt_long: Nhật ký": "4.11 Logs",
}
_PIPELINE_TABS = {
    "4.3 Phòng tiền xử lý": ":material/experiment: 1 · Tiền xử lý",
    "4.4 Chunking Lab": ":material/content_cut: 2 · Chunking",
    "4.5 Lập chỉ mục": ":material/storage: 3 · Lập chỉ mục",
    "4.6 Retrieval Playground": ":material/manage_search: 4 · Retrieval",
    "4.8 Đánh giá Retrieval (RQ1, RQ2)": ":material/analytics: 5 · Đánh giá",
}


def _set_admin_page() -> None:
    st.session_state["_active_page"] = _ADMIN_PAGES[st.session_state.admin_nav]


def _set_pipeline_tab() -> None:
    st.session_state["_active_page"] = st.session_state.pipeline_tab


if is_admin:
    active_page = st.session_state.get("_active_page", "4.7 Chat với LLM (RAG)")
    if active_page not in _ADMIN_PAGES.values() and active_page not in _PIPELINE_TABS:
        active_page = "4.7 Chat với LLM (RAG)"
    # Every pipeline step — including "5 · Đánh giá" (4.8) — highlights the
    # level-1 "Quy trình RAG" entry; the step itself shows in the pipeline tabs.
    if active_page in _PIPELINE_TABS:
        active_admin_label = ":material/account_tree: Quy trình RAG"
    else:
        active_admin_label = next(
            (label for label, target in _ADMIN_PAGES.items() if target == active_page),
            ":material/forum: Thử nghiệm Chat",
        )
    st.session_state["_active_page"] = active_page
    st.session_state["admin_nav"] = active_admin_label
    st.sidebar.radio(
        "Điều hướng quản trị",
        list(_ADMIN_PAGES),
        key="admin_nav",
        on_change=_set_admin_page,
        label_visibility="collapsed",
    )
    page = st.session_state["_active_page"]

    # -- Footer: identity + portal switch, pinned to the bottom of the rail --
    with st.sidebar:
        with st.container(key="admin_sidebar_footer"):
            st.caption("**Quản trị viên**")
            st.button(
                "Giao diện người dùng",
                icon=":material/arrow_back:",
                on_click=_set_user_role,
                help="Chuyển sang cổng thông tin người dùng",
                width="stretch",
            )
else:
    page = "5.1 Legal Portal"
    st.session_state["_active_page"] = page

if is_admin and page in _PIPELINE_TABS:
    st.caption(":material/account_tree: **RAG Pipeline** · Cấu hình, quan sát và đánh giá từng thành phần")
    st.session_state["pipeline_tab"] = page
    st.segmented_control(
        "Các bước RAG pipeline",
        list(_PIPELINE_TABS),
        format_func=lambda target: _PIPELINE_TABS[target],
        key="pipeline_tab",
        selection_mode="single",
        on_change=_set_pipeline_tab,
        width="stretch",
        label_visibility="collapsed",
    )
    st.caption(
        "Preprocessing  →  Text segmentation  →  Embedding & vector store  "
        "→  Search & ranking  →  Evaluation"
    )

# ---------------------------------------------------------------------------
# Screen 4.1 — Overview
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Đang xây dựng chunks...")
def get_built_chunks(
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    preprocessing_config_id: str | None = None,
):
    """Chunk corpus with the given configuration (cached)."""
    cfg = ChunkingConfig(strategy=strategy, chunk_size=chunk_size,
                         chunk_overlap=chunk_overlap)
    _, chunks, _, _ = get_corpus()
    if preprocessing_config_id:
        records = load_processed_records_by_id(preprocessing_config_id)
        chunks = [
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
    built, stats = build_chunks(chunks, cfg)
    return cfg, built, stats


@st.cache_resource(show_spinner="Đang build TF-IDF (preprocess corpus)...")
def get_tfidf_index(
    segmentation: str,
    chunk_artifact_id: str | None = None,
):
    """TF-IDF index over corpus preprocessed with the given segmentation."""
    import time

    from src.preprocessing.pipeline import run_pipeline as _rp

    _, chunks, dataset_version, _ = get_corpus()
    pcfg = PreprocessingConfig(word_segmentation=segmentation)
    texts, ids, metas = [], [], []
    t0 = time.time()
    if chunk_artifact_id:
        records = load_chunk_records(chunk_artifact_id)
        for record in records:
            texts.append(record["text"])
            ids.append(record["chunk_id"])
            metas.append(
                {
                    "law_id": record["law_id"],
                    "article_id": record["article_id"],
                }
            )
    elif artifact_is_current(pcfg, dataset_version):
        records = load_processed_records(pcfg)
        for record in records:
            texts.append(record["text"])
            ids.append(record["chunk_id"])
            metas.append(
                {
                    "law_id": record["law_id"],
                    "article_id": record["article_id"],
                }
            )
    else:
        for c in chunks:
            texts.append(_rp(c.text, pcfg).text)
            ids.append(c.chunk_id)
            metas.append({"law_id": c.law_id, "article_id": c.article_id})
    retriever = TfidfRetriever().fit(ids, texts, metas)
    index_id = chunk_artifact_id or pcfg.config_id()
    retriever.save(PROJECT_ROOT / "models" / f"tfidf_{index_id}.pkl")
    return retriever, round(time.time() - t0, 1)


def _set_admin_role() -> None:
    """Return from the public portal to the administrator interface."""
    st.session_state.app_role = "Quản trị viên"
    st.session_state._active_page = "4.7 Chat với LLM (RAG)"


def _queue_legal_question(question: str) -> None:
    st.session_state["legal_pending_question"] = question
    # Keep the assistant open across the app reruns that follow, so a
    # question queued from the portal (outside the dialog) still shows its
    # answer after the transcript refresh.
    st.session_state["legal_assistant_open"] = True


def _close_legal_assistant() -> None:
    """Dialog dismiss callback — drops the keep-open flag (X / ESC)."""
    st.session_state.pop("legal_assistant_open", None)


def _legal_rerun() -> None:
    """Refresh the assistant dialog, fragment-scoped when possible.

    ``scope="fragment"`` is only allowed while the dialog executes a
    fragment rerun (e.g. right after submitting a question inside the
    dialog). When the dialog renders during a full app rerun — such as when
    a portal suggestion button queues a question and the dialog opens in
    that same run — Streamlit raises ``StreamlitAPIException``, so fall back
    to a full rerun: the ``legal_assistant_open`` flag keeps the dialog on
    screen and the transcript shows the new messages.
    """
    try:
        st.rerun(scope="fragment")
    except StreamlitAPIException:
        st.rerun()


def _render_legal_chat_transcript() -> None:
    """Render the scrollable transcript without moving the input area."""
    if not st.session_state.legal_chat:
        st.subheader("Xin chào! Tôi là Trợ lý pháp lý AI.")
        st.write(
            "Tôi có thể hỗ trợ bạn tra cứu các quy định và văn bản pháp luật "
            "có trong hệ thống."
        )
        suggestions = [
            "Thời hiệu khởi kiện quyết định hành chính là bao lâu?",
            "Điều kiện khiếu nại quyết định hành chính là gì?",
            "Hồ sơ khởi kiện cần những tài liệu nào?",
        ]
        st.caption("Bạn có thể hỏi:")
        for index, suggestion in enumerate(suggestions):
            st.button(
                suggestion,
                key=f"legal_suggestion_{index}",
                on_click=_queue_legal_question,
                args=(suggestion,),
                width="stretch",
            )

    for message_index, message in enumerate(st.session_state.legal_chat):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                sources = message.get("sources", [])
                if sources:
                    with st.expander(
                        f":material/gavel: Nguồn tham khảo ({len(sources)})"
                    ):
                        for source in sources:
                            st.markdown(
                                f"**[{source['rank']}] Văn bản {source['law_id']}**  "
                                f"\nĐiều {source['article_id']}"
                            )
                            st.write(source["text"][:500])
                st.feedback("thumbs", key=f"legal_feedback_{message_index}")


def _toggle_legal_chat_expanded() -> None:
    """Toggle the chat panel between normal and expanded size."""
    st.session_state.legal_chat_expanded = not st.session_state.legal_chat_expanded


_LEGAL_THINKING_HTML = (
    '<div class="legal-thinking">'
    '<span class="legal-thinking-dot"></span>'
    '<span class="legal-thinking-dot"></span>'
    '<span class="legal-thinking-dot"></span>'
    '<span class="legal-thinking-label">{label}</span>'
    "</div>"
)


_LEGAL_COMPOSER_KEY_SHIM = """
<script>
if (!window.__legalChatComposerShim) {
  window.__legalChatComposerShim = true;

  function legalDialog() {
    return document.querySelector('[data-testid="stDialog"] [role="dialog"]');
  }
  function legalComposerTextarea() {
    var dialog = legalDialog();
    if (!dialog) return null;
    var form = dialog.querySelector(
      '.st-key-legal_chat_composer [data-testid="stForm"]'
    );
    return form ? form.querySelector('textarea') : null;
  }
  function legalTextareaMetrics(ta) {
    var cs = getComputedStyle(ta);
    var lineH = parseFloat(cs.lineHeight) || 20;
    var pad = (parseFloat(cs.paddingTop) || 0)
      + (parseFloat(cs.paddingBottom) || 0);
    return { lineH: lineH, pad: pad };
  }
  function legalUnclampRoot(ta) {
    var root = ta.closest('[data-testid="stTextAreaRootElement"]')
      || ta.parentElement;
    if (root && root !== ta) root.style.height = 'auto';
  }
  /* One row (default / after send). */
  function legalFitOneRow(ta) {
    if (!ta) return;
    var m = legalTextareaMetrics(ta);
    legalUnclampRoot(ta);
    ta.style.height = (m.lineH + m.pad) + 'px';
    ta.style.overflowY = 'hidden';
  }
  /* Auto-grow up to 4 lines, then scroll inside. Measuring starts from a
     one-row height: the textarea's intrinsic `rows` size would otherwise
     inflate scrollHeight when height is auto. */
  function legalAutoGrow(ta) {
    if (!ta) return;
    var m = legalTextareaMetrics(ta);
    var one = m.lineH + m.pad;
    var max = m.lineH * 4 + m.pad;
    legalUnclampRoot(ta);
    ta.style.height = one + 'px';
    var need = Math.max(ta.scrollHeight, one);
    ta.style.height = Math.min(need, max) + 'px';
    ta.style.overflowY = need > max ? 'auto' : 'hidden';
  }

  /* Enter sends, Shift+Enter inserts a newline. */
  document.addEventListener(
    'keydown',
    function (event) {
      if (event.key !== 'Enter' || event.shiftKey || event.ctrlKey
          || event.metaKey || event.altKey) return;
      var target = event.target;
      if (!target || target.tagName !== 'TEXTAREA') return;
      var dialog = legalDialog();
      if (!dialog || !dialog.contains(target)) return;
      var inComposer = dialog.querySelector(
        '.st-key-legal_chat_composer [data-testid="stForm"]'
      );
      if (!inComposer || !inComposer.contains(target)) return;
      var submit = dialog.querySelector(
        '[data-testid="stFormSubmitButton"] button'
      );
      if (submit && !submit.disabled) {
        event.preventDefault();
        submit.click();
      }
    },
    true
  );

  /* Typing grows the textarea. */
  document.addEventListener(
    'input',
    function (event) {
      var ta = event.target;
      if (!ta || ta.tagName !== 'TEXTAREA') return;
      var dialog = legalDialog();
      if (dialog && dialog.contains(ta)) legalAutoGrow(ta);
    },
    true
  );

  /* Streamlit re-renders the widget on rerun (height back to its own
     default): refit — one row when empty, grown when it has content. */
  new MutationObserver(function () {
    var ta = legalComposerTextarea();
    if (!ta) return;
    if (ta.value) legalAutoGrow(ta);
    else legalFitOneRow(ta);
  }).observe(document.body, { childList: true, subtree: true });
}
</script>
"""


@st.dialog("⚖ Trợ lý pháp lý AI", width="medium", on_dismiss=_close_legal_assistant)
def render_legal_assistant() -> None:
    """Public legal assistant using the existing grounded RAG backend."""
    st.session_state.setdefault("legal_chat", [])
    st.session_state.setdefault("legal_chat_expanded", False)

    # -- Panel size: expand / collapse (injected style + smooth transition) --
    if st.session_state.legal_chat_expanded:
        st.markdown(
            "<style>"
            "@media (min-width:768px){[data-testid='stDialog'] [role='dialog']{"
            "width:min(720px,calc(100vw - 48px))!important;"
            "height:min(820px,calc(100dvh - 48px))!important;}}"
            "@media (max-width:767px){[data-testid='stDialog'] [role='dialog']{"
            "top:12px!important;bottom:calc(12px + env(safe-area-inset-bottom))!important;"
            "left:12px!important;right:12px!important;width:auto!important;"
            "height:calc(100dvh - 24px - env(safe-area-inset-bottom))!important;}}"
            "</style>",
            unsafe_allow_html=True,
        )
    st.button(
        "Thu nhỏ" if st.session_state.legal_chat_expanded else "Mở rộng",
        icon=":material/close_fullscreen:"
        if st.session_state.legal_chat_expanded
        else ":material/open_in_full:",
        key="legal_chat_expand",
        help="Thu nhỏ khung chat" if st.session_state.legal_chat_expanded
        else "Mở rộng khung chat",
        on_click=_toggle_legal_chat_expanded,
        type="tertiary",
    )

    # -- Queue a newly typed question, then rerun so the user bubble and the
    #    thinking indicator render above the composer during processing --
    pending_question = st.session_state.pop("legal_pending_question", None)
    if pending_question:
        st.session_state.legal_chat.append(
            {"role": "user", "content": pending_question}
        )

    with st.container(key="legal_chat_subtitle"):
        st.caption("Hỗ trợ tra cứu thông tin pháp luật trong kho dữ liệu của hệ thống")

    # The only scrollable region of the panel (flex-grow, autoscroll).
    with st.container(
        height=300, border=False, key="legal_chat_transcript", autoscroll=True
    ):
        _render_legal_chat_transcript()

    # Placeholder rendered between the messages and the composer: the thinking
    # indicator fills it while the RAG pipeline runs, i.e. right ABOVE the
    # composer, never below it.
    thinking_slot = st.container(key="legal_chat_thinking")

    # -- Composer: one rounded box with textarea + bottom-right actions -----
    st.html(_LEGAL_COMPOSER_KEY_SHIM, unsafe_allow_javascript=True)
    with st.container(key="legal_chat_composer"):
        with st.form(
            "legal_assistant_form",
            border=False,
            enter_to_submit=True,
            clear_on_submit=True,
        ):
            typed_question = st.text_area(
                "Câu hỏi pháp luật",
                placeholder="Nhập câu hỏi pháp luật...",
                label_visibility="collapsed",
                height=68,
            )
            submitted = st.form_submit_button(
                "Gửi",
                icon=":material/arrow_upward:",
                type="primary",
                help="Gửi câu hỏi (Enter)",
            )
        with st.container(
            horizontal=True,
            horizontal_alignment="right",
            key="legal_chat_actions",
            gap="xsmall",
        ):
            with st.popover(
                "Lịch sử",
                icon=":material/history:",
                help="Lịch sử trò chuyện",
                use_container_width=False,
            ):
                user_questions = [
                    message["content"]
                    for message in st.session_state.legal_chat
                    if message["role"] == "user"
                ]
                if user_questions:
                    st.caption("**Hôm nay**")
                    for historic_question in reversed(user_questions):
                        st.write(f"• {historic_question[:80]}")
                else:
                    st.caption("Chưa có câu hỏi nào trong phiên hiện tại.")
            st.button(
                "Mới",
                icon=":material/add_comment:",
                key="legal_new_chat",
                help="Cuộc trò chuyện mới",
                on_click=lambda: st.session_state.update(legal_chat=[]),
            )
    st.caption(
        "Thông tin chỉ mang tính tham khảo, không thay thế tư vấn pháp lý "
        "chuyên nghiệp.",
    )

    if submitted and typed_question.strip():
        st.session_state["legal_pending_question"] = typed_question.strip()
        _legal_rerun()
    question = pending_question

    if question:
        try:
            phase = thinking_slot.empty()
            phase.markdown(
                _LEGAL_THINKING_HTML.format(
                    label="Đang tra cứu cơ sở pháp luật..."
                ),
                unsafe_allow_html=True,
            )
            # The public assistant runs entirely on the admin-managed active
            # configuration (src/system_config.py): end users cannot override
            # retriever, top_k, segmentation or any LLM parameter.
            portal_cfg = get_active_config()
            portal_pcfg = PreprocessingConfig(
                word_segmentation=portal_cfg.word_segmentation
            )
            processed_question = run_pipeline(question, portal_pcfg).text
            if portal_cfg.retriever == "dense":
                dr = DenseRetriever(model_name=DEFAULT_MODEL)
                col_name = dr.collection_name(
                    portal_pcfg.config_id(),
                    ChunkingConfig(strategy="article").config_id(),
                )
                if col_name not in dr.list_collections():
                    raise RuntimeError(
                        "Nguồn truy xuất chưa sẵn sàng, vui lòng thử lại sau."
                    )
                dr.use_collection(col_name)
                results = dr.search(
                    processed_question, top_k=int(portal_cfg.top_k)
                )
            else:
                retriever, _ = get_tfidf_index(portal_cfg.word_segmentation)
                results = retriever.search(
                    processed_question, top_k=int(portal_cfg.top_k)
                )
            if not results:
                raise ValueError("Không tìm thấy nguồn pháp luật phù hợp")
            phase.markdown(
                _LEGAL_THINKING_HTML.format(
                    label="Đang tổng hợp câu trả lời..."
                ),
                unsafe_allow_html=True,
            )
            available_models = list_ollama_models() if ollama_available() else []
            if not available_models or portal_cfg.llm_model not in available_models:
                raise ConnectionError("Trợ lý AI hiện chưa sẵn sàng")
            response = generate_answer_stream(
                question,
                results,
                model=portal_cfg.llm_model,
                temperature=float(portal_cfg.temperature),
                max_tokens=int(portal_cfg.max_tokens),
            )
            thinking_slot.empty()
            st.session_state.legal_chat.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": [
                        {
                            "rank": result.rank,
                            "law_id": result.law_id,
                            "article_id": result.article_id,
                            "text": result.text,
                        }
                        for result in results
                    ],
                }
            )
            _legal_rerun()
        except Exception as error:
            thinking_slot.empty()
            st.session_state.legal_chat.append(
                {
                    "role": "assistant",
                    "content": f"Hiện chưa thể hoàn tất yêu cầu: {error}",
                    "sources": [],
                }
            )
            _legal_rerun()


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------
if page.startswith("4.1 "):
    _dashboard_actions = admin_page_header(
        "Tổng quan",
        "Tổng quan dữ liệu và trạng thái các thành phần RAG từ nguồn thực tế.",
    )
    with _dashboard_actions:
        st.button(
            "Trang người dùng",
            icon=":material/arrow_back:",
            on_click=_set_user_role,
            help="Về cổng thông tin người dùng",
        )
    docs, chunks, version, report = get_corpus()
    questions = get_questions()

    with st.container(horizontal=True):
        st.metric("Documents", f"{len(docs):,}", border=True)
        st.metric("Chunks", f"{len(chunks):,}", border=True)
        st.metric("Labeled queries", f"{len(questions):,}", border=True)
        st.metric(
            "Valid articles",
            f"{report['valid_articles']:,}",
            border=True,
        )

    chart_data = pd.DataFrame(
        {
            "Thành phần": ["Documents", "Articles", "Questions"],
            "Số lượng": [len(docs), len(chunks), len(questions)],
        }
    )
    left, right = st.columns(2)
    with left.container(border=True, key="card_dashboard_scale"):
        st.subheader("Quy mô dữ liệu")
        st.bar_chart(chart_data, x="Thành phần", y="Số lượng")
    with right.container(border=True, key="card_dashboard_quality"):
        st.subheader("Data quality")
        st.metric("Empty articles removed", report["empty_articles"])
        st.metric("Duplicate articles removed", report["duplicate_articles"])
        st.caption("Các số liệu được tính trong quá trình nạp corpus, không phải mock data.")

    st.subheader("Trạng thái")
    st.write(f"- **Dataset version:** `{version}`")
    split_path = RAW_DIR.parent / "processed" / SPLIT_FILENAME
    if split_path.exists():
        with open(split_path, encoding="utf-8") as f:
            split = json.load(f)
        st.write("- **Split câu hỏi:** " + ", ".join(
            f"{k} = {len(v)}" for k, v in split["split_ids"].items()
        ) + f" (seed = {split['seed']})")
    else:
        if st.button("Tạo split train/dev/test (70/10/20, seed=42)"):
            split_questions(questions)
            st.rerun()

    st.write("- **Embedding / index:** Chưa xây dựng (tuần 2)")
    st.write("- **Kết quả thực nghiệm:** Chưa có kết quả thực nghiệm")

    with st.expander("Báo cáo kiểm tra dữ liệu"):
        st.json(report)

# ---------------------------------------------------------------------------
# Screen 4.2 — Data management
# ---------------------------------------------------------------------------
elif page.startswith("4.2 "):
    admin_page_header(
        "Tài liệu",
        "Quản lý corpus văn bản pháp luật và bổ sung văn bản mới vào chỉ mục.",
    )
    docs, chunks, version, report = get_corpus()

    with st.container(border=True, key="card_doc_check"):
        st.subheader("Kiểm tra dữ liệu")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Văn bản hợp lệ", report["valid_laws"], border=True)
        c2.metric("Article hợp lệ", report["valid_articles"], border=True)
        c3.metric("Article rỗng (loại)", report["empty_articles"], border=True)
        c4.metric("Article trùng (loại)", report["duplicate_articles"], border=True)

        if report["errors"]:
            with st.expander(f"Lỗi nạp ({len(report['errors'])})"):
                st.write(report["errors"][:20])

    with st.container(border=True, key="card_doc_preview"):
        st.subheader("Xem trước corpus")
        option = st.selectbox(
            "Chọn văn bản",
            range(len(docs)),
            format_func=lambda i: docs[i].title,
        )
        doc = docs[option]
        st.markdown(f"**document_id:** `{doc.document_id}` · **số article:** {doc.metadata['n_articles']}")
        st.text_area("Nội dung", doc.raw_text, height=300)

    with st.container(border=True, key="card_doc_questions"):
        st.subheader("Câu hỏi + nhãn relevance (mẫu)")
        questions = get_questions()
        qdf = pd.DataFrame(questions).head(20)
        st.dataframe(qdf, width="stretch")

    # -- Add new law incrementally (dense) + full refit (TF-IDF) ------------
    with st.container(border=True, key="card_doc_add"):
        st.subheader(":material/note_add: Thêm văn bản mới (incremental)")
        st.caption(
            "Dán nội dung văn bản dưới dạng JSON articles. Dense index: chỉ upsert "
            "các article mới (không rebuild). TF-IDF: cần fit lại toàn bộ (~1 phút) "
            "vì IDF thay đổi khi corpus thay đổi."
        )
        new_law_id = st.text_input(
            "law_id mới (ví dụ: 05/2025/qh15)", key="new_law_id"
        )
        new_articles_json = st.text_area(
            'Articles JSON — [{"article_id": "...", "title": "...", "text": "..."}]',
            height=180,
            key="new_articles_json",
        )

        def _parse_new_articles(raw: str) -> list[dict]:
            arts = json.loads(raw)
            if not isinstance(arts, list):
                raise ValueError("JSON phải là một list các article")
            out = []
            for a in arts:
                aid = (a.get("article_id") or "").strip()
                text = (a.get("text") or "").strip()
                title = (a.get("title") or "").strip()
                if aid and text:
                    out.append({"article_id": aid, "title": title, "text": text})
            return out

        if st.button(":material/add_circle: Thêm vào corpus & cập nhật index", type="primary",
                     disabled=not (new_law_id.strip() and new_articles_json.strip())):
            try:
                articles = _parse_new_articles(new_articles_json)
                if not articles:
                    st.warning("Không có article hợp lệ (cần article_id + text).")
                    st.stop()
                law_id = new_law_id.strip()

                # 1) Ghi đè vào corpus JSON (raw)
                corpus_path = RAW_DIR / CORPUS_FILENAME
                with open(corpus_path, encoding="utf-8") as f:
                    data = json.load(f)
                existing_ids = {law.get("law_id") for law in data}
                record = {"law_id": law_id, "articles": articles}
                if law_id in existing_ids:
                    data = [
                        record if law.get("law_id") == law_id else law for law in data
                    ]
                    action = "cập nhật"
                else:
                    data.append(record)
                    action = "thêm mới"
                with open(corpus_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                st.info(f"Đã {action} văn bản `{law_id}` ({len(articles)} article) vào corpus.")

                # 2) Dense: upsert incremental
                from src.preprocessing.pipeline import run_pipeline as _rp2

                pcfg = PreprocessingConfig(word_segmentation="none")
                dr = DenseRetriever(model_name=DEFAULT_MODEL)
                col_name = dr.collection_name(
                    pcfg.config_id(), ChunkingConfig(strategy="article").config_id()
                )
                if col_name in dr.list_collections():
                    ids = [f"{law_id}::{a['article_id']}" for a in articles]
                    texts = [
                        _rp2(f"{a['title']}\n{a['text']}" if a["title"] else a["text"],
                             pcfg).text
                        for a in articles
                    ]
                    metas = [{"law_id": law_id, "article_id": a["article_id"]}
                             for a in articles]
                    info = dr.add_documents(ids, texts, metas, col_name)
                    st.success(
                        f"Dense index (incremental): collection `{info.collection_name}` "
                        f"→ {info.n_vectors:,} vector."
                    )
                else:
                    st.warning(
                        f"Chưa có collection `{col_name}` — hãy xây dense index ở "
                        "màn 4.5 trước, hoặc bỏ qua bước này."
                    )

                # 3) TF-IDF: cần fit lại toàn bộ (IDF thay đổi)
                st.info("Đang fit lại TF-IDF toàn bộ (tách từ: none)...")
                get_tfidf_index.clear()
                get_tfidf_index("none")
                st.success("TF-IDF index đã được fit lại với corpus mới.")

                get_corpus.clear()
                st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"JSON không hợp lệ: {e}")
            except Exception as e:
                st.error(f"Lỗi: {e}")

# ---------------------------------------------------------------------------
# Screen 4.3 — Preprocessing lab
# ---------------------------------------------------------------------------
elif page.startswith("4.3 "):
    admin_page_header(
        "Tiền xử lý dữ liệu",
        "Preview trên đoạn mẫu hoặc materialize toàn bộ Zalo AI 2021 corpus "
        "thành artifact có thể tái lập.",
    )

    # -- Configuration lives in the main content: the sidebar is for
    #    navigation only (shared keys keep values across page switches).
    config_col, main_col = st.columns(
        [0.32, 0.68], gap="medium", vertical_alignment="top"
    )
    with config_col.container(border=True, key="card_prep_config"):
        st.subheader(":material/tune: Cấu hình")
        cfg = PreprocessingConfig(
            unicode_normalization=st.selectbox(
                "Unicode normalization",
                ["NFC", "none"],
                format_func=lambda value: (
                    "NFC — Khuyến nghị" if value == "NFC" else "Không chuẩn hóa"
                ),
                help="NFC giúp biểu diễn Unicode tiếng Việt nhất quán.",
                key="prep_unicode",
            ),
            whitespace_normalization=st.checkbox(
                "Whitespace normalization", value=True, key="prep_whitespace"
            ),
            remove_noise_chars=st.checkbox(
                "Loại ký tự nhiễu", value=True, key="prep_noise"
            ),
            lowercase=st.checkbox(
                "Chuẩn hóa chữ thường", value=False,
                help="Không khuyến nghị cho dense model",
                key="prep_lowercase",
            ),
            word_segmentation=st.selectbox(
                "Tách từ tiếng Việt",
                ["none", "underthesea", "pyvi"],
                format_func=lambda value: {
                    "none": "None — Khuyến nghị cho dense/baseline",
                    "underthesea": "Underthesea — Ưu tiên thử nghiệm TF-IDF",
                    "pyvi": "PyVi — Phương án đối chứng RQ1",
                }[value],
                help=(
                    "Không có phương pháp tốt nhất trước đánh giá. Hãy materialize từng "
                    "cấu hình và so sánh Recall@K/MRR."
                ),
                key="prep_segmentation",
            ),
            remove_stopwords=st.checkbox(
                "Bỏ stop-word", value=False,
                help="Cảnh báo: có thể làm mất nghĩa pháp lý",
                key="prep_stopwords",
            ),
        )
        st.caption(f"config_id: `{cfg.config_id()}`")

    with main_col:

        tab_sample, tab_dataset, tab_compare = st.tabs(
            ["Preview", "Xử lý toàn bộ dataset", "So sánh tách từ"],
            on_change="rerun",
        )

        # -- Tab 1: run pipeline on a sample, show step-by-step diff
        with tab_sample:
            docs, chunks, _, _ = get_corpus()
            default_text = chunks[0].text[:800]
            text = st.text_area("Văn bản mẫu", value=default_text, height=200)

            if st.button(":material/play_arrow: Chạy pipeline", type="primary"):
                stopwords = get_stopwords() if cfg.remove_stopwords else set()
                result = run_pipeline(text, cfg, stopwords)

                st.subheader("Kết quả từng bước")
                rows = [
                    {
                        "Bước": s.step,
                        "Ký tự trước": s.chars_before,
                        "Ký tự sau": s.chars_after,
                        "Thay đổi": "✓" if s.changed else "—",
                    }
                    for s in result.steps
                ]
                st.table(pd.DataFrame(rows))

                st.subheader("So sánh trước / sau")
                col_a, col_b = st.columns(2)
                col_a.text_area("Trước", text, height=250)
                col_b.text_area("Sau", result.text, height=250)

                st.subheader("Thống kê")
                s_before, s_after = compute_stats(text), compute_stats(result.text)
                st.write(pd.DataFrame({"Trước": s_before, "Sau": s_after}))

                st.subheader("Tách câu")
                for i, sent in enumerate(sentence_segment(text), 1):
                    st.write(f"{i}. {sent}")

        # -- Tab 2: materialize the full valid corpus as a versioned artifact
        if tab_dataset.open:
            with tab_dataset:
                docs, chunks, dataset_version, _ = get_corpus()
                artifact_path, manifest_path = artifact_paths(cfg)
                manifest = load_manifest(cfg)
                current_artifact = artifact_is_current(cfg, dataset_version)

                st.subheader("Zalo AI 2021 — Legal Text Retrieval")
                with st.container(horizontal=True):
                    st.metric("Documents", f"{len(docs):,}", border=True)
                    st.metric("Valid articles", f"{len(chunks):,}", border=True)
                    st.metric("Dataset version", dataset_version, border=True)
                    st.metric(
                        "Artifact",
                        "Sẵn sàng" if current_artifact else "Chưa có / hết hạn",
                        border=True,
                    )

                if current_artifact and manifest:
                    st.success(
                        f"Artifact `{artifact_path.name}` phù hợp với dataset và "
                        f"config `{cfg.config_id()}`.",
                        icon=":material/check_circle:",
                    )
                    with st.container(horizontal=True):
                        st.metric("Articles processed", f"{manifest['output_articles']:,}")
                        st.metric("Articles changed", f"{manifest['changed_articles']:,}")
                        st.metric("Tokens before", f"{manifest['tokens_before']:,}")
                        st.metric("Tokens after", f"{manifest['tokens_after']:,}")
                    st.download_button(
                        "Tải manifest",
                        data=json.dumps(manifest, ensure_ascii=False, indent=2),
                        file_name=manifest_path.name,
                        mime="application/json",
                        icon=":material/download:",
                    )
                else:
                    st.info(
                        "Chưa có artifact phù hợp. Dữ liệu raw sẽ chỉ được đọc; kết quả "
                        "được ghi riêng vào `data/processed/`.",
                        icon=":material/info:",
                    )

                force_reprocess = st.checkbox(
                    "Force reprocess (ghi đè artifact cùng config_id)",
                    value=False,
                    help="Chỉ bật khi cần tạo lại artifact dù dataset và cấu hình không đổi.",
                )
                run_dataset = st.button(
                    "Xử lý toàn bộ dataset",
                    icon=":material/play_arrow:",
                    type="primary",
                    disabled=current_artifact and not force_reprocess,
                )
                if run_dataset:
                    progress = st.progress(0.0, text="Đang chuẩn bị corpus...")

                    def _update_preprocessing_progress(done: int, total: int) -> None:
                        ratio = done / max(total, 1)
                        progress.progress(
                            ratio,
                            text=f"Đang tiền xử lý article {done:,}/{total:,} ({ratio:.0%})",
                        )

                    try:
                        with st.status(
                            "Đang tiền xử lý toàn bộ dataset...", expanded=True
                        ) as status:
                            st.write(f"Dataset version: `{dataset_version}`")
                            st.write(f"Preprocessing config: `{cfg.config_id()}`")
                            result_manifest = process_dataset(
                                cfg,
                                force=force_reprocess,
                                progress_callback=_update_preprocessing_progress,
                            )
                            status.update(
                                label="Tiền xử lý dataset hoàn tất",
                                state="complete",
                                expanded=False,
                            )
                        get_tfidf_index.clear()
                        st.success(
                            f"Đã lưu {result_manifest['output_articles']:,} article vào "
                            f"`{Path(result_manifest['artifact_path']).name}`."
                        )
                        st.rerun()
                    except Exception as error:
                        progress.empty()
                        st.error(f"Tiền xử lý dataset thất bại: {error}")

        # -- Tab 3: compare segmentation methods side by side
        if tab_compare.open:
            with tab_compare:
                st.caption(
                    "Cùng một văn bản, so sánh `none` / `underthesea` / `pyvi` (RQ1)"
                )
                docs, chunks, _, _ = get_corpus()
                sample_text = st.text_area(
                    "Văn bản so sánh", value=chunks[1].text[:400], height=150
                )
                if st.button("So sánh", type="primary"):
                    cols = st.columns(3)
                    for col, method in zip(cols, ["none", "underthesea", "pyvi"]):
                        with col:
                            st.markdown(f"**`{method}`**")
                            try:
                                out = segment_words(sample_text, method)
                                st.code(out, language=None)
                                st.caption(f"{len(out.split())} token")
                            except Exception as e:  # tool may not be installed yet
                                st.error(f"Lỗi: {e}")

# ---------------------------------------------------------------------------
# Screen 4.4 — Chunking Lab
# ---------------------------------------------------------------------------
# NOTE: get_built_chunks / get_tfidf_index are the @st.cache_resource versions
# defined above — do not redefine them here (a previous duplicate un-decorated
# copy silently disabled caching and re-fit TF-IDF on every query).

if page.startswith("4.4 "):
    admin_page_header(
        "Chunking Lab",
        "Cấu hình chiến lược chunking và materialize artifact cho bước lập chỉ mục.",
    )
    # -- Configuration lives in the main content: the sidebar is for
    #    navigation only (shared keys keep values across page switches).
    config_col, main_col = st.columns(
        [0.32, 0.68], gap="medium", vertical_alignment="top"
    )
    _, _, chunking_dataset_version, _ = get_corpus()
    preprocessing_manifests = [
        manifest
        for manifest in list_manifests()
        if manifest.get("dataset_version") == chunking_dataset_version
    ]
    manifests_by_id = {
        manifest["config_id"]: manifest for manifest in preprocessing_manifests
    }
    preprocessing_source_options = list(manifests_by_id) + ["raw"]
    with config_col.container(border=True, key="card_chunk_config"):
        st.subheader(":material/tune: Cấu hình chunking")
        preprocessing_source = st.selectbox(
            "Nguồn đầu vào",
            preprocessing_source_options,
            format_func=lambda source: (
                "Raw corpus — Chỉ dùng khi chưa có artifact"
                if source == "raw"
                else (
                    f"Processed — Khuyến nghị · "
                    f"{manifests_by_id[source]['preprocessing_config'].get('word_segmentation')} "
                    f"· {source}"
                )
            ),
            help="Ưu tiên artifact đã tạo ở bước Tiền xử lý để bảo đảm tái lập.",
            key="chunk_source",
        )
        strategy = st.selectbox(
            "Chiến lược",
            ["article", "fixed"],
            format_func=lambda value: {
                "article": "Article — Khuyến nghị cho văn bản pháp luật",
                "fixed": "Fixed size — Dùng để thực nghiệm chunk size",
            }[value],
            help="Article giữ nguyên ranh giới điều luật; fixed phù hợp nghiên cứu RQ về chunking.",
            key="chunk_strategy",
        )
        chunk_size = st.number_input(
            "chunk_size (token)", 64, 1024, 256, 64, key="chunk_size"
        )
        chunk_overlap = st.number_input(
            "chunk_overlap (token)", 0, chunk_size // 2, 0, key="chunk_overlap"
        )
        cfg, built, stats = get_built_chunks(
            strategy,
            chunk_size,
            chunk_overlap,
            None if preprocessing_source == "raw" else preprocessing_source,
        )
        st.caption(f"chunking config_id: `{cfg.config_id()}`")

    with main_col:

        if preprocessing_source == "raw":
            st.warning(
                "Chunking đang đọc raw corpus. Hãy tạo và chọn preprocessing artifact "
                "nếu cần một pipeline có thể tái lập.",
                icon=":material/warning:",
            )
        else:
            st.success(
                f"Đầu vào: preprocessing artifact `{preprocessing_source}`.",
                icon=":material/check_circle:",
            )

        source_id = preprocessing_source
        saved_chunk_manifest = load_chunk_manifest(source_id, cfg)
        current_chunk_artifact = chunk_artifact_is_current(
            source_id, cfg, chunking_dataset_version
        )
        chunk_data_path, chunk_manifest_path = chunk_artifact_paths(source_id, cfg)

        with st.container(border=True, key="card_chunk_generate"):
            st.subheader("Generate & Save Chunks")
            st.caption(
                "Materialize toàn bộ kết quả chunking để Indexing sử dụng trực tiếp. "
                "Raw corpus và preprocessing artifact không bị thay đổi."
            )
            if current_chunk_artifact and saved_chunk_manifest:
                st.success(
                    f"Artifact `{chunk_data_path.name}` đã sẵn sàng.",
                    icon=":material/check_circle:",
                )
                with st.container(horizontal=True):
                    st.metric(
                        "Input articles",
                        f"{saved_chunk_manifest['input_articles']:,}",
                    )
                    st.metric(
                        "Output chunks",
                        f"{saved_chunk_manifest['output_chunks']:,}",
                    )
                    st.metric("Average size", saved_chunk_manifest["length_mean"])
                    st.metric(
                        "Min / Max",
                        f"{saved_chunk_manifest['length_min']} / "
                        f"{saved_chunk_manifest['length_max']}",
                    )
                st.download_button(
                    "Tải chunking manifest",
                    data=json.dumps(saved_chunk_manifest, ensure_ascii=False, indent=2),
                    file_name=chunk_manifest_path.name,
                    mime="application/json",
                    icon=":material/download:",
                )
            else:
                st.info(
                    "Chưa có chunk artifact phù hợp với dataset, nguồn đầu vào và "
                    "cấu hình hiện tại.",
                    icon=":material/info:",
                )

            force_chunking = st.checkbox(
                "Force regenerate (ghi đè artifact cùng cấu hình)",
                value=False,
                help="Chỉ bật khi cần tạo lại artifact dù đầu vào và cấu hình không đổi.",
            )
            save_chunks = st.button(
                "Generate & Save Chunks",
                icon=":material/save:",
                type="primary",
                disabled=current_chunk_artifact and not force_chunking,
            )
            if save_chunks:
                chunk_progress = st.progress(0.0, text="Đang chuẩn bị chunks...")

                def _update_chunk_progress(done: int, total: int) -> None:
                    ratio = done / max(total, 1)
                    chunk_progress.progress(
                        ratio,
                        text=f"Đang lưu chunk {done:,}/{total:,} ({ratio:.0%})",
                    )

                try:
                    with st.status("Đang tạo chunk artifact...", expanded=True) as status:
                        st.write(f"Source: `{source_id}`")
                        st.write(f"Chunking config: `{cfg.config_id()}`")
                        result_manifest = generate_and_save_chunks(
                            source_id,
                            cfg,
                            force=force_chunking,
                            progress_callback=_update_chunk_progress,
                        )
                        status.update(
                            label="Chunk artifact đã được tạo",
                            state="complete",
                            expanded=False,
                        )
                    get_tfidf_index.clear()
                    st.success(
                        f"Đã lưu {result_manifest['output_chunks']:,} chunks vào "
                        f"`{Path(result_manifest['artifact_path']).name}`."
                    )
                    st.rerun()
                except Exception as error:
                    chunk_progress.empty()
                    st.error(f"Không thể tạo chunk artifact: {error}")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Số chunk", f"{stats.n_chunks:,}")
        c2.metric("Độ dài TB", stats.length_mean)
        c3.metric("Min / Max", f"{stats.n_chunks and stats.length_min} / {stats.length_max}")
        c4.metric("Vượt giới hạn 256", stats.n_over_limit)
        c5.metric("Chiến lược", strategy)

        st.subheader("Phân bố độ dài chunk")
        st.bar_chart({"token length": stats.length_distribution})

        st.subheader("Xem chunk")
        idx = st.number_input("chunk #", 0, max(stats.n_chunks - 1, 0), 0)
        ch = built[int(idx)]
        st.markdown(
            f"**chunk_id:** `{ch.chunk_id}` · **law/article:** "
            f"`{ch.law_id}` / `{ch.article_id}`"
        )
        st.text_area("Nội dung", ch.text, height=250)

# ---------------------------------------------------------------------------
# Screen 4.5 — Indexing & embedding
# ---------------------------------------------------------------------------
elif page.startswith("4.5 "):
    admin_page_header(
        "Lập chỉ mục và embedding",
        "Xây dựng TF-IDF baseline và dense index (Sentence-BERT + ChromaDB).",
    )

    _, _, index_dataset_version, _ = get_corpus()
    available_chunk_manifests = [
        manifest
        for manifest in list_chunk_manifests()
        if manifest.get("dataset_version") == index_dataset_version
    ]
    chunk_manifests_by_id = {
        manifest["artifact_id"]: manifest for manifest in available_chunk_manifests
    }
    selected_chunk_artifact = st.selectbox(
        "Chunk artifact",
        list(chunk_manifests_by_id),
        index=0 if chunk_manifests_by_id else None,
        placeholder="Hãy Generate & Save Chunks ở bước Chunking",
        format_func=lambda artifact_id: (
            f"{artifact_id} — "
            f"{chunk_manifests_by_id[artifact_id]['output_chunks']:,} chunks"
        ),
        help="Indexing chỉ sử dụng chunk artifact đã materialize để bảo đảm tái lập.",
    )
    selected_chunk_manifest = (
        chunk_manifests_by_id.get(selected_chunk_artifact)
        if selected_chunk_artifact
        else None
    )
    seg = (
        selected_chunk_manifest.get("preprocessing_config", {}).get(
            "word_segmentation", "none"
        )
        if selected_chunk_manifest
        else "none"
    )
    if selected_chunk_manifest:
        st.success(
            f"Indexing sẽ đọc trực tiếp `{selected_chunk_artifact}` · "
            f"preprocessing `{selected_chunk_manifest['source_id']}` · "
            f"chunking `{selected_chunk_manifest['chunking_config_id']}`.",
            icon=":material/recycling:",
        )
    else:
        st.warning(
            "Chưa có chunk artifact cho dataset hiện tại. Quay lại bước Chunking "
            "và chọn Generate & Save Chunks.",
            icon=":material/warning:",
        )

    st.subheader("1) Baseline: TF-IDF")
    st.caption(f"Word segmentation kế thừa từ artifact: `{seg}`.")
    if st.button(
        "Xây TF-IDF index",
        key="btn_tfidf",
        disabled=not selected_chunk_artifact,
    ):
        with st.spinner("Đang xây TF-IDF..."):
            try:
                retriever, elapsed = get_tfidf_index(seg, selected_chunk_artifact)
                st.success(
                    f"TF-IDF xong: {retriever.matrix.shape[0]:,} chunk × "
                    f"{retriever.matrix.shape[1]:,} term trong {elapsed}s "
                    f"(tách từ: {seg})"
                )
            except Exception as e:
                st.error(f"Lỗi: {e}")

    st.divider()
    st.subheader("2) Dense: Sentence-BERT + ChromaDB")
    model_name = st.text_input("Model", value=DEFAULT_MODEL)
    device = st.selectbox(
        "Device",
        ["auto", "cpu", "cuda"],
        format_func=lambda value: {
            "auto": "Auto — Khuyến nghị",
            "cpu": "CPU — Tương thích cao, chậm hơn",
            "cuda": "CUDA — Ưu tiên khi GPU khả dụng",
        }[value],
    )
    batch_size = st.number_input("Batch size", 16, 256, 64, 16)

    if st.button(
        "Xây dense index",
        key="btn_dense",
        type="primary",
        disabled=not selected_chunk_artifact,
    ):
        try:
            assert selected_chunk_manifest is not None
            dr = DenseRetriever(model_name=model_name,
                                device=None if device == "auto" else device)
            col_name = dr.collection_name(
                selected_chunk_manifest["source_id"],
                selected_chunk_manifest["chunking_config_id"],
            )
            progress = st.progress(0.0, text="Đang embed corpus...")
            chunk_records = load_chunk_records(selected_chunk_artifact)
            chunk_ids = [record["chunk_id"] for record in chunk_records]
            texts = [record["text"] for record in chunk_records]
            metas = [
                {
                    "law_id": record["law_id"],
                    "article_id": record["article_id"],
                }
                for record in chunk_records
            ]
            info = dr.build_index(
                chunk_ids, texts, metas, col_name,
                batch_size=int(batch_size),
                progress_callback=lambda p: progress.progress(
                    p, text=f"Đang embed corpus... {p:.0%}"
                ),
            )
            st.success(
                f"Xong: collection `{info.collection_name}` — "
                f"{info.n_vectors:,} vector, dim = {info.dimension}"
            )
        except Exception as e:
            st.error(f"Lỗi: {e}")

    st.divider()
    st.subheader("Collections hiện có")
    try:
        dr = DenseRetriever()
        cols = dr.list_collections()
        if cols:
            for c in cols:
                st.write(f"- `{c}` ({dr.client.get_collection(c).count():,} vector)")
        else:
            st.info("Chưa có collection nào.")
    except Exception as e:
        st.warning(f"ChromaDB chưa sẵn sàng: {e}")

# ---------------------------------------------------------------------------
# Screen 4.6 — Retrieval Playground
# ---------------------------------------------------------------------------
elif page.startswith("4.6 "):
    admin_page_header(
        "Retrieval Playground",
        "So sánh trực tiếp TF-IDF baseline và Sentence-BERT trên cùng một truy vấn.",
    )
    query = st.text_input(
        "Câu hỏi", value="Thời hiệu khởi kiện khiếu quyết định hành chính là bao lâu?"
    )
    top_k = st.selectbox(
        "top_k",
        TOP_K_CHOICES,
        index=TOP_K_CHOICES.index(5),
        format_func=lambda value: f"{value} — Khuyến nghị cân bằng" if value == 5 else str(value),
    )
    seg = st.selectbox(
        "Tách từ",
        ["none", "underthesea", "pyvi"],
        key="playground_seg",
        format_func=lambda value: {
            "none": "None — Khuyến nghị mặc định",
            "underthesea": "Underthesea — Dùng với index tương ứng",
            "pyvi": "PyVi — Dùng với index tương ứng",
        }[value],
    )

    from src.preprocessing.pipeline import run_pipeline as _rp

    pcfg = PreprocessingConfig(word_segmentation=seg)

    col_tfidf, col_dense = st.columns(2)

    # -- TF-IDF branch
    with col_tfidf:
        st.subheader("TF-IDF + Cosine (baseline)")
        if st.button("Tìm kiếm", key="btn_search_tfidf"):
            try:
                retriever, _ = get_tfidf_index(seg)
                q = _rp(query, pcfg).text
                results = retriever.search(q, top_k=int(top_k))
                if not results:
                    st.info("Không có kết quả (score = 0).")
                for r in results:
                    with st.expander(
                        f"#{r.rank} · score = {r.score:.4f} · `{r.chunk_id}`"
                    ):
                        st.write(r.text[:600])
            except Exception as e:
                st.error(f"Lỗi: {e}")

    # -- Dense branch
    with col_dense:
        st.subheader("Sentence-BERT + Cosine")
        if st.button("Tìm kiếm", key="btn_search_dense", type="primary"):
            try:
                dr = DenseRetriever(model_name=DEFAULT_MODEL)
                col_name = dr.collection_name(pcfg.config_id(),
                                              ChunkingConfig(strategy="article").config_id())
                existing = [c for c in dr.list_collections() if c == col_name]
                if not existing:
                    st.warning(
                        "Chưa có dense index cho cấu hình này — hãy xây ở màn 4.5 trước."
                    )
                else:
                    dr.use_collection(col_name)
                    results = dr.search(_rp(query, pcfg).text, top_k=int(top_k))
                    for r in results:
                        with st.expander(
                            f"#{r.rank} · cosine = {r.score:.4f} · `{r.chunk_id}`"
                        ):
                            st.write(r.text[:600])
            except Exception as e:
                st.error(f"Lỗi: {e}")

    st.caption(
        "Cosine score là độ tương tự vector, không phải 'xác suất đúng'; "
        "chỉ so sánh score trong cùng một mô hình/cấu hình."
    )

# ---------------------------------------------------------------------------
# Screen 4.7 — Chat with LLM (grounded RAG)
# ---------------------------------------------------------------------------
elif page.startswith("4.7 "):
    # -- Chat loop state (kept in session state, initialized up front so the
    #    config panel below can show the current conversation)
    if "rag_chat" not in st.session_state:
        st.session_state.rag_chat = []
    st.session_state.setdefault("rag_feedback", [])

    def _new_chat() -> None:
        st.session_state.rag_chat = []

    if is_admin:
        _playground_actions = admin_page_header(
            "Thử nghiệm Chat",
            "Kiểm thử toàn bộ luồng Query → Retrieval → Context → LLM → Answer.",
        )
        with _playground_actions:
            st.button(
                "Hội thoại mới",
                icon=":material/add_comment:",
                on_click=_new_chat,
                help="Bắt đầu một hội thoại mới",
            )
            st.button(
                "Trang người dùng",
                icon=":material/arrow_back:",
                on_click=_set_user_role,
                help="Về cổng thông tin người dùng",
            )

    # -- LLM availability
    models: list[str] = []
    if not ollama_available():
        st.warning(
            "Không kết nối được Ollama (`http://localhost:11434`). "
            "Cài đặt tại https://ollama.com rồi chạy:\n\n"
            "```\nollama pull qwen2.5:7b\nollama serve\n```"
        )
    else:
        models = list_ollama_models()
        if not models:
            st.warning("Ollama đang chạy nhưng chưa có model nào. Chạy: `ollama pull qwen2.5:7b`")

    # -- Retrieval & LLM settings come from the persisted admin configuration
    #    (managed on the "Cấu hình" screen via src/system_config.py). This
    #    chat surface only consumes them: neither admins nor end users can
    #    override retrieval/generation parameters per request, so the stored
    #    configuration stays authoritative for the whole pipeline.
    active_cfg, active_cfg_error = load_config()
    if active_cfg_error:
        st.warning(
            "Không thể tải cấu hình hệ thống — tạm dùng cấu hình mặc định."
        )
    if models and active_cfg.llm_model not in models:
        st.warning(
            f"Model `{active_cfg.llm_model}` chưa có trong Ollama — chạy "
            f"`ollama pull {active_cfg.llm_model}` hoặc đổi model ở trang "
            "Cấu hình."
        )
    st.caption(
        "**Cấu hình đang áp dụng** (quản trị tại trang **Cấu hình**): "
        f"retriever `{active_cfg.retriever}` · top_k = {active_cfg.top_k} · "
        f"tách từ `{active_cfg.word_segmentation}` · "
        f"LLM `{active_cfg.llm_model}` · "
        f"temperature = {active_cfg.temperature} · "
        f"max tokens = {active_cfg.max_tokens}"
    )
    chat_col = st.container()

    from src.preprocessing.pipeline import run_pipeline as _rp

    pcfg = PreprocessingConfig(word_segmentation=active_cfg.word_segmentation)

    def _retrieve(question: str):
        """Retrieve top-k chunks with the active admin configuration."""
        q = _rp(question, pcfg).text
        if active_cfg.retriever == "dense":
            dr = DenseRetriever(model_name=DEFAULT_MODEL)
            col = dr.collection_name(
                pcfg.config_id(), ChunkingConfig(strategy="article").config_id()
            )
            if col not in dr.list_collections():
                st.warning(
                    f"Chưa có dense collection `{col}` — hãy xây ở màn 4.5 "
                    "hoặc chuyển sang TF-IDF ở trang Cấu hình."
                )
                return None
            dr.use_collection(col)
            return dr.search(q, top_k=int(active_cfg.top_k))
        retriever, _ = get_tfidf_index(active_cfg.word_segmentation)
        return retriever.search(q, top_k=int(active_cfg.top_k))

    with chat_col:
        for message_index, msg in enumerate(st.session_state.rag_chat):
            if msg["role"] == "user":
                render_user_message(msg["content"], msg.get("sent_at"))
            else:
                with st.chat_message("assistant", avatar=":material/smart_toy:"):
                    render_assistant_time(msg.get("sent_at"))
                    st.markdown(msg["content"])
                    if msg.get("citations"):
                        with st.expander(":material/source: Nguồn tham chiếu", expanded=False):
                            st.caption(", ".join(f"`{c}`" for c in msg["citations"]))
                    rating = st.feedback("thumbs", key=f"answer_feedback_{message_index}")
                    if rating == 0:
                        reasons = st.multiselect(
                            "Câu trả lời có vấn đề gì?",
                            [
                                "Không chính xác",
                                "Không liên quan",
                                "Thiếu thông tin",
                                "Nguồn không phù hợp",
                                "Khác",
                            ],
                            key=f"feedback_reasons_{message_index}",
                        )
                        if st.button(
                            "Gửi phản hồi",
                            key=f"save_feedback_{message_index}",
                            disabled=not reasons,
                        ):
                            st.session_state.rag_feedback.append(
                                {"message_index": message_index, "reasons": reasons}
                            )
                            st.toast("Đã ghi nhận phản hồi", icon=":material/check:")

        if not st.session_state.rag_chat:
            st.markdown(
                '<div class="chat-empty-title">Bạn muốn hỏi điều gì?</div>'
                '<div class="chat-empty-sub">Tôi sẽ tìm kiếm trong kho tài liệu và '
                "tạo câu trả lời dựa trên các nguồn phù hợp nhất.</div>",
                unsafe_allow_html=True,
            )

            suggestions = [
                "Thời hiệu khởi kiện quyết định hành chính là bao lâu?",
                "Điều kiện để khiếu nại quyết định hành chính là gì?",
                "Hồ sơ khởi kiện cần những tài liệu nào?",
                "Phân biệt khiếu nại và khởi kiện hành chính.",
            ]

            def _queue_suggestion(suggestion: str) -> None:
                st.session_state["_pending_question"] = suggestion

            for row_start in (0, 2):
                suggestion_cols = st.columns(2)
                for suggestion_col, suggestion in zip(
                    suggestion_cols, suggestions[row_start:row_start + 2]
                ):
                    suggestion_col.button(
                        suggestion,
                        key=f"suggestion_{suggestions.index(suggestion)}",
                        on_click=_queue_suggestion,
                        args=(suggestion,),
                        width="stretch",
                    )

        pending_question = st.session_state.pop("_pending_question", None)
        typed_question = st.chat_input(
            "Nhập câu hỏi...",
            submit_mode="stop",
        )
        question = pending_question or typed_question
        if question:
            user_sent_at = current_message_time()
            st.session_state.rag_chat.append(
                {"role": "user", "content": question, "sent_at": user_sent_at}
            )
            render_user_message(question, user_sent_at)

            with st.chat_message("assistant", avatar=":material/smart_toy:"):
                assistant_meta = st.empty()
                reply = None
                try:
                    with st.status(":material/progress_activity: Đang xử lý câu hỏi...", expanded=True) as process_status:
                        st.write(":material/search: Truy xuất điều luật (retrieval)...")
                        results = _retrieve(question)
                        st.write(
                            f"✓ Truy xuất xong: {len(results) if results else 0} điều luật, "
                            f"top score = {results[0].score:.4f}" if results else "⚠ Không có kết quả"
                        )

                        if results:
                            st.markdown(f"**:material/source: Nguồn tham chiếu (top {len(results)})**")
                            for r in results:
                                st.markdown(
                                    f"**[Điều nguồn {r.rank}]** `{r.chunk_id}` · "
                                    f"score = {r.score:.4f}"
                                )
                                st.write(r.text[:400])

                            progress_box = st.empty()
                            log_lines: list[str] = []
                            t_start = time.time()

                            def _on_token(so_far: str, token: str):
                                # Keep the live generation log inside the single
                                # collapsible processing panel.
                                log_lines.append(token)
                                elapsed = max(time.time() - t_start, 1e-6)
                                progress_box.caption(
                                    f"Đang sinh câu trả lời: {len(log_lines)} token · "
                                    f"{len(so_far)} ký tự · "
                                    f"{len(log_lines) / elapsed:.1f} tok/s"
                                )

                            st.write(
                                f":material/model_training: Gọi Ollama model "
                                f"`{active_cfg.llm_model}` (stream)..."
                            )
                            cited = generate_answer_stream(
                                question,
                                results,
                                model=active_cfg.llm_model,
                                temperature=float(active_cfg.temperature),
                                max_tokens=int(active_cfg.max_tokens),
                                on_token=_on_token,
                            )
                            progress_box.caption(
                                f"✓ Sinh câu trả lời xong: {cited.eval_tokens} token · "
                                f"{cited.eval_duration_ms / 1000:.1f}s"
                            )
                            st.caption(
                                f"model = `{cited.model}` · prompt tokens = "
                                f"{cited.prompt_tokens:,} · output tokens = "
                                f"{cited.eval_tokens:,}"
                            )
                            process_status.update(
                                label=":material/manage_search: Quá trình xử lý & nguồn tham chiếu",
                                state="complete",
                                expanded=False,
                            )
                        else:
                            process_status.update(
                                label=":material/warning: Không tìm thấy nguồn phù hợp",
                                state="error",
                                expanded=False,
                            )

                    if results:
                        # Keep only the actual answer visible by default.
                        st.markdown(cited.answer)
                        reply = {
                            "role": "assistant",
                            "content": cited.answer,
                            "citations": cited.citations,
                        }
                    else:
                        st.warning("Không truy xuất được điều luật nào.")
                        reply = {"role": "assistant", "content": "_(không có kết quả truy xuất)_"}
                except Exception as e:
                    st.error(f"Lỗi: {e}")
                    reply = {"role": "assistant", "content": f"⚠️ Lỗi: {e}"}
                reply["sent_at"] = current_message_time()
                assistant_meta.markdown(
                    f'<div class="assistant-msg-time">Trợ lý · '
                    f'{html.escape(reply["sent_at"])}</div>',
                    unsafe_allow_html=True,
                )
            st.session_state.rag_chat.append(reply)


# ---------------------------------------------------------------------------
# Public legal information portal
# ---------------------------------------------------------------------------
elif page.startswith("5.1 "):
    portal_docs, portal_chunks, portal_version, _ = get_corpus()
    portal_questions = get_questions()
    open_assistant = render_legal_portal(
        portal_docs,
        portal_chunks,
        portal_questions,
        portal_version,
        on_admin=_set_admin_role,
        on_question=_queue_legal_question,
    )
    # ``legal_assistant_open`` is set when a portal suggestion queues a
    # question: the dialog then survives the full app rerun that refreshes
    # the transcript after answering (cleared by the dialog's on_dismiss).
    if open_assistant or st.session_state.get("legal_assistant_open", False):
        render_legal_assistant()


# ---------------------------------------------------------------------------
# Screen 4.8 — Retrieval Evaluation (Week 4: RQ1, RQ2)
# ---------------------------------------------------------------------------
elif page.startswith("4.8 "):
    from experiments.evaluate_retrieval import run_eval, save_result

    admin_page_header(
        "Đánh giá Retrieval (RQ1, RQ2)",
        "So sánh phương pháp tách từ và retriever bằng MRR / Hit@K / Recall@5 / P@5.",
    )
    st.markdown(
        "**RQ1** — Tách từ (none / underthesea / pyvi) ảnh hưởng thế nào đến retrieval?  \n"
        "**RQ2** — TF-IDF baseline so với dense SBERT?"
    )

    # -- Configuration lives in the main content: the sidebar is for
    #    navigation only. Run button sits inside the config card so
    #    parameter selection -> run is one continuous flow.
    config_col, result_col = st.columns(
        [0.34, 0.66], gap="medium", vertical_alignment="top"
    )
    with config_col.container(border=True, key="card_eval_config"):
        st.subheader(":material/tune: Cấu hình đánh giá")
        st.caption("Thiết lập tham số cho thí nghiệm.")
        rq = st.selectbox(
            "Câu hỏi nghiên cứu",
            ["RQ1 (tách từ)", "RQ2 (tfidf vs dense)"],
            format_func=lambda value: (
                f"{value} — Ưu tiên chạy trước" if value.startswith("RQ1") else value
            ),
            key="eval_rq",
        )
        retriever_eval = st.selectbox(
            "Retriever (RQ1)",
            ["tfidf", "dense"],
            format_func=lambda value: (
                "TF-IDF — Khuyến nghị để cô lập ảnh hưởng tách từ"
                if value == "tfidf"
                else "Dense — Thực nghiệm bổ sung"
            ),
            key="eval_retriever",
        )
        seg_eval = st.selectbox(
            "Tách từ (RQ2)",
            ["none", "underthesea", "pyvi"],
            format_func=lambda value: {
                "none": "None — Khuyến nghị để so sánh retriever công bằng",
                "underthesea": "Underthesea — Cấu hình bổ sung",
                "pyvi": "PyVi — Cấu hình bổ sung",
            }[value],
            key="eval_seg",
        )
        split_eval = st.selectbox(
            "Split dữ liệu",
            ["dev", "test", "train"],
            format_func=lambda value: {
                "dev": "Dev — Khuyến nghị khi phát triển",
                "test": "Test — Chỉ dùng cho đánh giá cuối cùng",
                "train": "Train — Chẩn đoán, không báo cáo kết quả cuối",
            }[value],
            key="eval_split",
        )
        max_q = st.number_input(
            "Số câu hỏi tối đa (0 = tất cả)", 0, 3000, 100, 50,
            help="Giảm để chạy nhanh khi thử nghiệm; 0 = toàn bộ split",
            key="eval_max_q",
        )
        run_eval_now = st.button(
            ":material/play_arrow: Chạy đánh giá", type="primary", width="stretch"
        )

    with result_col:
        with st.container(border=True, key="card_eval_result"):
            st.subheader(":material/analytics: Kết quả đánh giá")
            if not st.session_state.get("eval_ran_once") and not run_eval_now:
                st.markdown("**Chưa có kết quả đánh giá**")
                st.caption("Chọn cấu hình và nhấn **Chạy đánh giá**.")
        if run_eval_now:
            st.session_state["eval_ran_once"] = True
            configs = []
            if rq.startswith("RQ1"):
                for seg in ["none", "underthesea", "pyvi"]:
                    configs.append((retriever_eval, seg))
            else:
                configs = [("tfidf", seg_eval), ("dense", seg_eval)]

            all_results = []
            progress = st.progress(0.0, text="Đang chuẩn bị...")
            for i, (kind, seg) in enumerate(configs):
                progress.progress(
                    i / len(configs),
                    text=f"Đang chạy: {kind} + seg={seg}...",
                )
                try:
                    with st.spinner(f"Đang đánh giá {kind} + tách từ `{seg}`..."):
                        res = run_eval(
                            kind,
                            seg,
                            split=split_eval,
                            max_questions=(max_q or None),
                        )
                    path = save_result(res)
                    res["_path"] = path.name
                    all_results.append(res)
                    st.toast(f"Xong {kind}/{seg} → {path.name}", icon="✅")
                except Exception as e:
                    st.error(f"Lỗi khi chạy `{kind} + seg={seg}`: {e}")

            progress.empty()

            if all_results:
                st.subheader("Kết quả")
                rows = []
                for res in all_results:
                    m = res["metrics"]
                    rows.append(
                        {
                            "Cấu hình": f"{res['index'].get('type', '?')} / seg={res['preprocessing']['word_segmentation']}",
                            "nQuest": res["n_questions"],
                            "MRR": round(m["mrr"], 4),
                            "Hit@1": round(m["hit@k"]["1"], 4),
                            "Hit@3": round(m["hit@k"]["3"], 4),
                            "Hit@5": round(m["hit@k"]["5"], 4),
                            "Hit@10": round(m["hit@k"]["10"], 4),
                            "Recall@5": round(m["recall@k"]["5"], 4),
                            "P@5": round(m["precision@k"]["5"], 4),
                            "giây": res["elapsed_seconds"],
                        }
                    )
                st.dataframe(pd.DataFrame(rows), width="stretch")
                st.caption(
                    "Hit@K = tỉ lệ câu hỏi có ≥1 điều luật đúng trong top-K "
                    "(thước đo chính của Zalo AI 2021); MRR = trung bình nghịch đảo hạng."
                )

                # -- bar chart MRR / Hit@5
                st.subheader("So sánh trực quan")
                c1, c2 = st.columns(2)
                df_chart = pd.DataFrame(rows)
                c1.bar_chart(df_chart.set_index("Cấu hình")["MRR"])
                c2.bar_chart(df_chart.set_index("Cấu hình")["Hit@5"])

                with st.expander("Chi tiết JSON (từng thực nghiệm)"):
                    for res in all_results:
                        st.json(
                            {k: v for k, v in res.items() if not k.startswith("_")}
                        )

# ---------------------------------------------------------------------------
# Screen 4.9 — Model connections
# ---------------------------------------------------------------------------
elif page.startswith("4.9 "):
    admin_page_header(
        "Mô hình",
        "Quản lý kết nối LLM và cấu hình embedding dùng trong RAG.",
    )

    llm_tab, embedding_tab, reranker_tab = st.tabs(
        ["LLM", "Embedding", "Reranker"], on_change="rerun"
    )
    if llm_tab.open:
        with llm_tab:
            with st.container(border=True, key="card_models_llm"):
                st.subheader("Ollama")
                st.text_input(
                    "Base URL",
                    value="http://localhost:11434",
                    disabled=True,
                )
                available_models = list_ollama_models() if ollama_available() else []
                st.selectbox(
                    "Model",
                    available_models or ["qwen2.5:7b"],
                    key="models_llm_model",
                )
                if st.button(
                    "Kiểm tra kết nối",
                    icon=":material/cable:",
                    type="primary",
                ):
                    if ollama_available():
                        st.success("Kết nối Ollama thành công.")
                    else:
                        st.error("Không thể kết nối Ollama tại cổng 11434.")
    if embedding_tab.open:
        with embedding_tab:
            with st.container(border=True, key="card_models_embed"):
                st.subheader("Embedding model")
                st.text_input("Model", value=DEFAULT_MODEL, key="models_embedding")
                st.segmented_control(
                    "Device", ["Auto", "CPU", "CUDA"], default="Auto"
                )
                st.number_input("Batch size", 16, 256, 64, 16)
                st.caption(
                    "Việc kiểm tra embedding được thực hiện khi xây dense index để "
                    "tránh nạp model lớn chỉ cho thao tác xem cấu hình."
                )
    if reranker_tab.open:
        with reranker_tab:
            with st.container(border=True, key="card_models_rerank"):
                enabled = st.toggle("Bật reranker", value=False)
                st.text_input("Reranker model", disabled=not enabled)
                st.number_input("Top candidates", 5, 100, 20, disabled=not enabled)
                st.number_input("Final top-k", 1, 20, 5, disabled=not enabled)
                st.info(
                    "Backend hiện chưa triển khai reranker; cấu hình này chưa được "
                    "đưa vào thực nghiệm.",
                    icon=":material/info:",
                )

# ---------------------------------------------------------------------------
# Screen 4.10 — Settings
# ---------------------------------------------------------------------------
elif page.startswith("4.10 "):
    admin_page_header(
        "Cấu hình",
        "Quản trị cấu hình RAG & LLM áp dụng cho mọi lượt hỏi của hệ thống.",
    )

    # ---- RAG & LLM active configuration (persisted, admin-only) ----------
    # The stored configuration is loaded before the form renders and saving
    # is blocked entirely when loading fails, so a half-loaded form can
    # never overwrite the persisted values.
    with st.spinner("Đang tải cấu hình hệ thống..."):
        saved_cfg, cfg_load_error = load_config()
    if cfg_load_error:
        st.error("Không thể tải cấu hình hệ thống.")
        st.caption(f"Chi tiết: {cfg_load_error}")
        st.stop()
    if saved_cfg.updated_at:
        st.caption(
            f"Cập nhật lần cuối: {saved_cfg.updated_at} · bởi "
            f"{saved_cfg.updated_by or 'không rõ'}"
        )

    def _seed_ragcfg_widgets(cfg: RagLlmConfig) -> None:
        """Seed widget defaults from the stored config (snapped to the UI
        slider grids) and remember the snapshot for dirty-state tracking."""
        seeds = {
            "ragcfg_retriever": cfg.retriever
            if cfg.retriever in ("dense", "tfidf") else "dense",
            "ragcfg_top_k": min(max(int(cfg.top_k), 1), 20),
            "ragcfg_seg": cfg.word_segmentation
            if cfg.word_segmentation in ("none", "underthesea", "pyvi")
            else "none",
            "ragcfg_provider": "ollama",
            "ragcfg_model": (cfg.llm_model or "").strip() or "qwen2.5:7b",
            "ragcfg_temperature": round(
                min(max(float(cfg.temperature), 0.0), 1.0) * 20
            ) / 20,
            "ragcfg_max_tokens": min(
                max(round(int(cfg.max_tokens) / 128) * 128, 128), 4096
            ),
        }
        for key, value in seeds.items():
            st.session_state.setdefault(key, value)
        st.session_state.setdefault("ragcfg_snapshot", dict(seeds))

    _seed_ragcfg_widgets(saved_cfg)

    with st.container(border=True, key="card_settings_rag"):
        st.subheader(":material/tune: Cấu hình RAG")
        st.selectbox(
            "Retriever",
            ["dense", "tfidf"],
            key="ragcfg_retriever",
            format_func=lambda value: (
                "Dense (SBERT) — tìm kiếm ngữ nghĩa"
                if value == "dense"
                else "TF-IDF — baseline"
            ),
            help="Dense yêu cầu dense index đã được xây ở màn Lập chỉ mục.",
        )
        st.slider(
            "Top-K (số điều luật đưa vào prompt)",
            min_value=1, max_value=20, step=1,
            key="ragcfg_top_k",
            help="Số chunk tối đa được truy xuất và đưa vào ngữ cảnh cho LLM "
                 "trong mỗi lượt hỏi.",
        )
        st.selectbox(
            "Tách từ",
            ["none", "underthesea", "pyvi"],
            key="ragcfg_seg",
            format_func=lambda value: {
                "none": "None — khuyến nghị với dense",
                "underthesea": "Underthesea",
                "pyvi": "PyVi",
            }[value],
            help="Phải khớp với phương pháp tách từ đã dùng khi xây index.",
        )

    available_models = list_ollama_models() if ollama_available() else []
    with st.container(border=True, key="card_settings_llm"):
        st.subheader(":material/model_training: Cấu hình LLM")
        st.selectbox(
            "Provider",
            ["ollama"],
            key="ragcfg_provider",
            format_func=lambda value: "Ollama (local)",
            help="Hiện chỉ hỗ trợ Ollama; danh sách sẽ mở rộng khi bổ sung "
                 "provider khác.",
        )
        # Keep the widget type stable across reruns (Ollama up/down) so the
        # session key never binds to a different widget type; the configured
        # model always stays selectable.
        model_options = list(
            dict.fromkeys(
                [st.session_state.get("ragcfg_model", saved_cfg.llm_model)]
                + available_models
            )
        )
        st.selectbox(
            "Model",
            model_options,
            key="ragcfg_model",
            help="Danh sách model lấy từ Ollama. Khi Ollama chưa kết nối, "
                 "chỉ hiển thị model hiện tại.",
        )
        st.slider(
            "Temperature", min_value=0.0, max_value=1.0, step=0.05,
            key="ragcfg_temperature",
            help="0 = câu trả lời xác định cao; 0.2 khuyến nghị cho tra cứu "
                 "pháp lý.",
        )
        st.slider(
            "Max tokens trả lời", min_value=128, max_value=4096, step=128,
            key="ragcfg_max_tokens",
        )

    def _ragcfg_form_values() -> dict:
        return {
            "retriever": st.session_state.ragcfg_retriever,
            "top_k": int(st.session_state.ragcfg_top_k),
            "word_segmentation": st.session_state.ragcfg_seg,
            "llm_provider": st.session_state.ragcfg_provider,
            "llm_model": st.session_state.ragcfg_model.strip(),
            "temperature": float(st.session_state.ragcfg_temperature),
            "max_tokens": int(st.session_state.ragcfg_max_tokens),
        }

    def _reset_ragcfg_form() -> None:
        """Discard unsaved edits (widget state back to saved snapshot)."""
        for key, value in st.session_state.ragcfg_snapshot.items():
            st.session_state[f"ragcfg_{key}"] = value

    form_values = _ragcfg_form_values()
    dirty = form_values != st.session_state.ragcfg_snapshot

    save_col, reset_col = st.columns(2)
    reset_col.button(
        "Hoàn tác",
        icon=":material/undo:",
        disabled=not dirty,
        on_click=_reset_ragcfg_form,
        use_container_width=True,
    )
    save_clicked = save_col.button(
        "Lưu cấu hình",
        icon=":material/save:",
        type="primary",
        disabled=not dirty,
        use_container_width=True,
    )
    if save_clicked:
        try:
            with st.status("Đang lưu cấu hình...", expanded=False):
                save_config(
                    RagLlmConfig(**form_values), updated_by="Quản trị viên"
                )
            st.session_state.ragcfg_snapshot = _ragcfg_form_values()
            st.toast(
                "Đã lưu cấu hình RAG & LLM thành công.",
                icon=":material/check_circle:",
            )
            st.rerun()  # refresh "Cập nhật lần cuối" from disk
        except ConfigValidationError as error:
            st.error(f"Không thể lưu cấu hình. {error}")
        except OSError:
            st.error(
                "Không thể lưu cấu hình. Vui lòng kiểm tra lại thông tin hoặc "
                "kết nối hệ thống."
            )

    # ---- General session settings (unchanged scope) -----------------------
    st.divider()
    st.session_state.setdefault(
        "app_settings",
        {
            "system_name": "RAG Management & QA Platform",
            "language": "Tiếng Việt",
        },
    )
    saved_settings = st.session_state.app_settings
    with st.form("general_settings"):
        with st.container(border=True, key="card_settings_general"):
            st.subheader("General")
            system_name = st.text_input("Tên hệ thống", saved_settings["system_name"])
            language = st.selectbox(
                "Ngôn ngữ",
                ["Tiếng Việt", "English"],
                index=0 if saved_settings["language"] == "Tiếng Việt" else 1,
            )
        save_settings = st.form_submit_button(
            "Lưu thiết lập", icon=":material/save:", type="primary"
        )
    if save_settings:
        if not system_name.strip():
            st.error("Tên hệ thống không được để trống.")
        else:
            st.session_state.app_settings = {
                "system_name": system_name.strip(),
                "language": language,
            }
            st.toast("Đã lưu thiết lập cho phiên hiện tại", icon=":material/check:")

# ---------------------------------------------------------------------------
# Screen 4.11 — Observable system state
# ---------------------------------------------------------------------------
elif page.startswith("4.11 "):
    admin_page_header(
        "Nhật ký hệ thống",
        "Các sự kiện có thể xác minh trong phiên hiện tại và artifact thực nghiệm.",
    )
    metric_files = sorted((PROJECT_ROOT / "results" / "metrics").glob("*.json"))
    now_text = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    log_rows = [
        {
            "Time": now_text,
            "Level": "INFO" if dataset_ready() else "ERROR",
            "Module": "Data",
            "Message": "Dataset sẵn sàng" if dataset_ready() else "Thiếu dataset",
        },
        {
            "Time": now_text,
            "Level": "INFO" if ollama_available() else "WARN",
            "Module": "LLM",
            "Message": "Ollama đã kết nối" if ollama_available() else "Ollama chưa kết nối",
        },
        {
            "Time": now_text,
            "Level": "INFO",
            "Module": "Evaluation",
            "Message": f"Tìm thấy {len(metric_files)} artifact kết quả",
        },
    ]
    # Configuration changes are persisted to an append-only audit trail
    # (src/system_config.py) — surface the most recent ones here.
    for entry in read_audit_entries(10):
        new_values = entry.get("new", {})
        log_rows.append(
            {
                "Time": str(entry.get("timestamp", ""))[:19].replace("T", " "),
                "Level": "INFO",
                "Module": "Config",
                "Message": (
                    f"{entry.get('updated_by', 'không rõ')} cập nhật cấu hình: "
                    f"retriever={new_values.get('retriever')}, "
                    f"top_k={new_values.get('top_k')}, "
                    f"tách từ={new_values.get('word_segmentation')}, "
                    f"LLM={new_values.get('llm_model')} "
                    f"(temperature={new_values.get('temperature')}, "
                    f"max_tokens={new_values.get('max_tokens')})"
                ),
            }
        )
    level_filter = st.pills(
        "Level",
        ["INFO", "WARN", "ERROR"],
        default=["INFO", "WARN", "ERROR"],
        selection_mode="multi",
    )
    module_filter = st.multiselect(
        "Module",
        ["Data", "LLM", "Config", "Evaluation"],
        default=["Data", "LLM", "Config", "Evaluation"],
    )
    logs_df = pd.DataFrame(log_rows)
    filtered_logs = logs_df[
        logs_df["Level"].isin(level_filter) & logs_df["Module"].isin(module_filter)
    ]
    if filtered_logs.empty:
        st.info("Không có log nào khớp bộ lọc hiện tại.", icon=":material/info:")
    else:
        st.dataframe(filtered_logs, hide_index=True, width="stretch")

