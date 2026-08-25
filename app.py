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
from src.chunking.chunker import build_chunks  # noqa: E402
from src.retrieval.tfidf_retriever import TfidfRetriever  # noqa: E402
from src.retrieval.dense_retriever import DEFAULT_MODEL, DenseRetriever  # noqa: E402
from src.generation.ollama_client import (  # noqa: E402
    generate_answer_stream,
    list_ollama_models,
    ollama_available,
)
from src.evaluation.retrieval_metrics import evaluate_retrieval  # noqa: E402

st.set_page_config(page_title="Smart RAG QA", page_icon="⚖️", layout="wide")

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
    /* Larger, touch-friendly primary navigation icons. */
    [data-testid="stSidebar"] [class*="st-key-nav_"] button {
        min-height: 3.35rem;
        border-radius: 14px;
    }
    [data-testid="stSidebar"] [class*="st-key-nav_"] button span[role="img"] {
        font-size: 1.75rem;
    }
    /* Make the assistant identity visible at a glance. */
    [data-testid="stChatMessage"] [data-testid*="Avatar"] {
        width: 2.75rem;
        height: 2.75rem;
    }
    /* One visual group per assistant message.  Style the message content
       wrapper instead of every nested Markdown block (status/expanders also
       contain Markdown and must not become separate bubbles). */
    [data-testid="stChatMessage"] > [data-testid="stChatMessageContent"] {
        background: rgba(100, 151, 255, 0.10);
        border-left: 3px solid rgba(100, 151, 255, 0.65);
        border-radius: 4px 16px 16px 16px;
        padding: 10px 16px;
    }
    /* Custom user bubble (right-aligned) */
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
        background: linear-gradient(135deg, #1f8fff33, #1f8fff22);
        border: 1px solid #1f8fff55;
        border-right: 3px solid #1f8fff99;
        border-radius: 16px 4px 16px 16px;
        padding: 10px 16px;
        width: fit-content;
        max-width: 75%;
        color: inherit;
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
st.sidebar.title(":material/balance: Smart RAG QA")
st.sidebar.caption("Zalo AI 2021 — Legal Text Retrieval")

if not dataset_ready():
    st.warning("Chưa có dataset trong `data/raw/`. Xem hướng dẫn `data/DATASET_GUIDE.md` để tải.")
    st.stop()

# -- LLM connection status (always visible in sidebar)
st.sidebar.subheader("Kiểm tra LLM")
if ollama_available():
    _models = list_ollama_models()
    if _models:
        st.sidebar.success(f":material/check_circle: Ollama: {', '.join(_models[:3])}")
    else:
        st.sidebar.warning(":material/warning: Ollama chạy nhưng chưa có model")
else:
    st.sidebar.error(":material/error: Ollama chưa kết nối (11434)")

st.sidebar.divider()

# -- Navigation: 3 parent menus in sidebar, sub-pages as tabs in main area
# Chat (4.7) · Dữ liệu (4.1, 4.2) · Thí nghiệm & Đánh giá (4.3-4.6, 4.8)
_DEFAULT_PAGE = "4.7 Chat với LLM (RAG)"

_GROUP_OF = {
    "4.1 Tổng quan": "data",
    "4.2 Quản lý dữ liệu": "data",
    "4.3 Phòng tiền xử lý": "lab",
    "4.4 Chunking Lab": "lab",
    "4.5 Lập chỉ mục": "lab",
    "4.6 Retrieval Playground": "lab",
    "4.8 Đánh giá Retrieval (RQ1, RQ2)": "lab",
}

_GROUP_META = {
    "chat": (":material/chat_bubble:", "Chat / Tra cứu", _DEFAULT_PAGE),
    "data": (":material/database:", "Dữ liệu", "4.1 Tổng quan"),
    "lab": (":material/science:", "Thí nghiệm & Đánh giá", "4.3 Phòng tiền xử lý"),
}

_TAB_LABELS = {
    "4.1 Tổng quan": ":material/dashboard: Tổng quan",
    "4.2 Quản lý dữ liệu": ":material/folder_open: Quản lý dữ liệu",
    "4.3 Phòng tiền xử lý": ":material/experiment: Tiền xử lý",
    "4.4 Chunking Lab": ":material/content_cut: Chunking",
    "4.5 Lập chỉ mục": ":material/storage: Lập chỉ mục",
    "4.6 Retrieval Playground": ":material/manage_search: Retrieval",
    "4.8 Đánh giá Retrieval (RQ1, RQ2)": ":material/analytics: Đánh giá",
}

def _goto_group(group: str):
    st.session_state["_active_page"] = _GROUP_META[group][2]

def _set_page_from_tab(key: str):
    label = st.session_state[key]
    for pg, lb in _TAB_LABELS.items():
        if lb == label:
            st.session_state["_active_page"] = pg
            return

_active_page = st.session_state.get("_active_page", _DEFAULT_PAGE)
_active_group = _GROUP_OF.get(_active_page, "chat")

with st.sidebar:
    _cols = st.columns(len(_GROUP_META))
    for _col, (_gid, (_icon, _name, _)) in zip(_cols, _GROUP_META.items()):
        with _col:
            st.button(
                _icon,
                key=f"nav_{_gid}",
                help=_name,
                on_click=_goto_group,
                args=(_gid,),
                use_container_width=True,
                type="primary" if _gid == _active_group else "secondary",
            )
    st.caption(
        f"**{_GROUP_META[_active_group][0]} {_GROUP_META[_active_group][1]}**"
    )

# Tabs cho menu con (trừ nhóm Chat chỉ có 1 trang)
_group_pages = [p for p in _GROUP_OF if _GROUP_OF[p] == _active_group]
if len(_group_pages) > 1:
    _ordered = [p for p in _TAB_LABELS if p in _group_pages]
    _tab_labels = [_TAB_LABELS[p] for p in _ordered]
    st.segmented_control(
        "menu con",
        _tab_labels,
        selection_mode="single",
        default=_TAB_LABELS.get(_active_page, _tab_labels[0]),
        label_visibility="collapsed",
        key=f"tabs_{_active_group}",
        on_change=_set_page_from_tab,
        args=(f"tabs_{_active_group}",),
    )
    st.divider()

page = st.session_state.get("_active_page", _DEFAULT_PAGE)

# ---------------------------------------------------------------------------
# Screen 4.1 — Overview
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Đang xây dựng chunks...")
def get_built_chunks(strategy: str, chunk_size: int, chunk_overlap: int):
    """Chunk corpus with the given configuration (cached)."""
    cfg = ChunkingConfig(strategy=strategy, chunk_size=chunk_size,
                         chunk_overlap=chunk_overlap)
    _, chunks, _, _ = get_corpus()
    built, stats = build_chunks(chunks, cfg)
    return cfg, built, stats


@st.cache_resource(show_spinner="Đang build TF-IDF (preprocess corpus)...")
def get_tfidf_index(segmentation: str):
    """TF-IDF index over corpus preprocessed with the given segmentation."""
    import time

    from src.preprocessing.pipeline import run_pipeline as _rp

    _, chunks, _, _ = get_corpus()
    pcfg = PreprocessingConfig(word_segmentation=segmentation)
    texts, ids, metas = [], [], []
    t0 = time.time()
    for c in chunks:
        texts.append(_rp(c.text, pcfg).text)
        ids.append(c.chunk_id)
        metas.append({"law_id": c.law_id, "article_id": c.article_id})
    retriever = TfidfRetriever().fit(ids, texts, metas)
    retriever.save()
    return retriever, round(time.time() - t0, 1)


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------
if page.startswith("4.1"):
    st.header("Tổng quan hệ thống")
    docs, chunks, version, report = get_corpus()
    questions = get_questions()

    col1, col2, col3 = st.columns(3)
    col1.metric("Số văn bản (law)", f"{len(docs):,}")
    col2.metric("Số article (chunk)", f"{len(chunks):,}")
    col3.metric("Số câu hỏi có nhãn", f"{len(questions):,}")

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
elif page.startswith("4.2"):
    st.header("Quản lý dữ liệu")
    docs, chunks, version, report = get_corpus()

    st.subheader("Kiểm tra dữ liệu")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Văn bản hợp lệ", report["valid_laws"])
    c2.metric("Article hợp lệ", report["valid_articles"])
    c3.metric("Article rỗng (loại)", report["empty_articles"])
    c4.metric("Article trùng (loại)", report["duplicate_articles"])

    if report["errors"]:
        with st.expander(f"Lỗi nạp ({len(report['errors'])})"):
            st.write(report["errors"][:20])

    st.subheader("Xem trước corpus")
    option = st.selectbox(
        "Chọn văn bản",
        range(len(docs)),
        format_func=lambda i: docs[i].title,
    )
    doc = docs[option]
    st.markdown(f"**document_id:** `{doc.document_id}` · **số article:** {doc.metadata['n_articles']}")
    st.text_area("Nội dung", doc.raw_text, height=300)

    st.subheader("Câu hỏi + nhãn relevance (mẫu)")
    questions = get_questions()
    qdf = pd.DataFrame(questions).head(20)
    st.dataframe(qdf, use_container_width=True)

    # -- Add new law incrementally (dense) + full refit (TF-IDF) ------------
    st.divider()
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
elif page.startswith("4.3"):
    st.header("Phòng tiền xử lý dữ liệu")

    st.sidebar.subheader("Cấu hình tiền xử lý")
    cfg = PreprocessingConfig(
        unicode_normalization=st.sidebar.selectbox(
            "Unicode normalization", ["NFC", "none"]
        ),
        whitespace_normalization=st.sidebar.checkbox(
            "Whitespace normalization", value=True
        ),
        remove_noise_chars=st.sidebar.checkbox("Loại ký tự nhiễu", value=True),
        lowercase=st.sidebar.checkbox("Chuẩn hóa chữ thường", value=False,
                                      help="Không khuyến nghị cho dense model"),
        word_segmentation=st.sidebar.selectbox(
            "Tách từ tiếng Việt", ["none", "underthesea", "pyvi"]
        ),
        remove_stopwords=st.sidebar.checkbox("Bỏ stop-word", value=False,
                                             help="Cảnh báo: có thể làm mất nghĩa pháp lý"),
    )
    st.sidebar.caption(f"config_id: `{cfg.config_id()}`")

    tab_sample, tab_compare = st.tabs(["Thử trên một mẫu", "So sánh tách từ"])

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

    # -- Tab 2: compare segmentation methods side by side
    with tab_compare:
        st.caption("Cùng một văn bản, so sánh `none` / `underthesea` / `pyvi` (RQ1)")
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

if page.startswith("4.4"):
    st.header("Chunking Lab")
    st.sidebar.subheader("Cấu hình chunking")
    strategy = st.sidebar.selectbox(
        "Chiến lược", ["article", "fixed"], help="article = 1 điều luật / chunk"
    )
    chunk_size = st.sidebar.number_input("chunk_size (token)", 64, 1024, 256, 64)
    chunk_overlap = st.sidebar.number_input("chunk_overlap (token)", 0, chunk_size // 2, 0)

    cfg, built, stats = get_built_chunks(strategy, chunk_size, chunk_overlap)
    st.sidebar.caption(f"chunking config_id: `{cfg.config_id()}`")

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
elif page.startswith("4.5"):
    st.header("Lập chỉ mục và embedding")

    st.subheader("1) Baseline: TF-IDF")
    seg = st.selectbox("Tách từ (RQ1)", ["none", "underthesea", "pyvi"], key="tfidf_seg")
    if st.button("Xây TF-IDF index", key="btn_tfidf"):
        with st.spinner("Đang xây TF-IDF..."):
            try:
                retriever, elapsed = get_tfidf_index(seg)
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
    device = st.selectbox("Device", ["auto", "cpu", "cuda"])
    batch_size = st.number_input("Batch size", 16, 256, 64, 16)

    if st.button("Xây dense index", key="btn_dense", type="primary"):
        try:
            from src.preprocessing.pipeline import run_pipeline as _rp

            _, chunks, _, _ = get_corpus()
            pcfg = PreprocessingConfig(word_segmentation=seg)
            dr = DenseRetriever(model_name=model_name,
                                device=None if device == "auto" else device)
            col_name = dr.collection_name(
                PreprocessingConfig(word_segmentation=seg).config_id(),
                ChunkingConfig(strategy="article").config_id(),
            )
            progress = st.progress(0.0, text="Đang embed corpus...")
            texts = [_rp(c.text, pcfg).text for c in chunks]
            metas = [{"law_id": c.law_id, "article_id": c.article_id} for c in chunks]
            info = dr.build_index(
                [c.chunk_id for c in chunks], texts, metas, col_name,
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
elif page.startswith("4.6"):
    st.header("Retrieval Playground")
    query = st.text_input(
        "Câu hỏi", value="Thời hiệu khởi kiện khiếu quyết định hành chính là bao lâu?"
    )
    top_k = st.selectbox("top_k", TOP_K_CHOICES, index=TOP_K_CHOICES.index(5))
    seg = st.selectbox("Tách từ", ["none", "underthesea", "pyvi"], key="playground_seg")

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
elif page.startswith("4.7"):
    st.header("Chat với LLM (RAG)")

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

    # -- Retrieval settings
    st.sidebar.subheader("Retriever")
    retriever_kind = st.sidebar.selectbox(
        "Retriever", ["dense (SBERT)", "tfidf (baseline)"]
    )
    top_k_rag = st.sidebar.select_slider(
        "Số điều luật đưa vào prompt (top_k)", options=[1, 3, 5, 10], value=5
    )
    seg_rag = st.sidebar.selectbox("Tách từ", ["none", "underthesea", "pyvi"], key="rag_seg")

    st.sidebar.subheader("LLM")
    model_options = models if models else ["qwen2.5:7b"]
    default_idx = (
        model_options.index("qwen2.5:7b")
        if "qwen2.5:7b" in model_options
        else 0
    )
    llm_model = st.sidebar.selectbox("Model Ollama", model_options, index=default_idx)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.sidebar.slider("Max tokens trả lời", 128, 2048, 512, 128)

    from src.preprocessing.pipeline import run_pipeline as _rp

    pcfg = PreprocessingConfig(word_segmentation=seg_rag)

    # -- Chat loop (history kept in session state)
    if "rag_chat" not in st.session_state:
        st.session_state.rag_chat = []

    def _retrieve(question: str):
        """Retrieve top-k chunks with the selected retriever."""
        q = _rp(question, pcfg).text
        if retriever_kind.startswith("dense"):
            dr = DenseRetriever(model_name=DEFAULT_MODEL)
            col = dr.collection_name(
                pcfg.config_id(), ChunkingConfig(strategy="article").config_id()
            )
            if col not in dr.list_collections():
                st.warning(
                    f"Chưa có dense collection `{col}` — hãy xây ở màn 4.5 trước."
                )
                return None
            dr.use_collection(col)
            return dr.search(q, top_k=int(top_k_rag))
        retriever, _ = get_tfidf_index(seg_rag)
        return retriever.search(q, top_k=int(top_k_rag))

    for msg in st.session_state.rag_chat:
        if msg["role"] == "user":
            render_user_message(msg["content"], msg.get("sent_at"))
        else:
            with st.chat_message("assistant", avatar=":material/smart_toy:"):
                render_assistant_time(msg.get("sent_at"))
                st.markdown(msg["content"])
                if msg.get("citations"):
                    with st.expander(":material/source: Nguồn tham chiếu", expanded=False):
                        st.caption(", ".join(f"`{c}`" for c in msg["citations"]))

    question = st.chat_input("Đặt câu hỏi pháp luật (ví dụ: Thời hiệu khiếu kiện là bao lâu?)")
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

                        st.write(f":material/model_training: Gọi Ollama model `{llm_model}` (stream)...")
                        cited = generate_answer_stream(
                            question,
                            results,
                            model=llm_model,
                            temperature=temperature,
                            max_tokens=int(max_tokens),
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

        if st.sidebar.button(":material/delete: Xoá lịch sử chat"):
            st.session_state.rag_chat = []
            st.rerun()

# ---------------------------------------------------------------------------
# Screen 4.8 — Retrieval Evaluation (Week 4: RQ1, RQ2)
# ---------------------------------------------------------------------------
elif page.startswith("4.8"):
    from experiments.evaluate_retrieval import run_eval, save_result

    st.header("Đánh giá Retrieval (RQ1, RQ2)")
    st.markdown(
        "**RQ1** — Tách từ (none / underthesea / pyvi) ảnh hưởng thế nào đến retrieval?  \n"
        "**RQ2** — TF-IDF baseline so với dense SBERT?"
    )

    st.sidebar.subheader("Cấu hình đánh giá")
    rq = st.sidebar.selectbox("Câu hỏi nghiên cứu", ["RQ1 (tách từ)", "RQ2 (tfidf vs dense)"])
    retriever_eval = st.sidebar.selectbox("Retriever (RQ1)", ["tfidf", "dense"])
    seg_eval = st.sidebar.selectbox("Tách từ (RQ2)", ["none", "underthesea", "pyvi"])
    split_eval = st.sidebar.selectbox("Split dữ liệu", ["dev", "test", "train"])
    max_q = st.sidebar.number_input(
        "Số câu hỏi tối đa (0 = tất cả)", 0, 3000, 100, 50,
        help="Giảm để chạy nhanh khi thử nghiệm; 0 = toàn bộ split"
    )

    if st.button(":material/play_arrow: Chạy đánh giá", type="primary"):
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
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
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

