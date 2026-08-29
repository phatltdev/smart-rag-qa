"""Public legal-information portal presentation."""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

LEGAL_FIELDS = [
    ("Dân sự", ":material/groups:", "Quan hệ dân sự, nghĩa vụ và hợp đồng"),
    ("Doanh nghiệp", ":material/domain:", "Thành lập và hoạt động doanh nghiệp"),
    ("Hôn nhân & Gia đình", ":material/family_restroom:", "Hôn nhân, gia đình và hộ tịch"),
    ("Lao động", ":material/work:", "Việc làm và quan hệ lao động"),
    ("Đất đai", ":material/landscape:", "Quản lý và sử dụng đất"),
    ("Thuế", ":material/receipt_long:", "Nghĩa vụ và thủ tục thuế"),
    ("Giao thông", ":material/directions_car:", "Quy tắc và xử lý vi phạm"),
    ("Hình sự", ":material/security:", "Tội phạm và trách nhiệm hình sự"),
]


def _portal_styles() -> None:
    # st.markdown (not st.html) so the <style> block reliably lands in the
    # live DOM in this Streamlit version; style-only st.html is routed to the
    # event container and never rendered here.
    st.markdown(
        """
        <style>
        :root { --legal-primary:#0F2A44; --legal-bg:#F8FAFC;
          --legal-card:#FFFFFF; --legal-border:#D9E2EC; --legal-text:#1E293B;
          --legal-muted:#64748B; --legal-accent:#B88732;
          /* Chat panel palette (light) — the dialog is mounted on <body>,
             outside stAppViewContainer, so it must carry its own colors. */
          --chat-bg:#FFFFFF; --chat-text:#0F172A; --chat-secondary:#64748B;
          --chat-muted:#94A3B8; --chat-border:#E2E8F0;
          --chat-dark:#0F172A; --chat-accent:#EF4444; }
        [data-testid="stSidebar"] { display:none; }
        [data-testid="stAppViewContainer"] { background:var(--legal-bg); color:var(--legal-text); }
        [data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3, [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] label { color:var(--legal-text); }
        [data-testid="stAppViewContainer"] .stButton button {
          background:var(--legal-card); color:var(--legal-primary);
          border-color:var(--legal-border); }
        [data-testid="stAppViewContainer"] .stButton button[kind="primary"],
        [data-testid="stAppViewContainer"] .stButton button[data-variant="primary"],
        [data-testid="stAppViewContainer"] .stFormSubmitButton button[kind="primary"],
        [data-testid="stAppViewContainer"] .stFormSubmitButton button[data-variant="primary"] {
          background:var(--chat-accent); color:white; border-color:var(--chat-accent); }
        [data-testid="stAppViewContainer"] [data-baseweb="input"],
        [data-testid="stAppViewContainer"] [data-baseweb="textarea"],
        [data-testid="stAppViewContainer"] [data-baseweb="select"] {
          background:var(--legal-card); color:var(--legal-text); }

        /* ---- Page header: brand (left) + admin action (right) ---- */
        .legal-brand { padding:1.1rem 1.35rem; border:1px solid var(--legal-border);
          border-left:5px solid var(--legal-accent); border-radius:12px;
          background:var(--legal-card); }
        .legal-brand h1 { color:var(--legal-primary); font-size:clamp(1.3rem,2.6vw,1.9rem); margin:0; }
        .legal-brand p { color:var(--legal-muted); margin:.3rem 0 0; }

        /* ---- Primary navigation: a real navbar, not a segmented control ---- */
        [data-testid="stAppViewContainer"] [data-testid="stButtonGroup"] [role="radiogroup"] {
          display:flex; width:100%; background:var(--legal-card);
          border:1px solid var(--legal-border); border-radius:12px;
          padding:4px; gap:2px; box-shadow:0 1px 2px rgba(15,42,68,.05);
        }
        [data-testid="stAppViewContainer"] [data-testid="stButtonGroup"] [role="radiogroup"] button[role="radio"] {
          flex:1 1 0; min-height:42px; background:transparent; border:none;
          border-radius:9px; box-shadow:none; color:var(--legal-muted);
          font-size:.9rem; font-weight:500; padding:6px 4px;
          transition:background .15s ease, color .15s ease;
        }
        [data-testid="stAppViewContainer"] [data-testid="stButtonGroup"] [role="radiogroup"] button[role="radio"] p { color:inherit; }
        [data-testid="stAppViewContainer"] [data-testid="stButtonGroup"] [role="radiogroup"] button[role="radio"]:hover {
          background:#F1F5F9; color:var(--legal-primary);
        }
        [data-testid="stAppViewContainer"] [data-testid="stButtonGroup"] [role="radiogroup"] button[role="radio"][aria-checked="true"] {
          background:rgba(15,42,68,.07); color:var(--legal-primary); font-weight:600;
          box-shadow:inset 0 -2px 0 0 var(--legal-accent);
        }

        .legal-hero { padding:clamp(2rem,6vw,4.5rem) clamp(1rem,5vw,4rem);
          text-align:center; border-radius:18px; background:var(--legal-primary);
          margin:0 0 1.25rem; }
        .legal-hero h2,.legal-hero p { color:white!important; }
        .legal-hero h2 { font-size:clamp(1.6rem,4vw,2.7rem); }
        /* The hero now opens the page: keep the top gap compact instead of
           Streamlit's default main-container padding. */
        [data-testid="stAppViewContainer"] [data-testid="stMainBlockContainer"] {
          padding-top:1.5rem!important;
        }

        /* The trigger itself is the floating root. Unlike st.bottom, it does
           not create a full-width sticky bar or reserve document space. */
        .st-key-legal_assistant_trigger {
          position:fixed; right:24px; bottom:24px; width:max-content;
          z-index:9999; margin:0; padding:0;
        }
        .st-key-legal_assistant_trigger button { border-radius:999px;
          box-shadow:0 8px 24px rgba(15,42,68,.25); }
        /* While the chat panel is open the launcher is hidden, so the panel
           can use the full bottom:24px anchor. */
        body:has([data-testid="stDialog"]) .st-key-legal_assistant_trigger {
          display:none!important;
        }

        /* Streamlit mounts dialogs on <body> (outside the app view), which is
           why the dark theme colors leak into a white panel. Make the modal
           layer transparent/non-blocking and position only the actual panel. */
        [data-baseweb="modal"] {
          background:transparent!important; pointer-events:none!important;
        }
        [data-baseweb="modal"]::before {
          display:none!important; background:transparent!important;
        }

        /* ---- Chat panel: scoped light theme + viewport-aware height ---- */
        [data-testid="stDialog"] [role="dialog"] {
          position:fixed!important; right:24px!important; bottom:24px!important;
          top:auto!important; left:auto!important; margin:0!important;
          width:410px!important; max-width:calc(100vw - 32px)!important;
          height:min(620px,calc(100dvh - 48px))!important;
          max-height:calc(100dvh - 48px)!important;
          box-sizing:border-box!important;
          display:flex!important; flex-direction:column!important;
          overflow:hidden!important; overscroll-behavior:contain;
          background:var(--chat-bg)!important; color:var(--chat-text)!important;
          opacity:1!important;
          border:1px solid var(--chat-border)!important; border-radius:16px!important;
          box-shadow:0 20px 50px rgba(2,6,23,.22)!important;
          pointer-events:auto!important;
          transition:width .25s ease, height .25s ease;
        }
        /* Close button: visible slate icon in a round hover target */
        [data-testid="stDialog"] [role="dialog"] > button {
          position:absolute!important; top:10px!important; right:10px!important;
          width:36px!important; height:36px!important; margin:0!important;
          display:flex!important; align-items:center!important; justify-content:center!important;
          border-radius:999px!important; color:var(--chat-secondary)!important;
          background:transparent!important; border:none!important;
          transition:background .15s ease, color .15s ease;
        }
        [data-testid="stDialog"] [role="dialog"] > button:hover {
          background:#F1F5F9!important; color:var(--chat-text)!important;
        }
        [data-testid="stDialog"] [role="dialog"] > button svg { width:20px; height:20px; }
        /* Dialog title */
        [data-testid="stDialog"] [role="dialog"] > h2 {
          flex-shrink:0; margin:0!important; padding:14px 58px 10px 18px!important;
          color:var(--chat-text)!important; font-size:1.05rem!important; font-weight:600!important;
        }
        [data-testid="stDialog"] [role="dialog"] > h2 p { color:var(--chat-text)!important; }

        /* Content column: the only flexible region */
        [data-testid="stDialog"] [role="dialog"] > div {
          flex:1 1 auto!important; min-height:0!important; min-width:0;
          display:flex!important; flex-direction:column!important; overflow:hidden!important;
        }
        [data-testid="stDialog"] [role="dialog"] > div > [data-testid="stVerticalBlock"] {
          flex:1 1 auto!important; min-height:0!important;
          display:flex!important; flex-direction:column!important;
          overflow:hidden!important; gap:0!important;
        }
        /* The dialog body sits inside one extra layout wrapper — make that
           wrapper (and its vertical block) fill/constrain the panel too. */
        [data-testid="stDialog"] [role="dialog"] > div > [data-testid="stVerticalBlock"]
          > [data-testid="stLayoutWrapper"] {
          flex:1 1 auto!important; min-height:0!important;
          display:flex!important; flex-direction:column!important; overflow:hidden!important;
        }
        [data-testid="stDialog"] [role="dialog"] > div > [data-testid="stVerticalBlock"]
          > [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {
          flex:1 1 auto!important; min-height:0!important;
          display:flex!important; flex-direction:column!important;
          overflow:hidden!important; gap:10px!important;
        }
        /* Header, toolbar and composer keep their size... */
        [data-testid="stDialog"] [role="dialog"] [data-testid="stLayoutWrapper"] { flex-shrink:0; }
        /* ...except the wrapper that owns the transcript, which grows */
        [data-testid="stDialog"] [role="dialog"] [data-testid="stLayoutWrapper"]:has(> [data-testid="stVerticalBlock"].st-key-legal_chat_transcript) {
          flex:1 1 auto!important; min-height:0!important; height:auto!important;
          display:flex!important; flex-direction:column!important; overflow:hidden!important;
        }

        /* Header subtitle band */
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_subtitle {
          padding:0 18px 6px; border-bottom:1px solid #EEF2F7;
        }
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_subtitle p { color:#475569!important; }

        /* Expand toggle: ghost icon anchored next to the close button */
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_expand {
          position:absolute!important; top:10px!important; right:52px!important;
          margin:0!important; padding:0!important; z-index:20;
        }
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_expand button {
          width:36px!important; height:36px!important; min-height:36px!important;
          padding:0!important; border-radius:999px!important; border:none!important;
          background:transparent!important; color:var(--chat-secondary)!important;
          box-shadow:none!important; display:flex!important; align-items:center!important;
          justify-content:center!important;
        }
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_expand button:hover {
          background:#F1F5F9!important; color:var(--chat-text)!important;
        }
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_expand button p { font-size:0!important; }
        [data-testid="stDialog"] [role="dialog"] > h2 { padding-right:110px!important; }

        /* Thinking indicator: light row of pulsing dots above the composer */
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_thinking { flex-shrink:0; }
        .legal-thinking {
          display:flex; align-items:center; gap:5px;
          padding:4px 18px 2px; color:var(--chat-secondary); font-size:.85rem;
        }
        .legal-thinking-dot {
          width:6px; height:6px; border-radius:50%; background:var(--chat-secondary);
          animation:legalThinkingBlink 1.2s infinite ease-in-out;
        }
        .legal-thinking-dot:nth-child(2) { animation-delay:.2s; }
        .legal-thinking-dot:nth-child(3) { animation-delay:.4s; }
        .legal-thinking-label { margin-left:4px; }
        @keyframes legalThinkingBlink {
          0%,80%,100% { opacity:.25; transform:scale(.85); }
          40% { opacity:1; transform:scale(1); }
        }

        /* Transcript: the only scrollable area (flex-1 + min-height:0).
           The key class lands on the scrollable stVerticalBlock itself. */
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_transcript {
          flex:1 1 auto!important; min-height:0!important; height:auto!important;
          max-height:100%!important; overflow-y:auto!important;
          overscroll-behavior:contain; padding-right:6px;
        }

        /* Light text colors for everything rendered inside the panel */
        [data-testid="stDialog"] [role="dialog"] h1,
        [data-testid="stDialog"] [role="dialog"] h2,
        [data-testid="stDialog"] [role="dialog"] h3,
        [data-testid="stDialog"] [role="dialog"] h4 { color:var(--chat-text)!important; }
        [data-testid="stDialog"] [role="dialog"] [data-testid="stMarkdownContainer"] { color:var(--chat-text)!important; }
        [data-testid="stDialog"] [role="dialog"] [data-testid="stMarkdownContainer"] p { color:var(--chat-text)!important; }
        [data-testid="stDialog"] [role="dialog"] [data-testid="stCaptionContainer"] { color:var(--chat-muted)!important; }
        [data-testid="stDialog"] [role="dialog"] [data-testid="stCaptionContainer"] p { color:var(--chat-muted)!important; }
        [data-testid="stDialog"] [role="dialog"] code { background:#F1F5F9; color:var(--chat-text); }
        [data-testid="stDialog"] [role="dialog"] a { color:var(--legal-primary); }

        /* Suggested questions: light cards with a trailing arrow */
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_suggestion_0 button,
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_suggestion_1 button,
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_suggestion_2 button {
          background:#FFFFFF!important; color:#1E293B!important;
          border:1px solid #E2E8F0!important; border-radius:12px!important;
          box-shadow:none!important; text-align:left!important;
          padding:10px 14px!important; font-weight:500!important; min-height:42px!important;
          display:flex!important; align-items:center!important;
        }
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_suggestion_0 button::after,
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_suggestion_1 button::after,
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_suggestion_2 button::after {
          content:"→"; margin-left:auto; padding-left:8px; color:#94A3B8; font-weight:400;
        }
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_suggestion_0 button:hover,
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_suggestion_1 button:hover,
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_suggestion_2 button:hover {
          background:#F8FAFC!important; color:#0F172A!important; border-color:#CBD5E1!important;
        }

        /* Chat bubbles: user RIGHT (navy tint), bot LEFT (slate) */
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_transcript {
          gap:10px!important;
        }
        [data-testid="stDialog"] [role="dialog"] [data-testid="stChatMessage"] {
          background:transparent!important; border:none!important;
          padding:2px 0!important; gap:8px!important;
        }
        [data-testid="stDialog"] [role="dialog"] [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
          min-width:0; flex:0 1 auto!important;
        }
        /* User: row-reverse puts the avatar on the right, bubble right */
        [data-testid="stDialog"] [role="dialog"] [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
          flex-direction:row-reverse!important;
        }
        [data-testid="stDialog"] [role="dialog"] [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
          max-width:78%!important;
          background:rgba(15,42,68,.06)!important;
          border:1px solid rgba(15,42,68,.14)!important;
          border-radius:16px 16px 4px 16px!important;
          padding:9px 14px!important;
        }
        /* Bot: avatar left, slate bubble holding answer + sources */
        [data-testid="stDialog"] [role="dialog"] [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
          max-width:88%!important;
          background:#F8FAFC!important;
          border:1px solid #E2E8F0!important;
          border-radius:16px 4px 16px 16px!important;
          padding:10px 14px!important;
        }
        [data-testid="stDialog"] [role="dialog"] [data-testid="stChatMessage"] [data-testid="stExpander"] {
          background:#FFFFFF!important; border:1px solid #E2E8F0!important;
          border-radius:10px!important;
        }
        [data-testid="stDialog"] [role="dialog"] [data-testid="stExpander"] {
          border-color:var(--chat-border); background:var(--chat-bg);
        }
        [data-testid="stDialog"] [role="dialog"] [data-testid="stExpander"] summary
          { color:var(--chat-text); }

        /* Popover (history) is also body-mounted: give it light colors */
        [data-testid="stPopoverBody"] { background:var(--chat-bg)!important; color:var(--chat-text)!important; }
        [data-testid="stPopoverBody"] p { color:var(--chat-text)!important; }

        /* ---- Composer: one rounded box, ChatGPT style ---- */
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_composer {
          position:relative; background:#FFFFFF;
          border:1px solid #CBD5E1; border-radius:16px;
          box-shadow:0 1px 2px rgba(15,23,42,.06);
          padding:2px 12px 8px; flex-shrink:0;
          transition:border-color .15s ease, box-shadow .15s ease;
        }
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_composer:focus-within {
          border-color:var(--chat-accent);
          box-shadow:0 0 0 3px rgba(239,68,68,.10);
        }
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_composer [data-testid="stForm"] {
          border:none; background:transparent; padding:0;
        }
        /* Streamlit marks element containers position:relative (tooltip
           anchors). Neutralize inside the composer so the absolutely placed
           send button anchors to the composer box itself. */
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_composer [data-testid="stElementContainer"] {
          position:static!important;
        }
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_composer [data-testid="stTextArea"],
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_composer [data-testid="stTextAreaRootElement"],
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_composer [data-baseweb="textarea"] {
          background:#FFFFFF!important; border:none!important; box-shadow:none!important;
        }
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_composer textarea {
          background:transparent!important; color:var(--chat-text)!important;
          caret-color:var(--chat-text); resize:none!important;
          min-height:44px!important; max-height:130px!important;
          overflow-y:hidden;
        }
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_composer textarea::placeholder {
          color:var(--chat-muted)!important;
        }
        /* Send: round accent icon-only button, right-aligned in the form's
           last row (flow layout keeps it stable across responsive DOM). */
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_composer [data-testid="stElementContainer"]:has([data-testid="stFormSubmitButton"]) {
          margin-left:auto!important; flex:none!important;
        }
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_composer [data-testid="stFormSubmitButton"] {
          display:flex!important; justify-content:flex-end!important;
          margin:0!important; padding:0 2px 2px 0!important;
        }
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_composer [data-testid="stFormSubmitButton"] button {
          width:38px!important; min-width:38px!important; max-width:38px!important;
          height:38px!important; min-height:38px!important;
          padding:0!important; border-radius:999px!important; border:none!important;
          background:var(--chat-accent)!important; color:#FFFFFF!important;
          box-shadow:none!important; display:flex!important; flex:none!important;
          align-items:center!important; justify-content:center!important;
        }
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_composer [data-testid="stFormSubmitButton"] button:hover {
          background:#DC2626!important; color:#FFFFFF!important;
        }
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_composer [data-testid="stFormSubmitButton"] button p { font-size:0!important; }
        /* Ghost icon actions (history, new chat) share the composer bottom
           row with the send button: pulled up onto the same visual line. */
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_actions {
          margin-top:-52px!important; padding:6px 64px 0 4px;
          position:relative; z-index:1;
        }
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_actions button {
          min-height:34px!important; min-width:34px!important; padding:4px 8px!important;
          background:transparent!important; border:none!important; box-shadow:none!important;
          color:var(--chat-secondary)!important; border-radius:8px!important;
          font-size:13px!important;
        }
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_actions button:hover {
          background:#F1F5F9!important; color:var(--chat-text)!important;
        }
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_actions button p { font-size:0!important; }
        [data-testid="stDialog"] [role="dialog"] .st-key-legal_chat_actions [data-testid="stIconMaterial"] { font-size:1.25rem; }

        /* ---- Responsive ---- */
        @media (min-width:768px) and (max-width:1023px) {
          [data-testid="stDialog"] [role="dialog"] {
            width:380px!important;
            height:min(620px,calc(100dvh - 32px))!important;
            max-height:calc(100dvh - 32px)!important;
          }
        }
        @media (max-width:767px) {
          .st-key-legal_assistant_trigger {
            right:16px; bottom:calc(16px + env(safe-area-inset-bottom));
          }
          [data-testid="stAppViewContainer"] [data-testid="stButtonGroup"] [role="radiogroup"] {
            flex-wrap:wrap;
          }
          [data-testid="stAppViewContainer"] [data-testid="stButtonGroup"] [role="radiogroup"] button[role="radio"] {
            flex:1 1 30%; font-size:.82rem;
          }
          [data-testid="stDialog"] [role="dialog"] {
            left:12px!important; right:12px!important;
            bottom:calc(12px + env(safe-area-inset-bottom))!important;
            width:auto!important; max-width:none!important;
            height:calc(100dvh - 24px - env(safe-area-inset-bottom))!important;
            max-height:calc(100dvh - 24px - env(safe-area-inset-bottom))!important;
          }
          [data-testid="stDialog"] [role="dialog"] [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
            max-width:88%!important;
          }
          [data-testid="stDialog"] [role="dialog"] [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
            max-width:92%!important;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_legal_portal(
    docs,
    chunks,
    questions: list[dict],
    dataset_version: str,
    *,
    on_admin: Callable[[], None],
    on_question: Callable[[str], None],
) -> bool:
    """Render the public portal and return whether the assistant should open."""
    _portal_styles()
    st.session_state.setdefault("user_portal_page", "home")
    assistant_requested = False

    # Page order (render order, no CSS tricks): Hero first, then the system
    # header, dataset badge and navigation.
    st.html(
        """<section class="legal-hero"><h2>HỖ TRỢ TRA CỨU VÀ TƯ VẤN THÔNG TIN PHÁP LUẬT</h2>
        <p>Tìm kiếm văn bản, tra cứu quy định và nhận hỗ trợ từ Trợ lý pháp lý AI.</p></section>"""
    )

    st.html(
        """<section class="legal-brand"><h1>⚖ HỆ THỐNG TƯ VẤN PHÁP LUẬT</h1>
        <p>Nền tảng tra cứu và hỗ trợ thông tin pháp luật thông minh</p></section>"""
    )
    st.caption(f"Kho dữ liệu: `{dataset_version}`")

    portal_pages = {
        "home": ":material/home: Trang chủ",
        "fields": ":material/category: Lĩnh vực pháp luật",
        "documents": ":material/article: Văn bản pháp luật",
        "questions": ":material/help: Hỏi đáp pháp luật",
        "search": ":material/search: Tra cứu",
        "about": ":material/info: Giới thiệu",
        "admin": ":material/shield: Quản trị viên",
    }

    def _on_portal_nav_change() -> None:
        """The admin item switches the app role; it is not a portal page."""
        if st.session_state.user_portal_page == "admin":
            st.session_state.user_portal_page = "home"
            on_admin()

    page = st.segmented_control(
        "Điều hướng cổng thông tin", list(portal_pages),
        format_func=lambda target: portal_pages[target], key="user_portal_page",
        selection_mode="single", width="stretch", label_visibility="collapsed",
        on_change=_on_portal_nav_change,
    )
    if page == "admin":  # defensive: never render a portal branch for admin
        page = "home"

    def open_search(query: str) -> None:
        st.session_state.portal_search_query = query
        st.session_state.user_portal_page = "search"

    if page == "home":
        with st.form("portal_hero_search", border=False):
            query = st.text_input("Tra cứu pháp luật",
                placeholder="Nhập nội dung, văn bản hoặc vấn đề pháp luật cần tìm...",
                label_visibility="collapsed")
            submitted = st.form_submit_button("Tra cứu", icon=":material/search:", type="primary")
        if submitted and query.strip():
            open_search(query.strip())
            st.rerun()

        st.subheader("Lĩnh vực pháp luật", anchor=False)
        _render_fields(open_search, "home")
        st.subheader("Câu hỏi pháp luật được quan tâm", anchor=False)
        for index, item in enumerate(questions[:5], start=1):
            with st.container(border=True):
                st.caption(f"Câu hỏi {index}")
                st.write(item["text"])
                if st.button("Hỏi Trợ lý pháp lý AI", key=f"popular_question_{index}",
                             on_click=on_question, args=(item["text"],)):
                    assistant_requested = True
        st.subheader("Văn bản trong kho dữ liệu", anchor=False)
        st.dataframe(pd.DataFrame([{"Văn bản":doc.title,
            "Số article":doc.metadata.get("n_articles",0), "Nguồn":doc.source}
            for doc in docs[:10]]), hide_index=True, width="stretch")

    elif page == "fields":
        st.header("Lĩnh vực pháp luật")
        st.caption("Chọn lĩnh vực để tra cứu nội dung liên quan trong kho văn bản.")
        _render_fields(open_search, "all")

    elif page == "documents":
        st.header("Văn bản pháp luật")
        query = st.text_input("Tìm theo số hiệu văn bản",
                              placeholder="Ví dụ: luật, nghị định, thông tư...")
        filtered = [doc for doc in docs if not query or query.casefold() in doc.title.casefold()]
        st.caption(f"Tìm thấy {len(filtered):,} văn bản")
        selected = st.selectbox("Chọn văn bản", filtered, format_func=lambda doc:doc.title,
                                index=0 if filtered else None,
                                placeholder="Không có văn bản phù hợp")
        if selected:
            with st.container(border=True):
                st.subheader(selected.title)
                st.caption(f"Mã văn bản: `{selected.document_id}` · "
                           f"{selected.metadata.get('n_articles',0):,} article")
                st.text_area("Nội dung văn bản", selected.raw_text, height=500, disabled=True)

    elif page == "questions":
        st.header("Hỏi đáp pháp luật")
        query = st.text_input("Tìm câu hỏi", placeholder="Nhập tình huống pháp luật...")
        filtered = [item for item in questions if not query
                    or query.casefold() in item["text"].casefold()][:30]
        for index, item in enumerate(filtered):
            with st.container(border=True):
                st.write(item["text"])
                if st.button("Nhận hỗ trợ từ Trợ lý AI", key=f"question_assistant_{index}",
                             on_click=on_question, args=(item["text"],)):
                    assistant_requested = True

    elif page == "search":
        st.header("Tra cứu thông tin pháp luật")
        st.session_state.setdefault("portal_search_query", "")
        query = st.text_input("Nội dung cần tìm", key="portal_search_query")
        normalized = query.strip().casefold()
        if normalized:
            matching = [chunk for chunk in chunks if normalized in chunk.text.casefold()
                        or normalized in chunk.law_id.casefold()][:50]
            st.caption(f"Tìm thấy {len(matching):,} kết quả đầu tiên")
            if not matching:
                st.info("Không tìm thấy nội dung phù hợp trong kho dữ liệu.")
            for result in matching:
                with st.container(border=True):
                    st.subheader(result.title or f"Điều {result.article_id}")
                    st.caption(f"Văn bản {result.law_id} · Điều {result.article_id}")
                    st.write(result.text[:800])
        else:
            st.info("Nhập từ khóa để tra cứu trong toàn bộ kho văn bản.")

    elif page == "about":
        st.header("Giới thiệu hệ thống")
        st.write("Hệ thống hỗ trợ tra cứu văn bản pháp luật và cung cấp Trợ lý pháp lý "
                 "AI sử dụng Retrieval-Augmented Generation. Câu trả lời được tổng hợp "
                 "từ các nguồn trong kho dữ liệu và đi kèm căn cứ tham khảo.")
        st.warning("Nội dung nhằm mục đích tham khảo, không thay thế tư vấn của luật sư "
                   "hoặc cơ quan có thẩm quyền.")

    st.caption("© Hệ thống tư vấn pháp luật · Dữ liệu phục vụ nghiên cứu NLP và RAG")
    if st.button(
        "Trợ lý pháp lý AI",
        icon=":material/gavel:",
        type="primary",
        key="legal_assistant_trigger",
        help="Mở Trợ lý pháp lý AI",
    ):
        assistant_requested = True
    return assistant_requested


def _render_fields(open_search: Callable[[str], None], key_prefix: str) -> None:
    for row_start in range(0, len(LEGAL_FIELDS), 4):
        columns = st.columns(4)
        for column, (field, icon, description) in zip(columns, LEGAL_FIELDS[row_start:row_start+4]):
            with column.container(border=True, height="stretch"):
                st.markdown(f"### {icon} {field}")
                st.caption(description)
                st.button("Tra cứu", key=f"{key_prefix}_field_{field}",
                          on_click=open_search, args=(field,), width="stretch")
