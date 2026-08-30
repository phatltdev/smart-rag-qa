"""Shared design system for the Smart RAG QA application.

Single source of truth for the visual language used by both the public legal
portal and the admin dashboard: a light "Legal Portal" theme (navy primary,
gold accent, red primary action) so switching between the two surfaces reads
as one product instead of two different applications.

The canonical palette mirrors ``src/ui/legal_portal.py``. Widgets themselves
keep their Streamlit identity (theme tokens in ``.streamlit/config.toml``);
this module layers the shared component styles on top.
"""
from __future__ import annotations

import html

import streamlit as st

# Canonical palette — keep in sync with the portal variables in legal_portal.py.
DESIGN_TOKENS = {
    "primary": "#0F2A44",          # navy — headings, active navigation
    "secondary": "#1E3A5F",        # darker navy surface
    "background": "#F8FAFC",       # app background
    "card": "#FFFFFF",             # cards / sidebar surface
    "border": "#E2E8F0",           # hairline borders
    "text": "#0F172A",             # primary text
    "text_secondary": "#64748B",   # secondary text
    "text_muted": "#94A3B8",       # muted labels
    "accent": "#B88732",           # gold — active highlights (matches portal)
    "danger": "#EF4444",           # red — primary action (matches portal)
    "success": "#16A34A",
    "warning": "#F59E0B",
}

_ADMIN_THEME_CSS = """
<style>
:root {
  --sys-primary:#0F2A44;
  --sys-primary-soft:rgba(15,42,68,.07);
  --sys-secondary:#1E3A5F;
  --sys-bg:#F8FAFC;
  --sys-card:#FFFFFF;
  --sys-border:#E2E8F0;
  --sys-text:#0F172A;
  --sys-text-2:#64748B;
  --sys-text-muted:#94A3B8;
  --sys-accent:#B88732;
  --sys-danger:#EF4444;
  --sys-success:#16A34A;
  --sys-warning:#F59E0B;
  --sys-surface:#F1F5F9;
}

/* ===== Layout: light background, bounded content width ================== */
[data-testid="stAppViewContainer"] { background: var(--sys-bg); }
[data-testid="stMainBlockContainer"] {
  max-width: 1600px;
  margin-inline: auto;
  padding-top: 1.5rem !important;
}

/* ===== Sidebar: modern compact admin navigation ========================= */
/* Width: 248px expanded (content area reflows via Streamlit flex layout). */
[data-testid="stSidebar"] {
  background: var(--sys-card) !important;
  border-right: 1px solid var(--sys-border) !important;
  box-shadow: 1px 0 2px rgba(15, 42, 68, 0.04);
  width: 248px !important;
  min-width: 248px !important;
  max-width: 248px !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
  color: var(--sys-text-2);
}

/* Sticky-footer column layout: navigation grows, account block sits last. */
[data-testid="stSidebarContent"] {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}
[data-testid="stSidebarUserContent"] {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}
[data-testid="stSidebarUserContent"] > [data-testid="stVerticalBlock"] {
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  min-height: 0;
}
.st-key-admin_sidebar_footer {
  margin-top: auto;
  padding-top: 10px;
  border-top: 1px solid var(--sys-border);
}
.st-key-admin_sidebar_footer button {
  min-height: 38px;
  border-radius: 9px;
}

/* Brand row: 38px logo mark + compact name; collapse toggle sits inline. */
.admin-brand { display: flex; align-items: center; gap: 10px; padding: 2px 0 10px; min-width: 0; }
.admin-brand-mark {
  display: flex; align-items: center; justify-content: center;
  width: 36px; height: 36px; border-radius: 10px; flex: none;
  background: var(--sys-primary); color: #FFFFFF; font-size: 1.15rem;
}
.admin-brand-text { min-width: 0; }
.admin-brand-name {
  color: var(--sys-primary); font-weight: 600; font-size: .95rem;
  line-height: 1.15; white-space: nowrap;
}
.admin-brand-sub {
  color: var(--sys-text-muted); font-size: .7rem; margin-top: 1px;
  white-space: nowrap;
}

/* Primary navigation: each option renders as a BLOCK row, so the group
   heading ::before sits on its own full-width row and never squeezes the
   item pill beside it (that flex layout was what wrapped menu labels).
   Item pill = the content div; label text is forced to a single line. */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioGroup"] { gap: 4px; }
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioGroup"]
  > label[data-testid="stRadioOption"] {
  display: block;
  width: 100%;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioGroup"]
  > label[data-testid="stRadioOption"] > div {
  display: flex;
  /* Streamlit renders this wrapper as flex-column with align-items:center,
     which centers each row by its own text length and mis-aligns items.
     Stretch the row to full width so every item shares one left axis. */
  flex-direction: column;
  align-items: stretch;
  min-height: 40px;
  width: 100%;
  box-sizing: border-box;
  border-radius: 9px;
  padding: 8px 10px;
  transition: background 150ms ease, color 150ms ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioGroup"]
  > label[data-testid="stRadioOption"] > div > div {
  min-width: 0;
  display: flex;
  align-items: center;
  justify-content: flex-start;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioGroup"]
  > label[data-testid="stRadioOption"]:hover > div {
  background: #F8FAFC;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioGroup"]
  > label[data-testid="stRadioOption"] > div p {
  flex: 1;
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: #334155;
  font-weight: 500;
  font-size: .9rem;
}
/* Fixed icon slot: equal 20px for every glyph so labels share one text axis. */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioGroup"]
  > label[data-testid="stRadioOption"] p span[role="img"] {
  width: 20px;
  flex: none;
  display: inline-flex;
  justify-content: center;
  overflow: hidden;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioGroup"]
  > label[data-testid="stRadioOption"]:has(input[type="radio"]:checked) > div {
  background: var(--sys-surface);
  box-shadow: inset 3px 0 0 0 var(--sys-accent);
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioGroup"]
  > label[data-testid="stRadioOption"]:has(input[type="radio"]:checked) > div p {
  color: var(--sys-primary) !important;
  font-weight: 600;
}

/* Group headings — one standalone row above each menu group (menu order:
   overview(1) | RAG(2..4) | system(5..7)). */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioGroup"]
  > label[data-testid="stRadioOption"]::before {
  display: block;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--sys-text-muted);
  padding: 0 12px;
  margin-bottom: 6px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioGroup"]
  > label[data-testid="stRadioOption"]:nth-child(1)::before {
  content: "TỔNG QUAN";
  margin-top: 6px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioGroup"]
  > label[data-testid="stRadioOption"]:nth-child(2)::before {
  content: "RAG";
  margin-top: 18px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioGroup"]
  > label[data-testid="stRadioOption"]:nth-child(5)::before {
  content: "HỆ THỐNG";
  margin-top: 18px;
}

/* ===== Page header pattern ============================================== */
.admin-page-header {
  border-bottom: 1px solid var(--sys-border);
  padding-bottom: 12px;
}
.admin-page-header h1 {
  color: var(--sys-primary);
  font-size: 1.55rem;
  font-weight: 700;
  margin: 0;
}
.admin-page-header p {
  color: var(--sys-text-2);
  margin: 4px 0 0;
  font-size: .93rem;
}

/* ===== Cards & metrics ==================================================
   Streamlit >=1.62 renders bordered containers as plain stVerticalBlock
   without a dedicated testid, so cards carry a `key="card_*"` in app code
   and are matched via the documented st-key-<key> class. */
div[class*="st-key-card_"] {
  background: var(--sys-card);
  border: 1px solid var(--sys-border) !important;
  border-radius: 14px !important;
  box-shadow: 0 1px 2px rgba(15,42,68,.05);
}
[data-testid="stMetric"] { border-radius: 14px; }
[data-testid="stMetricLabel"] p {
  color: var(--sys-text-2);
  font-weight: 500;
  font-size: .85rem;
}
[data-testid="stMetricValue"] { color: var(--sys-primary); font-weight: 700; }

/* ===== Buttons ========================================================== */
.stButton > button { border-radius: 10px; font-weight: 500; }
.stButton > button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"] {
  border-color: var(--sys-border);
  color: #334155;
  background: var(--sys-card);
}
.stButton > button[kind="secondary"]:hover,
.stButton > button[data-testid="baseButton-secondary"]:hover {
  background: var(--sys-surface);
  border-color: #CBD5E1;
  color: var(--sys-primary);
}
.admin-header-actions button { white-space: nowrap; }

/* ===== Form controls ==================================================== */
[data-baseweb="input"]:focus,
[data-baseweb="textarea"]:focus {
  border-color: var(--sys-accent) !important;
}
[data-baseweb="input"]:focus-visible,
[data-baseweb="textarea"]:focus-visible {
  box-shadow: 0 0 0 3px rgba(184,135,50,.14) !important;
}

/* ===== Tables =========================================================== */
[data-testid="stTable"] {
  border: 1px solid var(--sys-border);
  border-radius: 12px;
  overflow: hidden;
  background: var(--sys-card);
}
[data-testid="stTable"] thead th {
  background: #F8FAFC;
  color: var(--sys-text-2);
  font-weight: 600;
}
[data-testid="stTable"] td,
[data-testid="stTable"] th {
  border-color: var(--sys-border) !important;
  color: var(--sys-text);
}
[data-testid="stTable"] tbody tr:hover td { background: #F8FAFC; }

/* ===== Expanders / status =============================================== */
[data-testid="stExpander"] {
  background: var(--sys-card);
  border: 1px solid var(--sys-border) !important;
  border-radius: 12px;
}
[data-testid="stExpander"] summary p {
  color: var(--sys-text);
  font-weight: 500;
}

/* ===== Segmented control / pills (RAG pipeline steps) =================== */
[data-testid="stMain"] [data-testid="stButtonGroup"] [role="radiogroup"] {
  background: var(--sys-card);
  border: 1px solid var(--sys-border);
  border-radius: 12px;
  padding: 4px;
  gap: 2px;
  box-shadow: 0 1px 2px rgba(15,42,68,.05);
}
[data-testid="stMain"] [data-testid="stButtonGroup"] [role="radiogroup"] button {
  border-radius: 9px;
  min-height: 40px;
  color: var(--sys-text-2);
  font-weight: 500;
  background: transparent;
  border: none;
}
[data-testid="stMain"] [data-testid="stButtonGroup"] [role="radiogroup"] button:hover {
  background: var(--sys-surface);
  color: var(--sys-primary);
}
[data-testid="stMain"] [data-testid="stButtonGroup"] [role="radiogroup"] button[aria-checked="true"] {
  background: var(--sys-primary-soft);
  color: var(--sys-primary);
  font-weight: 600;
  box-shadow: inset 0 -2px 0 0 var(--sys-accent);
}

/* ===== Chat playground ================================================== */
.chat-empty-title {
  text-align: center;
  color: var(--sys-primary);
  font-size: 1.45rem;
  font-weight: 700;
  margin-top: 2rem;
}
.chat-empty-sub {
  text-align: center;
  color: var(--sys-text-2);
  margin: 6px 0 1rem;
}
.st-key-suggestion_0 button,
.st-key-suggestion_1 button,
.st-key-suggestion_2 button,
.st-key-suggestion_3 button {
  background: var(--sys-card) !important;
  color: #1E293B !important;
  border: 1px solid var(--sys-border) !important;
  border-radius: 12px !important;
  box-shadow: none !important;
  text-align: left !important;
  min-height: 54px !important;
  display: flex !important;
  align-items: center !important;
  font-weight: 500 !important;
  padding: 10px 14px !important;
}
.st-key-suggestion_0 button::after,
.st-key-suggestion_1 button::after,
.st-key-suggestion_2 button::after,
.st-key-suggestion_3 button::after {
  content: "\\2192";
  margin-left: auto;
  padding-left: 8px;
  color: var(--sys-text-muted);
  font-weight: 400;
}
.st-key-suggestion_0 button:hover,
.st-key-suggestion_1 button:hover,
.st-key-suggestion_2 button:hover,
.st-key-suggestion_3 button:hover {
  background: #F8FAFC !important;
  color: #0F172A !important;
  border-color: #CBD5E1 !important;
}

/* ===== Responsive ======================================================= */
@media (max-width: 767px) {
  [data-testid="stHorizontalBlock"]:has(.admin-page-header) {
    flex-wrap: wrap;
  }
  [data-testid="stHorizontalBlock"]:has(.admin-page-header) > div {
    flex: 1 1 100% !important;
    min-width: 100% !important;
  }
  .admin-page-header h1 { font-size: 1.25rem; }
}
</style>
"""

# Icon-rail mode (68px): navigation collapses to icons and the active item
# keeps its tint + accent. The sidebar holds navigation only — all page
# configuration lives in the main content.
_ADMIN_SIDEBAR_COMPACT_CSS = """
<style>
[data-testid="stSidebar"] {
  width: 68px !important;
  min-width: 68px !important;
  max-width: 68px !important;
}

/* Brand row: logo only, centered above the rail. */
.admin-brand { justify-content: center; padding: 2px 0 10px; }
.admin-brand-text { display: none; }
.admin-brand-mark { width: 34px; height: 34px; font-size: 1.05rem; }

/* Nav items: icon-only, centered, active state preserved. With the pill
   wrapper stretched (expanded mode), the icon is re-centered via the
   column cross axis + centered inner row. */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioGroup"]
  > label[data-testid="stRadioOption"]::before { content: none !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioGroup"]
  > label[data-testid="stRadioOption"] > div {
  align-items: center;
  padding: 8px 0;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioGroup"]
  > label[data-testid="stRadioOption"] > div > div {
  justify-content: center;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioGroup"]
  > label[data-testid="stRadioOption"] p {
  font-size: 0 !important;
  min-width: 0;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioGroup"]
  > label[data-testid="stRadioOption"] p span[role="img"] {
  font-size: 1.3rem !important;
}

/* Footer: icon-only switch action. */
.st-key-admin_sidebar_footer { padding: 10px 4px 0; }
.st-key-admin_sidebar_footer [data-testid="stCaptionContainer"] { display: none; }
.st-key-admin_sidebar_footer button { min-width: 40px; padding: 8px 0 !important; }
.st-key-admin_sidebar_footer button p { font-size: 0 !important; }
.st-key-admin_sidebar_footer button p span[role="img"] { font-size: 1.25rem !important; }

/* Collapse toggle in the brand row: full-width icon hit target. */
[data-testid="stSidebar"] .stButton button { min-width: 32px; padding: 6px 0 !important; }
</style>
"""


def apply_admin_theme(compact: bool = False) -> None:
    """Inject the shared light design system (admin surface).

    Called once per admin rerun, before the admin layout renders. ``compact``
    switches the sidebar to the 68px icon rail. The public portal never calls
    this — it keeps its own portal stylesheet, so the two surfaces stay
    visually consistent yet independently styled.
    """
    st.markdown(_ADMIN_THEME_CSS, unsafe_allow_html=True)
    if compact:
        st.markdown(_ADMIN_SIDEBAR_COMPACT_CSS, unsafe_allow_html=True)


def admin_page_header(title: str, description: str | None = None):
    """Render the shared admin page header (title + description).

    Returns a right-aligned horizontal container the caller can fill with
    page-level action buttons (e.g. "Hội thoại mới", "Trang người dùng").
    """
    head, actions_col = st.columns(
        [0.6, 0.4], vertical_alignment="center", gap="medium"
    )
    with head:
        description_html = (
            f"<p>{html.escape(description)}</p>" if description else ""
        )
        st.markdown(
            f'<div class="admin-page-header"><h1>{html.escape(title)}</h1>'
            f"{description_html}</div>",
            unsafe_allow_html=True,
        )
    return actions_col.container(
        horizontal=True, horizontal_alignment="right", key=None
    )
