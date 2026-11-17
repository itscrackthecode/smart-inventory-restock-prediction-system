"""Shared application styling and page framing."""

from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root { --forest: #145044; --forest-deep: #0d392f; --ivory: #f6f7f1;
                --mint: #dcebe2; --ink: #1c2925; --muted: #63736d; --line: #dfe7e1; }
        [data-testid="stAppViewContainer"] { background: var(--ivory); color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(246,247,241,.94); }
        [data-testid="stSidebar"] { background: #102f29; }
        [data-testid="stSidebar"] * { color: #f3f5ed; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #d3dfd9; }
        h1, h2, h3 { color: var(--forest-deep); letter-spacing: -.025em; }
        .stockwise-hero { background: linear-gradient(116deg, #103c32 0%, #1d5b4a 100%);
            color: #f4f7f1; padding: clamp(1.5rem, 4vw, 3rem); border-radius: 20px;
            margin: .25rem 0 1.3rem; box-shadow: 0 12px 30px rgba(13,57,47,.12); }
        .stockwise-hero h1 { color: #fff; font-size: clamp(2rem, 4vw, 3.15rem);
            line-height: 1.06; letter-spacing: -.045em; margin: 0 0 .75rem; max-width: 800px; }
        .stockwise-hero p { color: #dcebe2; font-size: 1.05rem; line-height: 1.55;
            margin: 0; max-width: 780px; }
        .stockwise-cta { display: inline-flex; gap: .55rem; align-items: center;
            color: var(--forest-deep) !important; background: #fff; border: 1px solid var(--line);
            border-radius: 9px; padding: .65rem .9rem; margin: 0 0 1.1rem;
            font-weight: 650; text-decoration: none !important; transition: transform .16s ease, box-shadow .16s ease; }
        .stockwise-cta:hover { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(13,57,47,.09); }
        .source-note { color: var(--muted); font-size: .82rem; margin: -.45rem 0 1rem; }
        [data-testid="stMetric"] { background: #fff; border: 1px solid var(--line);
            border-radius: 12px; padding: 16px 18px; box-shadow: 0 2px 8px rgba(13,57,47,.045); }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMetricValue"] { color: var(--forest-deep); font-weight: 650; }
        div[data-testid="stForm"] { background: #fff; border: 1px solid var(--line);
            border-radius: 14px; padding: 18px; }
        .page-eyebrow { color: #507168; font-size: .78rem; text-transform: uppercase;
            letter-spacing: .11em; font-weight: 700; margin-bottom: .25rem; }
        .page-subtitle { color: var(--muted); margin-top: -.35rem; margin-bottom: 1.1rem; }
        .section-note { color: var(--muted); font-size: .88rem; margin-top: -.45rem; }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; }
        [data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"] { border-radius: 12px; }
        div.stButton > button[kind="primaryFormSubmit"], div.stButton > button[kind="primary"] {
            background: var(--forest); border-color: var(--forest); color: white; border-radius: 9px; }
        div.stButton > button[kind="primaryFormSubmit"]:hover, div.stButton > button[kind="primary"]:hover {
            background: var(--forest-deep); border-color: var(--forest-deep); }
        @media (max-width: 700px) { .block-container { padding-left: 1rem; padding-right: 1rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, *, eyebrow: str = "SMART INVENTORY") -> None:
    st.markdown(f'<div class="page-eyebrow">{eyebrow}</div><h1>{title}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="page-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def section_heading(title: str, explanation: str | None = None) -> None:
    st.subheader(title)
    if explanation:
        st.markdown(f'<p class="section-note">{explanation}</p>', unsafe_allow_html=True)
