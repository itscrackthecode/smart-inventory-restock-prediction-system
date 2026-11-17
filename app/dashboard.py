"""Streamlit application entry point."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.layout import apply_theme
from app.pages import about, analytics, inventory_explorer, model_performance, overview, predictions, sales_intelligence
from app.services.startup import ensure_database_ready


PAGE_SPECS = {
    "WORKSPACE": [
        (overview.render, "Overview", ":material/space_dashboard:", "overview", True),
        (sales_intelligence.render, "Sales intelligence", ":material/query_stats:", "sales-intelligence", False),
        (inventory_explorer.render, "Inventory explorer", ":material/inventory_2:", "inventory", False),
        (analytics.render, "Analytics", ":material/monitoring:", "analytics", False),
    ],
    "INTELLIGENCE": [
        (predictions.render, "Restock predictions", ":material/online_prediction:", "predictions", False),
        (model_performance.render, "Model performance", ":material/insights:", "model-performance", False),
    ],
    "PROJECT": [
        (about.render, "About & methodology", ":material/menu_book:", "about", False),
    ],
}


def main() -> None:
    st.set_page_config(
        page_title="Smart Inventory",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()
    st.sidebar.markdown(
        "<div style='font-size:1.18rem;font-weight:750;letter-spacing:-.02em;color:#f5f7f1'>"
        "◈ STOCKWISE</div><div style='font-size:.78rem;color:#bdcec7;margin:.2rem 0 1rem'>"
        "Inventory intelligence</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Synthetic inventory · Decision support")
    try:
        startup_report = ensure_database_ready()
    except (OSError, sqlite3.Error, ValueError, RuntimeError) as exc:
        st.error(f"The inventory database could not be prepared: {exc}")
        st.stop()
    if startup_report.seeded_rows:
        st.sidebar.caption(
            f"Initialized demo data · {startup_report.seeded_rows:,} inventory observations"
        )

    navigation_pages = {}
    for section, specs in PAGE_SPECS.items():
        navigation_pages[section] = []
        for renderer, title, icon, url_path, is_default in specs:
            page = st.Page(renderer, title=title, icon=icon, url_path=url_path, default=is_default)
            navigation_pages[section].append(page)
    navigation = st.navigation(navigation_pages, position="sidebar")
    navigation.run()


# Streamlit's script runner executes this file inside its runtime rather than
# invoking it as a conventional Python module entry point. Keep ordinary
# imports side-effect free for tests, but start the app whenever a Streamlit
# runtime is active.
if __name__ == "__main__" or st.runtime.exists():
    main()
