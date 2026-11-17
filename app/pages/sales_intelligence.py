"""Explore real public retail transactions separately from synthetic stock data."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from app.components.layout import page_header, section_heading
from app.config import PROJECT_ROOT
from src.uci_online_retail import external_data_paths, load_or_fetch_daily_sales


DATA_DIR = PROJECT_ROOT / "data" / "external"


def _load_local_sales() -> tuple[pd.DataFrame, dict]:
    return load_or_fetch_daily_sales(DATA_DIR)


def _daily_trend(data: pd.DataFrame, metric: str) -> alt.Chart:
    daily = data.groupby("date", as_index=False).agg(value=(metric, "sum"))
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date").reindex(pd.date_range(daily["date"].min(), daily["date"].max(), name="date"))
    daily["rolling_7d"] = daily["value"].rolling("7D", min_periods=1).mean()
    daily = daily.reset_index()
    y_title = "Units sold" if metric == "units_sold" else "Recorded sales value (£)"
    base = alt.Chart(daily).encode(
        x=alt.X("date:T", title="Transaction date"),
        tooltip=[alt.Tooltip("date:T", title="Date"),
                 alt.Tooltip("value:Q", title=y_title, format=",.0f")],
    )
    bars = base.mark_bar(color="#c3d8cd", opacity=0.65).encode(
        y=alt.Y("value:Q", title=y_title, scale=alt.Scale(zero=True)),
    )
    rolling = alt.Chart(daily).mark_line(color="#145044", strokeWidth=2.5).encode(
        x=alt.X("date:T", title="Transaction date"),
        y=alt.Y("rolling_7d:Q", title="7-day mean", scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("date:T", title="Date"),
                 alt.Tooltip("rolling_7d:Q", title="7-day mean", format=",.1f")],
    )
    return alt.layer(bars, rolling).resolve_scale(y="independent").properties(
        height=330, title="Recorded daily sales and 7-day mean"
    ).interactive()


def _rank_chart(data: pd.DataFrame, *, group: str, label: str, metric: str, title: str) -> alt.Chart:
    aggregations = {"value": (metric, "sum")}
    if group == "stock_code":
        aggregations["description"] = ("description", "first")
    ranked = data.groupby(group, as_index=False).agg(**aggregations)
    ranked = ranked.sort_values("value", ascending=False).head(15)
    if group == "stock_code":
        description = ranked["description"].fillna("Unnamed item").astype(str).str.slice(0, 34)
        ranked["product_label"] = description + " · " + ranked["stock_code"].astype(str)
        category_field = "product_label"
        tooltip_fields = [alt.Tooltip("stock_code:N", title="Product stock code"),
                          alt.Tooltip("description:N", title="Description")]
    else:
        category_field = group
        tooltip_fields = [alt.Tooltip(f"{group}:N", title=label)]
    return alt.Chart(ranked).mark_bar(color="#145044", cornerRadiusEnd=4).encode(
        x=alt.X("value:Q", title="Units sold", scale=alt.Scale(zero=True)),
        y=alt.Y(f"{category_field}:N", title=label, sort="-x"),
        tooltip=[*tooltip_fields,
                 alt.Tooltip("value:Q", title="Units sold", format=",.0f")],
    ).properties(height=390, title=title).interactive()


def render() -> None:
    page_header(
        "Sales intelligence",
        "Explore real historical retail transactions from UCI. Sales activity is useful demand context, but it does not tell us current stock or when a supplier replenished it.",
        eyebrow="PUBLIC DATA · UCI ONLINE RETAIL II",
    )
    paths = external_data_paths(DATA_DIR)
    if not paths["daily_sales"].is_file() or not paths["metadata"].is_file():
        st.markdown(
            "The public sales dataset is not cached in this app yet. Loading it downloads a 43.5 MB source archive, "
            "then creates a local daily product-and-country summary. This can take a few minutes the first time."
        )
        st.caption("This read-only import does not touch the synthetic, processed, engineered, or SQLite inventory data.")
        if st.button("Fetch UCI sales dataset", type="primary"):
            try:
                with st.spinner("Fetching and preparing UCI Online Retail II…"):
                    load_or_fetch_daily_sales(DATA_DIR)
                st.rerun()
            except (OSError, ValueError, TimeoutError, pd.errors.ParserError) as exc:
                st.error(f"The UCI dataset could not be loaded: {exc}")
        st.link_button("Open UCI dataset and license", "https://archive.ics.uci.edu/dataset/502/online+retail+ii")
        return

    try:
        data, metadata = _load_local_sales()
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        st.error(f"The cached UCI sales data could not be read: {exc}")
        return
    if data.empty:
        st.warning("The source workbook did not contain qualifying positive sale lines.")
        return

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])
    markets = sorted(data["country"].dropna().unique().tolist())
    controls = st.columns([1, 1.4, 1.2])
    chosen_market = controls[0].selectbox("Market", ["All markets", *markets])
    min_date, max_date = data["date"].min().date(), data["date"].max().date()
    date_range = controls[1].date_input(
        "Transaction dates", value=(min_date, max_date), min_value=min_date, max_value=max_date
    )
    metric_label = controls[2].selectbox("Rank products by", ["Units sold", "Recorded sales value (£)"])
    if isinstance(date_range, (tuple, list)):
        start_date, end_date = date_range[0], date_range[-1]
    else:
        start_date = end_date = date_range
    if start_date > end_date:
        st.error("The start date must be on or before the end date.")
        return

    filtered = data.loc[data["date"].dt.date.between(start_date, end_date)]
    if chosen_market != "All markets":
        filtered = filtered.loc[filtered["country"].eq(chosen_market)]
    if filtered.empty:
        st.info("No transaction summaries match these filters.")
        return

    ranking_metric = "units_sold" if metric_label == "Units sold" else "sales_value_gbp"
    units = int(filtered["units_sold"].sum())
    revenue = float(filtered["sales_value_gbp"].sum())
    product_count = int(filtered["stock_code"].nunique())
    market_count = int(filtered["country"].nunique())
    summary = st.columns(4)
    summary[0].metric("Units sold", f"{units:,}")
    summary[1].metric("Recorded sales value", f"£{revenue:,.0f}")
    summary[2].metric("Products", f"{product_count:,}")
    summary[3].metric("Markets", f"{market_count:,}")
    st.caption(
        f"Filtered period {start_date:%b %d, %Y}–{end_date:%b %d, %Y} · "
        f"{len(filtered):,} daily product-market summaries · historical source period {metadata['source_period']}"
    )

    section_heading(
        "Demand signal over time",
        "Bars show recorded units or value per calendar day; the line is a 7-calendar-day mean of dates with transactions. Dates without a recorded sale are not treated as zero-demand days.",
    )
    trend_metric = "units_sold" if metric_label == "Units sold" else "sales_value_gbp"
    st.altair_chart(_daily_trend(filtered, trend_metric), width="stretch")

    left, right = st.columns(2, gap="large")
    with left:
        section_heading("Products with the most recorded sales", "Ranks product stock codes over the selected period. Descriptions come from transaction records and may be missing or inconsistent.")
        if metric_label == "Units sold":
            st.altair_chart(_rank_chart(filtered, group="stock_code", label="Product stock code", metric="units_sold", title="Top 15 products by units sold"), width="stretch")
        else:
            ranked = filtered.groupby("stock_code", as_index=False).agg(value=("sales_value_gbp", "sum")).nlargest(15, "value")
            st.altair_chart(
                alt.Chart(ranked).mark_bar(color="#145044", cornerRadiusEnd=4).encode(
                    x=alt.X("value:Q", title="Recorded sales value (£)", scale=alt.Scale(zero=True)),
                    y=alt.Y("stock_code:N", title="Product stock code", sort="-x"),
                    tooltip=[alt.Tooltip("stock_code:N", title="Product stock code"), alt.Tooltip("value:Q", title="Value (£)", format="£,.2f")],
                ).properties(height=390, title="Top 15 products by recorded sales value").interactive(),
                width="stretch",
            )
    with right:
        section_heading("Markets by units sold", "Country is the source dataset's customer country, not a physical store location.")
        st.altair_chart(_rank_chart(filtered, group="country", label="Customer country", metric="units_sold", title="Top 15 customer countries by units sold"), width="stretch")

    top_product = filtered.groupby(["stock_code", "description"], as_index=False)["units_sold"].sum().sort_values("units_sold", ascending=False).head(1)
    top_market = filtered.groupby("country", as_index=False)["units_sold"].sum().sort_values("units_sold", ascending=False).head(1)
    if not top_product.empty and not top_market.empty:
        product_name = top_product.iloc[0]["description"]
        st.info(
            f"In this filtered history, **{product_name if pd.notna(product_name) else top_product.iloc[0]['stock_code']}** "
            f"had the most recorded units ({int(top_product.iloc[0]['units_sold']):,}); "
            f"{top_market.iloc[0]['country']} had the most units by customer country "
            f"({int(top_market.iloc[0]['units_sold']):,}). These are descriptive observations, not forecasts."
        )

    with st.expander("Data provenance and preparation"):
        st.markdown(
            f"**Source:** [UCI Online Retail II]({metadata['source_url']})  \n"
            f"**Citation:** {metadata['citation']}  \n"
            f"**License:** {metadata['license']}  \n"
            f"**Source rows:** {metadata['source_rows']:,} · **included positive sale lines:** {metadata['included_sale_lines']:,} · "
            f"**cancelled invoice lines excluded:** {metadata['cancelled_invoice_lines_excluded']:,} · "
            f"**invalid/non-sale lines excluded:** {metadata['invalid_non_cancelled_lines_excluded']:,}  \n"
            "Cancellation invoices, invalid dates, non-positive quantities or prices, and missing product/market keys are excluded. "
            "Sales value is recorded quantity multiplied by unit price in GBP; returns are excluded, so it is not net accounting revenue. "
            "The workbook timestamps have no timezone metadata and are displayed as source-local dates."
        )
        st.warning(
            "This dataset contains transactions, not on-hand stock, replenishment events, supplier lead times, or current retail activity. "
            "It can support historical demand analysis, but it cannot validate this project's restock classifier or produce an operational reorder quantity."
        )
        st.caption(f"Daily prepared file: {Path(paths['daily_sales']).name} · generated {metadata['created_at_utc']}")
