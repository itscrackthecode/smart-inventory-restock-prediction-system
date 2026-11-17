"""Reusable interactive Altair visualizations for operational inventory."""

from __future__ import annotations

import altair as alt
import pandas as pd


GREEN = "#145044"
GOLD = "#bd8c48"
RED = "#b24d42"
PALETTE = [GREEN, "#5a8273", "#91aa9c", GOLD, RED, "#8d785b"]


def availability_chart(data: pd.DataFrame, *, height: int = 290) -> alt.Chart:
    counts = data.groupby("availability_status", as_index=False).size().rename(columns={"size": "combinations"})
    return alt.Chart(counts).mark_arc(innerRadius=68, outerRadius=112).encode(
        theta=alt.Theta("combinations:Q", title="Product/store combinations"),
        color=alt.Color("availability_status:N", title="Availability", scale=alt.Scale(
            domain=["in_stock", "out_of_stock"], range=[GREEN, RED])),
        tooltip=[alt.Tooltip("availability_status:N", title="Availability"),
                 alt.Tooltip("combinations:Q", title="Combinations", format=",")],
    ).properties(height=height, title="Latest availability by product/store").interactive()


def stock_trend_chart(data: pd.DataFrame, *, height: int = 300) -> alt.Chart:
    daily = data.assign(observation_day=data["observation_date"].dt.floor("D")).groupby(
        "observation_day", as_index=False
    ).agg(mean_stock=("stock_quantity", "mean"), observations=("inventory_id", "count"))
    base = alt.Chart(daily).encode(
        x=alt.X("observation_day:T", title="Observation date"),
        tooltip=[alt.Tooltip("observation_day:T", title="Date"),
                 alt.Tooltip("mean_stock:Q", title="Mean stock", format=".1f"),
                 alt.Tooltip("observations:Q", title="Observations", format=",")],
    )
    return base.mark_line(color=GREEN, point=True).encode(
        y=alt.Y("mean_stock:Q", title="Mean units on hand", scale=alt.Scale(zero=False))
    ).properties(height=height, title="Daily mean stock across observations").interactive()


def category_stock_chart(data: pd.DataFrame, *, height: int = 300) -> alt.Chart:
    grouped = data.groupby("category", as_index=False).agg(
        mean_stock=("stock_quantity", "mean"), records=("inventory_id", "count")
    ).sort_values("mean_stock", ascending=False)
    return alt.Chart(grouped).mark_bar(color=GREEN, cornerRadiusEnd=4).encode(
        x=alt.X("mean_stock:Q", title="Mean stock quantity", scale=alt.Scale(zero=True)),
        y=alt.Y("category:N", title="Product category", sort="-x"),
        tooltip=[alt.Tooltip("category:N", title="Category"),
                 alt.Tooltip("mean_stock:Q", title="Mean stock", format=".1f"),
                 alt.Tooltip("records:Q", title="Inventory records", format=",")],
    ).properties(height=height, title="Stock quantity by category").interactive()


def store_performance_chart(data: pd.DataFrame, *, height: int = 330) -> alt.Chart:
    grouped = data.groupby(["store_location", "store_id"], as_index=False).agg(
        mean_stock=("stock_quantity", "mean"),
        stockouts=("availability_status", lambda values: int(values.eq("out_of_stock").sum())),
        combinations=("inventory_id", "count"),
    )
    return alt.Chart(grouped).mark_bar(color=GREEN, cornerRadiusEnd=4).encode(
        x=alt.X("mean_stock:Q", title="Mean stock quantity", scale=alt.Scale(zero=True)),
        y=alt.Y("store_location:N", title="Store", sort="-x"),
        tooltip=[alt.Tooltip("store_location:N", title="Location"),
                 alt.Tooltip("store_id:N", title="Store ID"),
                 alt.Tooltip("mean_stock:Q", title="Mean stock", format=".1f"),
                 alt.Tooltip("stockouts:Q", title="Out-of-stock pairs"),
                 alt.Tooltip("combinations:Q", title="Product/store pairs")],
    ).properties(height=height, title="Latest stock by store").interactive()


def restock_frequency_chart(data: pd.DataFrame, *, height: int = 340) -> alt.Chart:
    grouped = data.loc[data["restock_occurred"]].groupby(
        ["product_name", "product_id"], as_index=False
    ).size().rename(columns={"size": "restock_events"}).sort_values("restock_events", ascending=False).head(15)
    return alt.Chart(grouped).mark_bar(color=GOLD, cornerRadiusEnd=4).encode(
        x=alt.X("restock_events:Q", title="Observed restock events", scale=alt.Scale(zero=True)),
        y=alt.Y("product_name:N", title="Product (top 15)", sort="-x"),
        tooltip=[alt.Tooltip("product_name:N", title="Product"),
                 alt.Tooltip("product_id:N", title="Product ID"),
                 alt.Tooltip("restock_events:Q", title="Restock events")],
    ).properties(height=height, title="Restock events by product").interactive()


def category_store_heatmap(data: pd.DataFrame, *, height: int = 340) -> alt.Chart:
    grouped = data.groupby(["store_location", "category"], as_index=False).agg(
        mean_stock=("stock_quantity", "mean"), pairs=("inventory_id", "count")
    )
    return alt.Chart(grouped).mark_rect().encode(
        x=alt.X("category:N", title="Category", axis=alt.Axis(labelAngle=-28)),
        y=alt.Y("store_location:N", title="Store"),
        color=alt.Color("mean_stock:Q", title="Mean units", scale=alt.Scale(scheme="tealblues")),
        tooltip=[alt.Tooltip("store_location:N", title="Store"),
                 alt.Tooltip("category:N", title="Category"),
                 alt.Tooltip("mean_stock:Q", title="Mean stock", format=".1f"),
                 alt.Tooltip("pairs:Q", title="Records")],
    ).properties(height=height, title="Mean stock by store and category").interactive()


def monthly_pattern_chart(data: pd.DataFrame, *, height: int = 290) -> alt.Chart:
    monthly = data.assign(month=data["observation_date"].dt.month_name()).groupby(
        "month", as_index=False
    ).agg(mean_stock=("stock_quantity", "mean"), restocks=("restock_occurred", "sum"))
    month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    return alt.Chart(monthly).mark_bar(color=GREEN, cornerRadiusEnd=4).encode(
        x=alt.X("month:N", title="Calendar month", sort=month_order, axis=alt.Axis(labelAngle=-25)),
        y=alt.Y("mean_stock:Q", title="Mean stock quantity", scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("month:N", title="Month"),
                 alt.Tooltip("mean_stock:Q", title="Mean stock", format=".1f"),
                 alt.Tooltip("restocks:Q", title="Observed restock events")],
    ).properties(height=height, title="Mean stock by calendar month").interactive()


def weekday_pattern_chart(data: pd.DataFrame, *, height: int = 290) -> alt.Chart:
    daily = data.assign(day_of_week=data["observation_date"].dt.day_name()).groupby(
        "day_of_week", as_index=False
    ).agg(mean_stock=("stock_quantity", "mean"), restocks=("restock_occurred", "sum"))
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return alt.Chart(daily).mark_bar(color=GREEN, cornerRadiusEnd=4).encode(
        x=alt.X("day_of_week:N", title="Weekday", sort=weekday_order, axis=alt.Axis(labelAngle=-25)),
        y=alt.Y("mean_stock:Q", title="Mean stock quantity", scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("day_of_week:N", title="Weekday"),
                 alt.Tooltip("mean_stock:Q", title="Mean stock", format=".1f"),
                 alt.Tooltip("restocks:Q", title="Observed restock events")],
    ).properties(height=height, title="Mean stock by weekday").interactive()


def stock_distribution_chart(data: pd.DataFrame, *, height: int = 330) -> alt.Chart:
    binned = data.copy()
    binned["stock_range"] = pd.cut(
        binned["stock_quantity"], bins=20, include_lowest=True, duplicates="drop"
    ).astype(str)
    distribution = binned.groupby(["category", "stock_range"], as_index=False, observed=True).size()
    return alt.Chart(distribution).mark_bar().encode(
        x=alt.X("stock_range:O", title="Stock quantity range", axis=alt.Axis(labelAngle=-35), sort=None),
        y=alt.Y("size:Q", title="Observation count", scale=alt.Scale(zero=True)),
        color=alt.Color("category:N", title="Category", scale=alt.Scale(range=PALETTE)),
        xOffset=alt.XOffset("category:N"),
        tooltip=[alt.Tooltip("category:N", title="Category"),
                 alt.Tooltip("stock_range:O", title="Stock range"),
                 alt.Tooltip("size:Q", title="Observations", format=",")],
    ).properties(height=height, title="Stock quantity distribution by category").interactive()


def product_trend_chart(data: pd.DataFrame, *, height: int = 330) -> alt.Chart:
    return alt.Chart(data).mark_line(point=True).encode(
        x=alt.X("observation_date:T", title="Observation date"),
        y=alt.Y("stock_quantity:Q", title="Stock quantity", scale=alt.Scale(zero=True)),
        color=alt.Color("store_location:N", title="Store", scale=alt.Scale(range=PALETTE)),
        tooltip=[alt.Tooltip("observation_date:T", title="Date"),
                 alt.Tooltip("store_location:N", title="Store"),
                 alt.Tooltip("stock_quantity:Q", title="Stock"),
                 alt.Tooltip("availability_status:N", title="Availability"),
                 alt.Tooltip("restock_occurred:N", title="Restock observed?")],
    ).properties(height=height, title="Product stock history by store").interactive()
