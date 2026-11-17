"""Apply the saved Phase 6 pipelines to the latest selected inventory row."""

from __future__ import annotations

import sqlite3

import streamlit as st

from app.components.layout import page_header, section_heading
from app.config import database_path
from app.services.cached_data import cached_catalogs, cached_prediction_history, clear_inventory_caches
from app.services.inference import (
    DISPLAY_NAMES,
    MODEL_FILE_NAMES,
    format_model_inputs_for_display,
    load_inference_bundle,
    predict_for_product_store,
)


def render() -> None:
    page_header("Restock predictions", "Score the latest stored observation with the original fitted pipelines.")
    st.info(
        "The model estimates whether a restock event will occur on the next chronological observation. "
        "That is not a guarantee, and the next observation is not necessarily tomorrow."
    )
    db = str(database_path())
    products, stores = cached_catalogs(db)
    if products.empty or stores.empty:
        st.warning("Product and store catalogs are empty. Import inventory observations first.")
        return
    try:
        bundle = load_inference_bundle()
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(f"The trained model bundle is not ready: {exc}")
        return

    product_id = st.selectbox(
        "Product", products["product_id"].tolist(),
        format_func=lambda key: f"{products.loc[products.product_id.eq(key), 'product_name'].iloc[0]} · {key}",
        key="prediction_product",
    )
    store_ids = stores["store_id"].tolist()
    locations = dict(zip(stores["store_id"], stores["store_location"]))
    store_id = st.selectbox(
        "Store", store_ids, format_func=lambda key: f"{locations.get(key, key)} · {key}",
        key="prediction_store",
    )
    choices = list(MODEL_FILE_NAMES)
    compare_both = st.checkbox("Compare and save predictions from both models", value=True)
    if compare_both:
        selected_models = choices
    else:
        selected_models = [st.selectbox("Model", choices,
                                         format_func=lambda key: DISPLAY_NAMES[key])]

    st.caption("Validation-selected thresholds from the immutable Phase 7 report: " +
               " · ".join(f"{DISPLAY_NAMES[key]} {bundle['thresholds'][key]:.2f}" for key in choices))
    if st.button("Generate prediction", type="primary"):
        results = []
        for model_name in selected_models:
            try:
                results.append(predict_for_product_store(db, product_id=product_id, store_id=store_id,
                                                         model_name=model_name, bundle=bundle))
            except (ValueError, FileNotFoundError, sqlite3.Error, OSError) as exc:
                st.error(f"{DISPLAY_NAMES[model_name]} could not produce a prediction: {exc}")
        if results:
            clear_inventory_caches()
            st.session_state["last_prediction_results"] = results
    results = st.session_state.get("last_prediction_results", [])
    current_results = [result for result in results if result["product_id"] == product_id and result["store_id"] == store_id]
    if current_results:
        section_heading("Prediction result", "Scores are model estimates based on inputs available at the source observation.")
        cols = st.columns(len(current_results))
        for col, result in zip(cols, current_results):
            with col:
                label = "Restock event likely at the selected threshold" if result["predicted_restock"] else "Below the selected restock threshold"
                st.metric(DISPLAY_NAMES[result["model_key"]], f"{result['probability']:.1%}", label)
                st.progress(result["probability"], text=f"Probability {result['probability']:.1%} · threshold {result['threshold']:.0%}")
                st.write("**Class:**", "Restock" if result["predicted_restock"] else "No restock")
                st.write("**Source observation:**", result["observation_date"])
                st.write("**Stock at source:**", f"{result['stock_quantity']} units · {result['availability_status']}")
                st.write("**History available:**", f"{result['history_observation_count']} observations")
                if result["has_observation_gap"]:
                    st.warning("The stored timeline contains a date gap. Restock age uses calendar days since the latest earlier observed restock.")
                if not result["days_since_last_restock_valid"]:
                    st.caption("No earlier restock is recorded; restock age was masked and handled by the saved pipeline's training-fitted imputer.")
                st.caption(f"Saved at {result['created_at']} · model version {result['model_version']}")
                with st.expander("Model inputs used"):
                    st.dataframe(format_model_inputs_for_display(result["features"]), width="stretch")

        with st.expander("How to read this result"):
            st.markdown(
                "The probability is the pipeline's estimated chance of class 1 for the next qualifying observation. "
                "The class is positive when the probability meets the threshold selected on a chronological validation set. "
                "The system does not know inventory decisions or deliveries that have not yet been recorded."
            )

    section_heading("Prediction history", "Saved model name, version, probability, threshold, source observation, and prediction time.")
    history = cached_prediction_history(db, product_id, store_id)
    if history.empty:
        st.caption("No prediction has been saved for this product/store yet.")
    else:
        history["predicted_class"] = history["predicted_restock"].map({1: "Restock", 0: "No restock"})
        st.dataframe(history[["created_at", "model_name", "model_version", "observation_date", "stock_quantity",
                              "predicted_probability", "prediction_threshold", "predicted_class"]],
                     hide_index=True, width="stretch",
                     column_config={"predicted_probability": st.column_config.NumberColumn(format="%.1%%"),
                                    "prediction_threshold": st.column_config.NumberColumn(format="%.0%%")})
