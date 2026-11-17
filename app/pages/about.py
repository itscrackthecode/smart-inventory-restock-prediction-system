"""Project methodology, scope, and limitations."""

from __future__ import annotations

import streamlit as st

from app.components.layout import page_header


def render() -> None:
    page_header("About & methodology", "How the inventory workflow moves from generated observations to a saved model score.")
    st.markdown("### Project overview")
    st.write(
        "The Smart Inventory & Restock Prediction System is a portfolio project for exploring daily inventory records, "
        "summarizing stock patterns, and estimating a restock event on a product/store's next qualifying observation."
    )
    st.markdown("### Business objective")
    st.write(
        "Provide a decision-support view that helps teams inspect stock status and compare restock-risk signals. "
        "A model score is an estimate for review, not a purchase order or a certainty."
    )
    st.markdown("### Dataset and workflow")
    st.markdown(
        "1. **Synthetic source data:** deterministic generated observations across multiple products and stores. "
        "The data is explicitly synthetic.\n"
        "2. **Preprocessing:** timestamps are parsed consistently, records are validated, and the cleaned CSV is saved separately.\n"
        "3. **Feature engineering:** target labels use the next chronological daily observation for the same product/store. "
        "Historical features use past events only; the first restock-age values are treated as unverified.\n"
        "4. **Model development:** Logistic Regression and Random Forest pipelines learn imputers/encoders on the training partition.\n"
        "5. **Evaluation:** chronological holdout, with a boundary purge and validation-selected thresholds; saved in Phase 7 reports.\n"
        "6. **SQLite:** stores operational inventory and prediction history.\n"
        "7. **Application:** reads current inventory, creates exact Phase 5 inputs from as-of history, scores a saved pipeline, and stores the output."
    )
    st.markdown("### Prediction target and inputs")
    st.write(
        "The training target is whether a restock event is recorded on the next daily observation for the same product/store. "
        "A restock event on the selected source row is excluded from the current input features. "
        "When no prior restock is documented, restock age is marked invalid and left missing for the saved pipeline's fitted imputer. "
        "When the history has date gaps, the inference feature builder uses calendar-day distance to the latest earlier observed restock."
    )
    st.markdown("### Model evaluation")
    st.write(
        "The saved Phase 7 metrics evaluate a chronological holdout from the synthetic dataset. "
        "Thresholds were selected on a separate chronological validation set. The application displays these saved results without modifying them."
    )
    st.markdown("### Database architecture")
    st.write(
        "SQLite contains products, stores, inventory_records, and predictions. "
        "Inventory is linked by product/store foreign keys and unique per product/store observation timestamp. "
        "Predictions retain their source inventory record, model name/version, probability, threshold, class, and creation time."
    )
    st.markdown("### Technology stack")
    st.write("Python · Streamlit · Pandas · Altair · Scikit-learn · Joblib · SQLite")
    st.markdown("### Limitations")
    st.markdown(
        "- The inventory dataset is synthetic; reported metrics do not establish real-world accuracy.\n"
        "- Historical restock data is limited to observations present in the database. Unrecorded pre-window events are unknown.\n"
        "- The target concerns the next qualifying chronological daily observation, not a guaranteed delivery or a guaranteed event tomorrow.\n"
        "- No demand, supplier lead time, cost, promotion, or explicit reorder-point fields are present.\n"
        "- The dashboard's low-stock indicator is an adjustable review threshold, not a learned or authoritative business policy."
    )
    st.markdown("### Future improvements")
    st.write(
        "Validate with real, time-stamped inventory and delivery records; capture business reorder thresholds and lead times; "
        "monitor drift and calibration; compare operational costs of missed events and false alerts; and select durable hosted storage before deployment."
    )
