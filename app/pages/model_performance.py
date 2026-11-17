"""Read-only presentation of the Phase 7 evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.layout import page_header, section_heading
from app.config import FIGURES_DIR, REPORTS_DIR


MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}


def load_phase7_report(report_path: str | Path | None = None) -> dict:
    path = Path(report_path) if report_path is not None else REPORTS_DIR / "model_evaluation.json"
    if not path.is_file():
        raise FileNotFoundError(f"Phase 7 evaluation report not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not {"evaluation", "models", "figures"}.issubset(payload):
        raise ValueError("Phase 7 report does not contain the expected sections")
    return payload


def performance_table(report: dict) -> pd.DataFrame:
    rows = []
    for key, label in MODEL_LABELS.items():
        metrics = report["models"][key]["holdout_selected_threshold"]
        rows.append({
            "Model": label,
            "Threshold": metrics["threshold"],
            "Accuracy": metrics["accuracy"],
            "Precision (restock)": metrics["precision"],
            "Recall (restock)": metrics["recall"],
            "F1 (restock)": metrics["f1"],
            "ROC-AUC": metrics["roc_auc"],
            "Average precision": metrics["average_precision"],
        })
    return pd.DataFrame(rows)


def render() -> None:
    page_header("Model performance", "The holdout metrics below are read from the saved Phase 7 report; this page does not recalculate or tune them.")
    try:
        report = load_phase7_report()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Evaluation artifacts are unavailable: {exc}")
        return
    evaluation = report["evaluation"]
    distribution = evaluation["positive_negative_distribution"]
    st.warning(
        "Evaluation uses synthetic observations. These results do not establish model performance on real inventory operations."
    )
    st.caption(
        f"Chronological holdout: {evaluation['rows']:,} rows from {evaluation['feature_date_start'][:10]} "
        f"through {evaluation['feature_date_end'][:10]} (label dates through {evaluation['target_date_end'][:10]}). "
        f"Restock prevalence: {distribution['1']['percentage']:.2f}%."
    )
    st.markdown(
        "Thresholds were chosen by maximizing F1 on a separate chronological validation slice of the training data. "
        "The March 12 boundary rows were purged. Holdout labels were not used to select the thresholds."
    )
    table = performance_table(report)
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={column: st.column_config.NumberColumn(format="%.3f")
                       for column in table.columns if column != "Model"},
    )

    section_heading("Confusion matrices", "Rows are actual classes; columns are model predictions. The positive class is restock.")
    left, right = st.columns(2)
    left.image(str(FIGURES_DIR / "logistic_regression_confusion_matrix.png"),
               caption="Logistic Regression · selected threshold 0.75", width="stretch")
    right.image(str(FIGURES_DIR / "random_forest_confusion_matrix.png"),
                caption="Random Forest · selected threshold 0.53", width="stretch")

    section_heading("Ranking performance", "ROC-AUC summarizes class ranking across cutoffs; average precision focuses on the positive class under imbalance.")
    curve_left, curve_right = st.columns(2)
    curve_left.image(str(FIGURES_DIR / "roc_curve_comparison.png"),
                     caption="Saved Phase 7 ROC comparison", width="stretch")
    curve_right.image(str(FIGURES_DIR / "precision_recall_curve_comparison.png"),
                      caption="Saved Phase 7 precision–recall comparison", width="stretch")

    section_heading("Model feature analysis", "Importance and coefficient values describe fitted model associations, not causes.")
    explain_cols = st.columns(2)
    explain_cols[0].markdown("**Random Forest importance** ranks transformed input features by their contribution to fitted tree splits.")
    explain_cols[1].markdown("**Logistic Regression coefficients** show the direction and size of fitted log-odds contributions under its encoding.")
    left, right = st.columns(2)
    left.image(str(FIGURES_DIR / "random_forest_feature_importance.png"),
               caption="Saved Phase 7 feature importance", width="stretch")
    right.image(str(FIGURES_DIR / "logistic_regression_coefficients.png"),
                caption="Saved Phase 7 coefficient analysis", width="stretch")
    with st.expander("Metric interpretation for inventory teams"):
        st.markdown(
            "**Recall** asks: of restocks that occurred in the holdout, what share did the model identify? "
            "Higher recall can reduce missed restock events but may generate more alerts. "
            "**Precision** asks: of restock alerts, what share corresponded to a restock in the labeled next observation? "
            "Higher precision can reduce false alarms. Accuracy alone can be misleading when most observations are non-restocks."
        )
