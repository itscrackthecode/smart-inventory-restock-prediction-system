"""Evaluate saved Phase 6 pipelines on the untouched chronological holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.feature_engineering import MODEL_FEATURES, TARGET_COLUMN, select_model_features
from src.model_data import (
    DEFAULT_DATA_PATH,
    NEXT_LABEL_HORIZON,
    TEST_SIZE,
    chronological_train_test_split,
    load_engineered_dataset,
    target_distribution,
)
from src.model_persistence import DEFAULT_MODEL_DIR, load_model_pipeline
from src.model_training import fit_model_pipelines


DEFAULT_REPORT_DIR = Path("reports")
MODEL_NAMES = ("logistic_regression", "random_forest")
DEFAULT_THRESHOLD = 0.5
THRESHOLD_CANDIDATES = tuple(np.round(np.arange(0.05, 0.951, 0.01), 2))


def evaluate_binary_predictions(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Calculate threshold-specific and probability-based binary metrics."""
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if len(truth) != len(scores):
        raise ValueError("Labels and probabilities must have the same number of rows")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1")
    predicted = (scores >= threshold).astype(int)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    roc_auc = roc_auc_score(truth, scores) if np.unique(truth).size == 2 else None
    average_precision = average_precision_score(truth, scores) if np.any(truth == 1) else None
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, pos_label=1, zero_division=0)),
        "recall": float(recall_score(truth, predicted, pos_label=1, zero_division=0)),
        "f1": float(f1_score(truth, predicted, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "average_precision": float(average_precision) if average_precision is not None else None,
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_matrix_order": [["true_negative", "false_positive"], ["false_negative", "true_positive"]],
        "classification_report": classification_report(
            truth,
            predicted,
            labels=[0, 1],
            target_names=["no_restock", "restock"],
            output_dict=True,
            zero_division=0,
        ),
    }


def select_validation_threshold(
    y_validation: pd.Series | np.ndarray,
    validation_probabilities: np.ndarray,
    *,
    candidates: tuple[float, ...] = THRESHOLD_CANDIDATES,
) -> dict[str, Any]:
    """Choose a threshold by validation F1, then recall, precision, and threshold.

    This function receives validation labels/probabilities only; it has no
    holdout argument, so holdout results cannot influence threshold choice.
    """
    truth = np.asarray(y_validation, dtype=int)
    probabilities = np.asarray(validation_probabilities, dtype=float)
    if np.unique(truth).size < 2:
        raise ValueError("Threshold selection needs both classes in the validation partition")
    if len(truth) != len(probabilities):
        raise ValueError("Validation labels and probabilities have different lengths")
    if not candidates:
        raise ValueError("At least one threshold candidate is required")

    candidate_metrics = []
    for threshold in candidates:
        predicted = (probabilities >= threshold).astype(int)
        candidate_metrics.append(
            {
                "threshold": float(threshold),
                "f1": float(f1_score(truth, predicted, zero_division=0)),
                "recall": float(recall_score(truth, predicted, zero_division=0)),
                "precision": float(precision_score(truth, predicted, zero_division=0)),
            }
        )
    best = max(
        candidate_metrics,
        key=lambda row: (row["f1"], row["recall"], row["precision"], row["threshold"]),
    )
    return {
        "threshold": best["threshold"],
        "selection_metric": "validation F1",
        "tie_break_order": ["higher recall", "higher precision", "higher threshold"],
        "validation_f1": best["f1"],
        "validation_recall": best["recall"],
        "validation_precision": best["precision"],
        "validation_rows": int(len(truth)),
        "candidate_count": len(candidate_metrics),
    }


def _source_feature(transformed_name: str) -> str:
    """Map a ColumnTransformer/OneHotEncoder output name to its raw input."""
    name = transformed_name.split("__", 1)[-1]
    for feature in MODEL_FEATURES:
        if name == feature or name.startswith(f"{feature}_"):
            return feature
    return name


def extract_random_forest_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Return importances for each transformed input with original feature mapping."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    values = classifier.feature_importances_
    if len(names) != len(values):
        raise ValueError("Transformed feature names do not align with forest importances")
    result = pd.DataFrame(
        {
            "transformed_feature": names,
            "source_feature": [_source_feature(name) for name in names],
            "importance": values.astype(float),
        }
    )
    return result.sort_values("importance", ascending=False).reset_index(drop=True)


def extract_logistic_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    """Return signed Logistic Regression coefficients and encoded feature names."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    values = classifier.coef_[0]
    if len(names) != len(values):
        raise ValueError("Transformed feature names do not align with logistic coefficients")
    result = pd.DataFrame(
        {
            "transformed_feature": names,
            "source_feature": [_source_feature(name) for name in names],
            "coefficient": values.astype(float),
        }
    )
    return result.sort_values("coefficient", ascending=False).reset_index(drop=True)


def _write_confusion_matrix(
    metrics: dict[str, Any], *, model_name: str, path: Path
) -> None:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    figure, axis = plt.subplots(figsize=(7.5, 6.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No restock", "Restock"],
        yticklabels=["No restock", "Restock"],
        ax=axis,
    )
    axis.set(
        title=f"{model_name.replace('_', ' ').title()} confusion matrix\n(validation-selected threshold: {metrics['threshold']:.2f})",
        xlabel="Predicted class",
        ylabel="Actual class",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _create_plots(
    *,
    y_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    evaluation: dict[str, Any],
    forest_importance: pd.DataFrame,
    logistic_coefficients: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    for name in MODEL_NAMES:
        _write_confusion_matrix(
            evaluation[name]["holdout_selected_threshold"],
            model_name=name,
            path=figures_dir / f"{name}_confusion_matrix.png",
        )

    fig, ax = plt.subplots(figsize=(8.5, 7))
    for name, scores in probabilities.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores)
        auc = evaluation[name]["holdout_selected_threshold"]["roc_auc"]
        ax.plot(false_positive_rate, true_positive_rate, linewidth=2.4, label=f"{name.replace('_', ' ').title()} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="No-skill reference")
    ax.set(title="ROC curves on the chronological holdout", xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curve_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    for name, scores in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = evaluation[name]["holdout_selected_threshold"]["average_precision"]
        ax.plot(recall, precision, linewidth=2.4, label=f"{name.replace('_', ' ').title()} (AP={ap:.3f})")
    prevalence = float(np.mean(y_true))
    ax.axhline(prevalence, linestyle="--", color="gray", label=f"Positive prevalence={prevalence:.3f}")
    ax.set(title="Precision–recall curves on the chronological holdout", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "precision_recall_curve_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    metric_names = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
        "average_precision": "Average precision",
    }
    chart_rows = [
        {"model": name.replace("_", " ").title(), "metric": label, "value": evaluation[name]["holdout_selected_threshold"][metric]}
        for name in MODEL_NAMES
        for metric, label in metric_names.items()
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=pd.DataFrame(chart_rows), x="metric", y="value", hue="model", ax=ax)
    ax.set(title="Holdout metric comparison at validation-selected thresholds", xlabel="Metric", ylabel="Score", ylim=(0, 1))
    ax.tick_params(axis="x", labelrotation=18)
    for tick_label in ax.get_xticklabels():
        tick_label.set_ha("right")
    ax.legend(title="Model", loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(figures_dir / "model_metric_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    top_importance = forest_importance.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.barplot(data=top_importance, x="importance", y="transformed_feature", color="#4c78a8", ax=ax)
    ax.set(title="Random Forest: 15 largest encoded-feature importances", xlabel="Mean decrease in impurity importance", ylabel="Encoded model input")
    fig.tight_layout()
    fig.savefig(figures_dir / "random_forest_feature_importance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    extremes = pd.concat([logistic_coefficients.head(10), logistic_coefficients.tail(10)]).drop_duplicates()
    extremes = extremes.sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(11, 9))
    colors = ["#d95f5f" if value < 0 else "#3b8c72" for value in extremes["coefficient"]]
    ax.barh(extremes["transformed_feature"], extremes["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set(title="Logistic Regression: largest positive and negative coefficients", xlabel="Coefficient (standardized numeric inputs; one-hot categorical terms)", ylabel="Encoded model input")
    fig.tight_layout()
    fig.savefig(figures_dir / "logistic_regression_coefficients.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _make_comparison(evaluation: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str]]:
    rows = []
    for name in MODEL_NAMES:
        result = evaluation[name]
        baseline = result["holdout_default_threshold"]
        selected = result["holdout_selected_threshold"]
        row: dict[str, Any] = {
            "model": name,
            "validation_selected_threshold": result["threshold_selection"]["threshold"],
            "validation_f1_at_selected_threshold": result["threshold_selection"]["validation_f1"],
        }
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"):
            row[f"default_{metric}"] = baseline[metric]
            row[f"selected_{metric}"] = selected[metric]
        for label, index in (("true_negative", (0, 0)), ("false_positive", (0, 1)), ("false_negative", (1, 0)), ("true_positive", (1, 1))):
            row[f"selected_{label}"] = selected["confusion_matrix"][index[0]][index[1]]
        rows.append(row)
    frame = pd.DataFrame(rows)

    logistic = evaluation["logistic_regression"]["holdout_selected_threshold"]
    forest = evaluation["random_forest"]["holdout_selected_threshold"]
    recall_leader = max(MODEL_NAMES, key=lambda name: evaluation[name]["holdout_selected_threshold"]["recall"])
    false_alarm_leader = min(MODEL_NAMES, key=lambda name: evaluation[name]["holdout_selected_threshold"]["confusion_matrix"][0][1])
    precision_leader = max(MODEL_NAMES, key=lambda name: evaluation[name]["holdout_selected_threshold"]["precision"])
    narrative = {
        "more_actual_restocks_detected_by_recall": recall_leader,
        "fewer_false_alarms_by_count": false_alarm_leader,
        "higher_positive_precision": precision_leader,
        "tradeoff": (
            f"At validation-selected thresholds, Logistic Regression has recall {logistic['recall']:.3f} and precision {logistic['precision']:.3f}; "
            f"Random Forest has recall {forest['recall']:.3f} and precision {forest['precision']:.3f}. Higher recall captures more true restocks but may also increase false alarms."
        ),
        "practical_guidance": (
            f"For an alerting workflow that prioritizes finding restocks, {recall_leader.replace('_', ' ').title()} is the relative recall leader on this holdout. "
            f"For fewer false alerts, {false_alarm_leader.replace('_', ' ').title()} has fewer false positives. The best operating choice depends on the business cost of missed restocks versus unnecessary alerts."
        ),
        "reliability_statement": "These are results on synthetic data. Neither pipeline is established as reliable for retailer operations until evaluated on representative real inventory data and a cost-based alert policy.",
    }
    return frame, narrative


def run_model_evaluation(
    input_path: str | Path = DEFAULT_DATA_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
) -> dict[str, Any]:
    """Rebuild the Phase 6 split, calibrate thresholds on training-only validation,
    evaluate saved pipelines once on holdout, and write reports/figures.
    """
    root = Path(report_dir)
    figures_dir = root / "figures"
    root.mkdir(parents=True, exist_ok=True)

    data = load_engineered_dataset(input_path)
    phase6_split = chronological_train_test_split(data, test_size=TEST_SIZE)
    model_metadata_path = Path(model_dir) / "model_metadata.json"
    phase6_metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
    prior_split = phase6_metadata["split_strategy"]
    reconstructed = {
        "holdout_start": phase6_split.holdout_start.isoformat(),
        "train_rows": len(phase6_split.train),
        "holdout_rows": len(phase6_split.holdout),
        "purged_boundary_rows": len(phase6_split.purged),
    }
    expected = {
        "holdout_start": prior_split["holdout_start"],
        "train_rows": prior_split["train_rows"],
        "holdout_rows": prior_split["holdout_rows"],
        "purged_boundary_rows": prior_split["purged_boundary_rows"],
    }
    if reconstructed != expected:
        raise ValueError(f"Reconstructed Phase 6 split differs from saved model metadata: {reconstructed} != {expected}")

    # Threshold calibration uses only Phase 6 training dates. The later part of
    # that training window is validation; earlier dates fit temporary calibrators.
    validation_source = phase6_split.train.sort_values(
        ["product_id", "store_id", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)
    validation_split = chronological_train_test_split(validation_source, test_size=0.25)
    calibration_models = fit_model_pipelines(validation_split.train)
    y_validation = validation_split.holdout[TARGET_COLUMN].astype(int).to_numpy()
    x_validation = select_model_features(validation_split.holdout)
    thresholds: dict[str, dict[str, Any]] = {}
    for name, model in calibration_models.items():
        validation_probabilities = model.predict_proba(x_validation)[:, 1]
        thresholds[name] = select_validation_threshold(y_validation, validation_probabilities)

    # Load the saved Phase 6 pipelines and call predict_proba once per model on
    # holdout. No holdout labels are used until after probabilities are fixed.
    saved_models = {name: load_model_pipeline(name, model_dir) for name in MODEL_NAMES}
    x_holdout = select_model_features(phase6_split.holdout)
    y_holdout = phase6_split.holdout[TARGET_COLUMN].astype(int).to_numpy()
    holdout_probabilities = {
        name: model.predict_proba(x_holdout)[:, 1] for name, model in saved_models.items()
    }

    model_results: dict[str, Any] = {}
    for name in MODEL_NAMES:
        default_result = evaluate_binary_predictions(
            y_holdout, holdout_probabilities[name], threshold=DEFAULT_THRESHOLD
        )
        selected_threshold = thresholds[name]["threshold"]
        selected_result = evaluate_binary_predictions(
            y_holdout, holdout_probabilities[name], threshold=selected_threshold
        )
        model_results[name] = {
            "threshold_selection": thresholds[name],
            "holdout_default_threshold": default_result,
            "holdout_selected_threshold": selected_result,
        }

    forest_importance = extract_random_forest_importance(saved_models["random_forest"])
    logistic_coefficients = extract_logistic_coefficients(saved_models["logistic_regression"])
    comparison, narrative = _make_comparison(model_results)

    evaluation_details = {
        "dataset": str(input_path),
        "rows": len(phase6_split.holdout),
        "feature_date_start": phase6_split.holdout["timestamp"].min().isoformat(),
        "feature_date_end": phase6_split.holdout["timestamp"].max().isoformat(),
        "target_date_end": (phase6_split.holdout["timestamp"].max() + NEXT_LABEL_HORIZON).isoformat(),
        "positive_negative_distribution": target_distribution(phase6_split.holdout),
        "holdout_start": phase6_split.holdout_start.isoformat(),
        "training_rows": len(phase6_split.train),
        "purged_boundary_rows": len(phase6_split.purged),
        "purged_feature_date": phase6_split.purged["timestamp"].min().isoformat() if len(phase6_split.purged) else None,
        "training_label_date_end": (phase6_split.train["timestamp"].max() + NEXT_LABEL_HORIZON).isoformat(),
        "threshold_validation": {
            "rows": len(validation_split.holdout),
            "feature_date_start": validation_split.holdout["timestamp"].min().isoformat(),
            "feature_date_end": validation_split.holdout["timestamp"].max().isoformat(),
            "fit_rows": len(validation_split.train),
            "purged_boundary_rows": len(validation_split.purged),
            "method": "Temporary copies of the Phase 6 pipeline are fit on earlier training dates. Threshold maximizes validation F1 over 0.05–0.95 in 0.01 steps; ties prefer higher recall, then precision, then threshold.",
            "holdout_used_for_threshold_selection": False,
        },
        "holdout_used_for_fit_or_tuning": False,
        "saved_model_artifacts_modified": False,
    }
    report = {
        "evaluation": evaluation_details,
        "models": model_results,
        "model_comparison": narrative,
        "features": {
            "random_forest_importance": forest_importance.to_dict(orient="records"),
            "logistic_regression_coefficients": logistic_coefficients.to_dict(orient="records"),
            "interpretation_note": "These values describe fitted model associations, not causes. Positive logistic coefficients raise fitted log-odds and negative coefficients lower them, holding other encoded inputs fixed. Numeric coefficients are per standard deviation; one-hot terms are conditional contributions relative to the fitted intercept and other features, not simple category-versus-reference effects because all levels are encoded.",
        },
        "figures": [
            "figures/logistic_regression_confusion_matrix.png",
            "figures/random_forest_confusion_matrix.png",
            "figures/roc_curve_comparison.png",
            "figures/precision_recall_curve_comparison.png",
            "figures/model_metric_comparison.png",
            "figures/random_forest_feature_importance.png",
            "figures/logistic_regression_coefficients.png",
        ],
    }
    (root / "model_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    comparison.to_csv(root / "model_comparison.csv", index=False)
    forest_importance.to_csv(root / "random_forest_feature_importance.csv", index=False)
    logistic_coefficients.to_csv(root / "logistic_regression_coefficients.csv", index=False)
    _create_plots(
        y_true=y_holdout,
        probabilities=holdout_probabilities,
        evaluation=model_results,
        forest_importance=forest_importance,
        logistic_coefficients=logistic_coefficients,
        figures_dir=figures_dir,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    report = run_model_evaluation(args.input, args.model_dir, args.report_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
