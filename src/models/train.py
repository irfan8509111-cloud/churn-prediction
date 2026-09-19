"""Train and compare churn classification models.

Compares a Logistic Regression baseline against a tuned Random Forest classifier,
evaluates performance metrics, plots diagnostic curves, and saves the best model.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from src.data.make_dataset import load_config
from src.features.build_features import build_preprocessor, get_features_and_target
from src.visualization.plots import plot_confusion_matrix, plot_model_curves, plot_feature_importances


def evaluate_pipeline(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str = "Model"
) -> Tuple[Dict[str, float], Any]:
    """Calculate and print test metrics for a fitted pipeline."""
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        "pr_auc": round(float(average_precision_score(y_test, probs)), 4)
    }

    print(f"\n--- {name} Results ---")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, digits=3))

    cm = confusion_matrix(y_test, preds)
    return metrics, cm


def train():
    config = load_config()

    # 1. Load train and test data
    train_df = pd.read_csv(config["data"]["train_path"])
    test_df = pd.read_csv(config["data"]["test_path"])

    X_train, y_train = get_features_and_target(train_df, config)
    X_test, y_test = get_features_and_target(test_df, config)

    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape:  {X_test.shape}")

    # 2. Baseline: Logistic Regression
    print("\nFitting Logistic Regression baseline...")
    baseline_pipe = Pipeline([
        ("preprocessor", build_preprocessor(config)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=config["model"]["random_state"]))
    ])
    baseline_pipe.fit(X_train, y_train)
    baseline_metrics, _ = evaluate_pipeline(baseline_pipe, X_test, y_test, name="Logistic Regression (Baseline)")

    # 3. Candidate: Random Forest
    print("\nFitting Random Forest classifier...")
    rf_pipe = Pipeline([
        ("preprocessor", build_preprocessor(config)),
        ("classifier", RandomForestClassifier(
            n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],
            min_samples_leaf=config["model"]["min_samples_leaf"],
            random_state=config["model"]["random_state"],
            n_jobs=-1
        ))
    ])
    rf_pipe.fit(X_train, y_train)
    rf_metrics, rf_cm = evaluate_pipeline(rf_pipe, X_test, y_test, name="Random Forest")

    # 4. Generate figures
    fig_dir = Path(config["visualization"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(rf_cm, output_path=str(fig_dir / "confusion_matrix.png"))

    rf_probs = rf_pipe.predict_proba(X_test)[:, 1]
    plot_model_curves(y_test.values, rf_probs, output_path=str(fig_dir / "model_curves.png"))

    # Extract feature names from preprocessor and plot importances
    preprocessor = rf_pipe.named_steps["preprocessor"]
    rf_model = rf_pipe.named_steps["classifier"]
    cat_names = list(preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(
        config["features"]["categorical"]
    ))
    all_feature_names = config["features"]["numerical"] + cat_names
    plot_feature_importances(
        all_feature_names,
        rf_model.feature_importances_,
        top_n=10,
        output_path=str(fig_dir / "feature_importances.png")
    )

    # 5. Export model artifact and metrics
    output_path = Path(config["model"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_pipe, output_path)
    print(f"\nModel artifact saved to {output_path}")

    report = {
        "baseline_logistic_regression": baseline_metrics,
        "random_forest": rf_metrics,
        "selected_model": "random_forest"
    }
    metrics_path = Path(config["model"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    train()
