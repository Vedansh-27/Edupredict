"""
EduPredict Cloud — Model Evaluation and Visualization
=====================================================
Loads the saved model artifact and generates evaluation plots.
Run after train_model.py.

Usage:
    python ml/evaluate_model.py
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE    = PROJECT_ROOT / "data" / "student_performance.csv"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    # Load metadata
    metadata_path = MODELS_DIR / "model_metadata.json"
    if not metadata_path.exists():
        log.error("model_metadata.json not found. Run ml/train_model.py first.")
        sys.exit(1)

    with open(metadata_path) as f:
        metadata = json.load(f)

    feature_columns = metadata["feature_columns"]
    target_column   = metadata["target_column"]

    # Load model
    model_path = MODELS_DIR / "student_performance_model.joblib"
    pipeline = joblib.load(model_path)
    log.info(f"Loaded model: {metadata['model_name']}")

    # Load dataset
    df = pd.read_csv(DATA_FILE)
    df["sex_encoded"] = (df["sex"].str.strip() == "M").astype(int)
    X = df[feature_columns]
    y = df[target_column]

    # Predict
    y_pred = pipeline.predict(X)

    # Metrics
    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2   = r2_score(y, y_pred)

    log.info(f"Full-dataset metrics: MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    # ------------------------------------------------------------------ #
    # Plot 1: Actual vs Predicted                                         #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(y, y_pred, alpha=0.4, color="#2563EB", edgecolors="none", s=20)
    min_val, max_val = 0, 20
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual G3 (Final Grade)")
    ax.set_ylabel("Predicted G3")
    ax.set_title(f"Actual vs Predicted — {metadata['model_name']}")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.legend()
    ax.text(0.05, 0.95, f"MAE={mae:.2f}\nRMSE={rmse:.2f}\nR²={r2:.3f}",
            transform=ax.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # ------------------------------------------------------------------ #
    # Plot 2: Residuals                                                   #
    # ------------------------------------------------------------------ #
    residuals = y - y_pred
    ax2 = axes[1]
    ax2.scatter(y_pred, residuals, alpha=0.4, color="#7C3AED", edgecolors="none", s=20)
    ax2.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Predicted G3")
    ax2.set_ylabel("Residual (Actual - Predicted)")
    ax2.set_title("Residuals vs Predicted")

    plt.tight_layout()
    plot_path = REPORTS_DIR / "actual_vs_predicted.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"Plot saved to {plot_path}")

    # ------------------------------------------------------------------ #
    # Feature importance (if supported)                                   #
    # ------------------------------------------------------------------ #
    regressor = pipeline.named_steps.get("model")
    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
        indices = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(importances)), importances[indices], color="#2563EB", alpha=0.8)
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([feature_columns[i] for i in indices], rotation=45, ha="right")
        ax.set_title(f"Feature Importances — {metadata['model_name']}")
        ax.set_ylabel("Importance")
        plt.tight_layout()
        fi_path = REPORTS_DIR / "feature_importances.png"
        plt.savefig(fi_path, dpi=120, bbox_inches="tight")
        plt.close()
        log.info(f"Feature importance plot saved to {fi_path}")


if __name__ == "__main__":
    main()
