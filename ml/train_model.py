"""
EduPredict Cloud — Model Training Pipeline
==========================================
Trains and evaluates regression models to predict student final grade (G3, 0–20 scale).
Saves the best model pipeline and a machine-readable evaluation report.

Usage:
    python ml/train_model.py
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE    = PROJECT_ROOT / "data" / "student_performance.csv"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"
ML_DIR       = PROJECT_ROOT / "ml"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_COLUMN = "G3"

# Features used for training — order is critical and stored in metadata
FEATURE_COLUMNS = [
    "G1", "G2", "studytime", "failures", "absences",
    "sex_encoded", "age", "Medu", "Fedu",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
]

RANDOM_SEED = 42
TEST_SIZE   = 0.2


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV, validate columns, encode categoricals, return X and y."""
    log.info(f"Loading dataset from {path}")
    df = pd.read_csv(path)
    log.info(f"Dataset shape: {df.shape}")

    # Validate required columns
    required = {"G1", "G2", "G3", "studytime", "failures", "absences",
                "sex", "age", "Medu", "Fedu", "famrel", "freetime",
                "goout", "Dalc", "Walc", "health"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Drop rows with NaN in required columns
    before = len(df)
    df = df.dropna(subset=list(required))
    after = len(df)
    if before != after:
        log.warning(f"Dropped {before - after} rows with missing values.")

    # Encode sex: M=1, F=0
    le = LabelEncoder()
    df["sex_encoded"] = le.fit_transform(df["sex"].str.strip())
    log.info(f"sex encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Validate target range
    invalid_target = df[(df[TARGET_COLUMN] < 0) | (df[TARGET_COLUMN] > 20)]
    if len(invalid_target):
        log.warning(f"Clamping {len(invalid_target)} out-of-range target values.")
        df[TARGET_COLUMN] = df[TARGET_COLUMN].clip(0, 20)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    log.info(f"Feature matrix shape: {X.shape}")
    log.info(f"Target distribution:\n{y.describe()}")

    return X, y


# ---------------------------------------------------------------------------
# Model evaluation helper
# ---------------------------------------------------------------------------

def evaluate_model(name: str, pipeline: Pipeline, X_test: pd.DataFrame,
                   y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    log.info(f"[{name}] MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return {"model": name, "mae": round(mae, 4), "rmse": round(rmse, 4),
            "r2": round(r2, 4)}


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train():
    # ------------------------------------------------------------------ #
    # 1. Generate dataset if it doesn't exist                             #
    # ------------------------------------------------------------------ #
    if not DATA_FILE.exists():
        log.info("Dataset not found — generating synthetic data …")
        sys.path.insert(0, str(ML_DIR))
        from generate_synthetic_data import main as gen_main
        gen_main()

    # ------------------------------------------------------------------ #
    # 2. Load & preprocess                                                #
    # ------------------------------------------------------------------ #
    X, y = load_and_preprocess(DATA_FILE)

    # ------------------------------------------------------------------ #
    # 3. Train/test split                                                 #
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    log.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # ------------------------------------------------------------------ #
    # 4. Define candidate models (all inside a Pipeline with scaling)     #
    # ------------------------------------------------------------------ #
    candidates = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=200, max_depth=10,
                min_samples_leaf=3, random_state=RANDOM_SEED, n_jobs=-1
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=RANDOM_SEED
            )),
        ]),
    }

    # ------------------------------------------------------------------ #
    # 5. Train and evaluate each candidate                                #
    # ------------------------------------------------------------------ #
    results = []
    trained_pipelines = {}

    for name, pipeline in candidates.items():
        log.info(f"Training {name} …")
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(name, pipeline, X_test, y_test)

        # 5-fold cross-validation R² on training set
        cv_scores = cross_val_score(pipeline, X_train, y_train,
                                    cv=5, scoring="r2", n_jobs=-1)
        metrics["cv_r2_mean"] = round(float(cv_scores.mean()), 4)
        metrics["cv_r2_std"]  = round(float(cv_scores.std()), 4)
        log.info(f"  CV R² = {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")

        results.append(metrics)
        trained_pipelines[name] = pipeline

    # ------------------------------------------------------------------ #
    # 6. Select best model by test R²                                     #
    # ------------------------------------------------------------------ #
    results.sort(key=lambda x: x["r2"], reverse=True)
    best_result  = results[0]
    best_name    = best_result["model"]
    best_pipeline = trained_pipelines[best_name]

    log.info(f"\n{'='*60}")
    log.info(f"Best model: {best_name}")
    log.info(f"  MAE={best_result['mae']}  RMSE={best_result['rmse']}  R²={best_result['r2']}")
    log.info(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # 7. Save model artifact                                              #
    # ------------------------------------------------------------------ #
    model_path = MODELS_DIR / "student_performance_model.joblib"
    joblib.dump(best_pipeline, model_path)
    log.info(f"Model saved to {model_path}")

    # ------------------------------------------------------------------ #
    # 8. Save metadata                                                    #
    # ------------------------------------------------------------------ #
    metadata = {
        "model_name":      best_name,
        "model_version":   "1.0",
        "target_column":   TARGET_COLUMN,
        "target_scale":    "0-20",
        "feature_columns": FEATURE_COLUMNS,
        "feature_descriptions": {
            "G1":          "First period grade (0-20)",
            "G2":          "Second period grade (0-20)",
            "studytime":   "Weekly study time (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)",
            "failures":    "Number of past class failures (0-3)",
            "absences":    "Number of school absences (0-74)",
            "sex_encoded": "Student sex: Male=1, Female=0",
            "age":         "Student age (15-22)",
            "Medu":        "Mother education (0=none to 4=higher)",
            "Fedu":        "Father education (0=none to 4=higher)",
            "famrel":      "Family relationship quality (1=very bad to 5=excellent)",
            "freetime":    "Free time after school (1=very low to 5=very high)",
            "goout":       "Going out with friends (1=very low to 5=very high)",
            "Dalc":        "Weekday alcohol consumption (1=very low to 5=very high)",
            "Walc":        "Weekend alcohol consumption (1=very low to 5=very high)",
            "health":      "Current health status (1=very bad to 5=very good)",
        },
        "training_samples": int(len(X_train)),
        "test_samples":     int(len(X_test)),
        "dataset_source":   "synthetic (see data/README.md)",
        "random_seed":      RANDOM_SEED,
        "best_metrics": {
            "mae":  best_result["mae"],
            "rmse": best_result["rmse"],
            "r2":   best_result["r2"],
        },
        "selection_rationale": (
            f"{best_name} achieved the highest test R² ({best_result['r2']}) "
            f"among {len(candidates)} candidates (LinearRegression, Ridge, "
            f"RandomForest, GradientBoosting), evaluated on a held-out 20% "
            f"test split with random seed {RANDOM_SEED}."
        ),
    }

    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Metadata saved to {metadata_path}")

    # ------------------------------------------------------------------ #
    # 9. Save full evaluation report                                       #
    # ------------------------------------------------------------------ #
    report = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "dataset": {
            "path":      str(DATA_FILE),
            "source":    "synthetic (see data/README.md)",
            "n_rows":    int(len(X) + 0),
            "n_features": int(len(FEATURE_COLUMNS)),
        },
        "all_model_results": results,
        "selected_model": best_result,
    }

    report_path = REPORTS_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Evaluation report saved to {report_path}")

    log.info("\nTraining complete. All artifacts saved.\n")
    return best_pipeline, metadata


if __name__ == "__main__":
    train()
