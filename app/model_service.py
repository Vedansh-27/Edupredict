"""
EduPredict Cloud — Model Loading and Inference Service
======================================================
Loads the serialized sklearn pipeline and metadata at startup,
then exposes a safe inference function.
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — relative to project root, or resolved from this file's location
# ---------------------------------------------------------------------------
_HERE        = Path(__file__).resolve().parent          # app/
_PROJECT_ROOT = _HERE.parent                             # project root
_MODELS_DIR  = _PROJECT_ROOT / "models"
_MODEL_FILE  = _MODELS_DIR / "student_performance_model.joblib"
_META_FILE   = _MODELS_DIR / "model_metadata.json"


class ModelService:
    """
    Singleton-style service that loads the model once and exposes predict().
    Raises clear exceptions if files are missing rather than silently failing.
    """

    def __init__(self):
        self.pipeline     = None
        self.metadata: dict = {}
        self.is_loaded    = False
        self._load()

    # ------------------------------------------------------------------
    def _load(self):
        """Attempt to load model and metadata. Logs clearly on failure."""
        if not _MODEL_FILE.exists():
            log.error(
                f"Model artifact not found at {_MODEL_FILE}. "
                "Run 'python ml/train_model.py' to generate it."
            )
            return

        if not _META_FILE.exists():
            log.error(f"Model metadata not found at {_META_FILE}.")
            return

        try:
            self.pipeline = joblib.load(_MODEL_FILE)
            with open(_META_FILE) as f:
                self.metadata = json.load(f)
            self.is_loaded = True
            log.info(
                f"Model loaded: {self.metadata.get('model_name')} v{self.metadata.get('model_version')}"
            )
        except Exception as exc:
            log.exception(f"Failed to load model: {exc}")
            self.is_loaded = False

    # ------------------------------------------------------------------
    def predict(self, features: dict) -> float:
        """
        Run inference on a validated feature dict.
        Returns the predicted final grade (float, 0–20).
        Raises ValueError if the model is not loaded or output is non-finite.
        """
        if not self.is_loaded or self.pipeline is None:
            raise RuntimeError("Model is not loaded. Cannot predict.")

        feature_columns: list = self.metadata.get("feature_columns", [])
        if not feature_columns:
            raise RuntimeError("Feature column list is missing from metadata.")

        # Build ordered feature array
        row = [features.get(col) for col in feature_columns]
        if any(v is None for v in row):
            missing = [col for col in feature_columns if features.get(col) is None]
            raise ValueError(f"Missing feature values for: {missing}")

        X = pd.DataFrame([row], columns=feature_columns)

        # Safety: check for NaN / infinity in input
        if X.isnull().values.any() or np.isinf(X.values).any():
            raise ValueError("Input contains NaN or infinite values after preprocessing.")

        pred = self.pipeline.predict(X)[0]

        # Validate output
        if not math.isfinite(pred):
            raise RuntimeError(f"Model returned non-finite prediction: {pred}")

        # Clamp to valid grade range
        pred = float(np.clip(pred, 0.0, 20.0))
        return round(pred, 2)

    # ------------------------------------------------------------------
    @property
    def model_name(self) -> Optional[str]:
        return self.metadata.get("model_name")

    @property
    def feature_descriptions(self) -> dict:
        return self.metadata.get("feature_descriptions", {})

    @property
    def evaluation_metrics(self) -> dict:
        return self.metadata.get("best_metrics", {})

    @property
    def model_version(self) -> str:
        return self.metadata.get("model_version", "1.0")

    @property
    def training_samples(self) -> int:
        return self.metadata.get("training_samples", 0)

    @property
    def test_samples(self) -> int:
        return self.metadata.get("test_samples", 0)


# ---------------------------------------------------------------------------
# Module-level singleton — loaded once at import time
# ---------------------------------------------------------------------------
model_service = ModelService()
