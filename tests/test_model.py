"""
EduPredict Cloud — Tests: Model Service Unit Tests
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Test ModelService in isolation (does not require model artifact on disk)
# ---------------------------------------------------------------------------

class TestModelServicePredict:
    """Test predict() with a mocked pipeline."""

    def _make_service_with_mock(self, return_value: float):
        """Creates a ModelService with all attributes manually mocked."""
        from app.model_service import ModelService
        svc = object.__new__(ModelService)          # skip __init__
        svc.pipeline  = MagicMock()
        svc.pipeline.predict = MagicMock(return_value=np.array([return_value]))
        svc.is_loaded = True
        svc.metadata  = {
            "feature_columns": [
                "G1", "G2", "studytime", "failures", "absences",
                "sex_encoded", "age", "Medu", "Fedu",
                "famrel", "freetime", "goout", "Dalc", "Walc", "health",
            ],
            "model_name": "MockModel",
            "model_version": "1.0",
            "best_metrics": {"mae": 1.0, "rmse": 1.5, "r2": 0.80},
            "feature_descriptions": {},
            "training_samples": 519,
            "test_samples": 130,
        }
        return svc

    def _valid_features(self):
        return {
            "G1": 12, "G2": 13, "studytime": 2, "failures": 0,
            "absences": 4, "sex_encoded": 1, "age": 17,
            "Medu": 2, "Fedu": 2, "famrel": 4, "freetime": 3,
            "goout": 3, "Dalc": 1, "Walc": 2, "health": 4,
        }

    def test_predict_returns_float(self):
        svc = self._make_service_with_mock(13.5)
        result = svc.predict(self._valid_features())
        assert isinstance(result, float)

    def test_predict_returns_correct_value(self):
        svc = self._make_service_with_mock(13.5)
        result = svc.predict(self._valid_features())
        assert abs(result - 13.5) < 0.01

    def test_predict_clamps_above_20(self):
        svc = self._make_service_with_mock(25.0)
        result = svc.predict(self._valid_features())
        assert result == 20.0

    def test_predict_clamps_below_0(self):
        svc = self._make_service_with_mock(-5.0)
        result = svc.predict(self._valid_features())
        assert result == 0.0

    def test_predict_raises_when_not_loaded(self):
        from app.model_service import ModelService
        svc = object.__new__(ModelService)
        svc.pipeline  = None
        svc.is_loaded = False
        svc.metadata  = {}
        with pytest.raises(RuntimeError, match="not loaded"):
            svc.predict(self._valid_features())

    def test_predict_raises_on_missing_feature(self):
        svc = self._make_service_with_mock(12.0)
        features = self._valid_features()
        del features["G1"]
        with pytest.raises(ValueError, match="Missing feature values"):
            svc.predict(features)

    def test_predict_raises_on_nan_prediction(self):
        svc = self._make_service_with_mock(float("nan"))
        with pytest.raises(RuntimeError, match="non-finite"):
            svc.predict(self._valid_features())

    def test_predict_raises_on_inf_prediction(self):
        svc = self._make_service_with_mock(float("inf"))
        with pytest.raises(RuntimeError, match="non-finite"):
            svc.predict(self._valid_features())

    def test_predict_score_rounded_to_2dp(self):
        svc = self._make_service_with_mock(13.456789)
        result = svc.predict(self._valid_features())
        assert result == round(13.456789, 2)


# ---------------------------------------------------------------------------
# Test model artifact loading behaviour
# ---------------------------------------------------------------------------

class TestModelServiceLoading:

    def test_model_not_loaded_when_file_missing(self, tmp_path):
        """ModelService.is_loaded should be False if model file absent."""
        from app.model_service import ModelService
        svc = object.__new__(ModelService)
        svc.pipeline  = None
        svc.metadata  = {}
        svc.is_loaded = False
        assert not svc.is_loaded

    def test_model_name_none_when_not_loaded(self):
        from app.model_service import ModelService
        svc = object.__new__(ModelService)
        svc.pipeline  = None
        svc.metadata  = {}
        svc.is_loaded = False
        assert svc.model_name is None

    def test_feature_descriptions_empty_when_not_loaded(self):
        from app.model_service import ModelService
        svc = object.__new__(ModelService)
        svc.pipeline  = None
        svc.metadata  = {}
        svc.is_loaded = False
        assert svc.feature_descriptions == {}


# ---------------------------------------------------------------------------
# Test schema validation (Pydantic)
# ---------------------------------------------------------------------------

class TestSchemas:

    def _valid_data(self):
        return {
            "G1": 12, "G2": 13, "studytime": 2, "failures": 0,
            "absences": 4, "sex": "M", "age": 17,
            "Medu": 2, "Fedu": 2, "famrel": 4, "freetime": 3,
            "goout": 3, "Dalc": 1, "Walc": 2, "health": 4,
        }

    def test_valid_request_parses(self):
        from app.schemas import PredictionRequest
        req = PredictionRequest(**self._valid_data())
        assert req.G1 == 12

    def test_sex_normalised_to_uppercase(self):
        from app.schemas import PredictionRequest
        data = {**self._valid_data(), "sex": "f"}
        req = PredictionRequest(**data)
        assert req.sex == "F"

    def test_invalid_sex_raises(self):
        from pydantic import ValidationError
        from app.schemas import PredictionRequest
        with pytest.raises(ValidationError):
            PredictionRequest(**{**self._valid_data(), "sex": "X"})

    def test_G1_too_high_raises(self):
        from pydantic import ValidationError
        from app.schemas import PredictionRequest
        with pytest.raises(ValidationError):
            PredictionRequest(**{**self._valid_data(), "G1": 25})

    def test_failures_too_high_raises(self):
        from pydantic import ValidationError
        from app.schemas import PredictionRequest
        with pytest.raises(ValidationError):
            PredictionRequest(**{**self._valid_data(), "failures": 4})

    def test_age_too_low_raises(self):
        from pydantic import ValidationError
        from app.schemas import PredictionRequest
        with pytest.raises(ValidationError):
            PredictionRequest(**{**self._valid_data(), "age": 10})


# ---------------------------------------------------------------------------
# ML pipeline test — reproducing training
# ---------------------------------------------------------------------------

class TestMLPipeline:

    def test_synthetic_data_generation(self):
        """Confirm synthetic data generator produces expected shape."""
        sys.path.insert(0, str(PROJECT_ROOT / "ml"))
        from generate_synthetic_data import generate_dataset
        df = generate_dataset(n=100, seed=0)
        assert df.shape == (100, 16)
        assert "G3" in df.columns
        assert df["G3"].between(0, 20).all()

    def test_G3_in_valid_range(self):
        from generate_synthetic_data import generate_dataset
        df = generate_dataset(n=500, seed=42)
        assert df["G3"].min() >= 0
        assert df["G3"].max() <= 20

    def test_reproducibility(self):
        from generate_synthetic_data import generate_dataset
        df1 = generate_dataset(n=100, seed=7)
        df2 = generate_dataset(n=100, seed=7)
        assert df1.equals(df2)
