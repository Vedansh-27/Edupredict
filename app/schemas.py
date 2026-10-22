"""
EduPredict Cloud — Pydantic Request/Response Schemas
=====================================================
Defines validated input/output shapes for the FastAPI prediction endpoint.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """
    Input features for student final grade prediction.
    All values must match the ranges present in the training data.
    """

    G1: float = Field(..., ge=0, le=20, description="First period grade (0-20)")
    G2: float = Field(..., ge=0, le=20, description="Second period grade (0-20)")
    studytime: int = Field(..., ge=1, le=4,
                           description="Weekly study time: 1=<2h, 2=2-5h, 3=5-10h, 4=>10h")
    failures: int = Field(..., ge=0, le=3,
                          description="Number of past class failures (0-3)")
    absences: int = Field(..., ge=0, le=74,
                          description="Number of school absences (0-74)")
    sex: str = Field(..., description="Student sex: 'M' or 'F'")
    age: int = Field(..., ge=15, le=22, description="Student age (15-22)")
    Medu: int = Field(..., ge=0, le=4,
                      description="Mother education: 0=none, 1=primary, 2=5th-9th, 3=secondary, 4=higher")
    Fedu: int = Field(..., ge=0, le=4,
                      description="Father education: 0=none to 4=higher")
    famrel: int = Field(..., ge=1, le=5,
                        description="Family relationship quality (1=very bad to 5=excellent)")
    freetime: int = Field(..., ge=1, le=5,
                          description="Free time after school (1=very low to 5=very high)")
    goout: int = Field(..., ge=1, le=5,
                       description="Going out with friends (1=very low to 5=very high)")
    Dalc: int = Field(..., ge=1, le=5,
                      description="Weekday alcohol consumption (1=very low to 5=very high)")
    Walc: int = Field(..., ge=1, le=5,
                      description="Weekend alcohol consumption (1=very low to 5=very high)")
    health: int = Field(..., ge=1, le=5,
                        description="Current health status (1=very bad to 5=very good)")

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if cleaned not in ("M", "F"):
            raise ValueError("sex must be 'M' or 'F'")
        return cleaned

    model_config = {
        "json_schema_extra": {
            "example": {
                "G1": 12,
                "G2": 13,
                "studytime": 2,
                "failures": 0,
                "absences": 4,
                "sex": "M",
                "age": 17,
                "Medu": 2,
                "Fedu": 2,
                "famrel": 4,
                "freetime": 3,
                "goout": 3,
                "Dalc": 1,
                "Walc": 2,
                "health": 4,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Prediction API response."""
    predicted_score: float = Field(..., description="Predicted final grade (0-20 scale)")
    score_scale: str = Field(default="0-20", description="Scale of the predicted score")
    message: str = Field(..., description="Human-readable interpretation")
    model_version: str = Field(default="1.0")
    interpretation: str = Field(..., description="Performance-level interpretation")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    target_column: str
    target_scale: str
    feature_descriptions: dict
    evaluation_metrics: dict
    dataset_source: str
    training_samples: int
    test_samples: int
