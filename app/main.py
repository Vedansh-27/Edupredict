"""
EduPredict Cloud — FastAPI Application
=======================================
Entry point for the prediction service.
Serves the frontend and exposes the ML prediction REST API.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Logging setup — must happen before any local imports
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
_HERE         = Path(__file__).resolve().parent   # app/
_PROJECT_ROOT = _HERE.parent                       # project root
_STATIC_DIR   = _PROJECT_ROOT / "static"
_TEMPLATE_DIR = _PROJECT_ROOT / "templates"

# Add project root to sys.path so app/ imports work regardless of CWD
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from app.model_service import model_service
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="EduPredict Cloud",
    description=(
        "A cloud-deployed machine learning system for student academic "
        "performance prediction. Predicts final examination score (0–20 scale) "
        "from historical academic features.\n\n"
        "**Disclaimer:** Predictions are statistical estimates only and must not "
        "be used to make consequential decisions about individual students."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — allow same-origin + localhost during development.
# In production on Render, the frontend and API share the same origin,
# so CORS is not strictly required. The list below is explicit and restricted.
# ---------------------------------------------------------------------------
_ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    # Add your Render URL here after deployment, e.g.:
    # "https://edupredict-cloud.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)

# ---------------------------------------------------------------------------
# Static files and templates
# ---------------------------------------------------------------------------
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))


# ---------------------------------------------------------------------------
# Lifespan handler (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application):
    # Startup
    if model_service.is_loaded:
        log.info(f"[startup] Model ready: {model_service.model_name}")
    else:
        log.warning(
            "[startup] Model NOT loaded. The /predict endpoint will return 503. "
            "Run 'python ml/train_model.py' and restart the server."
        )
    yield
    # Shutdown (nothing to clean up)


# Recreate app with lifespan (must happen before route registration below,
# so we patch the lifespan onto the already-created app object)
app.router.lifespan_context = lifespan


# ===========================================================================
# Endpoints
# ===========================================================================

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root(request: Request):
    """Serve the main prediction dashboard."""
    return templates.TemplateResponse(request, "index.html")


# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    Returns the application health status and whether the ML model is loaded.
    Use this endpoint for Render health checks and uptime monitoring.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model_service.is_loaded,
        model_name=model_service.model_name if model_service.is_loaded else None,
    )


# ---------------------------------------------------------------------------

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """
    Returns metadata about the deployed model, including supported features,
    their valid ranges, and evaluation metrics measured on held-out test data.
    """
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. The service is starting up or the model artifact is missing.",
        )
    return ModelInfoResponse(
        model_name=model_service.model_name,
        model_version=model_service.model_version,
        target_column="G3",
        target_scale="0-20",
        feature_descriptions=model_service.feature_descriptions,
        evaluation_metrics=model_service.evaluation_metrics,
        dataset_source="Synthetic dataset mirroring UCI Student Performance (see data/README.md)",
        training_samples=model_service.training_samples,
        test_samples=model_service.test_samples,
    )


# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predicts a student's final examination grade (G3, 0–20 scale).

    Submit validated student features as JSON. The model returns an estimated
    final grade.

    **Important:** This prediction is a statistical estimate. It should not be
    used to make decisions about individual students without human oversight.
    """
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not currently available. Please try again later.",
        )

    # Encode 'sex' to numeric as done during training
    sex_encoded = 1 if request.sex.upper() == "M" else 0

    # Build feature dict matching training feature order exactly
    features = {
        "G1":          request.G1,
        "G2":          request.G2,
        "studytime":   request.studytime,
        "failures":    request.failures,
        "absences":    request.absences,
        "sex_encoded": sex_encoded,
        "age":         request.age,
        "Medu":        request.Medu,
        "Fedu":        request.Fedu,
        "famrel":      request.famrel,
        "freetime":    request.freetime,
        "goout":       request.goout,
        "Dalc":        request.Dalc,
        "Walc":        request.Walc,
        "health":      request.health,
    }

    try:
        score = model_service.predict(features)
    except ValueError as exc:
        log.warning(f"Prediction rejected (bad input): {exc}")
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        log.error(f"Prediction runtime error: {exc}")
        raise HTTPException(status_code=503, detail="Prediction failed. Please try again later.")
    except Exception:
        log.exception("Unexpected prediction error")
        raise HTTPException(status_code=500, detail="An internal error occurred.")

    # Derive interpretation band
    if score >= 17:
        interpretation = "Excellent — student is performing at a very high level."
    elif score >= 14:
        interpretation = "Good — student is performing above average."
    elif score >= 10:
        interpretation = "Satisfactory — student meets passing threshold."
    elif score >= 7:
        interpretation = "At risk — student may need additional academic support."
    else:
        interpretation = "Critical — student is significantly below the passing threshold."

    log.info(f"Prediction complete: score={score}, interpretation='{interpretation}'")

    return PredictionResponse(
        predicted_score=score,
        score_scale="0-20",
        message="Estimated final examination score (G3). This is a statistical estimate only.",
        model_version=model_service.model_version,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Global exception handler — prevent stack trace leakage
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please contact the administrator."},
    )


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
