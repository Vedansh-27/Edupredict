# Implementation Walkthrough — EduPredict Cloud
*Technical explanation of the actual source code for FT-III presentation.*

---

## 1. Repository Structure

```
EduPredict-Cloud/
├── app/
│   ├── __init__.py          # Python package marker
│   ├── main.py              # FastAPI app, endpoints, startup
│   ├── schemas.py           # Pydantic request/response models
│   └── model_service.py     # Model loading + inference singleton
├── ml/
│   ├── generate_synthetic_data.py   # Reproducible dataset generator
│   ├── train_model.py               # Model training pipeline
│   └── evaluate_model.py            # Evaluation plots
├── data/
│   ├── README.md                    # Dataset documentation
│   └── student_performance.csv      # Synthetic dataset (committed)
├── models/
│   ├── student_performance_model.joblib  # Trained sklearn pipeline
│   └── model_metadata.json               # Feature list, metrics
├── static/
│   ├── css/styles.css       # Responsive dashboard stylesheet
│   └── js/app.js            # Frontend API calls and rendering
├── templates/
│   └── index.html           # Jinja2 HTML template
├── tests/
│   ├── test_health.py       # Health endpoint tests
│   ├── test_predict.py      # Prediction endpoint tests
│   └── test_model.py        # Model service unit tests
├── scripts/
│   └── performance_test.py  # Latency benchmarking script
├── reports/
│   └── evaluation_report.json  # Actual measured ML metrics
├── presentation_materials/  # FT-III presentation support
├── requirements.txt
├── requirements-dev.txt
├── render.yaml              # Render PaaS configuration
├── .gitignore
├── .env.example
└── README.md
```

---

## 2. ML Pipeline (ml/train_model.py)

### Data Generation
`ml/generate_synthetic_data.py` creates 649 rows with 16 columns using NumPy's `default_rng(42)`.
Grades G1 and G2 are generated with realistic statistical correlations with study time, failures, and absences.
G3 is derived as `0.6*G2 + 0.3*G1 + noise - failures_penalty`, then clamped to [0, 20].

### Feature Engineering
`sex` (categorical: M/F) is label-encoded to `sex_encoded` (1/0) before training.
All 15 numeric features are standardised via `StandardScaler` inside the sklearn `Pipeline`.

### Training Pipeline
```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                         learning_rate=0.05, random_state=42))
])
```
The Pipeline ensures the scaler is fit only on training data and consistently applied at inference time, preventing data leakage.

### Model Selection
Four candidates trained: LinearRegression, Ridge, RandomForest, GradientBoosting.
Selected by highest test R² on a stratified 80/20 split (seed=42).
5-fold CV R² also computed and logged for robustness.

### Artifacts
- `models/student_performance_model.joblib` — serialised sklearn Pipeline
- `models/model_metadata.json` — feature list, descriptions, metrics
- `reports/evaluation_report.json` — full comparison table

---

## 3. Backend (app/)

### app/main.py — FastAPI Application

**Startup:**
```python
@app.on_event("startup")
async def startup_event():
    if model_service.is_loaded:
        log.info(f"✓ Model ready: {model_service.model_name}")
```
The model loads at import time via `model_service = ModelService()`. FastAPI's startup event logs the status.

**Endpoints:**
- `GET /` → renders `templates/index.html` via Jinja2
- `GET /health` → returns `{status, model_loaded, model_name}`
- `GET /model-info` → returns metadata and metrics
- `POST /predict` → validates → encodes sex → calls `model_service.predict()` → returns result

**Error handling:** A global exception handler catches unhandled exceptions and returns a generic 500 response, preventing stack trace leakage.

### app/schemas.py — Pydantic Validation

```python
class PredictionRequest(BaseModel):
    G1: float = Field(..., ge=0, le=20)
    sex: str  # validated by @field_validator
    studytime: int = Field(..., ge=1, le=4)
    ...
```
FastAPI automatically returns 422 with field-level messages when validation fails.

### app/model_service.py — Model Service

```python
class ModelService:
    def predict(self, features: dict) -> float:
        X = pd.DataFrame([row], columns=feature_columns)
        pred = self.pipeline.predict(X)[0]
        if not math.isfinite(pred):
            raise RuntimeError(...)
        return round(float(np.clip(pred, 0, 20)), 2)
```
Features are assembled in fixed order from `model_metadata.json`.
Output is clamped to [0, 20] and checked for NaN/infinity.

---

## 4. Frontend (templates/, static/)

### index.html
Structured as:
1. Header with branding and live health indicator
2. Hero section with disclaimer banner
3. Main grid: form panel (left) + result/model-info panel (right)
4. Footer with API links

### static/js/app.js

**Health polling:**
```javascript
const resp = await fetch("/health");
// Updates green/red dot in header
```

**Prediction flow:**
```javascript
form.addEventListener("submit", async (e) => {
    if (!validateForm()) return;
    const payload = buildPayload();
    const resp = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
    });
    const data = await resp.json();
    renderResult(data);  // shows score bar, badge, interpretation
});
```
No hardcoded predictions — all data comes from the live API.

---

## 5. Deployment Configuration (render.yaml)

```yaml
buildCommand: |
  pip install -r requirements.txt &&
  python ml/generate_synthetic_data.py &&
  python ml/train_model.py

startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
healthCheckPath: /health
```

`$PORT` is assigned by Render at runtime (typically 10000). Uvicorn binds to `0.0.0.0` to accept connections from outside the container. The build step ensures the model artifact exists before the server starts.

---

## 6. Testing Strategy

| Test Type | File | Tool |
|-----------|------|------|
| Health endpoint | tests/test_health.py | FastAPI TestClient |
| Prediction (valid/invalid) | tests/test_predict.py | FastAPI TestClient |
| Model service unit tests | tests/test_model.py | pytest + MagicMock |
| ML pipeline reproducibility | tests/test_model.py | Direct function calls |
| API performance | scripts/performance_test.py | urllib + stdlib |

Run all tests: `pytest tests/ -v`
