"""
EduPredict Cloud — Tests: Prediction Endpoint
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Valid baseline payload
# ---------------------------------------------------------------------------
VALID_PAYLOAD = {
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


# ===========================================================================
# Happy-path tests
# ===========================================================================

def test_predict_valid_returns_200():
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code in (200, 503), (
        f"Expected 200 (model loaded) or 503 (model missing). Got {resp.status_code}"
    )


def test_predict_response_schema():
    resp = client.post("/predict", json=VALID_PAYLOAD)
    if resp.status_code == 503:
        pytest.skip("Model artifact not present — skipping inference test.")
    data = resp.json()
    assert "predicted_score" in data
    assert "score_scale" in data
    assert "message" in data
    assert "model_version" in data
    assert "interpretation" in data


def test_predict_score_in_valid_range():
    resp = client.post("/predict", json=VALID_PAYLOAD)
    if resp.status_code == 503:
        pytest.skip("Model artifact not present.")
    data = resp.json()
    score = data["predicted_score"]
    assert 0.0 <= score <= 20.0, f"Score {score} is out of 0–20 range."


def test_predict_female_student():
    payload = {**VALID_PAYLOAD, "sex": "F", "G1": 15, "G2": 16}
    resp = client.post("/predict", json=payload)
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        assert 0 <= resp.json()["predicted_score"] <= 20


def test_predict_high_failures():
    payload = {**VALID_PAYLOAD, "failures": 3, "G1": 7, "G2": 6}
    resp = client.post("/predict", json=payload)
    assert resp.status_code in (200, 503)


def test_predict_zero_grades():
    payload = {**VALID_PAYLOAD, "G1": 0, "G2": 0, "absences": 74}
    resp = client.post("/predict", json=payload)
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        assert resp.json()["predicted_score"] >= 0


def test_predict_max_grades():
    payload = {**VALID_PAYLOAD, "G1": 20, "G2": 20, "studytime": 4}
    resp = client.post("/predict", json=payload)
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        assert resp.json()["predicted_score"] <= 20


# ===========================================================================
# Missing-field tests
# ===========================================================================

def test_predict_missing_G1():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "G1"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_missing_G2():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "G2"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_missing_sex():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "sex"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_empty_payload():
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


# ===========================================================================
# Invalid type / out-of-range tests
# ===========================================================================

def test_predict_G1_out_of_range_high():
    payload = {**VALID_PAYLOAD, "G1": 25}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_G1_out_of_range_low():
    payload = {**VALID_PAYLOAD, "G1": -1}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_G2_out_of_range():
    payload = {**VALID_PAYLOAD, "G2": 21}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_invalid_sex():
    payload = {**VALID_PAYLOAD, "sex": "X"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_invalid_sex_empty():
    payload = {**VALID_PAYLOAD, "sex": ""}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_failures_out_of_range():
    payload = {**VALID_PAYLOAD, "failures": 5}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_absences_out_of_range():
    payload = {**VALID_PAYLOAD, "absences": 100}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_age_out_of_range():
    payload = {**VALID_PAYLOAD, "age": 30}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_studytime_out_of_range():
    payload = {**VALID_PAYLOAD, "studytime": 5}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_string_where_number():
    payload = {**VALID_PAYLOAD, "G1": "abc"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_null_value():
    payload = {**VALID_PAYLOAD, "G1": None}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


# ===========================================================================
# Malformed JSON
# ===========================================================================

def test_predict_malformed_json():
    resp = client.post(
        "/predict",
        content=b"this is not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


def test_predict_array_not_object():
    resp = client.post("/predict", json=[1, 2, 3])
    assert resp.status_code == 422


# ===========================================================================
# Model-info endpoint
# ===========================================================================

def test_model_info_endpoint_exists():
    resp = client.get("/model-info")
    assert resp.status_code in (200, 503)


def test_model_info_schema_when_loaded():
    resp = client.get("/model-info")
    if resp.status_code == 503:
        pytest.skip("Model not loaded.")
    data = resp.json()
    assert "model_name" in data
    assert "feature_descriptions" in data
    assert "evaluation_metrics" in data


# ===========================================================================
# Root / static
# ===========================================================================

def test_root_returns_200():
    resp = client.get("/")
    assert resp.status_code == 200


def test_root_content_type_html():
    resp = client.get("/")
    ct = resp.headers.get("content-type", "")
    assert "text/html" in ct
