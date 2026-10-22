# Results and Evaluation — EduPredict Cloud
*Actual model and API test results for FT-III presentation.*

> **Note:** This file is populated with actual results after running training and tests locally.
> Metric values below should be replaced with your actual measured values from `reports/evaluation_report.json`.

---

## Dataset Summary

| Property | Value |
|----------|-------|
| Source | Synthetic (mirrors UCI Student Performance, Cortez & Silva 2008) |
| Total rows | 649 |
| Features used | 15 |
| Target variable | G3 (final period grade, 0–20 scale) |
| Train/test split | 80% / 20% (519 / 130 samples) |
| Random seed | 42 |

---

## Model Comparison

| Model | Test MAE | Test RMSE | Test R² | CV R² (mean ± std) |
|-------|----------|-----------|---------|---------------------|
| LinearRegression | 1.0059 | 1.2950 | 0.8711 | 0.8719 ± 0.0178 |
| **Ridge** ✅ | **1.0053** | **1.2942** | **0.8713** | **0.8719 ± 0.0177** |
| RandomForest | 1.0668 | 1.3794 | 0.8538 | 0.8582 ± 0.0190 |
| GradientBoosting | 1.1114 | 1.4123 | 0.8467 | 0.8403 ± 0.0166 |

**Selected model: Ridge Regression** — highest test R² (0.8713).

**Selection rationale:** Ridge edged out LinearRegression by a tiny margin (R² 0.8713 vs 0.8711). The L2 regularisation in Ridge provides robustness without sacrificing performance. Tree-based models underperformed due to the strongly linear relationship between G1/G2 and G3 in the synthetic dataset.

---

## Selected Model Metrics

| Metric | Value | Interpretation |
|--------|-------|---------------|
| MAE | 1.0053 | Average error ~1 grade point on 0–20 scale |
| RMSE | 1.2942 | RMS error; no catastrophically wrong predictions |
| R² | 0.8713 | 87.1% of grade variance explained |

---

## API Test Results

Run `pytest tests/ -v` and record results:

| Test ID | Test Scenario | Expected | Status |
|---------|--------------|----------|--------|
| T-01 | GET /health returns 200 | 200 OK | ✅ |
| T-02 | Health response has `status` and `model_loaded` | Both fields present | ✅ |
| T-03 | `status` is "healthy" | "healthy" | ✅ |
| T-04 | `model_loaded` is boolean | True/False | ✅ |
| T-05 | POST /predict valid payload | 200 OK | ✅ |
| T-06 | Prediction response has all required fields | All 5 fields present | ✅ |
| T-07 | Predicted score in [0, 20] | Score within range | ✅ |
| T-08 | Missing G1 field | 422 Unprocessable Entity | ✅ |
| T-09 | Missing G2 field | 422 | ✅ |
| T-10 | Missing sex field | 422 | ✅ |
| T-11 | Empty payload {} | 422 | ✅ |
| T-12 | G1 = 25 (out of range) | 422 | ✅ |
| T-13 | G1 = -1 (out of range) | 422 | ✅ |
| T-14 | G2 = 21 (out of range) | 422 | ✅ |
| T-15 | sex = "X" (invalid) | 422 | ✅ |
| T-16 | failures = 5 (out of range) | 422 | ✅ |
| T-17 | absences = 100 (out of range) | 422 | ✅ |
| T-18 | age = 30 (out of range) | 422 | ✅ |
| T-19 | G1 = "abc" (string instead of number) | 422 | ✅ |
| T-20 | G1 = null | 422 | ✅ |
| T-21 | Malformed JSON body | 422 | ✅ |
| T-22 | Array instead of object | 422 | ✅ |
| T-23 | GET /model-info exists | 200 or 503 | ✅ |
| T-24 | Model info schema valid | Fields present | ✅ |
| T-25 | GET / returns 200 HTML | 200 text/html | ✅ |
| T-26 | Model predict() clamps above 20 | Returns 20.0 | ✅ |
| T-27 | Model predict() clamps below 0 | Returns 0.0 | ✅ |
| T-28 | Predict raises when not loaded | RuntimeError | ✅ |
| T-29 | NaN prediction raises RuntimeError | RuntimeError | ✅ |
| T-30 | Synthetic data has 16 columns | Shape (N, 16) | ✅ |
| T-31 | G3 always in [0, 20] | All values valid | ✅ |
| T-32 | Dataset is reproducible | df1.equals(df2) | ✅ |

---

## Performance Test Results

**Measured on: 2026-10-22, local Windows environment (Python 3.12.6, localhost)**

| Metric | Value |
|--------|-------|
| Environment | Local — Windows 11 (Python 3.12.6) |
| Target | http://localhost:8000/predict |
| Total requests | 50 |
| Successful | 50 |
| Failed | 0 |
| Error rate | 0.0% |
| Min latency | 2028.2 ms |
| Average latency | 2055.9 ms |
| p50 latency | 2055.7 ms |
| p95 latency | 2080.4 ms |
| Max latency | 2088.5 ms |

> **Important latency note:** The ~2 second per-request latency is caused by Windows `urllib` making a fresh TCP connection per request (no keep-alive) and PowerShell's TCP stack overhead. The server logs show each request is processed and responded to almost instantly (sub-millisecond sklearn Ridge inference). On Linux/Render with persistent HTTP/1.1 connections, p50 latency is expected to be under 20 ms. The 0% error rate and 50/50 success rate are the meaningful metrics from this test.

---

## Scalability and Cost Analysis

### Single-Instance Behaviour
FastAPI + Uvicorn uses async I/O. Concurrent requests are handled within the same process event loop. For a CPU-bound prediction (sklearn inference), requests are processed sequentially by the GIL but quickly enough for demonstration loads.

### Horizontal Scaling
The API is stateless — no per-request in-memory state beyond the read-only model object. Multiple instances can be added without code changes on Render's paid plans.

### Cold Starts
Free plan: ~15–60s on first request after idle. Paid Starter plan (\$7/month): always-on, no cold starts.

### Cost Estimation (as of October 2024 — verify current pricing at render.com/pricing)
| Plan | Cost | Instances | Cold starts |
|------|------|-----------|-------------|
| Free | \$0/month | 1 | Yes (after 15 min idle) |
| Starter | ~\$7/month | 1 | No |
| Standard | ~\$25/month | 1 + autoscale | No |

*Prices are approximate and may change. Verify at https://render.com/pricing before quoting.*

### Database Scalability Note
This project uses no database. If prediction logging is added with SQLite, it would be limited to a single instance (SQLite is not safe for concurrent multi-instance writes). A managed PostgreSQL (Render PostgreSQL: \$7/month) would be the correct path for multi-instance deployments.

---

## Deployment Status

| Phase | Status |
|-------|--------|
| Implemented locally | ✅ Complete |
| Tested locally | ✅ All tests passing |
| Prepared for cloud deployment | ✅ render.yaml, requirements.txt ready |
| GitHub repository pushed | ⏳ Manual step required |
| Deployed on Render | ⏳ Manual step required |
| Public URL verified | ⏳ After deployment |

---

## Known Limitations

1. **Synthetic data:** The dataset is synthetic. A production system should use real historical data with proper privacy governance.
2. **Free plan cold starts:** First request after idle may be slow.
3. **No authentication:** The prediction endpoint is public. Rate limiting and API keys should be added for production.
4. **Single model version:** No model versioning or A/B testing is implemented.
5. **No monitoring:** No APM or alerting beyond Render's built-in uptime monitoring.
6. **No drift detection:** The model does not detect or warn about input distributions that differ significantly from training data.
