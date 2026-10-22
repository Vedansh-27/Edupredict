# Screenshot Checklist — EduPredict Cloud
*FT-III Submission Evidence. Capture these screenshots after deployment.*

---

## Pre-deployment Screenshots (Local)

- [ ] **SS-01: Project folder structure**
  - Command: `tree /F` in `EduPredict-Cloud/` directory
  - Shows: Complete file tree of the project

- [ ] **SS-02: Local server running**
  - Command: `uvicorn app.main:app --reload`
  - Shows: Terminal with "Application startup complete"

- [ ] **SS-03: Home / Dashboard**
  - URL: `http://localhost:8000/`
  - Shows: EduPredict Cloud dashboard with form visible

- [ ] **SS-04: Prediction form filled**
  - URL: `http://localhost:8000/`
  - Shows: All form fields populated with sample values

- [ ] **SS-05: Prediction result**
  - After submitting the form
  - Shows: Predicted score, score bar, badge, interpretation text

- [ ] **SS-06: FastAPI Swagger documentation**
  - URL: `http://localhost:8000/docs`
  - Shows: /health, /model-info, /predict endpoints listed

- [ ] **SS-07: Swagger /predict Try It Out**
  - URL: `http://localhost:8000/docs` → POST /predict → Try it out
  - Shows: Request body and 200 response with predicted_score

- [ ] **SS-08: Health check response**
  - URL: `http://localhost:8000/health`
  - Shows: `{"status":"healthy","model_loaded":true,"model_name":"..."}`

- [ ] **SS-09: Model info response**
  - URL: `http://localhost:8000/model-info`
  - Shows: JSON with MAE, RMSE, R² and feature descriptions

- [ ] **SS-10: Model evaluation metrics**
  - File: `reports/evaluation_report.json`
  - Shows: All four model comparison results + selected model metrics

- [ ] **SS-11: Evaluation plots**
  - File: `reports/actual_vs_predicted.png`
  - Shows: Scatter plot of actual vs predicted grades

- [ ] **SS-12: Pytest test results**
  - Command: `pytest tests/ -v`
  - Shows: All tests passing (green) with test names

- [ ] **SS-13: Performance test output**
  - Command: `python scripts/performance_test.py --n 50`
  - Shows: Latency table with p50, p95, error rate

---

## Post-deployment Screenshots (Render)

- [ ] **SS-14: GitHub repository**
  - URL: `https://github.com/YourUsername/EduPredict-Cloud`
  - Shows: Repository with all committed files and README

- [ ] **SS-15: Render dashboard — service created**
  - URL: Render dashboard
  - Shows: edupredict-cloud service in the services list

- [ ] **SS-16: Render build logs**
  - Shows: Successful build with dependency install, training, startup

- [ ] **SS-17: Render deploy confirmation**
  - Shows: "Live" status and the public deployment URL

- [ ] **SS-18: Live public dashboard**
  - URL: `https://your-app.onrender.com/`
  - Shows: EduPredict Cloud dashboard in browser with Render URL in address bar

- [ ] **SS-19: Live prediction on deployed app**
  - URL: `https://your-app.onrender.com/`
  - Shows: Prediction result card with score from the deployed API

- [ ] **SS-20: Live health check on deployed app**
  - URL: `https://your-app.onrender.com/health`
  - Shows: `{"status":"healthy","model_loaded":true}` from live service

- [ ] **SS-21: Live Swagger on deployed app**
  - URL: `https://your-app.onrender.com/docs`
  - Shows: Swagger UI with live Render URL in browser address bar

- [ ] **SS-22: Performance test against deployed service**
  - Command: `python scripts/performance_test.py --url https://your-app.onrender.com --n 20`
  - Shows: Latency measurements from remote service

---

## Evidence for Each Rubric Item

| Rubric Area | Key Screenshots |
|-------------|-----------------|
| Problem Analysis & Cloud Design | SS-18, SS-14, architecture.md diagram |
| Cloud service configuration | SS-15, SS-16, SS-17, render.yaml |
| Working implementation | SS-03, SS-04, SS-05, SS-12 |
| Database/storage (model file) | SS-09, SS-10, models/ directory |
| Networking and HTTPS | SS-18 (https:// in URL bar), SS-20 |
| Security (input validation) | SS-07 (422 error response) |
| Performance & scalability | SS-13, SS-22 |
| Results and evaluation | SS-10, SS-11, SS-12 |
| Viva preparation | viva_questions.md |

---

## How to Capture Screenshots on Windows

```powershell
# Full screen: Win + PrtScn (saves to Pictures/Screenshots)
# Selected area: Win + Shift + S (opens Snipping Tool)
# Active window: Alt + PrtScn
```

Recommended: Use Win + Shift + S for precise captures. Save all screenshots in a `submission_screenshots/` folder organised by SS number.
