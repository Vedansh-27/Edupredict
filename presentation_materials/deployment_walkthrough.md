# Deployment Walkthrough — EduPredict Cloud
*Step-by-step deployment guide for Render. Complete these steps in order.*

---

## Prerequisites

- A GitHub account
- A Render account (free tier): https://render.com
- Git installed and configured locally
- All local tests passing (`pytest tests/ -v`)

---

## Step 1: Initialize Git Repository

Open PowerShell in the `EduPredict-Cloud/` directory:

```powershell
cd "C:\Users\Vedansh Garg\OneDrive\Desktop\EduPredict-Cloud"

git init
git add .
git commit -m "feat: initial EduPredict Cloud implementation

- ML training pipeline with 4 candidate models
- FastAPI backend with /health, /model-info, /predict endpoints
- Responsive prediction dashboard (HTML/CSS/JS)
- Automated tests (pytest)
- Render deployment configuration"
```

---

## Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `EduPredict-Cloud`
3. Set to **Public** (required for Render free plan)
4. Do NOT initialise with README (we have one)
5. Click **Create repository**
6. Copy the repository URL shown (e.g., `https://github.com/YourUsername/EduPredict-Cloud.git`)

---

## Step 3: Push to GitHub

```powershell
git remote add origin https://github.com/YourUsername/EduPredict-Cloud.git
git branch -M main
git push -u origin main
```

You will be prompted for your GitHub credentials. Use a personal access token (Settings → Developer settings → Personal access tokens) as your password if prompted.

---

## Step 4: Create Render Web Service

1. Go to https://dashboard.render.com
2. Click **New** → **Web Service**
3. Select **Connect a repository** → authorise GitHub access
4. Select `EduPredict-Cloud`
5. Configure:
   - **Name:** `edupredict-cloud` (or similar)
   - **Region:** Singapore or nearest to you
   - **Branch:** `main`
   - **Runtime:** Python 3
   - **Build Command:**
     ```
     pip install -r requirements.txt && python ml/generate_synthetic_data.py && python ml/train_model.py
     ```
   - **Start Command:**
     ```
     uvicorn app.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Plan:** Free
6. Click **Create Web Service**

---

## Step 5: Monitor Deployment

1. Render will show build logs in real time.
2. Watch for:
   - `Collecting fastapi==0.115.6` — dependencies installing
   - `[INFO] Synthetic dataset written to: ...` — data generated
   - `[INFO] Best model: ...` — training complete
   - `Application startup complete.` — server ready
3. Build typically takes 3–6 minutes on first deploy.

---

## Step 6: Identify Public URL

After a successful deploy:
- Render shows the URL at the top of the service page.
- Format: `https://edupredict-cloud-xxxx.onrender.com`
- This is your live application URL.

---

## Step 7: Update CORS (after getting URL)

In `app/main.py`, add your Render URL:
```python
_ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://edupredict-cloud-xxxx.onrender.com",  # <-- add this
]
```

Commit and push — Render auto-deploys:
```powershell
git add app/main.py
git commit -m "config: add Render URL to CORS allowed origins"
git push
```

---

## Step 8: Verify Deployment

Test each endpoint (replace URL with your actual Render URL):

```powershell
# Home page
Invoke-WebRequest https://edupredict-cloud-xxxx.onrender.com/ -UseBasicParsing

# Health check
Invoke-WebRequest https://edupredict-cloud-xxxx.onrender.com/health | Select-Object -ExpandProperty Content

# Model info
Invoke-WebRequest https://edupredict-cloud-xxxx.onrender.com/model-info | Select-Object -ExpandProperty Content

# Prediction
$body = '{"G1":12,"G2":13,"studytime":2,"failures":0,"absences":4,"sex":"M","age":17,"Medu":2,"Fedu":2,"famrel":4,"freetime":3,"goout":3,"Dalc":1,"Walc":2,"health":4}'
Invoke-RestMethod -Uri https://edupredict-cloud-xxxx.onrender.com/predict -Method POST -ContentType "application/json" -Body $body

# Swagger docs (open in browser)
Start-Process "https://edupredict-cloud-xxxx.onrender.com/docs"
```

---

## Troubleshooting

### Build fails: Module not found
- Check `requirements.txt` is present and committed.
- Verify build command starts with `pip install -r requirements.txt`.

### Server starts but model not loaded
- Check build logs: did `python ml/train_model.py` complete without errors?
- Ensure `models/` directory is not in `.gitignore`.
- Verify `models/student_performance_model.joblib` was committed.

### 502 Bad Gateway on Render
- Application crashed on startup.
- Check "Logs" tab in Render dashboard for Python exceptions.
- Common cause: port binding — ensure `--port $PORT` is in start command.

### Cold start (first request takes 30–60s)
- Normal on the free plan.
- Free plan instances spin down after 15 minutes of inactivity.
- Solution: upgrade to the Starter plan (\$7/month) for always-on.

### CORS error in browser console
- Add your Render URL to `_ALLOWED_ORIGINS` in `app/main.py`.
- Commit and push to trigger a new deploy.

---

## Running Performance Tests Against the Deployed Service

After getting your Render URL:

```powershell
python scripts/performance_test.py --url https://edupredict-cloud-xxxx.onrender.com --n 20 --delay 0.5
```

Keep requests ≤ 50 on the free tier. Use `--delay 0.5` to avoid overwhelming the instance.
