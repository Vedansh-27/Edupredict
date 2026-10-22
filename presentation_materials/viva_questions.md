# Viva Questions — EduPredict Cloud
*Cloud Computing FT-III — At least 20 expected viva questions with technically accurate answers.*

---

## Section A: Cloud Concepts and Architecture

**Q1. What cloud service model does EduPredict Cloud use and why?**

EduPredict Cloud uses Platform as a Service (PaaS) via Render. PaaS abstracts infrastructure management — we don't provision or maintain VMs, install OS packages, or manage TLS certificates. We only manage the application code, which is appropriate for a student project where development speed and low operational overhead are priorities. IaaS (like AWS EC2) would give more control but require more configuration. SaaS is too restrictive for a custom ML application.

---

**Q2. What is the difference between IaaS, PaaS, and SaaS? Give examples.**

- **IaaS (Infrastructure as a Service):** Provides virtualised compute, storage, and networking. You manage the OS upward. Examples: AWS EC2, Google Compute Engine, Azure VMs.
- **PaaS (Platform as a Service):** Provides a managed runtime environment. You manage only the application and data. Examples: Render, Heroku, Google App Engine, AWS Elastic Beanstalk.
- **SaaS (Software as a Service):** Fully managed applications delivered to end users. Examples: Gmail, Salesforce, Zoom, GitHub.

---

**Q3. How does Render deploy the application?**

Render connects to the GitHub repository via a webhook. When code is pushed to `main`, Render triggers a new build. The build command installs dependencies and trains the ML model. Then the start command launches Uvicorn. Render manages HTTPS, DNS, container restarts, and health checks automatically.

---

**Q4. What is a cold start and how does it affect this application?**

A cold start occurs when a service instance has been idle (spun down) and must restart before handling a new request. On Render's free plan, services spin down after 15 minutes of inactivity. The first request after a cold start may experience 30–60 seconds of delay while the instance restarts and the model reloads. Paid plans keep instances always-on. In this project, the `/health` endpoint can be used to confirm the instance is warm before demonstrations.

---

**Q5. How is HTTPS implemented in this project?**

HTTPS is provided automatically by Render via Let's Encrypt TLS certificates. The certificate is provisioned and renewed without any manual configuration. All traffic to the Render domain uses TLS 1.2 or 1.3. The application itself does not need to manage certificates because the platform terminates TLS at the edge.

---

**Q6. What is horizontal scaling and can this application support it?**

Horizontal scaling means adding more instances of a service to handle higher load, rather than making one instance more powerful (vertical scaling). Render's paid plans support multiple instances. This FastAPI application is stateless — each request is independent. No shared in-memory session state means horizontal scaling can be applied without code changes. The model is loaded from a file on each instance, so multiple instances can serve predictions simultaneously. SQLite would be a constraint if prediction history were stored locally, but this project uses a stateless API.

---

## Section B: Machine Learning Integration

**Q7. What dataset was used and why was synthetic data chosen?**

A synthetic dataset was generated using `ml/generate_synthetic_data.py`. It mirrors the statistical structure of the UCI Student Performance dataset (Cortez and Silva, 2008). Synthetic data was chosen because: (1) automated downloading of the UCI dataset is unreliable in CI/CD pipelines due to URL changes; (2) synthetic data is fully reproducible with a fixed random seed; (3) it avoids licensing ambiguity; (4) it is clearly documented as non-real student data.

---

**Q8. What is the prediction target and what scale is it on?**

The target variable is `G3` — the student's final period grade — on a 0–20 integer scale, matching the UCI dataset's Portuguese secondary school grading system. The model predicts a continuous floating-point value that is clamped to [0, 20].

---

**Q9. What models were trained and how was the best one selected?**

Four candidate regression models were trained: Linear Regression, Ridge Regression, Random Forest Regressor, and Gradient Boosting Regressor. Each was wrapped in a sklearn Pipeline with a StandardScaler. They were evaluated on a held-out 20% test split (random seed 42) using MAE, RMSE, and R². The model with the highest test R² was selected. Five-fold cross-validation on the training set was also computed to detect overfitting.

---

**Q10. What do MAE, RMSE, and R² measure? Why use all three?**

- **MAE (Mean Absolute Error):** Average of |predicted - actual|. Interpretable in the same units as the target (grade points). Robust to outliers.
- **RMSE (Root Mean Square Error):** Square root of the average squared error. Penalises large errors more heavily. Good for catching catastrophic predictions.
- **R² (Coefficient of Determination):** Proportion of variance in the target explained by the model. 1.0 = perfect; 0.0 = predicts the mean; negative = worse than the mean.

Using all three gives a complete picture: MAE and RMSE capture error magnitude, R² captures explanatory power.

---

**Q11. How is data leakage prevented?**

The final grade G3 is the target variable and is never used as an input feature. G1 and G2 (first and second period grades) are used as inputs — this is appropriate because in a real scenario, the prediction would be made after these grades are available but before the final exam. The train/test split was performed before any model fitting. The StandardScaler was fit only on the training set and applied to the test set.

---

**Q12. How is the trained model loaded and used at inference time?**

The trained sklearn Pipeline (StandardScaler + Regressor) is serialised using `joblib.dump()` to `models/student_performance_model.joblib`. The `ModelService` class loads it once during application startup using `joblib.load()`. Each prediction request calls `pipeline.predict()` on a single-row DataFrame with features in exactly the same order as during training. The feature order is stored in `models/model_metadata.json`.

---

**Q13. What prevents a NaN or infinity from reaching the model?**

Multiple layers: (1) Pydantic validates numeric ranges on input — values outside bounds are rejected with 422. (2) After building the feature DataFrame, `X.isnull().any()` and `np.isinf(X.values).any()` are checked before calling predict. (3) After prediction, `math.isfinite(pred)` is checked — if the model returns a non-finite value, a RuntimeError is raised and the API returns 503.

---

## Section C: API and Backend

**Q14. Why was FastAPI chosen over Flask or Django?**

FastAPI offers: automatic Swagger UI at `/docs`; Pydantic-based request validation with automatic 422 errors; native async support for high concurrency; type annotations for self-documenting code; and significantly higher performance than Flask (comparable to Node.js in benchmarks). Django is better for full-stack web apps with ORM requirements — unnecessary for this ML API.

---

**Q15. What does the /predict endpoint validate?**

Pydantic validates: required fields are present; numeric types match; G1 and G2 are in [0, 20]; studytime in [1, 4]; failures in [0, 3]; absences in [0, 74]; sex is 'M' or 'F'; age in [15, 22]; Medu/Fedu/famrel/freetime/goout/Dalc/Walc/health in [0–4] or [1–5] as appropriate. Invalid requests receive a 422 response with field-level error messages. No stack traces are exposed.

---

**Q16. What is CORS and how is it configured here?**

CORS (Cross-Origin Resource Sharing) is a browser security mechanism that restricts JavaScript from making requests to a different origin than the page. In this project, the frontend and API share the same origin (both served by FastAPI), so CORS is not strictly needed. A restrictive CORS policy is still configured: only `localhost:8000`, `127.0.0.1:8000`, and `localhost:3000` are allowed, plus the Render deployment URL when added. Unrestricted `*` origins are not used.

---

**Q17. How are errors handled without leaking sensitive information?**

FastAPI's exception handlers catch all unhandled exceptions and return a generic `{"detail": "An internal server error occurred."}` JSON response with status 500. Specific, user-safe error messages are returned for validation errors (422) and model unavailability (503). Stack traces, file paths, environment variables, and internal state are never included in API responses.

---

## Section D: Security and DevOps

**Q18. What security measures are implemented?**

- Input validation on frontend (JS) and backend (Pydantic).
- No hardcoded credentials or secrets.
- `.gitignore` excludes `.env` files.
- No secrets committed to GitHub.
- HTTPS provided by Render (TLS termination at platform).
- Restricted CORS policy.
- No unnecessary PII storage.
- NaN/infinity checks prevent model poisoning.
- Generic error messages prevent information leakage.

---

**Q19. Why is SQLite not used in this project? When would you add it?**

This project uses a stateless prediction API — each request is independent and nothing is persisted. SQLite is appropriate for prediction history, but it introduces complications on PaaS: the SQLite file is ephemeral on Render's free plan (reset on each deploy). If prediction logging were required, a managed database (e.g., Render PostgreSQL or Supabase) would be the correct choice. Adding SQLite locally and on a shared-disk persistent plan is possible, but not justified for this demonstration.

---

**Q20. What happens if the model file is missing on Render?**

The `ModelService.__init__()` checks for the model file on startup. If missing, `is_loaded` is set to `False` and a clear log warning is emitted. The `/health` endpoint returns `{"model_loaded": false}`. The `/predict` endpoint returns a `503 Service Unavailable` with the message "Model is not currently available." The application does not crash — it degrades gracefully. The Render build command trains the model before starting the service, so this condition should not occur in normal deployment.

---

**Q21. How would you scale this application to handle 10,000 requests per day?**

- Upgrade to a Render paid plan (removes cold starts, allows multiple instances).
- Enable autoscaling in Render's dashboard.
- Add a CDN (e.g., Cloudflare) in front of static assets.
- Replace SQLite with a managed PostgreSQL database if persistence is needed.
- Cache model predictions for identical inputs using Redis.
- Consider moving the model to a dedicated inference endpoint (e.g., AWS SageMaker) if latency becomes critical.

---

**Q22. What is Uvicorn and why is it used?**

Uvicorn is an ASGI (Asynchronous Server Gateway Interface) server. FastAPI is an ASGI framework, which means it needs an ASGI-compatible server to handle HTTP connections. Uvicorn uses uvloop (or asyncio on Windows) for high-performance async I/O. It is Render-compatible and is the recommended production server for FastAPI. The start command binds to `0.0.0.0` and reads the port from the `PORT` environment variable, which Render sets automatically.

---

**Q23. How do you add the Render deployment URL after getting it?**

1. In `app/main.py`, add `"https://your-app-name.onrender.com"` to the `_ALLOWED_ORIGINS` list.
2. Commit and push — Render auto-deploys.
3. Optionally add it as an environment variable for cleaner configuration.

---

**Q24. What is the difference between regression and classification, and why is regression used here?**

Classification predicts a discrete class label (e.g., pass/fail). Regression predicts a continuous numerical value. G3 is a numerical grade (0–20) with meaningful ordering and distance between values — predicting 13.4 vs 14.6 is more informative than predicting "pass" vs "fail." Regression is the appropriate choice. R² and RMSE are the correct metrics — accuracy percentage would be meaningless for a regression task.
