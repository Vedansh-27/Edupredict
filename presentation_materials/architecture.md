# Architecture Specification — EduPredict Cloud

## Logical System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER / BROWSER                           │
│           (Desktop, Tablet, or Mobile — any browser)           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ HTTPS (TLS 1.2/1.3)
                            │ Port 443 (Render-managed SSL/TLS)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                RENDER WEB SERVICE (PaaS)                        │
│          edupredict-cloud.onrender.com (example URL)           │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │               FastAPI / Uvicorn Application                │ │
│  │                                                            │ │
│  │  ┌──────────────────┐    ┌──────────────────────────────┐ │ │
│  │  │  Static Files    │    │     API Endpoints             │ │ │
│  │  │  /static/css     │    │  GET  /              (HTML)   │ │ │
│  │  │  /static/js      │    │  GET  /health                 │ │ │
│  │  │  Jinja2 Template │    │  GET  /model-info             │ │ │
│  │  └──────────────────┘    │  POST /predict                │ │ │
│  │           │               │  GET  /docs  (Swagger)       │ │ │
│  │           │               └────────────┬─────────────────┘ │ │
│  │           │                            │                    │ │
│  │           │               ┌────────────▼─────────────────┐ │ │
│  │           │               │  Request Validation Layer     │ │ │
│  │           │               │  (Pydantic schemas)           │ │ │
│  │           │               └────────────┬─────────────────┘ │ │
│  │           │                            │                    │ │
│  │           │               ┌────────────▼─────────────────┐ │ │
│  │           │               │  ModelService (Singleton)     │ │ │
│  │           │               │  Loaded once at startup       │ │ │
│  │           │               └────────────┬─────────────────┘ │ │
│  │           │                            │                    │ │
│  │           │               ┌────────────▼─────────────────┐ │ │
│  │           │               │  Scikit-learn Pipeline        │ │ │
│  │           │               │  StandardScaler + Regressor   │ │ │
│  │           │               │  (models/*.joblib)            │ │ │
│  │           │               └────────────────────────────── ┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Render Platform Layer                         │ │
│  │  • Auto-managed HTTPS certificate (Let's Encrypt)         │ │
│  │  • Automatic restarts on crash                            │ │
│  │  • GitHub → Render CD pipeline                            │ │
│  │  • Environment variable management                        │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

           ▲
           │  Source code push (git push origin main)
           │
┌──────────┴───────────────────────────────────────────────────┐
│                    GITHUB REPOSITORY                          │
│            github.com/[your-username]/EduPredict-Cloud       │
│                                                               │
│   • Source code (app/, ml/, static/, templates/, tests/)     │
│   • requirements.txt, render.yaml, .gitignore                │
│   • Training data (synthetic, committed)                      │
│   • Model artifact (models/*.joblib, committed)              │
│   • Documentation (README.md, presentation_materials/)       │
└───────────────────────────────────────────────────────────────┘
```

## Request-Response Data Flow

### Prediction Request

```
1. User fills form in browser (HTML/CSS/JS)
   ↓
2. Frontend JavaScript validates inputs (client-side)
   ↓
3. fetch() POST /predict with JSON body
   {G1, G2, studytime, failures, absences, sex, age, Medu, Fedu,
    famrel, freetime, goout, Dalc, Walc, health}
   ↓ HTTPS
4. FastAPI receives request — Pydantic validates schema & ranges
   - Invalid → 422 Unprocessable Entity with field-level errors
   ↓
5. app/main.py → encode sex (M→1, F→0), build feature dict
   ↓
6. ModelService.predict() → assemble DataFrame in fixed feature order
   ↓
7. sklearn Pipeline: StandardScaler transform → Regressor predict()
   ↓
8. Clamp output to [0, 20]; check for NaN/inf
   ↓
9. Build PredictionResponse → JSON response
   {"predicted_score": 13.4, "score_scale": "0-20",
    "message": "...", "model_version": "1.0",
    "interpretation": "..."}
   ↓ HTTPS
10. JavaScript renders result card, score bar, badge, interpretation
```

### Health Check Flow

```
GET /health
  → ModelService.is_loaded check (no disk I/O after startup)
  → {"status": "healthy", "model_loaded": true, "model_name": "..."}
```

## Component Roles

| Component | Role |
|-----------|------|
| GitHub | Version control, CI/CD source trigger |
| Render | PaaS host — build, deploy, HTTPS, restarts |
| Uvicorn | ASGI server — handles HTTP connections |
| FastAPI | API routing, Swagger docs, CORS |
| Pydantic | Input validation and serialisation |
| ModelService | Singleton model loader — loads once at startup |
| sklearn Pipeline | StandardScaler + trained regressor |
| Jinja2 | HTML template rendering for the dashboard |
| Static files | CSS, JS served by FastAPI's StaticFiles |

## Cloud Service Model Selection

**PaaS (Platform as a Service) — Render**

| Criterion | IaaS (e.g. AWS EC2) | PaaS (Render) | SaaS |
|-----------|---------------------|---------------|------|
| Infrastructure management | You manage VM, OS, networking | Platform manages all | Fully managed |
| Deployment complexity | High | Low | None |
| Customisability | Maximum | High (within platform) | Minimal |
| Suitable for this project? | Over-engineered | ✅ Ideal | Too restrictive |
| Cost (student budget) | Pay for idle VMs | Free tier available | Varies |

**Why PaaS is ideal here:**
- No VM provisioning or OS maintenance required.
- Git-push-to-deploy workflow fits academic timelines.
- Free tier supports the demonstration load.
- HTTPS is automatic — no certificate management.
- Health checks and auto-restart are built in.
