/**
 * EduPredict Cloud — Frontend JavaScript
 * =======================================
 * Handles form submission, validation, API communication,
 * and result rendering. All results come from the actual
 * backend prediction API — no hardcoded values.
 */

"use strict";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const API_BASE   = "";               // same origin — no need for absolute URL
const PREDICT_URL = `${API_BASE}/predict`;
const HEALTH_URL  = `${API_BASE}/health`;
const MODEL_URL   = `${API_BASE}/model-info`;

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const form              = document.getElementById("predictionForm");
const predictBtn        = document.getElementById("predictBtn");
const btnText           = document.getElementById("btnText");
const btnSpinner        = document.getElementById("btnSpinner");
const resetBtn          = document.getElementById("resetBtn");
const formError         = document.getElementById("formError");

const resultPlaceholder = document.getElementById("resultPlaceholder");
const resultCard        = document.getElementById("resultCard");
const scoreValue        = document.getElementById("scoreValue");
const scoreBarFill      = document.getElementById("scoreBarFill");
const resultBadge       = document.getElementById("resultBadge");
const resultInterp      = document.getElementById("resultInterpretation");
const modelVersion      = document.getElementById("modelVersion");
const scoreScale        = document.getElementById("scoreScale");

const healthDot         = document.getElementById("healthDot");
const healthText        = document.getElementById("healthText");
const modelInfoContent  = document.getElementById("modelInfoContent");


// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------
async function checkHealth() {
  try {
    const resp = await fetch(HEALTH_URL);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    if (data.status === "healthy" && data.model_loaded) {
      healthDot.className = "health-dot ok";
      healthText.textContent = "Online";
    } else {
      healthDot.className = "health-dot err";
      healthText.textContent = data.model_loaded ? "Degraded" : "Model unavailable";
    }
  } catch {
    healthDot.className = "health-dot err";
    healthText.textContent = "Offline";
  }
}


// ---------------------------------------------------------------------------
// Load model info
// ---------------------------------------------------------------------------
async function loadModelInfo() {
  try {
    const resp = await fetch(MODEL_URL);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const d = await resp.json();

    const m = d.evaluation_metrics || {};
    const mae  = m.mae  != null ? m.mae.toFixed(3)  : "—";
    const rmse = m.rmse != null ? m.rmse.toFixed(3) : "—";
    const r2   = m.r2   != null ? m.r2.toFixed(3)   : "—";

    modelInfoContent.innerHTML = `
      <div class="metrics-grid">
        <div class="metric-box">
          <div class="metric-label">MAE</div>
          <div class="metric-value">${mae}</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">RMSE</div>
          <div class="metric-value">${rmse}</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">R²</div>
          <div class="metric-value">${r2}</div>
        </div>
      </div>
      <div class="info-row"><span class="info-key">Model</span><span class="info-value">${esc(d.model_name)}</span></div>
      <div class="info-row"><span class="info-key">Version</span><span class="info-value">${esc(d.model_version)}</span></div>
      <div class="info-row"><span class="info-key">Target</span><span class="info-value">${esc(d.target_column)} (${esc(d.target_scale)})</span></div>
      <div class="info-row"><span class="info-key">Training samples</span><span class="info-value">${d.training_samples}</span></div>
      <div class="info-row"><span class="info-key">Test samples</span><span class="info-value">${d.test_samples}</span></div>
      <div class="info-row"><span class="info-key">Dataset</span><span class="info-value">Synthetic (UCI-structured)</span></div>
      <p style="font-size:0.72rem;color:var(--slate-500);margin-top:.75rem;">
        Metrics are measured on a held-out test split. MAE and RMSE are on the 0–20 grade scale.
        R² indicates proportion of variance explained by the model.
      </p>
    `;
  } catch (err) {
    modelInfoContent.innerHTML = `
      <p style="font-size:.85rem;color:var(--slate-500);">
        Could not load model information. The API may still be starting up.
      </p>`;
  }
}

// Safe HTML escape
function esc(str) {
  if (str == null) return "—";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}


// ---------------------------------------------------------------------------
// Form validation
// ---------------------------------------------------------------------------
function validateForm() {
  let valid = true;

  const fields = [
    { id: "G1",        min: 0,  max: 20,  type: "number",  label: "G1" },
    { id: "G2",        min: 0,  max: 20,  type: "number",  label: "G2" },
    { id: "studytime", type: "select", label: "Study time" },
    { id: "failures",  type: "select", label: "Failures" },
    { id: "absences",  min: 0,  max: 74,  type: "number",  label: "Absences" },
    { id: "sex",       type: "select", label: "Sex" },
    { id: "age",       min: 15, max: 22,  type: "number",  label: "Age" },
    { id: "Medu",      type: "select", label: "Mother education" },
    { id: "Fedu",      type: "select", label: "Father education" },
    { id: "famrel",    type: "select", label: "Family relations" },
    { id: "freetime",  type: "select", label: "Free time" },
    { id: "goout",     type: "select", label: "Going out" },
    { id: "health",    type: "select", label: "Health" },
    { id: "Dalc",      type: "select", label: "Weekday alcohol" },
    { id: "Walc",      type: "select", label: "Weekend alcohol" },
  ];

  fields.forEach(f => {
    const el  = document.getElementById(f.id);
    const err = document.getElementById(`${f.id}-error`);
    if (!el) return;

    el.classList.remove("invalid");
    if (err) err.textContent = "";

    const val = el.value.trim();

    if (!val) {
      el.classList.add("invalid");
      if (err) err.textContent = `${f.label} is required.`;
      valid = false;
      return;
    }

    if (f.type === "number") {
      const num = parseFloat(val);
      if (isNaN(num)) {
        el.classList.add("invalid");
        if (err) err.textContent = `${f.label} must be a number.`;
        valid = false;
      } else if (num < f.min || num > f.max) {
        el.classList.add("invalid");
        if (err) err.textContent = `${f.label} must be between ${f.min} and ${f.max}.`;
        valid = false;
      }
    }
  });

  return valid;
}


// ---------------------------------------------------------------------------
// Build request payload from form
// ---------------------------------------------------------------------------
function buildPayload() {
  return {
    G1:        parseFloat(document.getElementById("G1").value),
    G2:        parseFloat(document.getElementById("G2").value),
    studytime: parseInt(document.getElementById("studytime").value, 10),
    failures:  parseInt(document.getElementById("failures").value, 10),
    absences:  parseInt(document.getElementById("absences").value, 10),
    sex:       document.getElementById("sex").value,
    age:       parseInt(document.getElementById("age").value, 10),
    Medu:      parseInt(document.getElementById("Medu").value, 10),
    Fedu:      parseInt(document.getElementById("Fedu").value, 10),
    famrel:    parseInt(document.getElementById("famrel").value, 10),
    freetime:  parseInt(document.getElementById("freetime").value, 10),
    goout:     parseInt(document.getElementById("goout").value, 10),
    Dalc:      parseInt(document.getElementById("Dalc").value, 10),
    Walc:      parseInt(document.getElementById("Walc").value, 10),
    health:    parseInt(document.getElementById("health").value, 10),
  };
}


// ---------------------------------------------------------------------------
// Render prediction result
// ---------------------------------------------------------------------------
function renderResult(data) {
  const score = data.predicted_score;
  const pct   = Math.min((score / 20) * 100, 100).toFixed(1);

  scoreValue.textContent   = score.toFixed(1);
  scoreBarFill.style.width = `${pct}%`;
  if (modelVersion) modelVersion.textContent = data.model_version || "1.0";
  if (scoreScale)   scoreScale.textContent   = data.score_scale   || "0–20";

  // Badge
  let badgeClass = "badge-satisfactory";
  let badgeText  = "Satisfactory";
  if (score >= 17)      { badgeClass = "badge-excellent";    badgeText = "Excellent"; }
  else if (score >= 14) { badgeClass = "badge-good";         badgeText = "Good"; }
  else if (score >= 10) { badgeClass = "badge-satisfactory"; badgeText = "Satisfactory"; }
  else if (score >= 7)  { badgeClass = "badge-at-risk";      badgeText = "At Risk"; }
  else                   { badgeClass = "badge-critical";     badgeText = "Critical"; }

  resultBadge.className   = `result-badge ${badgeClass}`;
  resultBadge.textContent = badgeText;

  resultInterp.textContent = data.interpretation || data.message || "";

  resultPlaceholder.hidden = true;
  resultCard.hidden        = false;
}


// ---------------------------------------------------------------------------
// Form submit
// ---------------------------------------------------------------------------
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  formError.hidden = true;
  formError.textContent = "";

  if (!validateForm()) return;

  // Loading state
  predictBtn.disabled  = true;
  btnText.textContent  = "Predicting…";
  btnSpinner.hidden    = false;

  try {
    const payload = buildPayload();
    const resp = await fetch(PREDICT_URL, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    const data = await resp.json();

    if (!resp.ok) {
      // Show API error message — safe, never a stack trace
      const msg = data.detail
        || (Array.isArray(data.detail) ? data.detail.map(d => d.msg).join("; ") : null)
        || `Prediction failed (HTTP ${resp.status}).`;
      formError.textContent = msg;
      formError.hidden      = false;
      return;
    }

    renderResult(data);

  } catch (err) {
    formError.textContent = "Could not reach the prediction API. Please check your connection and try again.";
    formError.hidden      = false;
  } finally {
    predictBtn.disabled  = false;
    btnText.textContent  = "Predict Final Grade";
    btnSpinner.hidden    = true;
  }
});


// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------
resetBtn.addEventListener("click", () => {
  form.reset();

  // Clear all error states
  document.querySelectorAll(".invalid").forEach(el => el.classList.remove("invalid"));
  document.querySelectorAll(".field-error").forEach(el => el.textContent = "");
  formError.hidden = true;

  // Hide result
  resultCard.hidden        = false;
  resultPlaceholder.hidden = false;
  resultCard.hidden        = true;
  resultPlaceholder.hidden = false;
});


// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  loadModelInfo();
});
