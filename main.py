"""
Standalone SMS phishing API (FastAPI).
Run on your PC; point the Flutter app to http://<YOUR_PC_LAN_IP>:8000

Models go in: backend/models/
  - tfidf_vectorizer.pkl
  - sms_rf_model (2).pkl
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make terminal printing reliable on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

APP_ROOT = Path(__file__).resolve().parent
# Allow Railway Storage (or any external volume) to mount models elsewhere.
# Example Railway env var: MODELS_DIR=/mnt/models
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(APP_ROOT / "models")))

VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
ALT_VECTORIZER_PATHS = [
    MODELS_DIR / "tfidf.pkl",
]

SMS_MODEL_PATH = MODELS_DIR / "sms_rf_model (2).pkl"
ALT_SMS_MODEL_PATHS = [
    MODELS_DIR / "sms_rf_model.pkl",
    MODELS_DIR / "sms_rf_model(2).pkl",
    MODELS_DIR / "sms_rf_model_2.pkl",
    MODELS_DIR / "xgb_sms_model (1).pkl",
    MODELS_DIR / "xgb_sms_model.pkl",
]

# Optional: let Railway download model files at boot.
# Set MODEL_BASE_URL to something like:
#   https://raw.githubusercontent.com/<user>/<repo>/main/models
# and ensure the files are publicly accessible at:
#   <MODEL_BASE_URL>/tfidf_vectorizer.pkl
#   <MODEL_BASE_URL>/sms_rf_model%20(2).pkl   (or one of the ALT names above)
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL")


def _ascii_preview(s: str, limit: int = 600) -> str:
    s = s.replace("\r", " ").replace("\n", " ")
    if len(s) > limit:
        s = s[:limit] + "..."
    # Keep only ASCII so Windows terminals won't crash with UnicodeEncodeError.
    return "".join((ch if ord(ch) < 128 else "?") for ch in s)


class SmsRequest(BaseModel):
    message: str = Field(..., min_length=1, description="SMS text to analyze")


class PredictionResponse(BaseModel):
    prediction: int
    result: str
    phishing_probability: float | None = None


app = FastAPI(title="Phishing SMS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_threshold_env = os.getenv("PHISHING_THRESHOLD")
# If PHISHING_THRESHOLD is not set, we use the model's raw `predict()` output (no thresholding).
PHISHING_THRESHOLD = float(_threshold_env) if _threshold_env is not None else None


def _load_joblib(path: Path) -> Any:
    if not path.exists():
        raise RuntimeError(f"Missing model file: {path}")
    return joblib.load(path)

def _ensure_models_present() -> None:
    has_vec = VECTORIZER_PATH.exists() or any(p.exists() for p in ALT_VECTORIZER_PATHS)
    has_sms = SMS_MODEL_PATH.exists() or any(p.exists() for p in ALT_SMS_MODEL_PATHS)
    if has_vec and has_sms:
        return

    if not MODEL_BASE_URL:
        return

    import urllib.parse
    import urllib.request

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def _download(name: str, dest: Path) -> None:
        url = MODEL_BASE_URL.rstrip("/") + "/" + urllib.parse.quote(name)
        with urllib.request.urlopen(url) as resp:  # nosec - controlled by deployment env var
            dest.write_bytes(resp.read())

    if not VECTORIZER_PATH.exists() and not any(p.exists() for p in ALT_VECTORIZER_PATHS):
        # try canonical name first, then alternates
        try:
            _download("tfidf_vectorizer.pkl", VECTORIZER_PATH)
        except Exception:
            last_err: Exception | None = None
            for p in ALT_VECTORIZER_PATHS:
                try:
                    _download(p.name, p)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
            if last_err is not None:
                raise last_err

    if not SMS_MODEL_PATH.exists() and not any(p.exists() for p in ALT_SMS_MODEL_PATHS):
        # try canonical name first, then alternates
        try:
            _download("sms_rf_model (2).pkl", SMS_MODEL_PATH)
        except Exception:
            last_err: Exception | None = None
            for p in ALT_SMS_MODEL_PATHS:
                try:
                    _download(p.name, p)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
            if last_err is not None:
                raise last_err

def _resolve_sms_model_path() -> Path:
    if SMS_MODEL_PATH.exists():
        return SMS_MODEL_PATH
    for p in ALT_SMS_MODEL_PATHS:
        if p.exists():
            return p
    return SMS_MODEL_PATH

def _resolve_vectorizer_path() -> Path:
    if VECTORIZER_PATH.exists():
        return VECTORIZER_PATH
    for p in ALT_VECTORIZER_PATHS:
        if p.exists():
            return p
    return VECTORIZER_PATH


@app.on_event("startup")
def _startup_load_models() -> None:
    _ensure_models_present()
    app.state.vectorizer = _load_joblib(_resolve_vectorizer_path())
    app.state.sms_model = _load_joblib(_resolve_sms_model_path())


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _predict_sms(message: str) -> PredictionResponse:
    message = _normalize_text(message)
    if not message:
        raise HTTPException(status_code=400, detail="message must not be empty")

    vectorizer = app.state.vectorizer
    model = app.state.sms_model

    features = vectorizer.transform([message])
    pred = int(model.predict(features)[0])
    phishing_proba: float | None = None
    if PHISHING_THRESHOLD is not None and hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        if len(proba) >= 2:
            phishing_proba = float(proba[1])
            pred = 1 if float(proba[1]) >= PHISHING_THRESHOLD else 0
    elif hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        if len(proba) >= 2:
            phishing_proba = float(proba[1])

    result = "Phishing" if pred == 1 else "Safe"
    print(_ascii_preview(f"[check_sms] message={message}", limit=300))
    if phishing_proba is not None:
        print(
            _ascii_preview(
                f"[check_sms] prediction={pred} result={result} phishing_proba={phishing_proba:.4f}",
                limit=300,
            )
        )
    else:
        print(_ascii_preview(f"[check_sms] prediction={pred} result={result}", limit=300))
    return PredictionResponse(
        prediction=pred,
        result=result,
        phishing_probability=phishing_proba,
    )


@app.post("/check_sms", response_model=PredictionResponse)
def check_sms(req: SmsRequest) -> PredictionResponse:
    return _predict_sms(req.message)


@app.post("/check_message", response_model=PredictionResponse)
def check_message(req: SmsRequest) -> PredictionResponse:
    return _predict_sms(req.message)
