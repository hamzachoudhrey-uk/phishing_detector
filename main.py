"""
Phishing detection API (FastAPI): SMS and email text only (TF-IDF + text classifier).

All user payloads use the same pipeline: visible plain text only (HTML → text), then
every URL-like token removed (http(s), www, scheme://, bare domains, tel/sms/mailto,
data: URIs, image/link src in HTML). The model only sees remaining words.

- POST /check_sms — same extraction as /check_message; response `content_mode` is "sms" for plain input.
- POST /check_message — identical preprocessing; `content_mode` is "plain_email" or "html_email".

Models in backend/models/ (or MODELS_DIR / MODEL_BASE_URL on Railway):
  - tfidf_vectorizer.pkl (or tfidf.pkl)
  - SMS classifier pickle (e.g. xgb_sms_model (1).pkl)
"""
from __future__ import annotations

import sys
import os
import re
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup

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

# Gmail/HTML bodies: `_extract_text_for_classifier` turns HTML into visible text and strips
# URLs before the SMS/Tfidf model runs — same for `/check_sms` when input looks like HTML.
TEXT_CLASSIFIER_PREVIEW_LEN = int(os.getenv("TEXT_CLASSIFIER_PREVIEW_LEN", "3800"))


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


class CombinedPredictionResponse(PredictionResponse):
    """Top-level prediction/result/phishing_probability are from the text (SMS/Tfidf) model only."""

    text_check: PredictionResponse | None = None
    # Plain-language body (after HTML strip where applicable) passed to SMS/Tfidf model.
    text_input_preview: str | None = None
    # "sms" | "plain_email" | "html_email" — clarifies extraction path for clients.
    content_mode: str = "sms"


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


def _normalize_url(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    parsed = urlparse(url)
    if parsed.scheme:
        return url
    return f"http://{url}"


def _strip_url_trailing_punctuation(url: str) -> str:
    while url and url[-1] in ".,;:!?)>]}'\"":
        url = url[:-1]
    return url


def _extract_urls_scheme_any(text: str) -> list[str]:
    """`ftp://`, `market://`, `https://`, etc."""
    pat = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s<>'\"`\]]+")
    return _urls_from_pattern(text, pat, group=0)


def _urls_from_pattern(text: str, pat: re.Pattern[str], group: int = 0) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for match in pat.finditer(text):
        candidate = _strip_url_trailing_punctuation(match.group(group).strip())
        if not candidate:
            continue
        normalized = _normalize_url(candidate) if "://" in candidate else candidate
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _extract_urls_bare_domain(text: str) -> list[str]:
    """`evil.com/path` without scheme — skips `user@gmail.com` (after @)."""
    pat = re.compile(
        r"(?i)(?<![@\w/.])"
        r"(?:[a-z](?:[a-z0-9-]*[a-z0-9])?\.)+"
        r"[a-z]{2,63}"
        r"(?:/[^\s<>'\"`\]]*)?"
    )
    return _urls_from_pattern(text, pat, group=0)


def _extract_special_uri_prefixes(text: str) -> list[str]:
    """Smishing / mailto style."""
    pat = re.compile(r"(?i)\b(?:mailto|tel|sms):[^\s<>'\"`\]]+")
    return _urls_from_pattern(text, pat, group=0)


def _extract_data_uris(text: str) -> list[str]:
    pat = re.compile(r"(?i)\bdata:[^\s<>'\"`\]]+")
    return _urls_from_pattern(text, pat, group=0)


def _extract_ipv4_hosts(text: str) -> list[str]:
    """Literal IPv4 (optional :port / path). Octets 0–255 only so dates like 2024.05.03 are not matched."""
    octet = r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
    pat = re.compile(
        rf"(?<![\w.])(?:(?:{octet})\.){{3}}{octet}"
        rf"(?::\d{{1,5}})?(?:/[^\s<>'\"`\]]*)?",
        re.I,
    )
    return _urls_from_pattern(text, pat, group=0)


def _collect_urls_to_strip_from_plain(plain: str) -> list[str]:
    """All URL-like spans to remove before the text model (SMS and non-HTML email)."""
    n = _normalize_text(plain)
    if not n:
        return []
    return _merge_unique_urls(
        _extract_urls_conservative_plain(n),
        _extract_urls_scheme_any(n),
        _extract_urls_bare_domain(n),
        _extract_special_uri_prefixes(n),
        _extract_data_uris(n),
        _extract_ipv4_hosts(n),
    )


def _strip_urls_plain_email_style(plain: str) -> str:
    """Strip every URL-like token; same for SMS and plain email."""
    n = _normalize_text(plain)
    if not n:
        return ""
    detected = _collect_urls_to_strip_from_plain(n)
    return _plain_with_urls_removed(n, detected)


def _extract_text_for_classifier(raw_message: str) -> tuple[str, bool]:
    """Single preprocessing path for every endpoint: plain visible words only, no URLs.

    Returns ``(cleaned_text, used_html_branch)``. The string ``cleaned_text`` is what
    must be passed to the vectorizer/model (no HTML, no URL-like tokens).
    """
    raw = raw_message.strip()
    if not raw:
        return "", False
    if _looks_like_html(raw):
        plain = _visible_text_from_html(raw)
        anchor_urls = _hrefs_from_html(raw)
        src_urls = _http_srcs_from_html(raw)
        from_plain = _collect_urls_to_strip_from_plain(plain)
        detected_urls = _merge_unique_urls(anchor_urls, src_urls, from_plain)
        cleaned = _plain_with_urls_removed(plain, detected_urls)
        return (cleaned, True)
    cleaned = _strip_urls_plain_email_style(raw)
    return (cleaned, False)


def _visible_text_from_html(html: str) -> str:
    """Visible words from HTML: drop script/style/noscript, prefer <body> text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    root = soup.body if soup.body is not None else soup
    return _normalize_text(root.get_text(" ", strip=True))


def _prediction_response(pred: int, phishing_proba: float | None) -> PredictionResponse:
    result = "Phishing" if pred == 1 else "Safe"
    return PredictionResponse(
        prediction=pred,
        result=result,
        phishing_probability=phishing_proba,
    )


def _preview_classifier_text(text: str) -> str | None:
    """Readable preview returned to clients (not truncated for model input)."""
    if not text:
        return None
    t = _normalize_text(text)
    limit = TEXT_CLASSIFIER_PREVIEW_LEN
    if len(t) > limit:
        return t[:limit] + "\n…(truncated in preview)"
    return t


def _looks_like_html(s: str) -> bool:
    t = s.strip().lower()[:48000]
    if "<html" in t[:8000]:
        return True
    if "<body" in t[:8000]:
        return True
    return len(re.findall(r"<[a-zA-Z][a-zA-Z0-9:-]*\b", t)) >= 6


def _extract_urls_conservative_plain(text: str) -> list[str]:
    """Only schemes or www.* — avoids matching user@gmail.com as gmail.com."""

    urls: list[str] = []
    seen: set[str] = set()
    patterns = [
        re.compile(r"(?i)\b(https?://[^\s<>'\"`\]]+)"),
        re.compile(r"(?i)(?<![@\w])(www\.[^\s<>'\"`\]]+)"),
    ]
    for pat in patterns:
        for match in pat.finditer(text):
            candidate = _strip_url_trailing_punctuation(match.group(1).strip())
            if not candidate:
                continue
            normalized = _normalize_url(candidate)
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            urls.append(normalized)
    return urls


def _hrefs_from_html(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    urls: list[str] = []
    for tag in soup.find_all("a", href=True):
        raw = tag.get("href")
        href = unescape(((raw if isinstance(raw, str) else str(raw))) or "").strip()
        if not href or href.lower().startswith(("javascript:", "mailto:", "tel:", "sms:", "#")):
            continue
        if href.startswith("//"):
            href = "https:" + href
        if not href.startswith(("http://", "https://")):
            continue
        candidate = _strip_url_trailing_punctuation(href)
        normalized = _normalize_url(candidate)
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        urls.append(normalized)
    return urls


def _http_srcs_from_html(html: str) -> list[str]:
    """http(s) in img/iframe/etc. ``src`` (visible text often omits these)."""
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    urls: list[str] = []
    for tag in soup.find_all(src=True):
        raw = tag.get("src")
        src = unescape(((raw if isinstance(raw, str) else str(raw))) or "").strip()
        if not src or src.lower().startswith(("javascript:", "mailto:", "tel:", "sms:", "#")):
            continue
        if src.lower().startswith("data:"):
            continue
        if src.startswith("//"):
            src = "https:" + src
        if not src.startswith(("http://", "https://")):
            continue
        candidate = _strip_url_trailing_punctuation(src)
        normalized = _normalize_url(candidate)
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        urls.append(normalized)
    return urls


def _merge_unique_urls(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        for url in group:
            key = url.lower().rstrip(".")
            if key in seen:
                continue
            seen.add(key)
            out.append(url)
    return out


def _plain_with_urls_removed(plain_text: str, urls: list[str]) -> str:
    rest = plain_text
    for u in sorted({*urls}, key=len, reverse=True):
        if len(u) < 4:
            continue
        try:
            rest = re.sub(re.escape(u), " ", rest, flags=re.IGNORECASE)
        except re.error:
            rest = rest.replace(u, " ")
    return _normalize_text(rest)


def _combined_prediction_response(
    text_check: PredictionResponse | None,
    *,
    text_input_preview: str | None = None,
    content_mode: str = "sms",
) -> CombinedPredictionResponse:
    overall_pred = 0
    overall_prob: float | None = None
    if text_check is not None:
        overall_pred = int(text_check.prediction)
        overall_prob = (
            float(text_check.phishing_probability)
            if text_check.phishing_probability is not None
            else None
        )

    return CombinedPredictionResponse(
        prediction=overall_pred,
        result="Phishing" if overall_pred == 1 else "Safe",
        phishing_probability=overall_prob,
        text_check=text_check,
        text_input_preview=text_input_preview,
        content_mode=content_mode,
    )


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

    print(_ascii_preview(f"[check_sms] message={message}", limit=300))
    if phishing_proba is not None:
        print(
            _ascii_preview(
                f"[check_sms] prediction={pred} result={'Phishing' if pred == 1 else 'Safe'} phishing_proba={phishing_proba:.4f}",
                limit=300,
            )
        )
    else:
        print(
            _ascii_preview(
                f"[check_sms] prediction={pred} result={'Phishing' if pred == 1 else 'Safe'}",
                limit=300,
            )
        )
    return _prediction_response(pred, phishing_proba)


def _predict_email_text_only(
    raw_message: str,
    *,
    plain_content_mode: str = "plain_email",
    log_tag: str = "check_message",
) -> CombinedPredictionResponse:
    """Classify after `_extract_text_for_classifier` — model input is plain words only."""
    raw = raw_message.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="message must not be empty")

    cleaned_text, is_html = _extract_text_for_classifier(raw_message)
    content_mode = "html_email" if is_html else plain_content_mode

    text_check: PredictionResponse | None = None
    if cleaned_text:
        text_check = _predict_sms(cleaned_text)

    combined = _combined_prediction_response(
        text_check=text_check,
        text_input_preview=_preview_classifier_text(cleaned_text) if cleaned_text else None,
        content_mode=content_mode,
    )
    print(
        _ascii_preview(
            f"[{log_tag}] mode={content_mode} text={'yes' if text_check is not None else 'no'} overall={combined.result}",
            limit=300,
        )
    )
    return combined


@app.post("/check_sms", response_model=CombinedPredictionResponse)
def check_sms(req: SmsRequest) -> CombinedPredictionResponse:
    return _predict_email_text_only(
        req.message,
        plain_content_mode="sms",
        log_tag="check_sms",
    )


@app.post("/check_message", response_model=CombinedPredictionResponse)
def check_message(req: SmsRequest) -> CombinedPredictionResponse:
    return _predict_email_text_only(req.message)


@app.post("/check_message_text_only", response_model=CombinedPredictionResponse)
def check_message_text_only(req: SmsRequest) -> CombinedPredictionResponse:
    """Same as /check_message — one text-only pipeline for the model."""
    return _predict_email_text_only(req.message)
