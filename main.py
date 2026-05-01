"""
Phishing detection API (FastAPI): SMS text + URL analysis.

Models in backend/models/ (or MODELS_DIR / MODEL_BASE_URL on Railway):
  - tfidf_vectorizer.pkl (or tfidf.pkl)
  - SMS classifier pickle (e.g. xgb_sms_model (1).pkl)
  - url_phishing_model.pkl
"""
from __future__ import annotations

import sys
import os
import ipaddress
import re
import socket
import warnings
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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

URL_MODEL_PATH = MODELS_DIR / "url_phishing_model.pkl"
ALT_URL_MODEL_PATHS = [
    MODELS_DIR / "url_model.pkl",
    MODELS_DIR / "phishing_url_model.pkl",
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


class UrlRequest(BaseModel):
    url: str = Field(..., min_length=1, description="URL to analyze")


class PredictionResponse(BaseModel):
    prediction: int
    result: str
    phishing_probability: float | None = None


class UrlPredictionResponse(PredictionResponse):
    url: str
    error: str | None = None


class CombinedPredictionResponse(PredictionResponse):
    text_check: PredictionResponse | None = None
    url_checks: list[UrlPredictionResponse] = Field(default_factory=list)
    extracted_urls: list[str] = Field(default_factory=list)


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
    has_url = URL_MODEL_PATH.exists() or any(p.exists() for p in ALT_URL_MODEL_PATHS)
    if has_vec and has_sms and has_url:
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

    if not URL_MODEL_PATH.exists() and not any(p.exists() for p in ALT_URL_MODEL_PATHS):
        try:
            _download("url_phishing_model.pkl", URL_MODEL_PATH)
        except Exception:
            last_err: Exception | None = None
            for p in ALT_URL_MODEL_PATHS:
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


def _resolve_url_model_path() -> Path:
    if URL_MODEL_PATH.exists():
        return URL_MODEL_PATH
    for p in ALT_URL_MODEL_PATHS:
        if p.exists():
            return p
    return URL_MODEL_PATH


@app.on_event("startup")
def _startup_load_models() -> None:
    _ensure_models_present()
    app.state.vectorizer = _load_joblib(_resolve_vectorizer_path())
    app.state.sms_model = _load_joblib(_resolve_sms_model_path())
    app.state.url_model = _load_joblib(_resolve_url_model_path())
    app.state.url_feature_names = list(
        getattr(
            app.state.url_model,
            "feature_names_in_",
            [
                "having_IPhaving_IP_Address",
                "URLURL_Length",
                "Shortining_Service",
                "having_At_Symbol",
                "double_slash_redirecting",
                "Prefix_Suffix",
                "having_Sub_Domain",
                "SSLfinal_State",
                "Domain_registeration_length",
                "Favicon",
                "port",
                "HTTPS_token",
                "Request_URL",
                "URL_of_Anchor",
                "Links_in_tags",
                "SFH",
                "Submitting_to_email",
                "Abnormal_URL",
                "Redirect",
                "on_mouseover",
                "RightClick",
                "popUpWidnow",
                "Iframe",
                "age_of_domain",
                "DNSRecord",
                "web_traffic",
                "Page_Rank",
                "Google_Index",
                "Links_pointing_to_page",
                "Statistical_report",
            ],
        )
    )


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


def _extract_urls(text: str) -> list[str]:
    # Supports:
    # - https://example.com/path
    # - http://example.com
    # - www.example.com
    # - example.com/path
    pattern = re.compile(
        r"(?i)\b((?:https?://|www\.)[^\s<>'\"`]+|(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s<>'\"`]+)?)"
    )
    seen: set[str] = set()
    urls: list[str] = []
    for match in pattern.finditer(text):
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


def _remove_urls_from_text(text: str) -> str:
    pattern = re.compile(
        r"(?i)\b((?:https?://|www\.)[^\s<>'\"`]+|(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s<>'\"`]+)?)"
    )
    no_urls = pattern.sub(" ", text)
    return _normalize_text(no_urls)


def _prediction_response(pred: int, phishing_proba: float | None) -> PredictionResponse:
    result = "Phishing" if pred == 1 else "Safe"
    return PredictionResponse(
        prediction=pred,
        result=result,
        phishing_probability=phishing_proba,
    )


def _combined_prediction_response(
    text_check: PredictionResponse | None,
    url_checks: list[UrlPredictionResponse],
    extracted_urls: list[str],
) -> CombinedPredictionResponse:
    phishing_candidates: list[float] = []
    any_phishing = False

    if text_check is not None:
        any_phishing = any_phishing or text_check.prediction == 1
        if text_check.phishing_probability is not None:
            phishing_candidates.append(float(text_check.phishing_probability))

    for check in url_checks:
        if check.error is None:
            any_phishing = any_phishing or check.prediction == 1
            if check.phishing_probability is not None:
                phishing_candidates.append(float(check.phishing_probability))

    overall_pred = 1 if any_phishing else 0
    overall_prob = max(phishing_candidates) if phishing_candidates else None

    return CombinedPredictionResponse(
        prediction=overall_pred,
        result="Phishing" if overall_pred == 1 else "Safe",
        phishing_probability=overall_prob,
        text_check=text_check,
        url_checks=url_checks,
        extracted_urls=extracted_urls,
    )


def _is_ip_host(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _same_or_subdomain(hostname: str, other: str) -> bool:
    if not hostname or not other:
        return False
    return other == hostname or other.endswith(f".{hostname}")


def _feature_from_ratio(ratio: float, legit_threshold: float, suspicious_threshold: float) -> int:
    if ratio < legit_threshold:
        return 1
    if ratio <= suspicious_threshold:
        return 0
    return -1


def _safe_hostname(candidate: str | None) -> str:
    return (candidate or "").strip().lower()


def _fetch_url_context(url: str) -> dict[str, Any]:
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
            )
        },
    )
    try:
        with urlopen(req, timeout=6) as resp:  # nosec - fetching user-provided URLs for analysis
            content_type = resp.headers.get("Content-Type", "")
            body = resp.read(250_000)
            final_url = resp.geturl()
        text = body.decode("utf-8", errors="replace")
        return {
            "ok": True,
            "html": text,
            "content_type": content_type,
            "final_url": final_url,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "html": "", "content_type": "", "final_url": url}


def _extract_url_features(url: str) -> list[int]:
    parsed = urlparse(url)
    hostname = _safe_hostname(parsed.hostname)
    fetch = _fetch_url_context(url)
    final_url = fetch["final_url"]
    final_parsed = urlparse(final_url)
    final_host = _safe_hostname(final_parsed.hostname)
    html = fetch["html"] if fetch["ok"] else ""
    soup = BeautifulSoup(html, "html.parser") if html else None

    features: dict[str, int] = {}
    features["having_IPhaving_IP_Address"] = -1 if _is_ip_host(hostname) else 1

    url_len = len(url)
    features["URLURL_Length"] = 1 if url_len < 54 else (0 if url_len <= 75 else -1)

    shorteners = (
        "bit.ly",
        "goo.gl",
        "tinyurl.com",
        "ow.ly",
        "t.co",
        "is.gd",
        "buff.ly",
        "adf.ly",
        "bitly.com",
        "cutt.ly",
    )
    features["Shortining_Service"] = -1 if any(s in hostname for s in shorteners) else 1
    features["having_At_Symbol"] = -1 if "@" in url else 1
    features["double_slash_redirecting"] = -1 if url.rfind("//") > 7 else 1
    features["Prefix_Suffix"] = -1 if "-" in hostname else 1

    domain_parts = [part for part in hostname.split(".") if part]
    features["having_Sub_Domain"] = 1 if len(domain_parts) <= 2 else (0 if len(domain_parts) == 3 else -1)
    features["SSLfinal_State"] = 1 if final_parsed.scheme.lower() == "https" else -1
    features["Domain_registeration_length"] = 0
    features["port"] = -1 if parsed.port not in (None, 80, 443) else 1
    features["HTTPS_token"] = -1 if "https" in hostname.replace(".", "") else 1
    features["Abnormal_URL"] = 1 if hostname else -1
    features["Redirect"] = 0 if final_url != url else 1

    try:
        socket.gethostbyname(hostname)
        features["DNSRecord"] = 1
    except Exception:
        features["DNSRecord"] = -1

    if soup is None:
        features["Favicon"] = 0
        features["Request_URL"] = 0
        features["URL_of_Anchor"] = 0
        features["Links_in_tags"] = 0
        features["SFH"] = 0
        features["Submitting_to_email"] = 0
        features["on_mouseover"] = 0
        features["RightClick"] = 0
        features["popUpWidnow"] = 0
        features["Iframe"] = 0
        features["Links_pointing_to_page"] = 0
    else:
        favicon_links = soup.find_all("link", rel=lambda value: value and "icon" in " ".join(value).lower())
        if favicon_links:
            href = favicon_links[0].get("href", "")
            icon_host = _safe_hostname(urlparse(href).hostname) if href.startswith(("http://", "https://")) else hostname
            features["Favicon"] = 1 if _same_or_subdomain(hostname, icon_host) else -1
        else:
            features["Favicon"] = 0

        resource_tags = soup.find_all(["img", "audio", "embed", "iframe", "source"])
        if resource_tags:
            external = 0
            for tag in resource_tags:
                src = tag.get("src", "")
                src_host = _safe_hostname(urlparse(src).hostname) if src.startswith(("http://", "https://")) else hostname
                if src and not _same_or_subdomain(hostname, src_host):
                    external += 1
            features["Request_URL"] = _feature_from_ratio(external / len(resource_tags), 0.22, 0.61)
        else:
            features["Request_URL"] = 1

        anchors = soup.find_all("a")
        if anchors:
            unsafe = 0
            for tag in anchors:
                href = unescape((tag.get("href", "") or "").strip())
                if not href or href.startswith(("#", "javascript:void", "mailto:")):
                    unsafe += 1
                    continue
                href_host = _safe_hostname(urlparse(href).hostname) if href.startswith(("http://", "https://")) else hostname
                if not _same_or_subdomain(hostname, href_host):
                    unsafe += 1
            features["URL_of_Anchor"] = _feature_from_ratio(unsafe / len(anchors), 0.31, 0.67)
        else:
            features["URL_of_Anchor"] = 1

        linked_tags = soup.find_all("link") + soup.find_all("script") + soup.find_all("meta")
        if linked_tags:
            external = 0
            checked = 0
            for tag in linked_tags:
                attr = tag.get("href") or tag.get("src") or tag.get("content") or ""
                if not attr:
                    continue
                checked += 1
                attr_host = _safe_hostname(urlparse(attr).hostname) if attr.startswith(("http://", "https://")) else hostname
                if attr and not _same_or_subdomain(hostname, attr_host):
                    external += 1
            features["Links_in_tags"] = (
                _feature_from_ratio(external / checked, 0.17, 0.81) if checked else 1
            )
        else:
            features["Links_in_tags"] = 1

        form = soup.find("form")
        if form is None:
            features["SFH"] = 1
        else:
            action = (form.get("action", "") or "").strip().lower()
            if not action or action == "about:blank":
                features["SFH"] = -1
            elif action.startswith(("http://", "https://")):
                action_host = _safe_hostname(urlparse(action).hostname)
                features["SFH"] = 1 if _same_or_subdomain(hostname, action_host) else 0
            else:
                features["SFH"] = 1

        features["Submitting_to_email"] = -1 if "mailto:" in html.lower() else 1
        features["on_mouseover"] = -1 if "onmouseover" in html.lower() else 1
        features["RightClick"] = -1 if re.search(r"event\.button\s*==\s*2", html, re.I) else 1
        features["popUpWidnow"] = -1 if re.search(r"alert\s*\(", html, re.I) else 1
        features["Iframe"] = -1 if soup.find("iframe") is not None else 1
        features["Links_pointing_to_page"] = 0

    features["age_of_domain"] = 0
    features["web_traffic"] = 0
    features["Page_Rank"] = 0
    features["Google_Index"] = 0
    features["Statistical_report"] = -1 if (_is_ip_host(hostname) or any(s in hostname for s in shorteners)) else 1

    feature_names = app.state.url_feature_names
    return [int(features.get(name, 0)) for name in feature_names]


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


def _predict_url(url: str) -> PredictionResponse:
    url = _normalize_url(url)
    if not url:
        raise HTTPException(status_code=400, detail="url must not be empty")

    model = app.state.url_model
    feature_vector = _extract_url_features(url)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but .* was fitted with feature names",
        )
        pred = int(model.predict([feature_vector])[0])
    phishing_proba: float | None = None
    if hasattr(model, "predict_proba"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but .* was fitted with feature names",
            )
            proba = model.predict_proba([feature_vector])[0]
            if len(proba) >= 2:
                phishing_proba = float(proba[1])

    print(_ascii_preview(f"[check_url] url={url}", limit=300))
    if phishing_proba is not None:
        print(
            _ascii_preview(
                f"[check_url] prediction={pred} result={'Phishing' if pred == 1 else 'Safe'} phishing_proba={phishing_proba:.4f}",
                limit=300,
            )
        )
    else:
        print(
            _ascii_preview(
                f"[check_url] prediction={pred} result={'Phishing' if pred == 1 else 'Safe'}",
                limit=300,
            )
        )
    return _prediction_response(pred, phishing_proba)


def _predict_split_content(raw_message: str) -> CombinedPredictionResponse:
    message = _normalize_text(raw_message)
    if not message:
        raise HTTPException(status_code=400, detail="message must not be empty")

    extracted_urls = _extract_urls(message)
    cleaned_text = _remove_urls_from_text(message)

    text_check: PredictionResponse | None = None
    if cleaned_text:
        text_check = _predict_sms(cleaned_text)

    url_checks: list[UrlPredictionResponse] = []
    for url in extracted_urls:
        try:
            url_result = _predict_url(url)
            url_checks.append(
                UrlPredictionResponse(
                    url=url,
                    prediction=url_result.prediction,
                    result=url_result.result,
                    phishing_probability=url_result.phishing_probability,
                )
            )
        except Exception as exc:
            url_checks.append(
                UrlPredictionResponse(
                    url=url,
                    prediction=0,
                    result="Unknown",
                    phishing_probability=None,
                    error=str(exc),
                )
            )

    combined = _combined_prediction_response(
        text_check=text_check,
        url_checks=url_checks,
        extracted_urls=extracted_urls,
    )
    print(
        _ascii_preview(
            (
                f"[split] urls={len(extracted_urls)} "
                f"text={'yes' if text_check is not None else 'no'} "
                f"overall={combined.result}"
            ),
            limit=300,
        )
    )
    return combined


@app.post("/check_sms", response_model=CombinedPredictionResponse)
def check_sms(req: SmsRequest) -> CombinedPredictionResponse:
    # Backward compatible: still includes prediction/result/phishing_probability
    # while also returning split text/url analysis details.
    return _predict_split_content(req.message)


@app.post("/check_message", response_model=CombinedPredictionResponse)
def check_message(req: SmsRequest) -> CombinedPredictionResponse:
    # Email-body scan path. Same split behavior as SMS.
    return _predict_split_content(req.message)


@app.post("/check_url", response_model=PredictionResponse)
def check_url(req: UrlRequest) -> PredictionResponse:
    return _predict_url(req.url)
