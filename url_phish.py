"""
URL phishing via the project's XGBoost pickle + developer ``extract_features``.
API matches ``predict_url`` logic from ``test_url_model.py`` (no argparse / prints).
"""
from __future__ import annotations

import ipaddress
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import pandas as pd

# Same basename order as developer ``MODEL_FILE`` (try this first).
MODEL_FILENAMES: tuple[str, ...] = (
    "final_xgboost_model.pkl",
    "url_final_xgboost_model.pkl",
)

# IMPORTANT:
# Your model output shows: 0 = phishing and 1 = legitimate.
# Google predicted raw class 1, so class 1 is treated as safe.
PHISHING_CLASS = 0

SHORTENING_SERVICES = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd", "buff.ly",
    "adf.ly", "bitly.com", "cutt.ly", "rebrand.ly", "rb.gy", "s.id",
    "shorturl.at", "tiny.cc", "lnkd.in"
}

SUSPICIOUS_WORDS = [
    "login", "verify", "account", "update", "secure", "bank", "paypal",
    "signin", "confirm", "password", "free", "bonus", "win", "gift"
]


def ensure_scheme(url: str) -> str:
    """Add http:// if user enters example.com without scheme."""
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url


def is_ip_address(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def count_subdomains(host: str) -> int:
    parts = [p for p in host.split(".") if p]
    # example.com => 2 parts, so 0 subdomains
    return max(0, len(parts) - 2)


def extract_features(url: str, feature_columns: list[str]) -> pd.DataFrame:
    """
    Extract the exact feature columns expected by the model.

    Note:
    Some original dataset features require live webpage/domain lookups
    such as Domain Age, Traffic, PageRank, Google Index, page links, iframe, popup, etc.
    For command-line raw URL testing, those are set to neutral values.
    """
    url = ensure_scheme(url)
    parsed = urlparse(url)
    host = parsed.netloc.lower().split("@")[-1].split(":")[0]
    path = parsed.path.lower()
    full_url = url.lower()

    subdomain_count = count_subdomains(host)
    has_https = parsed.scheme.lower() == "https"
    has_ip = is_ip_address(host)
    has_non_standard_port = parsed.port not in (None, 80, 443)
    is_short_url = host in SHORTENING_SERVICES or any(host.endswith("." + s) for s in SHORTENING_SERVICES)
    has_redirecting_double_slash = "//" in url.replace(parsed.scheme + "://", "", 1)
    has_prefix_suffix = "-" in host
    suspicious_text = any(word in full_url for word in SUSPICIOUS_WORDS)

    # Create neutral/default values first.
    # Your feature_columns.pkl has these 31 columns:
    features = {col: 0 for col in feature_columns}

    # URL-based features we can calculate from a raw URL.
    features.update({
        "Index": 0,
        "UsingIP": 1 if has_ip else 0,
        "LongURL": 1 if len(url) >= 75 else 0,
        "ShortURL": 1 if is_short_url else 0,
        "Symbol@": 1 if "@" in url else 0,
        "Redirecting//": 1 if has_redirecting_double_slash else 0,
        "PrefixSuffix-": 1 if has_prefix_suffix else 0,
        "SubDomains": 1 if subdomain_count >= 2 else 0,
        "HTTPS": 0 if has_https else 1,
        "DomainRegLen": 0,
        "Favicon": 0,
        "NonStdPort": 1 if has_non_standard_port else 0,
        "HTTPSDomainURL": 1 if "https" in host else 0,
        "RequestURL": 0,
        "AnchorURL": 0,
        "LinksInScriptTags": 0,
        "ServerFormHandler": 0,
        "InfoEmail": 1 if "mailto:" in full_url or "email" in full_url else 0,
        "AbnormalURL": 1 if suspicious_text or has_ip else 0,
        "WebsiteForwarding": 0,
        "StatusBarCust": 0,
        "DisableRightClick": 0,
        "UsingPopupWindow": 0,
        "IframeRedirection": 0,
        "AgeofDomain": 0,
        "DNSRecording": 0,
        "WebsiteTraffic": 0,
        "PageRank": 0,
        "GoogleIndex": 0,
        "LinksPointingToPage": 0,
        "StatsReport": 1 if has_ip or suspicious_text else 0,
    })

    # Keep only the columns and order expected by the model.
    row = {col: features.get(col, 0) for col in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def load_url_classifier(models_dir: Path) -> tuple[list[str], Any]:
    cols_path = models_dir / "feature_columns.pkl"
    if not cols_path.exists():
        raise FileNotFoundError(f"Missing {cols_path}")

    model_path: Path | None = None
    for name in MODEL_FILENAMES:
        candidate = models_dir / name
        if candidate.exists():
            model_path = candidate
            break
    if model_path is None:
        raise FileNotFoundError(
            "Missing URL model; tried: "
            + ", ".join(str(models_dir / n) for n in MODEL_FILENAMES)
        )

    # Match developer ``predict_url``: columns come only from the pickle — never reorder.
    columns: list[str] = joblib.load(cols_path)
    model = joblib.load(model_path)
    return columns, model


def _class_probability_index(classes: Any, label: int) -> int:
    """``predict_proba`` column index matching ``label`` (e.g. phishing ``PHISHING_CLASS``)."""
    for i, c in enumerate(classes):
        if float(c) == float(label):
            return i
    raise ValueError(f"class label {label!r} not in model.classes_: {classes!r}")


def predict_url_phishing(
    raw_url: str,
    *,
    model: Any,
    feature_columns: list[str],
) -> tuple[str, str, float, int]:
    """Same ``predict`` as developer script; ``probability`` returned is ``P(phishing)`` (0..1), not %-scaled."""
    if raw_url == "":
        raise ValueError("empty url")

    X = extract_features(raw_url, feature_columns)

    prediction = int(model.predict(X)[0])

    probability_phish = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        phish_ix = _class_probability_index(model.classes_, PHISHING_CLASS)
        probability_phish = float(probabilities[phish_ix])

    pred_1_if_phishing = 1 if prediction == PHISHING_CLASS else 0
    label = "Phishing Website" if pred_1_if_phishing == 1 else "Legitimate Website"
    prob_out = round(float(probability_phish), 4) if probability_phish is not None else 0.0
    return raw_url, label, prob_out, pred_1_if_phishing
