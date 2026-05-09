"""
URL phishing via the project's XGBoost pickle.

Feature extraction and class semantics match the Kaggle / training pipeline
(``-1`` / ``0`` / ``1`` feature coding, ``0`` = legitimate / ``1`` = phishing,
``predict_proba`` column for label ``1``, decision threshold ``0.30`` by default).
"""
from __future__ import annotations

import os
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

# Training uses 0 = legitimate, 1 = phishing (``model.classes_`` is ``[0, 1]``).
PHISHING_CLASS = 1

# Match developer training script: lower threshold improves recall vs 0.5.
URL_PHISHING_THRESHOLD = float(os.getenv("URL_PHISHING_THRESHOLD", "0.30"))


def ensure_scheme(url: str) -> str:
    """Add https:// if the user omits a scheme so ``urlparse`` yields a real host."""
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "https://" + url
    return url


def extract_features(url: str, feature_columns: list[str]) -> pd.DataFrame:
    """
    Same feature logic as the Kaggle / FastAPI training reference.
    Column order follows ``feature_columns``; any name not computed here defaults to ``1``
    (same as ``DataFrame.reindex(..., fill_value=1)`` in the reference ``predict_url``).
    """
    url = ensure_scheme(url.strip())
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    features: dict[str, int] = {}

    ip_pattern = r"(([0-9]{1,3}\.){3}[0-9]{1,3})"
    features["UsingIP"] = -1 if re.search(ip_pattern, url) else 1

    if len(url) < 54:
        features["LongURL"] = 1
    elif len(url) <= 75:
        features["LongURL"] = 0
    else:
        features["LongURL"] = -1

    shortening = r"bit\.ly|goo\.gl|tinyurl|t\.co|ow\.ly"
    features["ShortURL"] = -1 if re.search(shortening, url) else 1

    features["Symbol@"] = -1 if "@" in url else 1

    features["Redirecting//"] = -1 if url.rfind("//") > 7 else 1

    features["PrefixSuffix-"] = -1 if "-" in domain else 1

    dots = domain.count(".")
    if dots == 1:
        features["SubDomains"] = 1
    elif dots == 2:
        features["SubDomains"] = 0
    else:
        features["SubDomains"] = -1

    features["HTTPS"] = 1 if parsed.scheme == "https" else -1

    features["DomainRegLen"] = -1 if len(domain) > 20 else 1

    features["HTTPSDomainURL"] = -1 if "https" in domain else 1

    features["RequestURL"] = -1 if len(url) > 100 else 1

    suspicious_words = [
        "login",
        "verify",
        "secure",
        "account",
        "update",
        "bank",
        "free",
        "gift",
        "paypal",
    ]

    features["AnchorURL"] = (
        -1 if any(word in url.lower() for word in suspicious_words) else 1
    )

    features["WebsiteTraffic"] = -1 if re.search(shortening, url) else 1
    features["GoogleIndex"] = -1 if re.search(shortening, url) else 1

    features["StatsReport"] = (
        -1 if any(word in url.lower() for word in suspicious_words) else 1
    )

    remaining_cols = [
        "Favicon",
        "NonStdPort",
        "ServerFormHandler",
        "InfoEmail",
        "AbnormalURL",
        "WebsiteForwarding",
        "StatusBarCust",
        "DisableRightClick",
        "UsingPopupWindow",
        "IframeRedirection",
        "AgeofDomain",
        "DNSRecording",
        "PageRank",
        "LinksPointingToPage",
        "LinksInScriptTags",
    ]

    for col in remaining_cols:
        features[col] = 1

    row = {col: int(features[col]) if col in features else 1 for col in feature_columns}
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
    """
    Returns ``P(phishing)`` from the column for class ``1``, and a binary phishing flag
    using ``URL_PHISHING_THRESHOLD`` (default ``0.30``) when ``predict_proba`` exists.
    """
    if raw_url == "":
        raise ValueError("empty url")

    X = extract_features(raw_url, feature_columns)

    prediction = int(model.predict(X)[0])

    probability_phish: float | None = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        phish_ix = _class_probability_index(model.classes_, PHISHING_CLASS)
        probability_phish = float(probabilities[phish_ix])

    if probability_phish is not None:
        pred_1_if_phishing = 1 if probability_phish >= URL_PHISHING_THRESHOLD else 0
    else:
        pred_1_if_phishing = 1 if prediction == PHISHING_CLASS else 0

    label = "Phishing Website" if pred_1_if_phishing == 1 else "Legitimate Website"
    prob_out = round(float(probability_phish), 4) if probability_phish is not None else 0.0
    return raw_url, label, prob_out, pred_1_if_phishing
