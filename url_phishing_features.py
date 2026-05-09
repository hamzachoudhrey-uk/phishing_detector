"""
Lexical + lightweight HTML/WHOIS/Alexa features for ``url_final_xgboost_model.pkl``.

Column names align with ``feature_columns.pkl`` (classic phishing URL datasets).
Values use {-1, 0, 1} tri-class encoding expected by the trained booster.

Adapted from commonly published phishing URL extractors; HTTP/WHOIS calls may fail —
fail-soft branches mirror upstream ``except: return -1`` patterns.
"""
from __future__ import annotations

import ipaddress
import re
from datetime import date, datetime
from typing import Any
from urllib.parse import quote, urlparse

import requests
from bs4 import BeautifulSoup

SHORTENING_SERVICES = re.compile(
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"
    r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"
    r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
    r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|db\.tt|"
    r"qr\.ae|adf\.ly|bitly\.com|cur\.lv|tinyurl\.com|ity\.im|q\.gs|po\.st|bc\.vc|twitthis\.com|"
    r"u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|prettylinkpro\.com|scrnch\.me|"
    r"filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|link\.zip\.net",
    re.I,
)


class UrlPhishingFeatures:
    """Produce one row matching ``feature_columns.pkl`` order for ``url``."""

    _COLUMN_ORDER: tuple[str, ...] = (
        "UsingIP",
        "LongURL",
        "ShortURL",
        "Symbol@",
        "Redirecting//",
        "PrefixSuffix-",
        "SubDomains",
        "HTTPS",
        "DomainRegLen",
        "Favicon",
        "NonStdPort",
        "HTTPSDomainURL",
        "RequestURL",
        "AnchorURL",
        "LinksInScriptTags",
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
        "WebsiteTraffic",
        "PageRank",
        "GoogleIndex",
        "LinksPointingToPage",
        "StatsReport",
    )

    def __init__(
        self,
        url: str,
        *,
        fetch_timeout_s: float = 10.0,
        user_agent: str = "PhishingDetectorBot/1.0 (+https://localhost)",
    ) -> None:
        self.url = url.strip()
        self.fetch_timeout_s = fetch_timeout_s
        self.user_agent = user_agent
        self.response: requests.Response | None = None
        self.soup: BeautifulSoup | None = None
        self.domain = ""
        self.urlparse = urlparse(self.url)
        self.whois_response: Any = None

        try:
            self.domain = self.urlparse.netloc or ""
        except Exception:
            self.domain = ""

        try:
            import whois  # type: ignore

            host = self.urlparse.hostname
            if host:
                self.whois_response = whois.whois(host)
        except Exception:
            self.whois_response = None

        headers = {"User-Agent": user_agent}
        try:
            self.response = requests.get(
                self.url,
                timeout=fetch_timeout_s,
                allow_redirects=True,
                headers=headers,
            )
            text = self.response.text if self.response.text else ""
            self.soup = BeautifulSoup(text, "html.parser")
        except Exception:
            self.response = None
            self.soup = None

    @classmethod
    def column_names(cls) -> tuple[str, ...]:
        return cls._COLUMN_ORDER

    def extract_row(self) -> dict[str, int]:
        """Return feature name -> value in {-1, 0, 1}."""
        return {
            "UsingIP": self._using_ip(),
            "LongURL": self._long_url(),
            "ShortURL": self._short_url(),
            "Symbol@": self._symbol_at(),
            "Redirecting//": self._redirecting(),
            "PrefixSuffix-": self._prefix_suffix(),
            "SubDomains": self._sub_domains(),
            "HTTPS": self._https_scheme(),
            "DomainRegLen": self._domain_reg_len(),
            "Favicon": self._favicon(),
            "NonStdPort": self._non_std_port(),
            "HTTPSDomainURL": self._https_domain_url(),
            "RequestURL": self._request_url(),
            "AnchorURL": self._anchor_url(),
            "LinksInScriptTags": self._links_in_script_tags(),
            "ServerFormHandler": self._server_form_handler(),
            "InfoEmail": self._info_email(),
            "AbnormalURL": self._abnormal_url(),
            "WebsiteForwarding": self._website_forwarding(),
            "StatusBarCust": self._status_bar_cust(),
            "DisableRightClick": self._disable_right_click(),
            "UsingPopupWindow": self._using_popup_window(),
            "IframeRedirection": self._iframe_redirection(),
            "AgeofDomain": self._age_of_domain(),
            "DNSRecording": self._dns_recording(),
            "WebsiteTraffic": self._website_traffic(),
            "PageRank": self._page_rank(),
            "GoogleIndex": self._google_index(),
            "LinksPointingToPage": self._links_pointing_to_page(),
            "StatsReport": self._stats_report(),
        }

    # --- Published-style helpers (tri-class {-1,0,1}) ---

    def _using_ip(self) -> int:
        """Match legacy extractor: attempt to parse full URL string as IP (usually fails → 1)."""
        try:
            ipaddress.ip_address(self.url)
            return -1
        except Exception:
            return 1

    def _long_url(self) -> int:
        if len(self.url) < 54:
            return 1
        if 54 <= len(self.url) <= 75:
            return 0
        return -1

    def _short_url(self) -> int:
        if SHORTENING_SERVICES.search(self.url):
            return -1
        return 1

    def _symbol_at(self) -> int:
        return -1 if "@" in self.url else 1

    def _redirecting(self) -> int:
        pos = self.url.rfind("//")
        return -1 if pos > 6 else 1

    def _prefix_suffix(self) -> int:
        try:
            dom = self.domain or (self.urlparse.netloc or "")
            return -1 if "-" in dom else 1
        except Exception:
            return -1

    def _sub_domains(self) -> int:
        dot_count = len(re.findall(r"\.", self.url))
        if dot_count == 1:
            return 1
        if dot_count == 2:
            return 0
        return -1

    def _https_scheme(self) -> int:
        try:
            return 1 if self.urlparse.scheme.lower() == "https" else -1
        except Exception:
            return 1

    def _domain_reg_len(self) -> int:
        wr = self.whois_response
        if wr is None:
            return -1
        try:
            expiration_date = wr.expiration_date
            creation_date = wr.creation_date
            if isinstance(expiration_date, list):
                expiration_date = expiration_date[0]
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            if expiration_date is None or creation_date is None:
                return -1
            age = (expiration_date.year - creation_date.year) * 12 + (
                expiration_date.month - creation_date.month
            )
            return 1 if age >= 12 else -1
        except Exception:
            return -1

    def _favicon(self) -> int:
        if self.soup is None:
            return -1
        try:
            dom = self.domain
            for link in self.soup.find_all("link", href=True):
                href = link["href"]
                dots = [x.start(0) for x in re.finditer(r"\.", href)]
                if self.url in href or len(dots) == 1 or (dom and dom in href):
                    return 1
            return -1
        except Exception:
            return -1

    def _non_std_port(self) -> int:
        try:
            netloc = self.urlparse.netloc or self.domain
            parts = netloc.split(":")
            return -1 if len(parts) > 1 else 1
        except Exception:
            return -1

    def _https_domain_url(self) -> int:
        try:
            dom = self.domain.lower()
            return -1 if "https" in dom else 1
        except Exception:
            return -1

    def _request_url(self) -> int:
        if self.soup is None:
            return -1
        dom = self.domain
        i, success = 0, 0
        try:
            for tag_name in ("img", "audio", "embed", "iframe"):
                for node in self.soup.find_all(tag_name, src=True):
                    src = node["src"]
                    dots = [x.start(0) for x in re.finditer(r"\.", src)]
                    if self.url in src or (dom and dom in src) or len(dots) == 1:
                        success += 1
                    i += 1
            if i == 0:
                return 0
            pct = success / float(i) * 100
            if pct < 22.0:
                return 1
            if 22.0 <= pct < 61.0:
                return 0
            return -1
        except Exception:
            return -1

    def _anchor_url(self) -> int:
        if self.soup is None:
            return -1
        dom = self.domain
        try:
            unsafe = 0
            i = 0
            for a in self.soup.find_all("a", href=True):
                href = a["href"]
                if (
                    "#" in href
                    or "javascript" in href.lower()
                    or "mailto" in href.lower()
                    or not (self.url in href or (dom and dom in href))
                ):
                    unsafe += 1
                i += 1
            if i == 0:
                return -1
            pct = unsafe / float(i) * 100
            if pct < 31.0:
                return 1
            if 31.0 <= pct < 67.0:
                return 0
            return -1
        except Exception:
            return -1

    def _links_in_script_tags(self) -> int:
        if self.soup is None:
            return -1
        dom = self.domain
        i, success = 0, 0
        try:
            for link in self.soup.find_all("link", href=True):
                href = link["href"]
                dots = [x.start(0) for x in re.finditer(r"\.", href)]
                if self.url in href or (dom and dom in href) or len(dots) == 1:
                    success += 1
                i += 1
            for script in self.soup.find_all("script", src=True):
                src = script["src"]
                dots = [x.start(0) for x in re.finditer(r"\.", src)]
                if self.url in src or (dom and dom in src) or len(dots) == 1:
                    success += 1
                i += 1
            if i == 0:
                return 0
            pct = success / float(i) * 100
            if pct < 17.0:
                return 1
            if 17.0 <= pct < 81.0:
                return 0
            return -1
        except Exception:
            return -1

    def _server_form_handler(self) -> int:
        if self.soup is None:
            return -1
        dom = self.domain
        try:
            forms = self.soup.find_all("form", action=True)
            if not forms:
                return 1
            for form in forms:
                act = form.get("action") or ""
                if act == "" or act == "about:blank":
                    return -1
                if self.url not in act and (not dom or dom not in act):
                    return 0
            return 1
        except Exception:
            return -1

    def _info_email(self) -> int:
        if self.response is None:
            return -1
        try:
            return -1 if re.findall(r"[mail\(\)|mailto:?]", self.response.text) else 1
        except Exception:
            return -1

    def _abnormal_url(self) -> int:
        return -1

    def _website_forwarding(self) -> int:
        if self.response is None:
            return -1
        try:
            hist = len(self.response.history)
            if hist <= 1:
                return 1
            if hist <= 4:
                return 0
            return -1
        except Exception:
            return -1

    def _status_bar_cust(self) -> int:
        if self.response is None:
            return -1
        try:
            return 1 if re.findall(r".+onmouseover.+ ", self.response.text) else -1
        except Exception:
            return -1

    def _disable_right_click(self) -> int:
        if self.response is None:
            return -1
        try:
            return 1 if re.findall(r"event.button ?== ?2", self.response.text) else -1
        except Exception:
            return -1

    def _using_popup_window(self) -> int:
        if self.response is None:
            return -1
        try:
            return 1 if re.findall(r"alert\(", self.response.text) else -1
        except Exception:
            return -1

    def _iframe_redirection(self) -> int:
        if self.response is None:
            return -1
        try:
            return 1 if re.findall(r"[ |]", self.response.text) else -1
        except Exception:
            return -1

    def _age_of_domain(self) -> int:
        wr = self.whois_response
        if wr is None:
            return -1
        try:
            creation_date = wr.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            if creation_date is None:
                return -1
            today = date.today()
            age = (today.year - creation_date.year) * 12 + (
                today.month - creation_date.month
            )
            return 1 if age >= 6 else -1
        except Exception:
            return -1

    def _dns_recording(self) -> int:
        return self._age_of_domain()

    def _website_traffic(self) -> int:
        try:
            import urllib.request

            qurl = quote(self.url, safe="")
            with urllib.request.urlopen(
                "http://data.alexa.com/data?cli=10&dat=s&url=" + qurl,
                timeout=5,
            ) as resp:
                xml_text = resp.read().decode("utf-8", errors="replace")
            soup = BeautifulSoup(xml_text, "xml")
            reach = soup.find("REACH")
            if reach is None or reach.get("RANK") is None:
                return -1
            rank = int(reach["RANK"])
            return 1 if rank < 100000 else 0
        except Exception:
            return -1

    def _page_rank(self) -> int:
        dom = self.domain
        if not dom:
            return -1
        try:
            rank_checker_response = requests.post(
                "https://www.checkpagerank.net/index.php",
                {"name": dom},
                timeout=8,
                headers={"User-Agent": self.user_agent},
            )
            gr = re.findall(r"Global Rank: ([0-9]+)", rank_checker_response.text)
            if not gr:
                return -1
            global_rank = int(gr[0])
            return 1 if 0 < global_rank < 100000 else -1
        except Exception:
            return -1

    def _google_index(self) -> int:
        try:
            from googlesearch import search  # type: ignore

            for _ in search(self.url, num_results=3):
                return 1
            return -1
        except Exception:
            return 1

    def _links_pointing_to_page(self) -> int:
        if self.soup is None:
            return -1
        try:
            n = len(self.soup.find_all("a", href=True))
            if n == 0:
                return -1
            if n <= 2:
                return 0
            return 1
        except Exception:
            return -1

    def _stats_report(self) -> int:
        if self.response is None:
            return -1
        try:
            if re.findall(
                r"statcounter\.com|Histats\.com|analytics\.google\.com",
                self.response.text,
                re.I,
            ):
                return 1
            return -1
        except Exception:
            return -1


def build_feature_matrix(url: str, column_names: list[str], **kwargs: Any) -> Any:
    """One-row dense matrix for sklearn/XGBoost (columns in ``column_names`` order)."""
    import numpy as np

    ext = UrlPhishingFeatures(url, **kwargs)
    row = ext.extract_row()
    values: list[float] = []
    for c in column_names:
        if not c:
            continue
        if c == "Index":
            values.append(0.0)
        else:
            values.append(float(row[c]))
    return np.array([values], dtype=np.float32)
