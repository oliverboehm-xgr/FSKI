from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import List, Optional
from app.net import http_get

from urllib.parse import urlparse, urljoin, quote_plus, parse_qs



@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass
class FetchResult:
    title: str
    url: str
    text: str
    snippet: str
    body: str
    hash: str
    fetched_at: str
    domain: str


DEFAULT_HEADERS = {
    # Browser-like UA reduces 403 on many sites.
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.7",
    "Connection": "close",
}


def _now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _strip_html(s: str) -> str:
    s = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?is)<[^>]+>", " ", s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _extract_title(page: str) -> str:
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", page)
    if not m:
        return ""
    return _strip_html(m.group(1))


def _normalize_result_url(u: str) -> str:
    u = (u or "").strip()
    if u.startswith("//"):
        return "https:" + u
    # DDG redirect: /l/?uddg=...
    if u.startswith("/l/?"):
        return _decode_ddg_redirect("https://duckduckgo.com" + u)
    if u.startswith("https://duckduckgo.com/l/?") or u.startswith("http://duckduckgo.com/l/?"):
        return _decode_ddg_redirect(u)
    return u


def _decode_ddg_redirect(ddg: str) -> str:
    try:
        pu = urlparse(ddg)
        q = parse_qs(pu.query)
        uddg = (q.get("uddg") or [""])[0]
        if not uddg:
            return ddg
        # uddg is already decoded by parse_qs
        return uddg
    except Exception:
        return ddg


def _ddg_parse_html(page: str, k: int) -> List[SearchResult]:
    # Newer DDG HTML variants can differ slightly; support a few patterns.
    # 1) Standard: <a class="result__a" href="...">Title</a>
    re_a1 = re.compile(r'(?is)<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>')
    a = re_a1.findall(page)

    # 2) Fallback: sometimes title anchor uses "result__url" wrapper
    if not a:
        re_a2 = re.compile(r'(?is)<a[^>]*href="([^"]+)"[^>]*class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>')
        a = re_a2.findall(page)

    # Snippets
    snippets: List[str] = []
    re_s1 = re.compile(r'(?is)<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>')
    snippets = [_strip_html(x) for x in re_s1.findall(page)[:k]]
    if not snippets:
        re_s2 = re.compile(r'(?is)<div[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</div>')
        snippets = [_strip_html(x) for x in re_s2.findall(page)[:k]]

    out: List[SearchResult] = []
    for i, (href, title_html) in enumerate(a[:k]):
        url = _normalize_result_url(href)
        title = _strip_html(title_html)
        snip = snippets[i] if i < len(snippets) else ""
        out.append(SearchResult(title=title, url=url, snippet=snip))
    return out


def _ddg_parse_lite(page: str, k: int) -> List[SearchResult]:
    """Parse DuckDuckGo lite HTML.

    Lite pages are much simpler but vary a bit; we support a couple common patterns.
    """
    # Typical: <a rel="nofollow" class="result-link" href="...">Title</a>
    re_a1 = re.compile(r'(?is)<a[^>]*class="[^"]*result-link[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>')
    a = re_a1.findall(page)
    if not a:
        # Fallback: any link inside a result row.
        re_a2 = re.compile(r'(?is)<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>')
        a = re_a2.findall(page)

    # Snippets in lite are often in <td class="result-snippet"> or plain <td>.
    re_s1 = re.compile(r'(?is)<td[^>]*class="[^"]*result-snippet[^"]*"[^>]*>(.*?)</td>')
    snippets = [_strip_html(x) for x in re_s1.findall(page)[:k]]
    if not snippets:
        re_s2 = re.compile(r'(?is)<td[^>]*>(.*?)</td>')
        snippets = [_strip_html(x) for x in re_s2.findall(page)[:k]]

    out: List[SearchResult] = []
    for i, (href, title_html) in enumerate(a[:k]):
        url = _normalize_result_url(href)
        title = _strip_html(title_html)
        snip = snippets[i] if i < len(snippets) else ""
        if not url or not title:
            continue
        out.append(SearchResult(title=title, url=url, snippet=snip))
    return out


def search_ddg(query: str, k: int = 6, timeout_s: float = 12.0) -> List[SearchResult]:
    """DuckDuckGo HTML search (no API key). Tries multiple DDG HTML endpoints and parses several variants.

    Raises RuntimeError on block/captcha or when no results are extractable.
    """
    k = max(1, int(k or 6))
    # Two endpoints; the html.duckduckgo.com host often works when duckduckgo.com/html is blocked.
    # kl=de-de nudges DDG to German locale; improves results for German queries.
    q = quote_plus(query)
    endpoints = [
        ("html", f"https://duckduckgo.com/html/?q={q}&kl=de-de"),
        ("html2", f"https://html.duckduckgo.com/html/?q={q}&kl=de-de"),
        ("lite", f"https://lite.duckduckgo.com/lite/?q={q}&kl=de-de"),
    ]

    last_err: Optional[str] = None
    for kind, u in endpoints:
        try:
            resp = http_get(u, headers=DEFAULT_HEADERS, timeout=timeout_s)
            if resp.status == 0 or resp.status >= 400:
                last_err = f"ddg_http_{resp.status}" if resp.status else resp.text
                continue
            page = resp.text or ""
            low = page.lower()

            # Basic block/captcha detection
            if "captcha" in low or "unusual traffic" in low or "verify you are a human" in low:
                raise RuntimeError("ddg_block_or_captcha")

            out = _ddg_parse_lite(page, k) if kind=='lite' else _ddg_parse_html(page, k)
            if out:
                return out

            # If HTML endpoint returned an empty template, treat as failure so caller can react.
            last_err = "ddg_no_results"
        except Exception as e:
            last_err = str(e)
            continue

    raise RuntimeError(last_err or "ddg_search_failed")


def fetch(url: str, timeout_s: float = 12.0) -> FetchResult:
    url_n = _normalize_result_url(url)
    pu = urlparse(url_n)
    if not pu.scheme:
        raise ValueError("fetch: missing scheme")

    resp = http_get(url_n, headers=DEFAULT_HEADERS, timeout=timeout_s)
    if resp.status == 0 or resp.status >= 400:
        raise RuntimeError(f"fetch_http_{resp.status}: {resp.text[:120]}")
    ct = (resp.content_type or "").lower()
    raw = resp.text

    if "text/plain" in ct:
        text = re.sub(r"\s+", " ", raw.replace("\u00a0", " ")).strip()
        title = ""
    else:
        title = _extract_title(raw) if ("text/html" in ct or ct == "") else ""
        text = _strip_html(raw)

    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    snippet = text[:420]
    body = text[:3000]
    return FetchResult(
        title=title,
        url=url_n,
        text=text,
        snippet=snippet,
        body=body,
        hash=h,
        fetched_at=_now_iso(),
        domain=pu.hostname or "",
    )


@dataclass
class SpiderBudget:
    max_pages: int = 6
    max_bytes_total: int = 5_000_000
    per_domain_max: int = 3
    timeout_s: float = 12.0
    max_links_per_page: int = 12


def _extract_links(html_page: str, base: str, max_links: int) -> List[str]:
    re_href = re.compile(r"(?is)href=[\"']([^\"'#]+)[\"']")
    links = []
    for href in re_href.findall(html_page)[:max_links]:
        href = (href or "").strip()
        if not href:
            continue
        links.append(urljoin(base, href))
    return links


def spider(seeds: List[str], bud: Optional[SpiderBudget] = None) -> List[FetchResult]:
    if not seeds:
        raise ValueError("no seeds")
    bud = bud or SpiderBudget()

    seen = set()
    dcount = {}
    q: List[str] = []
    for s in seeds:
        u = _normalize_result_url(s)
        if u and u not in seen:
            seen.add(u)
            q.append(u)

    used = 0
    out: List[FetchResult] = []

    while q and len(out) < bud.max_pages and used < bud.max_bytes_total:
        u = q.pop(0)
        pu = urlparse(u)
        dom = (pu.hostname or "").lower()
        if not dom:
            continue
        if dcount.get(dom, 0) >= bud.per_domain_max:
            continue

        try:
            resp = http_get(u, headers=DEFAULT_HEADERS, timeout=bud.timeout_s)
            if resp.status == 0 or resp.status >= 400:
                continue
            raw = resp.text
            used += len(raw.encode("utf-8", errors="ignore"))
            if used >= bud.max_bytes_total:
                break

            ct = (resp.content_type or "").lower()
            if "text/html" in ct or ct == "":
                for lk in _extract_links(raw, u, bud.max_links_per_page):
                    lk_n = _normalize_result_url(lk)
                    if lk_n and lk_n not in seen:
                        seen.add(lk_n)
                        q.append(lk_n)

            txt = _strip_html(raw)
            h = hashlib.sha256(txt.encode("utf-8", errors="ignore")).hexdigest()
            out.append(
                FetchResult(
                    title=_extract_title(raw),
                    url=u,
                    text=txt,
                    snippet=txt[:420],
                    body=txt[:3000],
                    hash=h,
                    fetched_at=_now_iso(),
                    domain=dom,
                )
            )
            dcount[dom] = dcount.get(dom, 0) + 1
        except Exception:
            # keep crawling other URLs
            continue

    return out
