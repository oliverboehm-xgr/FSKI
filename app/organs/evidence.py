from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.net import http_post_json


@dataclass
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    # Default to a non-"*-instruct" model name. Operator may override via env.
    model: str = "llama3.1:8b"
    temperature: float = 0.1
    num_ctx: int = 2048
    stream: bool = False


def _ollama_chat(cfg: OllamaConfig, system: str, user: str) -> str:
    payload = {
        "model": cfg.model,
        "stream": cfg.stream,
        "options": {"temperature": cfg.temperature, "num_ctx": cfg.num_ctx},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    status, txt = http_post_json(cfg.host.rstrip("/") + "/api/chat", payload, timeout_s=60.0)
    if status == 0:
        raise RuntimeError(txt)
    if status >= 400:
        raise RuntimeError(f"ollama /api/chat HTTP {status}: {txt[:200]}")
    data = json.loads(txt or "{}")
    msg = data.get("message") or {}
    if isinstance(msg, dict) and "content" in msg:
        return str(msg.get("content") or "")
    if "response" in data:
        return str(data.get("response") or "")
    return ""


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    i = s.find("{")
    j = s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(s[i : j + 1])
    except Exception:
        return None


def extract_evidence_claims(
    cfg: OllamaConfig,
    question: str,
    query: str,
    serp_lines: List[str],
    page_lines: List[str],
) -> Dict[str, Any]:
    """Convert raw WebSense evidence into structured claims.

    No keyword heuristics. LLM decides what is supported, what is missing, and produces a compact claims JSON.

    Returns JSON dict with keys:
      answer: short answer if possible
      claims: [{text, value(optional), unit(optional), time(optional), confidence, support:[urls]}]
      missing: [str]
      uncertainty: 0..1
      notes: str
    """

    now = time.strftime('%Y-%m-%d', time.gmtime())

    system = (
        "You are an evidence extractor. Given a USER QUESTION and web evidence (SERP snippets and page excerpts), "
        "produce ONLY valid JSON. No prose. "
        "Rules:\n"
        "- Only state claims that are supported by the evidence.\n"
        "- Prefer concrete values (numbers, units, timestamps) when present.\n"
        "- Today is NOW_DATE; if the question asks for current/up-to-date information, treat it as time-sensitive and prefer recent sources.\n"
        "- For time-sensitive questions: every claim should include a 'time' or 'as_of' field if available; otherwise increase uncertainty and add an item to 'missing'.\n"
        "- For time-sensitive questions: do not present a strong single-source claim as certain; if you only have one source, increase uncertainty and mention the need for confirmation.\n"
        "- If evidence is insufficient, list what is missing in 'missing'.\n"
        "- uncertainty is 0..1 (higher if evidence is weak/contradictory).\n"
        "- 'support' should include the URLs you used.\n"
        "- Keep the answer short and factual if possible.\n"
    )

    user = (
        f"NOW_DATE: {now}\n"
        f"QUESTION: {question}\n"
        f"SEARCH_QUERY: {query}\n\n"
        "SERP_SNIPPETS:\n" + "\n\n".join(serp_lines[:6]) + "\n\n"
        "PAGE_EXCERPTS:\n" + "\n\n".join(page_lines[:6]) + "\n\n"
        "Return JSON with keys: answer, claims, missing, uncertainty, notes."
    )

    raw = _ollama_chat(cfg, system, user)
    out = _extract_json(raw) or {}

    # Extract a small fallback URL set from the provided evidence lines.
    # (Used if the model forgets to attach 'support' URLs.)
    fallback_urls: List[str] = []
    try:
        for ln in (serp_lines or []) + (page_lines or []):
            s = str(ln or "")
            if "URL:" not in s:
                continue
            for part in s.split("URL:")[1:]:
                u = part.strip().splitlines()[0].strip()
                if u and u not in fallback_urls:
                    fallback_urls.append(u)
            if len(fallback_urls) >= 6:
                break
    except Exception:
        fallback_urls = []

    # sanitize
    if not isinstance(out.get("answer"), str):
        out["answer"] = ""
    if not isinstance(out.get("claims"), list):
        out["claims"] = []
    # Drop unsupported/hallucinated claims: each claim should cite at least one URL.
    # If the model forgot to include support URLs but we have evidence URLs available,
    # attach the top fallback URL so the downstream system can stay grounded.
    clean_claims: List[Dict[str, Any]] = []
    for c in (out.get("claims") or []):
        if not isinstance(c, dict):
            continue
        txt = str(c.get("text") or "").strip()
        sup = c.get("support") if isinstance(c.get("support"), list) else []
        sup2 = [str(u or "").strip() for u in sup if str(u or "").strip()][:3]
        if not txt:
            continue
        if not sup2 and fallback_urls:
            sup2 = [fallback_urls[0]]
        if not sup2:
            continue
        c2 = dict(c)
        c2["text"] = txt[:420]
        c2["support"] = sup2
        try:
            c2["confidence"] = float(c2.get("confidence", 0.6) or 0.6)
        except Exception:
            c2["confidence"] = 0.6
        c2["confidence"] = max(0.0, min(1.0, float(c2["confidence"])))
        clean_claims.append(c2)
    out["claims"] = clean_claims[:12]

    # If the model produced an answer but no claims, promote it into a single claim
    # with fallback URL support (keeps WebSense usable on small models).
    try:
        if (not out.get('claims')) and isinstance(out.get('answer'), str) and out.get('answer') and fallback_urls:
            out['claims'] = [{
                'text': str(out.get('answer') or '')[:420],
                'support': [fallback_urls[0]],
                'confidence': 0.55,
            }]
    except Exception:
        pass
    if not isinstance(out.get("missing"), list):
        out["missing"] = []
    try:
        out["uncertainty"] = float(out.get("uncertainty", 0.5) or 0.5)
    except Exception:
        out["uncertainty"] = 0.5
    out["uncertainty"] = max(0.0, min(1.0, out["uncertainty"]))
    if not isinstance(out.get("notes"), str):
        out["notes"] = ""

    return out


def refine_search_query(
    cfg: OllamaConfig,
    question: str,
    current_query: str,
    claims: Dict[str, Any],
) -> Dict[str, Any]:
    """Propose an improved search query based on what is missing.

    This is intentionally LLM-based (no keyword heuristics). The model sees the
    question, current query, and extracted claims/missing list and proposes a
    tighter query.

    Returns JSON dict with keys:
      query: str
      reason: str
      confidence: 0..1
    """

    system = (
        "You are a search query refiner. Given the USER QUESTION, the CURRENT_SEARCH_QUERY, and extracted claims JSON (including missing items), "
        "propose ONE improved web search query that is more likely to retrieve the missing evidence. "
        "Return ONLY valid JSON. No prose. "
        "Rules:\n"
        "- The refined query MUST stay tightly aligned to the original question (same entity + relation).\n"
        "- Prefer explicit role/attribute terms implied by the question (do not drift into generic words like 'situation' or 'news').\n"
        "- If the question is time-sensitive, include a recency cue (year or 'current' equivalent) and prefer primary/official sources.\n"
        "- Use the missing list to add the specific missing term(s) to the query.\n"
        "- Keep the query short; include location/time qualifiers if relevant; do not add quotation marks unless needed."
    )

    user = (
        f"QUESTION: {question}\n"
        f"CURRENT_SEARCH_QUERY: {current_query}\n\n"
        f"CLAIMS_JSON: {json.dumps(claims or {}, ensure_ascii=False)}\n\n"
        "Return JSON with keys: query, reason, confidence."
    )

    raw = _ollama_chat(cfg, system, user)
    out = _extract_json(raw) or {}
    q = out.get("query")
    if not isinstance(q, str):
        q = ""
    out["query"] = q.strip()[:180]
    if not isinstance(out.get("reason"), str):
        out["reason"] = ""
    try:
        out["confidence"] = float(out.get("confidence", 0.5) or 0.5)
    except Exception:
        out["confidence"] = 0.5
    out["confidence"] = max(0.0, min(1.0, out["confidence"]))
    return out


def seed_search_query(
    cfg: OllamaConfig,
    question: str,
    *,
    active_topic: str = "",
    locale_hint: str = "de",
) -> Dict[str, Any]:
    """Create a concrete initial web search query for a question.

    This exists to avoid heuristic fallbacks like "use the full user message".
    The model must produce ONE concise query or an empty string.

    Returns JSON dict with keys:
      query: str
      reason: str
      confidence: 0..1
    """

    system = (
        "You are a web search query builder. Given a USER QUESTION, produce ONE concrete search query. "
        "Return ONLY valid JSON. No prose. "
        "Rules:\n"
        "- The query MUST include the key entity/entities and the asked attribute/role.\n"
        "- Keep it short; include location/time qualifiers if relevant (especially for time-sensitive questions).\n"
        "- Prefer primary/official sources for time-sensitive facts.\n"
        "- Avoid generic filler words (e.g., 'situation', 'overview').\n"
        "- Do not echo the whole question; do not add quotation marks unless needed."
    )

    user = (
        f"QUESTION: {question}\n"
        f"ACTIVE_TOPIC: {active_topic}\n"
        f"LOCALE_HINT: {locale_hint}\n"
        "Return JSON with keys: query, reason, confidence."
    )

    raw = _ollama_chat(cfg, system, user)
    out = _extract_json(raw) or {}
    q = out.get("query")
    if not isinstance(q, str):
        q = ""
    out["query"] = q.strip()[:180]
    if not isinstance(out.get("reason"), str):
        out["reason"] = ""
    try:
        out["confidence"] = float(out.get("confidence", 0.5) or 0.5)
    except Exception:
        out["confidence"] = 0.5
    out["confidence"] = max(0.0, min(1.0, out["confidence"]))
    return out


def rank_serp_results(
    cfg: OllamaConfig,
    question: str,
    query: str,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Rank SERP results by relevance to the question.

    Input results are dicts: {title,url,snippet,trust(optional)}.

    Returns JSON dict:
      order: [int]   (indices into the input list, best first)
      confidence: 0..1
      notes: str
    """

    now = time.strftime('%Y-%m-%d', time.gmtime())
    system = (
        "You are a SERP ranker. Given a USER QUESTION and a list of search results (title/url/snippet), "
        "return a ranking by relevance and expected factual usefulness. "
        "Return ONLY valid JSON. No prose. "
        "Rules: Only use the given results; do not invent URLs. Prefer official/authoritative sources when relevance is similar. "
        "Today is NOW_DATE. If the question asks for current/up-to-date information, prefer results that appear recent (snippet/title contains a recent date or clear recency cues)."
    )

    user = (
        f"NOW_DATE: {now}\n"
        f"QUESTION: {question}\n"
        f"SEARCH_QUERY: {query}\n\n"
        f"RESULTS_JSON: {json.dumps(results or [], ensure_ascii=False)[:6000]}\n\n"
        "Return JSON with keys: order (list of indices), confidence, notes."
    )

    raw = _ollama_chat(cfg, system, user)
    out = _extract_json(raw) or {}
    order = out.get("order")
    if not isinstance(order, list):
        order = []
    clean: List[int] = []
    n = len(results or [])
    for x in order:
        try:
            i = int(x)
        except Exception:
            continue
        if 0 <= i < n and i not in clean:
            clean.append(i)
    out["order"] = clean
    try:
        out["confidence"] = float(out.get("confidence", 0.5) or 0.5)
    except Exception:
        out["confidence"] = 0.5
    out["confidence"] = max(0.0, min(1.0, out["confidence"]))
    if not isinstance(out.get("notes"), str):
        out["notes"] = ""
    out["notes"] = str(out.get("notes") or "")[:300]
    return out
