from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.net import http_post_json


@dataclass
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    model: str = "llama3.1:8b-instruct"
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
    r = http_post_json(cfg.host.rstrip("/") + "/api/chat", payload, timeout_s=60.0)
    if isinstance(r, dict):
        msg = r.get("message") or {}
        if isinstance(msg, dict) and "content" in msg:
            return str(msg.get("content") or "")
        if "response" in r:
            return str(r.get("response") or "")
    return str(r)


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

    system = (
        "You are an evidence extractor. Given a USER QUESTION and web evidence (SERP snippets and page excerpts), "
        "produce ONLY valid JSON. No prose. "
        "Rules:\n"
        "- Only state claims that are supported by the evidence.\n"
        "- Prefer concrete values (numbers, units, timestamps) when present.\n"
        "- If evidence is insufficient, list what is missing in 'missing'.\n"
        "- uncertainty is 0..1 (higher if evidence is weak/contradictory).\n"
        "- 'support' should include the URLs you used.\n"
        "- Keep the answer short and factual if possible.\n"
    )

    user = (
        f"QUESTION: {question}\n"
        f"SEARCH_QUERY: {query}\n\n"
        "SERP_SNIPPETS:\n" + "\n\n".join(serp_lines[:6]) + "\n\n"
        "PAGE_EXCERPTS:\n" + "\n\n".join(page_lines[:6]) + "\n\n"
        "Return JSON with keys: answer, claims, missing, uncertainty, notes."
    )

    raw = _ollama_chat(cfg, system, user)
    out = _extract_json(raw) or {}

    # sanitize
    if not isinstance(out.get("answer"), str):
        out["answer"] = ""
    if not isinstance(out.get("claims"), list):
        out["claims"] = []
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
        "You are a search query refiner. Given the USER QUESTION, the CURRENT_SEARCH_QUERY, and extracted claims JSON, "
        "propose ONE improved web search query that is more likely to retrieve missing evidence. "
        "Return ONLY valid JSON. No prose. "
        "Rules: keep the query short; include location/time qualifiers if relevant; do not add quotation marks unless needed."
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
