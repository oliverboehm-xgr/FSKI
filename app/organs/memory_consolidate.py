from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from app.net import http_post_json


@dataclass
class OllamaConfig:
    # Prefer IPv4 loopback to avoid Windows/MSYS IPv6 localhost quirks.
    host: str = "http://127.0.0.1:11434"
    model: str = "llama3.3"
    temperature: float = 0.15
    num_ctx: int = 2048
    stream: bool = False


_SYSTEM = """You are the Memory Consolidation organ in a digital organism.
You decide what becomes durable long-term memory.

You MUST return STRICT JSON only. No prose.

Constraints:
- Choose only facts that are useful to satisfy the organism's axioms.
- Prefer crisp, verifiable statements. No motivational fluff.
- Do not duplicate existing long-term memories.
- Keep each summary <= 220 characters.
- Return at most `limit` memories.

JSON schema:
{
  "memory_long_writes": [
    {
      "summary": "string<=220",
      "topic": "string",
      "salience": 0.0,
      "axioms": ["A1","A2","A3","A4"],
      "sources": ["url"],
      "confidence": 0.0
    }
  ],
  "beliefs": [
    {
      "subject": "string",
      "predicate": "string",
      "object": "string",
      "confidence": 0.0,
      "provenance": "memory_consolidate"
    }
  ],
  "notes": "short"
}
"""


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def consolidate_memories(
    cfg: OllamaConfig,
    axioms: Dict[str, str],
    state_summary: str,
    active_topic: str,
    trigger: str,
    candidates: List[Dict[str, Any]],
    existing_memory_long: List[str],
    limit: int = 2,
) -> Dict[str, Any]:
    """Select a small set of candidate facts into durable long-term memory.

    This organ is deliberately conservative: it should write few, high-signal memories.
    """

    url = cfg.host.rstrip("/") + "/api/chat"
    limit = max(0, min(6, int(limit or 0)))

    # Keep payload small and structured.
    cand_small: List[Dict[str, Any]] = []
    for c in (candidates or [])[:12]:
        if not isinstance(c, dict):
            continue
        txt = str(c.get("text") or "").strip()
        if not txt:
            continue
        try:
            conf = float(c.get("confidence", 0.6) or 0.6)
        except Exception:
            conf = 0.6
        srcs = c.get("sources") if isinstance(c.get("sources"), list) else []
        srcs2 = [str(u or "").strip() for u in srcs if str(u or "").strip()][:3]
        cand_small.append({"text": txt[:520], "confidence": max(0.0, min(1.0, conf)), "sources": srcs2})

    existing_small = [str(s or "").strip()[:240] for s in (existing_memory_long or []) if str(s or "").strip()][:18]

    user = {
        "trigger": trigger,
        "active_topic": active_topic or "",
        "axioms": axioms,
        "state_summary": state_summary[:1200],
        "limit": limit,
        "existing_memory_long": existing_small,
        "candidates": cand_small,
        "rules": [
            "Write 0 memories if candidates are low-signal or duplicates.",
            "Summaries must be factual, compact, and not generic advice.",
            "If a memory is derived from a candidate with sources, copy up to 2 URLs into sources.",
            "Use salience to indicate how important this memory is for future decisions (0..1).",
            "Only include axioms that the memory directly supports.",
        ],
    }

    payload = {
        "model": cfg.model,
        "stream": cfg.stream,
        "options": {"temperature": float(cfg.temperature), "num_ctx": int(cfg.num_ctx)},
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    }

    status, txt = http_post_json(url, payload, timeout=90)
    if status == 0:
        raise RuntimeError(txt)
    if status >= 400:
        raise RuntimeError(f"ollama /api/chat HTTP {status}: {txt[:200]}")

    data = json.loads(txt or "{}")
    content = ((data.get("message") or {}).get("content") or "").strip()
    out = _extract_json(content) or {}

    # Normalize outputs.
    mems = out.get("memory_long_writes") if isinstance(out.get("memory_long_writes"), list) else []
    beliefs = out.get("beliefs") if isinstance(out.get("beliefs"), list) else []
    norm_mems: List[Dict[str, Any]] = []
    seen = set(s.lower() for s in existing_small)

    for m in mems[:limit]:
        if not isinstance(m, dict):
            continue
        summ = str(m.get("summary") or "").strip()
        if not summ:
            continue
        summ = summ.replace("\n", " ").strip()[:220]
        key = summ.lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            sal = float(m.get("salience", 0.6) or 0.6)
        except Exception:
            sal = 0.6
        ax = m.get("axioms") if isinstance(m.get("axioms"), list) else []
        ax2 = [str(a) for a in ax if str(a) in ("A1", "A2", "A3", "A4")]
        srcs = m.get("sources") if isinstance(m.get("sources"), list) else []
        srcs2 = [str(u or "").strip() for u in srcs if str(u or "").strip()][:2]
        topic = str(m.get("topic") or active_topic or "").strip()[:80]
        norm_mems.append(
            {
                "summary": summ,
                "topic": topic,
                "salience": max(0.0, min(1.0, sal)),
                "axioms": ax2,
                "sources": srcs2,
            }
        )

    norm_beliefs: List[Dict[str, Any]] = []
    for b in beliefs[:10]:
        if not isinstance(b, dict):
            continue
        subj = str(b.get("subject") or "").strip()[:120]
        pred = str(b.get("predicate") or "").strip()[:80]
        obj = str(b.get("object") or "").strip()[:200]
        if not subj or not pred or not obj:
            continue
        try:
            conf = float(b.get("confidence", 0.6) or 0.6)
        except Exception:
            conf = 0.6
        norm_beliefs.append(
            {
                "subject": subj,
                "predicate": pred,
                "object": obj,
                "confidence": max(0.0, min(1.0, conf)),
                "provenance": str(b.get("provenance") or "memory_consolidate"),
            }
        )

    return {
        "memory_long_writes": norm_mems,
        "beliefs": norm_beliefs,
        "notes": str(out.get("notes") or "")[:240],
    }
