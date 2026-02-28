from __future__ import annotations

"""Belief extraction organ.

Goal: extract durable, user-asserted facts/preferences from *any* user message.

Hard requirement for this project:
  - No keyword routing, no hard-coded user-intent heuristics.
  - The output must be conservative (prefer 0 beliefs over hallucinations).

Contract:
  extract_user_beliefs(...) -> {"beliefs": [...], "notes": str}

Beliefs are triples {subject,predicate,object} with confidence 0..1.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.net import http_post_json


@dataclass
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    model: str = "llama3.2:3b"
    temperature: float = 0.2
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


def extract_user_beliefs(
    cfg: OllamaConfig,
    axioms: Dict[str, str],
    state_summary: str,
    user_text: str,
    *,
    active_topic: str = "",
    workspace: Any = None,
    needs: Any = None,
    wishes: Any = None,
) -> Dict[str, Any]:
    """Extract stable beliefs from a user message.

    This is NOT a feedback interpreter. It runs on every user message.
    It must be conservative: prefer returning 0 beliefs over hallucinating.
    """

    ax = "\n".join([f"{k}: {v}" for k, v in (axioms or {}).items()])
    topic = (active_topic or "").strip()[:80]

    system = (
        "You are a BELIEF extraction organ inside a digital organism. "
        "Your job is to extract durable, user-asserted facts/preferences from the USER message. "
        "Return ONLY valid JSON. No prose.\n"
        "Rules:\n"
        "- Be conservative: only extract beliefs that the USER explicitly states or clearly implies.\n"
        "- NEVER invent facts, identities, or preferences. If uncertain, return an empty list.\n"
        "- Extract 0..3 beliefs. Each belief is {subject,predicate,object,confidence,provenance}.\n"
        "- confidence in [0,1]. Use <=0.6 if slightly uncertain; >=0.85 only if explicit.\n"
        "- provenance must be one of: 'user_utterance', 'user_preference', 'user_identity'.\n"
        "- Predicates should be short and generic (e.g. name, preference, goal, constraint, role).\n"
        "Output JSON keys: beliefs (list), notes (short string)."
    )

    user = (
        f"AXIOMS:\n{ax}\n\n"
        f"ACTIVE_TOPIC: {topic}\n\n"
        f"INTERNAL_STATE_SUMMARY:\n{state_summary}\n\n"
        f"WORKSPACE_JSON:\n{json.dumps(workspace or [], ensure_ascii=False)[:2000]}\n\n"
        f"NEEDS_JSON:\n{json.dumps(needs or {}, ensure_ascii=False)[:1200]}\n\n"
        f"WISHES_JSON:\n{json.dumps(wishes or {}, ensure_ascii=False)[:1200]}\n\n"
        f"USER:\n{user_text}\n\n"
        "Return JSON."
    )

    raw = _ollama_chat(cfg, system, user)
    out = _extract_json(raw) or {}

    if not isinstance(out.get("beliefs"), list):
        out["beliefs"] = []
    if not isinstance(out.get("notes"), str):
        out["notes"] = ""

    clean = []
    for b in out.get("beliefs") or []:
        if not isinstance(b, dict):
            continue
        subj = str(b.get("subject") or "").strip()
        pred = str(b.get("predicate") or "").strip()
        obj = str(b.get("object") or "").strip()
        if not (subj and pred and obj):
            continue
        try:
            conf = float(b.get("confidence", 0.75) or 0.75)
        except Exception:
            conf = 0.75
        conf = 0.0 if conf < 0.0 else 1.0 if conf > 1.0 else conf
        prov = str(b.get("provenance") or "user_utterance")[:40]
        if prov not in ("user_utterance", "user_preference", "user_identity"):
            prov = "user_utterance"
        clean.append(
            {
                "subject": subj[:200],
                "predicate": pred[:60],
                "object": obj[:600],
                "confidence": conf,
                "provenance": prov,
            }
        )
        if len(clean) >= 3:
            break

    out["beliefs"] = clean
    out["notes"] = (out.get("notes") or "")[:300]
    return out
