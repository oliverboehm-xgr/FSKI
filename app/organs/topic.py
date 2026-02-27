from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

from app.net import http_post_json


@dataclass
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    model: str = "llama3.2:3b-instruct"
    temperature: float = 0.1
    num_ctx: int = 1024
    stream: bool = False


SYSTEM = """You are a topic anchorer for a long-running conversation.
Return STRICT JSON only. No prose.

Goal:
- Infer the current active topic of the user message within the ongoing conversation.
- Prefer stable topics; change only when clearly needed.
- Output a concise topic label in German (2-6 words) and a confidence 0..1.

JSON schema:
{
  "active_topic": "string",
  "confidence": 0.0,
  "reason": "short"
}
"""


def detect_topic(cfg: OllamaConfig, user_text: str, prev_topic: str, context_hint: str) -> Dict[str, Any]:
    url = cfg.host.rstrip("/") + "/api/chat"
    user = {
        "user_text": user_text,
        "previous_topic": prev_topic or "",
        "context_hint": context_hint or "",
        "rules": [
            "If the user message is short and does not introduce new subject matter, keep previous_topic.",
            "If the user asks about a new domain, switch to that domain topic.",
            "Do not output greetings as topics.",
        ],
    }
    payload = {
        "model": cfg.model,
        "stream": cfg.stream,
        "options": {"temperature": cfg.temperature, "num_ctx": int(cfg.num_ctx)},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    }
    status, txt = http_post_json(url, payload, timeout=30)
    if status == 0:
        raise RuntimeError(txt)
    if status >= 400:
        raise RuntimeError(f"ollama /api/chat HTTP {status}: {txt[:200]}")
    data = json.loads(txt or "{}")
    content = ((data.get("message") or {}).get("content") or "")
    # strict json parse (best effort)
    try:
        return json.loads(content)
    except Exception:
        # salvage between braces
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except Exception:
                pass
    return {"active_topic": prev_topic or "Allgemein", "confidence": 0.2, "reason": "parse_failed"}
