from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class OllamaConfig:
    host: str = "http://localhost:11434"
    model: str = "llama3.3"
    temperature: float = 0.2
    num_ctx: int = 2048
    stream: bool = False


def _ollama_chat(cfg: OllamaConfig, system: str, user: str) -> str:
    url = cfg.host.rstrip("/") + "/api/chat"
    payload = {
        "model": cfg.model,
        "stream": cfg.stream,
        "options": {"temperature": cfg.temperature, "num_ctx": cfg.num_ctx},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    msg = data.get("message") or {}
    return (msg.get("content") or "").strip()


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return None


def decide(
    cfg: OllamaConfig,
    axioms: Dict[str, str],
    state_summary: str,
    input_text: str,
    scope: str = "user",
) -> Dict[str, Any]:
    """Small decision model that converts state + input into pressures/actions.

    Returns dict:
      {
        "drives": {axis_name: delta_float, ...},
        "actions": {"websense":0..1, "daydream":0..1, "reply":0..1},
        "web_query": "..." | "",
        "notes": "short rationale"
      }

    This keeps triggering logic 'AI-like' (no keyword heuristics).
    """

    system = (
        "You are a tiny decision model inside a digital organism. "
        "Your job: map INTERNAL_STATE + INPUT into numeric drives and action scores. "
        "Return ONLY valid JSON. No prose. If actions.websense > 0, set web_query to a concrete search query string (natural language ok), never placeholders like "search for user query" or "direct_response_to_user_question". If no web search is needed, set web_query to empty string."
    )

    # Strict JSON schema to keep parsing robust.
    user = {
        "scope": scope,
        "axioms": axioms,
        "internal_state": state_summary,
        "input": input_text,
        "output_schema": {
            "drives": {
                "energy": "-1..1",
                "stress": "-1..1",
                "curiosity": "-1..1",
                "confidence": "-1..1",
                "uncertainty": "-1..1",
                "social_need": "-1..1",
                "urge_reply": "-1..1",
                "urge_share": "-1..1",
                "pressure_websense": "-1..1",
                "pressure_daydream": "-1..1",
                "purpose_a1": "-1..1",
                "purpose_a2": "-1..1",
                "purpose_a3": "-1..1",
                "purpose_a4": "-1..1",
                "tension_a1": "-1..1",
                "tension_a2": "-1..1",
                "tension_a3": "-1..1",
                "tension_a4": "-1..1",
            },
            "actions": {"websense": "0..1", "daydream": "0..1", "reply": "0..1"},
            "web_query": "string (empty if not needed)",
            "notes": "short string",
        },
        "rules": [
            "Do not use keyword heuristics. Base decisions on uncertainty/confidence, curiosity, and teleology tensions.",
            "If uncertainty is high and evidence is needed, increase pressure_websense and actions.websense.",
            "If idle scope and curiosity/uncertainty suggests exploration, increase pressure_daydream and actions.daydream.",
            "If the user asked something directly, actions.reply should be high.",
            "web_query should be a concise search query when actions.websense is high; otherwise empty.",
            "Always include pressure_websense and pressure_daydream in drives (use 0 if no change).",
            "All numbers must be valid JSON numbers.",
        ],
    }

    raw = _ollama_chat(cfg, system, json.dumps(user, ensure_ascii=False))
    parsed = _extract_json(raw) or {}

    drives = parsed.get("drives") if isinstance(parsed.get("drives"), dict) else {}
    actions = parsed.get("actions") if isinstance(parsed.get("actions"), dict) else {}
    out = {
        "drives": drives,
        "actions": {
            "websense": float(actions.get("websense", 0.0) or 0.0),
            "daydream": float(actions.get("daydream", 0.0) or 0.0),
            "reply": float(actions.get("reply", 0.0) or 0.0),
        },
        "web_query": str(parsed.get("web_query") or ""),
        "notes": str(parsed.get("notes") or ""),
        "_raw": raw[:2000],
    }
    return out
