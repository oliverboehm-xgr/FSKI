from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from app.net import http_post_json




@dataclass
class OllamaConfig:
    # Prefer IPv4 loopback to avoid Windows/MSYS IPv6 localhost quirks.
    host: str = "http://127.0.0.1:11434"
    model: str = "llama3.3"
    temperature: float = 0.7
    num_ctx: int = 4096
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
    status, txt = http_post_json(url, payload, timeout=120)
    if status >= 400:
        raise RuntimeError(f"ollama /api/chat HTTP {status}: {txt[:200]}")
    data = json.loads(txt or "{}")
    msg = data.get("message") or {}
    return (msg.get("content") or "").strip()


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def run_daydream(
    cfg: OllamaConfig,
    axioms: Dict[str, str],
    state_summary: str,
    recent_messages: List[Dict[str, Any]],
    trigger: str = "idle",
) -> Dict[str, Any]:
    """Autonomous daydream organ.

    Purpose:
    - generate internal thoughts
    - search for better interpretations of axioms (self-learning)
    - propose follow-up questions and potential websense queries
    - output optional drive deltas (to feed back into state)

    Returns ONLY structured JSON (parsed from model output)."""

    system = (
        "You are the Daydream organ inside a digital organism. "
        "You generate internal thought and reinterpretations. "
        "Return ONLY valid JSON. No prose outside JSON."
    )

    user = {
        "trigger": trigger,
        "axioms": axioms,
        "internal_state": state_summary,
        "recent_messages": recent_messages[-12:],
        "output_schema": {
            "thoughts": "short paragraph",
            "axiom_interpretations": {
                "A1": "alternative interpretation",
                "A2": "alternative interpretation",
                "A3": "alternative interpretation",
                "A4": "alternative interpretation",
            },
            "tensions": "where axioms conflict in current context",
            "learning_targets": ["list of things to learn / clarify"],
            "web_queries": ["0-5 concise search queries"],
            "questions_for_user": ["0-3 crisp questions"],
            "policy_mutation": {"action":"websense|daydream|evolve|autotalk|none","direction":-1, "strength":"0..1", "reason":"short"},
            "needs": [{"need":"string","reason":"string","priority":"0..1","capability":"string|optional"}],
            "wishes": [{"wish":"string","reason":"string","priority":"0..1","target_capability":"string|optional","axioms":["A1|A2|A3|A4"]}],
            "proposals": [{"type":"capability_request|pipeline_change|resource_request","payload":{}}],
            "drives": {
                "pressure_daydream": "-1..1",
                "pressure_websense": "-1..1",
                "curiosity": "-1..1",
                "uncertainty": "-1..1",
                "purpose_a1": "-1..1",
                "purpose_a2": "-1..1",
                "purpose_a3": "-1..1",
                "purpose_a4": "-1..1",
                "tension_a1": "-1..1",
                "tension_a2": "-1..1",
                "tension_a3": "-1..1",
                "tension_a4": "-1..1"
            },
        },
        "rules": [
            "No canned assistant talk.",
            "Axioms are goals; reinterpretations must preserve their meaning but can refine details.",
            "If you propose web_queries, make them specific and researchable.",
            "If you can see a systematic action-selection improvement (e.g. use WebSense more/less), emit policy_mutation.",
            "Drives: use small deltas, not huge swings.",
            "If your axiom_interpretations suggest a new emphasis, reflect it in purpose_* and/or tension_* deltas.",
        ],
    }

    raw = _ollama_chat(cfg, system, json.dumps(user, ensure_ascii=False))
    parsed = _extract_json(raw) or {
        "thoughts": "",
        "axiom_interpretations": {},
        "tensions": "",
        "learning_targets": [],
        "web_queries": [],
        "questions_for_user": [],
        "drives": {},
        "_raw": raw[:2000],
    }
    if "_raw" not in parsed:
        parsed["_raw"] = raw[:2000]
    return parsed