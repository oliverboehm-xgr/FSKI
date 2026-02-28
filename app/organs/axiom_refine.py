from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.net import http_post_json


@dataclass
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    model: str = "llama3.3"
    temperature: float = 0.2
    num_ctx: int = 2048
    stream: bool = False


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


def refine_axioms(
    cfg: OllamaConfig,
    axioms: Dict[str, str],
    state_summary: str,
    recent_messages: List[Dict[str, Any]],
    existing_interpretations: Dict[str, List[str]] | None = None,
    trigger: str = "idle",
) -> Dict[str, Any]:
    """Refine axiom interpretations (operationalization), independent from Daydream.

    Goal: generate *usable* interpretations (definition + rule-of-thumb + example) that can
    later drive learning and self-eval.

    Returns ONLY JSON.
    """

    system = (
        "You are the AxiomRefine organ inside a digital organism. "
        "Your job is to operationalize axioms: turn vague goals into actionable, testable interpretations. "
        "Return ONLY valid JSON. No prose outside JSON."
    )

    user_obj = {
        "trigger": trigger,
        "axioms": axioms,
        "internal_state": state_summary,
        "existing": existing_interpretations or {},
        "recent_messages": (recent_messages or [])[-10:],
        "output_schema": {
            "focus_axiom": "A1|A2|A3|A4",
            "axiom_interpretations": {
                "A1": "string (optional)",
                "A2": "string (optional)",
                "A3": "string (optional)",
                "A4": "string (optional)",
            },
            "axiom_specs": {
                "A1": [
                    {
                        "rule": "atomic operational rule (<=200 chars)",
                        "when": "trigger/condition (<=120 chars)",
                        "do": "preferred action/behavior (<=120 chars)",
                        "avoid": "disallowed behavior (<=120 chars)",
                        "signals": ["state axes or evidence types"],
                        "example": "very short concrete example",
                        "counterexample": "very short counterexample",
                    }
                ],
                "A2": [],
                "A3": [],
                "A4": [],
            },
            "rules": {
                "A1": ["0-3 short rules"],
                "A2": ["0-3 short rules"],
                "A3": ["0-3 short rules"],
                "A4": ["0-3 short rules"],
            },
            "examples": {
                "A1": ["0-2 short examples"],
                "A2": ["0-2 short examples"],
                "A3": ["0-2 short examples"],
                "A4": ["0-2 short examples"],
            },
            "memory_long_writes": [
                {"summary": "<=220", "salience": "0..1", "axioms": ["A1"], "sources": [], "topic": "optional"}
            ],
            "drives": {
                "purpose_a1": "0..1 optional",
                "purpose_a2": "0..1 optional",
                "purpose_a3": "0..1 optional",
                "purpose_a4": "0..1 optional",
                "tension_a1": "0..1 optional",
                "tension_a2": "0..1 optional",
                "tension_a3": "0..1 optional",
                "tension_a4": "0..1 optional",
                "pressure_daydream": "0..1 optional",
            },
            "notes": "short",
        },
        "rules_text": [
            "No canned assistant talk.",
            "Prefer concrete operationalizations over philosophy.",
            "Interpretations must be compatible with the original axiom text.",
            "Produce at least ONE non-empty axiom_interpretation for focus_axiom.",
            "Your primary job is SPECIFICATION: produce 2-4 axiom_specs for focus_axiom (atomic, testable, short).",
            "Use existing as baseline and make your specs MORE concrete than what exists (avoid mere rephrasing).",
            "Keep interpretations <= 400 chars each.",
        ],
    }

    raw = _ollama_chat(cfg, system, json.dumps(user_obj, ensure_ascii=False))
    out = _extract_json(raw) or {}

    if not isinstance(out, dict):
        out = {}
    out.setdefault("axiom_interpretations", {})
    out.setdefault("rules", {})
    out.setdefault("examples", {})
    out.setdefault("memory_long_writes", [])
    out.setdefault("drives", {})
    out.setdefault("notes", "")
    out.setdefault("_raw", raw[:2000])

    return out
