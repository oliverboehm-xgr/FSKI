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
    recent_evidence: List[Dict[str, Any]] | None = None,
    existing_interpretations: Dict[str, Any] | None = None,
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
        "You generate internal thought, consolidate experience, and propose self-development steps. "
        "Return ONLY valid JSON. No prose outside JSON."
    )

    user = {
        "trigger": trigger,
        "axioms": axioms,
        "internal_state": state_summary,
        "existing_axiom_interpretations": existing_interpretations or {},
        "recent_messages": recent_messages[-12:],
        "recent_evidence": (recent_evidence or [])[-3:],
        "output_schema": {
            "thoughts": "short paragraph",
            "axiom_interpretations": {
                "A1": "alternative interpretation",
                "A2": "alternative interpretation",
                "A3": "alternative interpretation",
                "A4": "alternative interpretation",
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
            "tensions": "where axioms conflict in current context",
            "learning_targets": ["list of things to learn / clarify"],
            "web_queries": ["0-5 concise search queries"],
            "questions_for_user": ["0-3 crisp questions"],
            "needs": [{"need":"string","reason":"string","priority":"0..1","capability":"string|optional"}],
            "wishes": [{"wish":"string","reason":"string","priority":"0..1","target_capability":"string|optional","axioms":["A1|A2|A3|A4"]}],
            "proposals": [{"type":"capability_request|pipeline_change|resource_request","payload":{}}],
            "memory_long_writes": [
                {"summary":"string<=220","salience":"0..1","axioms":["A1","A2","A3","A4"],"sources":["url"],"topic":"string(optional)"}
            ],
            "beliefs": [
                {"subject":"string","predicate":"string","object":"string","confidence":"0..1","provenance":"daydream"}
            ],
            "drives": {
                "pressure_daydream": "0..1",
                "pressure_websense": "0..1",
                "curiosity": "0..1",
                "uncertainty": "0..1",
                "purpose_a1": "0..1",
                "purpose_a2": "0..1",
                "purpose_a3": "0..1",
                "purpose_a4": "0..1",
                "tension_a1": "0..1",
                "tension_a2": "0..1",
                "tension_a3": "0..1",
                "tension_a4": "0..1"
            },
        },
        "rules": [
            "No canned assistant talk.",
            "Axioms are goals; reinterpretations must preserve their meaning but can refine details.",
            "Your primary job is SPECIFICATION: turn at least ONE axiom into 2-4 axiom_specs with triggers, do/avoid, and measurable signals.",
            "Use existing_axiom_interpretations as baseline and make your specs MORE concrete than what exists (avoid rephrasing).",
            "Keep specs short, atomic, and testable. Avoid philosophical fluff.",
            "If you propose web_queries, make them specific and researchable.",
            "If you consolidate knowledge from recent_evidence, prefer memory_long_writes over long free-form text.",
            "Drives are target levels in 0..1 (not deltas). Use small moves; avoid saturating everything to 0 or 1.",
            "drives MUST include at least: uncertainty, pressure_daydream, and one of purpose_a* or tension_a*.",
            "If your axiom_interpretations suggest a new emphasis, reflect it in purpose_* and/or tension_* deltas.",
            "memory_long_writes must be short, non-fluffy, and not duplicates of existing memories.",
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
        "memory_long_writes": [],
        "beliefs": [],
        "drives": {},
        "_raw": raw[:2000],
    }
    if "_raw" not in parsed:
        parsed["_raw"] = raw[:2000]
    return parsed