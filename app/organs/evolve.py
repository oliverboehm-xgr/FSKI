from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.net import http_post_json


@dataclass
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    model: str = "llama3.3"
    temperature: float = 0.4
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
    if status == 0:
        raise RuntimeError(txt)
    if status >= 400:
        raise RuntimeError(f"ollama /api/chat HTTP {status}: {txt[:200]}")
    data = json.loads(txt or "{}")
    msg = data.get("message") or {}
    return (msg.get("content") or "").strip()


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        if not text:
            return None
        i = text.find("{")
        if i < 0:
            return None
        obj, _end = json.JSONDecoder().raw_decode(text[i:])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def propose_mutations(
    cfg: OllamaConfig,
    axioms: Dict[str, str],
    state_summary: str,
    recent_messages: List[Dict[str, Any]],
    beliefs: List[Dict[str, Any]],
    self_model: Dict[str, Any],
    trigger: str = "idle",
    existing_proposals: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Self-development organ.

    Produces wishes and mutation proposals that a human can approve.
    No auto-apply. No code generation here; only structured proposals.
    """

    system = (
        "You are the EVOLVE organ inside a digital organism. "
        "You propose self-development steps (new organs, matrix coupling changes, new axes) "
        "based on AXIOMS + SELF_MODEL + INTERNAL_STATE + RECENT_MESSAGES + BELIEFS. "
        "Return ONLY valid JSON, no prose. "
        "You MUST stay generic: do not hardcode user-specific hacks. "
        "You MUST propose changes as Events->Matrix->Vector compatible components."
    )

    user = {
        "trigger": trigger,
        "axioms": axioms,
        "internal_state": state_summary,
        "recent_messages": recent_messages[-20:],
        "beliefs": beliefs[-24:],
        "self_model": self_model,
        "existing_proposals": existing_proposals or [],
        "output_schema": {
            "wishes": [
                {
                    "wish": "short string",
                    "why": "short rationale",
                    "linked_axioms": ["A1|A2|A3|A4"],
                    "priority": "0..1",
                }
            ],
            "mutation_proposals": [
                {
                    "type": "new_organ | new_axis | adapter_binding | matrix_tuning | trust_policy",
                    "name": "identifier",
                    "description": "what and why",
                    "events": ["event_types involved"],
                    "state_axes": ["axes to add or strengthen"],
                    "matrix_targets": ["matrix names affected"],
                    "expected_effect": {
                        "uncertainty": "-1..1",
                        "capability_gap": "-1..1",
                        "desire_upgrade": "-1..1",
                        "pressure_evolve": "-1..1"
                    },
                    "axiom_alignment": {"A1": "0..1", "A2": "0..1", "A3": "0..1", "A4": "0..1"},
                    "risk": "0..1",
                    "notes": "short"
                }
            ],
            "drives": {
                "pressure_evolve": "-1..1",
                "capability_gap": "-1..1",
                "desire_upgrade": "-1..1",
                "curiosity": "-1..1"
            },
        },
        "rules": [
            "Do NOT generate code. Only propose structured mutations.",
            "Each proposal must map to events/matrices/adapters/axes; keep it composable.",
            "If no useful mutation is needed, return empty lists and drives close to 0.",
            "Be conservative: propose at most 3 mutations per call.",
            "If trigger is 'refine' and existing_proposals is non-empty: do NOT create many new proposals. Improve/clarify the existing proposal(s) instead (make them more actionable, reduce risk, align better with axioms).",
        ],
    }

    raw = _ollama_chat(cfg, system, json.dumps(user, ensure_ascii=False))
    parsed = _extract_json(raw) or {}
    if "_raw" not in parsed:
        parsed["_raw"] = raw[:2000]
    # normalize
    if not isinstance(parsed.get("wishes"), list):
        parsed["wishes"] = []
    if not isinstance(parsed.get("mutation_proposals"), list):
        parsed["mutation_proposals"] = []
    if not isinstance(parsed.get("drives"), dict):
        parsed["drives"] = {}
    return parsed
