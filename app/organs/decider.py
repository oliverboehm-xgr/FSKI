from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.net import http_post_json
@dataclass
class OllamaConfig:
    # Prefer IPv4 loopback to avoid Windows/MSYS IPv6 localhost quirks.
    host: str = "http://127.0.0.1:11434"
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
        # Robustly decode the first JSON object even if the model adds extra text.
        obj, _end = json.JSONDecoder().raw_decode(text[i:])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def decide(
    cfg: OllamaConfig,
    axioms: Dict[str, str],
    state_summary: str,
    input_text: str,
    beliefs: Any = None,
    scope: str = "user",
    workspace: Any = None,
    needs: Any = None,
    wishes: Any = None,
    self_report: Any = None,
    active_topic: str = '',
    policy_hint: Any = None,
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

    # NOTE: Keep this a single well-formed Python string (avoid unescaped quotes).
    system = (
        "You are a tiny decision model inside a digital organism. "
        "Your job: map INTERNAL_STATE + INPUT into numeric drives and action scores, especially epistemic signals (uncertainty, freshness_need). "
        "Return ONLY valid JSON. No prose. "
        "If actions.websense > 0, set web_query to a concrete search query string (natural language ok), "
        "never placeholders like 'search for user query' or 'direct_response_to_user_question'. "
        "If no web search is needed, set web_query to empty string. "
        "Prefer the same language as the INPUT when forming web_query (e.g. German input -> German query). "
        "If the input asks about your state/goals/needs/wishes, you should set reply=1 and websense=0 unless uncertainty/freshness is high. "
        "If the input asks about time-sensitive facts (TV program, weather, news, prices, current office holders like president/CEO/minister), raise freshness_need and actions.websense. If INTERNAL_STATE shows error_signal high, treat that as a recent correction and prefer verification (actions.websense) over guessing for factual questions. "
        "If the user asks for a concrete external fact and you cannot answer from BELIEFS with high confidence, prefer actions.websense over asking the user for more context. "
        "Do NOT treat channel+time questions (e.g. 'RTL 20:15') as ambiguous; they have a single correct lookup answer. "
        "Be conservative with self-upgrade: only raise capability_gap/desire_upgrade/actions.evolve when you can point to a concrete missing capability or recurring failure; otherwise keep them low (<=0.2)."
    )

    # Strict JSON schema to keep parsing robust.
    user = {
        "scope": scope,
        "axioms": axioms,
        "beliefs": beliefs or [],
        "internal_state": state_summary,
        "workspace": workspace or [],
        "needs": needs or {},
        "wishes": wishes or {},
        "self_report": self_report or {},
        "active_topic": active_topic or "",
        "policy_hint": policy_hint or {},
        "input": input_text,
        "output_schema": {
            "drives": {
                "confidence": "0..1",
                "uncertainty": "0..1",
                "freshness_need": "0..1",
                "curiosity": "0..1",
                "stress": "0..1",
                "social_need": "0..1",
                "urge_reply": "0..1",
                "urge_share": "0..1",
                "pressure_websense": "0..1",
                "pressure_daydream": "0..1",
                "pressure_evolve": "0..1",
                "capability_gap": "0..1",
                "desire_upgrade": "0..1",
                "purpose_a1": "0..1",
                "purpose_a2": "0..1",
                "purpose_a3": "0..1",
                "purpose_a4": "0..1",
                "tension_a1": "0..1",
                "tension_a2": "0..1",
                "tension_a3": "0..1",
                "tension_a4": "0..1"
            },
            "actions": {"websense": "0..1", "daydream": "0..1", "evolve": "0..1", "reply": "0..1"},
            "web_query": "string (empty if not needed)",
            "notes": "short string",
        },
	        "rules": [
	            "All drive values are bounded 0..1.",
	            "For epistemic drives (uncertainty, confidence, freshness_need, pressure_websense): do not output all zeros by default; reflect the situation.",
            "Do not use string/keyword heuristics. Base decisions on epistemic signals (uncertainty/confidence, freshness_need), curiosity, and teleology tensions.",
            "Use BELIEFS to reduce uncertainty when they directly answer the INPUT; in that case actions.websense should be low and confidence higher.",
            "If uncertainty is high and evidence is needed, increase pressure_websense and actions.websense.",
            "If freshness_need is high (time-sensitive facts like weather/news/prices), increase pressure_websense and actions.websense.",
	            "If INPUT is about your internal state, goals, needs, wishes, identity, or capabilities (self-queries), set actions.websense=0 and keep uncertainty low unless the user asked for external facts.",
	            "If INPUT explicitly asks to search the internet / look something up, treat that as freshness_need high and set actions.websense high (>=0.8) with a concrete web_query.",            "If INPUT asks for a concrete external fact and you are not confident, do NOT ask for clarification by default; set actions.websense high and form a query.",
            "If the user asks for a concrete fact, web_query should include the key entity + attribute (e.g. 'Wetter Berlin aktuell Temperatur').",
            "If scope is idle: prefer daydream unless uncertainty or freshness_need is meaningfully high AND you can form a concrete web_query.",
            "If capability_gap is high or desire_upgrade is high (often linked to A4), actions.evolve may be increased (>=0.6) to propose mutations.",
            "If the user asked something directly, actions.reply should be high.",
            "web_query should be a concise concrete search query when actions.websense is high; otherwise empty.",
            "Always include pressure_websense and pressure_daydream in drives (use 0 if no change).",
            "All numbers must be valid JSON numbers."
        ],
    }

    raw = _ollama_chat(cfg, system, json.dumps(user, ensure_ascii=False))
    parsed = _extract_json(raw) or {}


    drives = parsed.get("drives") if isinstance(parsed.get("drives"), dict) else {}
    actions = parsed.get("actions") if isinstance(parsed.get("actions"), dict) else {}

    def _cl01(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            return 0.0
        return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v

    # Enforce bounded drives (0..1).
    for k, v in list(drives.items()):
        drives[k] = _cl01(v)

    out = {
        "drives": drives,
        "actions": {
            "websense": _cl01(actions.get("websense", 0.0) or 0.0),
            "daydream": _cl01(actions.get("daydream", 0.0) or 0.0),
            "evolve": _cl01(actions.get("evolve", 0.0) or 0.0),
            "reply": _cl01(actions.get("reply", 0.0) or 0.0),
        },
        "web_query": str(parsed.get("web_query") or ""),
        "notes": str(parsed.get("notes") or ""),
        "_raw": raw[:2000],
    }
    return out