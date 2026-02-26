from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.net import http_post_json

@dataclass
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    model: str = "llama3.1:8b-instruct"
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
    r = http_post_json(cfg.host.rstrip("/") + "/api/chat", payload, timeout_s=60.0)
    # Ollama returns either {"message":{"content":...}} (non-stream) or concatenated.
    if isinstance(r, dict):
        msg = r.get("message") or {}
        if isinstance(msg, dict) and "content" in msg:
            return str(msg.get("content") or "")
        if "response" in r:
            return str(r.get("response") or "")
    return str(r)

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    # robust: find first '{' and last '}' and parse
    if not text:
        return None
    s = text.strip()
    i = s.find("{")
    j = s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(s[i:j+1])
    except Exception:
        return None

def interpret_feedback(
    cfg: OllamaConfig,
    axioms: Dict[str, str],
    state_summary: str,
    last_assistant: str,
    user_text: str,
) -> Dict[str, Any]:
    """Interpret whether user_text is feedback/correction about last_assistant.

    Returns JSON dict with:
      is_feedback: 0..1
      feedback_type: "correction"|"praise"|"preference"|"other"
      delta_reward: -1..+1
      desired_state_delta: {axis: float}  # target movement in state space
      desired_drive_delta: {axis: float}  # how decision drives should change next time
      domains_reward: ["example.com", ...]
      domains_penalty: ["example.com", ...]
      axiom_scores: {A1..:0..1}
      beliefs: [{"subject":str,"predicate":str,"object":str,"confidence":0..1,"provenance":str}, ...]
      notes: str
    """

    # Keep axioms compact
    ax = "\n".join([f"{k}: {v}" for k, v in (axioms or {}).items()])

    system = (
        "You are a feedback interpreter inside a digital organism. "
        "Your job is to decide if the USER message is feedback/correction about the LAST_ASSISTANT response. "
        "If yes, extract a reward signal and the intended correction in a GENERIC way. "
        "Return ONLY valid JSON. No prose.\n"
        "Rules:\n"
        "- The user may express feedback indirectly (e.g. 'I see it differently', 'I'd expect...', 'Actually...', "
        "  'That interpretation doesn't fit', 'Not quite', 'Could be seen as...'). Treat such disagreement as feedback.\n"
        "- The user may add missing details without saying 'wrong'. If it corrects/improves the last response, treat as feedback.\n"
        "- is_feedback is a probability 0..1.\n"
        "- delta_reward in [-1,+1]. Negative means user says it was wrong/bad, positive means good/correct.\n"
        "- desired_state_delta and desired_drive_delta are dictionaries of axis->delta in [-1,+1]. "
        "Use axes like uncertainty, confidence, stress, curiosity, pressure_websense, usefulness, etc if present. "
        "Do NOT invent many axes; prefer a few strong ones.\n"
        "- If the feedback implies some sources were bad/good, include domains_penalty/domains_reward as lists of domains (no URLs).\n"
        "- If the user provides the correct facts/definition, include it in notes but keep notes short.\n"
        "- If the user states a corrected definition, fact, preference, or expectation that should be remembered, extract 1..3 BELIEFS as subject/predicate/object triples.\n"
        "  Keep beliefs generic (e.g. subject='Begriff X', predicate='means', object='...'). Add provenance='user_feedback'.\n"
        "- Include axiom_scores (0..1 per axiom key you see), based on how the LAST_ASSISTANT response violated or satisfied goals.\n"
        "Output contract:\n"
        "- If NOT feedback, set is_feedback<=0.3, delta_reward near 0, and leave deltas empty.\n"
        "- If feedback, set is_feedback>=0.7 when clearly tied to LAST_ASSISTANT.\n"
    )
    user = (
        f"AXIOMS:\n{ax}\n\n"
        f"INTERNAL_STATE_SUMMARY:\n{state_summary}\n\n"
        f"LAST_ASSISTANT:\n{last_assistant}\n\n"
        f"USER:\n{user_text}\n\n"
        "Return JSON with keys: is_feedback, feedback_type, delta_reward, desired_state_delta, desired_drive_delta, domains_reward, domains_penalty, axiom_scores, beliefs, notes."
    )
    raw = _ollama_chat(cfg, system, user)
    out = _extract_json(raw) or {}
    # sanitize
    try:
        out["is_feedback"] = float(out.get("is_feedback", 0.0) or 0.0)
    except Exception:
        out["is_feedback"] = 0.0
    out["is_feedback"] = max(0.0, min(1.0, out["is_feedback"]))
    try:
        out["delta_reward"] = float(out.get("delta_reward", 0.0) or 0.0)
    except Exception:
        out["delta_reward"] = 0.0
    out["delta_reward"] = max(-1.0, min(1.0, out["delta_reward"]))
    if not isinstance(out.get("desired_state_delta"), dict):
        out["desired_state_delta"] = {}
    if not isinstance(out.get("desired_drive_delta"), dict):
        out["desired_drive_delta"] = {}
    if not isinstance(out.get("axiom_scores"), dict):
        out["axiom_scores"] = {}
    if not isinstance(out.get("domains_reward"), list):
        out["domains_reward"] = []
    if not isinstance(out.get("domains_penalty"), list):
        out["domains_penalty"] = []
    if not isinstance(out.get("beliefs"), list):
        out["beliefs"] = []
    if not isinstance(out.get("feedback_type"), str):
        out["feedback_type"] = "other"
    if not isinstance(out.get("notes"), str):
        out["notes"] = ""
    return out
