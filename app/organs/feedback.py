from __future__ import annotations

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
      organ_hints: {"websense":0..1,"daydream":0..1,"evolve":0..1}
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
        "Use axes like uncertainty, confidence, stress, curiosity, pressure_websense, capability_gap, desire_upgrade, etc if present. "
        "Do NOT invent many axes; prefer a few strong ones.\n"
        "- If the feedback implies some sources were bad/good, include domains_penalty/domains_reward as lists of domains (no URLs).\n"
        "- If the user suggests or instructs a tool/organ strategy (web search / WebSense, daydream, evolve/upgrade), encode that as organ_hints with values 0..1. Infer intent from meaning (no keyword lists).\n"
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
        "Return JSON with keys: is_feedback, feedback_type, delta_reward, desired_state_delta, desired_drive_delta, organ_hints, domains_reward, domains_penalty, axiom_scores, beliefs, notes."
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
    if not isinstance(out.get("organ_hints"), dict):
        out["organ_hints"] = {}
    if not isinstance(out.get("beliefs"), list):
        out["beliefs"] = []
    if not isinstance(out.get("feedback_type"), str):
        out["feedback_type"] = "other"
    if not isinstance(out.get("notes"), str):
        out["notes"] = ""

    # Fallback: if the model omitted deltas but produced axiom_scores, derive a minimal
    # desired_state_delta from axiom_scores to keep plasticity functional.
    try:
        if (not out.get("desired_state_delta")) and isinstance(out.get("axiom_scores"), dict):
            ax = out.get("axiom_scores") or {}
            ds: Dict[str, float] = {}
            for k, v in ax.items():
                kk = str(k).strip().lower()
                if not kk.startswith('a'):
                    continue
                try:
                    n = int(kk[1:])
                except Exception:
                    continue
                try:
                    sc = float(v)
                except Exception:
                    continue
                sc = max(0.0, min(1.0, sc))
                d = (sc - 0.5) * 0.25  # small, bounded
                ds[f"purpose_a{n}"] = d
                ds[f"tension_a{n}"] = -d
            out["desired_state_delta"] = ds
    except Exception:
        pass
    # sanitize organ_hints (keep only known organ names and clamp 0..1)
    try:
        oh = out.get("organ_hints") if isinstance(out.get("organ_hints"), dict) else {}
        clean = {}
        for k, v in (oh or {}).items():
            kk = str(k).strip().lower()
            if kk not in ("websense","daydream","evolve","autotalk"):
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            fv = max(0.0, min(1.0, fv))
            if fv > 0.0:
                clean[kk] = fv
        out["organ_hints"] = clean
    except Exception:
        out["organ_hints"] = {}
    return out
