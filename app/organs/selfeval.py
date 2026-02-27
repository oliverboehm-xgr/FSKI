from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.net import http_post_json


@dataclass
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    model: str = "llama3.2:3b-instruct"
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


def evaluate_outcome(
    cfg: OllamaConfig,
    axioms: Dict[str, str],
    state_summary: str,
    question: str,
    answer: str,
    websense_claims_json: str = "",
) -> Dict[str, Any]:
    """Self-evaluate the last answer outcome.

    This is internal learning: it should work even if the user does not explicitly rate.

    Returns JSON with:
      delta_reward: -1..+1
      drives_delta: {axis: float}
      domains_reward/domains_penalty: [domain]
      beliefs: [{subject,predicate,object,confidence,provenance}]
      axiom_scores: {A1..A4:0..1}
      notes: short
    """

    ax = "\n".join([f"{k}: {v}" for k, v in (axioms or {}).items()])

    system = (
        "You are an internal self-evaluator inside a digital organism. "
        "Evaluate whether the ANSWER correctly addresses the QUESTION, using WEBSENSE_CLAIMS_JSON if present. "
        "Return ONLY valid JSON. No prose.\n"
        "Rules:\n"
        "- delta_reward in [-1,+1]. Negative if answer is unhelpful/incorrect/too vague; positive if correct and useful.\n"
        "- drives_delta is a small adjustment suggestion (few axes): uncertainty, confidence, usefulness, pressure_websense.\n"
        "- If evidence was insufficient, increase pressure_websense and uncertainty.\n"
        "- If answer is well-supported, reduce uncertainty and pressure_websense, increase confidence/usefulness.\n"
        "- If you can extract stable facts/definitions from ANSWER supported by claims, emit 0..3 beliefs with provenance='self_eval'.\n"
        "- If you can identify good/bad domains from claims, output domains_reward/domains_penalty as domains only.\n"
    )

    user = (
        f"AXIOMS:\n{ax}\n\n"
        f"INTERNAL_STATE_SUMMARY:\n{state_summary}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"WEBSENSE_CLAIMS_JSON:\n{websense_claims_json}\n\n"
        "Return JSON keys: delta_reward, eval_scores, drives_delta, domains_reward, domains_penalty, beliefs, axiom_scores, notes."
    )

    raw = _ollama_chat(cfg, system, user)
    out = _extract_json(raw) or {}

    # sanitize
    try:
        out["delta_reward"] = float(out.get("delta_reward", 0.0) or 0.0)
    except Exception:
        out["delta_reward"] = 0.0
    out["delta_reward"] = max(-1.0, min(1.0, out["delta_reward"]))

    if not isinstance(out.get("eval_scores"), dict):
        out["eval_scores"] = {}
    if not isinstance(out.get("drives_delta"), dict):
        out["drives_delta"] = {}
    if not isinstance(out.get("domains_reward"), list):
        out["domains_reward"] = []
    if not isinstance(out.get("domains_penalty"), list):
        out["domains_penalty"] = []
    if not isinstance(out.get("beliefs"), list):
        out["beliefs"] = []
    if not isinstance(out.get("axiom_scores"), dict):
        out["axiom_scores"] = {}
    if not isinstance(out.get("notes"), str):
        out["notes"] = ""

    # Fallback: if the model omitted drives_delta but produced axiom_scores, derive a small
    # drives_delta to keep the learning loop (matrix plasticity) alive.
    try:
        if (not out.get("drives_delta")) and isinstance(out.get("axiom_scores"), dict):
            ax = out.get("axiom_scores") or {}
            dd: Dict[str, float] = {}
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
                d = (sc - 0.5) * 0.20
                dd[f"purpose_a{n}"] = d
                dd[f"tension_a{n}"] = -d
            out["drives_delta"] = dd
    except Exception:
        pass

    return out
