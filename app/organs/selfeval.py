from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.net import http_post_json


@dataclass
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    # NOTE: default should be an installed Ollama model name (no '-instruct' suffix assumptions).
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
    meta_json: str = "",
) -> Dict[str, Any]:
    """Self-evaluate the last answer outcome.

    This is internal learning: it should work even if the user does not explicitly rate.

    Returns JSON with:
      delta_reward: -1..+1
      drives_delta: {axis: float}  (few axes only)
      domains_reward/domains_penalty: [domain]
      beliefs: [{subject,predicate,object,confidence,provenance}]
      axiom_scores: {A1..A4:0..1}
      eval_scores: dict
      notes: short

    meta_json is optional contextual telemetry (JSON string), e.g. whether WebSense was available/used.
    """

    ax = "\n".join([f"{k}: {v}" for k, v in (axioms or {}).items()])

    # Only propose deltas on axes that actually exist in BunnyCore (avoid 'usefulness' etc.).
    allowed_axes = (
        "uncertainty, confidence, curiosity, freshness_need, social_need, urge_reply, "
        "pressure_websense, pressure_daydream, pressure_evolve, capability_gap, desire_upgrade, "
        "purpose_a1..purpose_a4, tension_a1..tension_a4, valence, arousal, stress"
    )

    system = (
        "You are an internal self-evaluator inside a digital organism. "
        "Evaluate whether the OUTPUT is good given the INPUT context and AXIOMS. "
        "The INPUT may be a normal user QUESTION, or an internal/idle evaluation. "
        "Use WEBSENSE_CLAIMS_JSON if present. Return ONLY valid JSON. No prose.\n"
        "Rules:\n"
        "- delta_reward in [-1,+1]. Negative if answer is unhelpful/incorrect/too vague/avoids the task; positive if correct and useful.\n"
        "- drives_delta is a small adjustment suggestion (few axes only) using ONLY these axes: " + allowed_axes + ".\n"
        "- META_JSON may include flags like idle=true or mode='silence_eval'. In these cases, you are evaluating INTERNAL behavior, not a user-facing fact answer. Do NOT penalize for not using WebSense unless the internal task explicitly required it.\n"
        "- If META_JSON.mode == 'silence_eval', interpret lack of user response as WEAK / AMBIGUOUS feedback. Keep |delta_reward| modest (prefer |delta_reward| <= 0.3) unless the prior answer is clearly wrong or harmful.\n"
        "- META_JSON may tell you whether WebSense exists/was used. If this is a normal external fact question AND WebSense is available but the answer refuses to look things up, that is BAD: delta_reward should be strongly negative (<= -0.6) and drives_delta must raise pressure_websense, uncertainty, and capability_gap.\n"
        "- If evidence was insufficient, increase pressure_websense and uncertainty.\n"
        "- If repeated failure is implied (user dissatisfaction, inability to perform a concrete lookup), slightly increase capability_gap and pressure_evolve/desire_upgrade.\n"
        "- If answer is well-supported, reduce uncertainty/pressure_websense and increase confidence.\n"
        "- If you can extract stable facts/definitions from ANSWER supported by claims, emit 0..3 beliefs with provenance='self_eval'.\n"
        "- If you can identify good/bad domains from claims, output domains_reward/domains_penalty as domains only.\n"
    )

    user = (
        f"AXIOMS:\n{ax}\n\n"
        f"INTERNAL_STATE_SUMMARY:\n{state_summary}\n\n"
        f"META_JSON:\n{meta_json}\n\n"
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

    # Filter drives_delta to known axis names (defensive: models sometimes emit extras).
    allowed = {
        "uncertainty",
        "confidence",
        "curiosity",
        "freshness_need",
        "social_need",
        "urge_reply",
        "pressure_websense",
        "pressure_daydream",
        "pressure_evolve",
        "capability_gap",
        "desire_upgrade",
        "valence",
        "arousal",
        "stress",
        "purpose_a1",
        "purpose_a2",
        "purpose_a3",
        "purpose_a4",
        "tension_a1",
        "tension_a2",
        "tension_a3",
        "tension_a4",
    }
    dd: Dict[str, float] = {}
    for k, v in (out.get("drives_delta") or {}).items():
        kk = str(k)
        if kk not in allowed:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        dd[kk] = max(-1.0, min(1.0, fv))
    out["drives_delta"] = dd

    # Fallback: if the model omitted drives_delta but produced axiom_scores, derive a small
    # drives_delta to keep the learning loop (matrix plasticity) alive.
    try:
        if (not out.get("drives_delta")) and isinstance(out.get("axiom_scores"), dict):
            axm = out.get("axiom_scores") or {}
            dd2: Dict[str, float] = {}
            for k, v in axm.items():
                kk = str(k).strip().lower()
                if not kk.startswith("a"):
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
                dd2[f"purpose_a{n}"] = d
                dd2[f"tension_a{n}"] = -d
            out["drives_delta"] = dd2
    except Exception:
        pass

    return out
