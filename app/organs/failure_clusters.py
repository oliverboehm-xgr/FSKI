from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.net import http_post_json


@dataclass
class OllamaConfig:
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
        obj, _end = json.JSONDecoder().raw_decode(text[i:])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def assign_failure_cluster(
    cfg: OllamaConfig,
    axioms: Dict[str, str],
    user_text: str,
    last_assistant: str,
    selfeval: Dict[str, Any] | None = None,
    feedback: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assign a recurring failure/capability gap to a stable cluster key.

    This is intentionally LLM-based (no keyword heuristics). Output is small + stable.
    """
    system = (
        "You are Bunny's Failure-Cluster organ. "
        "Given a user message, the assistant reply, and optional self-eval/feedback, "
        "infer the most likely recurring failure mode or missing capability. "
        "Return STRICT JSON only."
    )

    prompt = {
        "axioms": axioms,
        "user_text": user_text,
        "assistant_text": last_assistant,
        "selfeval": selfeval or {},
        "feedback": feedback or {},
        "output_schema": {
            "cluster_key": "stable identifier, lowercase snake_case, <=40 chars",
            "label": "short human-readable label",
            "why": "1-2 sentences",
            "severity": "0..1",
            "missing_capability": "optional string; empty if none",
            "suggested_skill": "optional short strategy name; empty if none",
        },
        "rules": [
            "Do not invent tools you don't have.",
            "If this is not a failure, cluster_key should be 'none' with severity 0.",
            "Prefer stable reusable categories (e.g. 'topic_bleed', 'needs_websense', 'missing_modality_vision', 'clarify_loop').",
        ],
    }

    txt = _ollama_chat(cfg, system, json.dumps(prompt, ensure_ascii=False))
    obj = _extract_json(txt) or {}
    ck = str(obj.get("cluster_key") or "none").strip().lower()
    ck = re.sub(r"[^a-z0-9_]+", "_", ck)[:40].strip("_") or "none"
    obj["cluster_key"] = ck
    obj["label"] = str(obj.get("label") or ck).strip()[:120]
    try:
        obj["severity"] = float(obj.get("severity") or 0.0)
    except Exception:
        obj["severity"] = 0.0
    return obj
