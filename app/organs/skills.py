from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.net import http_post_json


@dataclass
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    model: str = "llama3.3"
    temperature: float = 0.2
    num_ctx: int = 3072
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


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        i = (text or "").find("{")
        if i < 0:
            return {}
        obj, _ = json.JSONDecoder().raw_decode(text[i:])
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def build_skill(
    cfg: OllamaConfig,
    axioms: Dict[str, str],
    cluster: Dict[str, Any],
    examples: List[Dict[str, Any]],
    existing_skill: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create or refine a compact, testable strategy for a failure cluster.

    Output:
      - strategy_digest: 5-12 bullet lines, stable wording, no fluff
      - tests: list of natural-language acceptance tests
      - confidence: 0..1
    """
    system = (
        "You are Bunny's Skill organ. "
        "Your job is to compress repeated experiences into a reusable strategy. "
        "Return STRICT JSON only."
    )
    prompt = {
        "axioms": axioms,
        "cluster": cluster,
        "examples": examples[-12:],
        "existing_skill": existing_skill or {},
        "output_schema": {
            "strategy_digest": "string <= 900 chars, concise, actionable (lines or bullets)",
            "tests": [{"name":"str","prompt":"str","pass_criteria":"str"}],
            "confidence": "0..1",
        },
        "rules": [
            "No domain-specific hacks; strategy must generalize.",
            "Must be compatible with Events->Matrix->Vector architecture.",
            "Do not claim capabilities you don't have. If a capability is missing, suggest a proposal path via DevLab instead of pretending.",
        ],
    }
    txt = _ollama_chat(cfg, system, json.dumps(prompt, ensure_ascii=False))
    obj = _extract_json(txt)
    if "tests" not in obj or not isinstance(obj["tests"], list):
        obj["tests"] = []
    try:
        obj["confidence"] = float(obj.get("confidence") or 0.4)
    except Exception:
        obj["confidence"] = 0.4
    obj["strategy_digest"] = str(obj.get("strategy_digest") or "").strip()[:900]
    return obj
