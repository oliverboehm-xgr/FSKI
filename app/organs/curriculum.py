
from __future__ import annotations

import json, time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from app.net import http_post_json


@dataclass
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    # Default to a non-"*-instruct" model name. Operator may override.
    model: str = "llama3.2:3b"
    temperature: float = 0.2
    num_ctx: int = 2048
    stream: bool = False


def _ollama_chat(cfg: OllamaConfig, system: str, user: str, timeout_s: float = 90.0) -> str:
    payload = {
        "model": cfg.model,
        "stream": cfg.stream,
        "options": {"temperature": cfg.temperature, "num_ctx": cfg.num_ctx},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    status, txt = http_post_json(cfg.host.rstrip("/") + "/api/chat", payload, timeout_s=timeout_s)
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


def _extract_json(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    i = s.find("{")
    j = s.rfind("}")
    if i < 0 or j <= i:
        return {}
    try:
        return json.loads(s[i:j+1])
    except Exception:
        return {}


def build_task_variants(
    cfg: OllamaConfig,
    seeds: List[Dict[str, str]],
    variants_per_seed: int = 2,
) -> List[Dict[str, Any]]:
    """Generate paraphrase/variant tasks for self-training.

    Each seed: {"question": str, "failure": str(optional)}.
    Returns list of tasks: {"question":..., "seed":..., "notes":...}
    """
    if not seeds:
        return []
    system = (
        "You are a curriculum generator for a self-learning digital organism. "
        "Given seed user questions and short failure notes, generate diverse but equivalent variants. "
        "Do not add domain-specific heuristics. Keep the intent identical. "
        "Return STRICT JSON only."
    )
    user = json.dumps({
        "seeds": seeds[:12],
        "variants_per_seed": int(max(0, variants_per_seed)),
        "output_schema": {
            "tasks": [
                {"question": "str", "seed_index": "int", "notes": "str"}
            ]
        },
        "rules": [
            "Variants must preserve meaning and be answerable without extra context unless the seed required context.",
            "Prefer realistic user phrasing, including short and demanding forms.",
            "No greetings, no meta."
        ]
    }, ensure_ascii=False)
    raw = _ollama_chat(cfg, system, user, timeout_s=60.0)
    obj = _extract_json(raw)
    tasks = obj.get("tasks") if isinstance(obj, dict) else None
    out: List[Dict[str, Any]] = []
    if isinstance(tasks, list):
        for t in tasks:
            if not isinstance(t, dict):
                continue
            q = str(t.get("question") or "").strip()
            if not q:
                continue
            out.append({"question": q, "seed_index": int(t.get("seed_index") or 0), "notes": str(t.get("notes") or "")[:240]})
    # include seeds as tasks too
    for i, s in enumerate(seeds[:12]):
        q = str(s.get("question") or "").strip()
        if q:
            out.append({"question": q, "seed_index": i, "notes": str(s.get("failure") or "")[:240]})
    return out
