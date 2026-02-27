from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from app.net import http_post_json


@dataclass
class SleepConfig:
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "llama3.2:3b-instruct"
    ctx: int = 2048
    temperature: float = 0.2


def sleep_consolidate(
    cfg: SleepConfig,
    axioms: Dict[str, str],
    state_summary: str,
    workspace_items: List[Dict[str, Any]],
    beliefs: List[Dict[str, Any]],
    recent_messages: List[Dict[str, Any]],
    axiom_interpretations: Dict[str, List[Dict[str, Any]]] | None = None,
) -> Dict[str, Any]:
    """Consolidate experiences into durable structures (belief merges, TODOs, self-model notes).

    Returns a JSON dict with:
      - merged_beliefs: optional list of belief updates/additions
      - prune_beliefs: optional list of belief ids/keys to downrank
      - summary: short natural-language consolidation
      - next_focus: list of next-step items
    """
    prompt = {
        "task": "sleep_consolidation",
        "axioms": axioms,
        "state": state_summary,
        "workspace": workspace_items[:20],
        "beliefs": beliefs[:40],
        "axiom_interpretations": axiom_interpretations or {},
        "recent": recent_messages[-40:],
        "output_schema": {
            "summary": "string (<=1200 chars)",
            "merged_beliefs": [{"subject":"str","predicate":"str","object":"str","confidence":"0..1","provenance":"str"}],
            "downgrade_beliefs": [{"subject":"str","predicate":"str","object":"str","reason":"str"}],
            "next_focus": [{"text":"str","importance":"0..1"}],
            "axiom_digests": {"A1":"str<=600","A2":"str<=600","A3":"str<=600","A4":"str<=600"},
        },
        "rules": [
            "Do NOT hallucinate factual claims. Only consolidate from user feedback, beliefs, or explicit evidence in recent messages/workspace.",
            "Prefer merging redundant beliefs and lowering confidence of conflicting ones.",
            "Keep it compact and actionable.",
        ],
    }

    sys = (
        "You are Bunny's Sleep/Consolidation organ. "
        "Your job is to consolidate recent experience into durable beliefs and priorities, respecting the axioms. "
        "Return STRICT JSON matching output_schema. No prose outside JSON. Also compress axiom_interpretations into axiom_digests: short, stable, non-redundant summaries (<=600 chars each)."
    )

    payload = {
        "model": cfg.model,
        "stream": False,
        "options": {"temperature": cfg.temperature, "num_ctx": int(cfg.ctx)},
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    }

    status, txt = http_post_json(cfg.ollama_url.rstrip("/") + "/api/chat", payload, timeout_s=60)
    if status == 0:
        raise RuntimeError(txt)
    if status >= 400:
        raise RuntimeError(f"ollama /api/chat HTTP {status}: {txt[:200]}")
    data = json.loads(txt or "{}")
    content = (((data or {}).get("message") or {}).get("content") or "").strip()
    try:
        return json.loads(content)
    except Exception:
        # fallback: wrap raw output
        return {"summary": content[:1200], "merged_beliefs": [], "downgrade_beliefs": [], "next_focus": []}
