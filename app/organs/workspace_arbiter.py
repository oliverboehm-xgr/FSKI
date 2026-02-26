from __future__ import annotations

import json
from typing import Any, Dict, List

from app.net import http_post_json


def arbitrate_workspace(
    ollama_url: str,
    model: str,
    candidates: List[Dict[str, Any]],
    prev_items: List[Dict[str, Any]] | None = None,
    max_items: int = 12,
) -> List[Dict[str, Any]]:
    """Select a compact global workspace from candidate items.

    This is intentionally LLM-based (no string heuristics) and must be robust.
    Falls back to a simple weight sort if the model fails.
    """
    prev_items = prev_items or []
    max_items = int(max(4, min(20, max_items)))

    system = (
        "You are a workspace arbiter for a cognitive system. "
        "Given candidate items, select a small set of the most salient items "
        "to keep in the GLOBAL WORKSPACE. "
        "Return STRICT JSON only.\n\n"
        "Rules:\n"
        "- Keep at most max_items.\n"
        "- Prefer items that reduce uncertainty, track goals/axioms, and support next action.\n"
        "- Keep the active topic item if present.\n"
        "- Do not invent new items; only select from candidates or prev_items.\n"
        "Output schema: {\"items\": [ {item...}, ... ], \"note\": \"...\" }\n"
    )
    user = {
        "max_items": max_items,
        "prev_items": prev_items,
        "candidates": candidates,
    }

    try:
        resp = http_post_json(
            f"{ollama_url.rstrip('/')}/api/generate",
            {
                "model": model,
                "prompt": json.dumps(user, ensure_ascii=False),
                "system": system,
                "stream": False,
                "format": "json",
            },
            timeout_s=35,
        )
        txt = (resp or {}).get("response") or ""
        data = json.loads(txt)
        items = data.get("items")
        if isinstance(items, list) and items:
            out: List[Dict[str, Any]] = []
            for it in items:
                if isinstance(it, dict) and "kind" in it and "text" in it:
                    out.append(it)
            if out:
                return out[:max_items]
    except Exception:
        pass

    # fallback (deterministic, but used only when the arbiter fails)
    try:
        merged = list(prev_items or []) + list(candidates or [])
        merged2: List[Dict[str, Any]] = []
        for it in merged:
            if isinstance(it, dict) and it.get("kind") and it.get("text"):
                merged2.append(it)
        merged2.sort(key=lambda x: float(x.get("w", 0.5) or 0.5), reverse=True)
        return merged2[:max_items]
    except Exception:
        return candidates[:max_items]
