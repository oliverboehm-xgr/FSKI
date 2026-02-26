from __future__ import annotations

"""Introspection organ (non-LLM).

Goal: provide a compact, generic self-model snapshot that can be fed into
daydream/evolve organs.

No heuristics about user text. This is purely a capability inventory.
"""

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List


def now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def list_organs(app_dir: str) -> List[str]:
    organs_dir = os.path.join(app_dir, "organs")
    out: List[str] = []
    try:
        for fn in os.listdir(organs_dir):
            if not fn.endswith(".py"):
                continue
            if fn.startswith("__"):
                continue
            out.append(fn[:-3])
    except Exception:
        return []
    out.sort()
    return out


def build_self_model(
    *,
    app_dir: str,
    axes: List[str],
    adapters: List[Dict[str, Any]],
    matrices: List[Dict[str, Any]],
    model_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a compact self-model snapshot.

    This is intentionally small and stable. It should not include volatile UI state.
    """
    return {
        "created_at": now_iso(),
        "organs": list_organs(app_dir),
        "axes": list(axes),
        "adapters": adapters,
        "matrices": matrices,
        "models": model_cfg,
    }
