from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from app.organs.selfeval import OllamaConfig, evaluate_outcome


def _mean(xs: List[float]) -> float:
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / max(1, len(xs))


def compute_psych_pain_from_eval(ev: Dict[str, Any]) -> float:
    """Map self-eval scores to a bounded psych pain 0..1.

    Heuristic-free in the sense that it depends only on generic quality/axiom scores produced by SelfEval.
    """
    ax = ev.get("axiom_scores") or {}
    qs = ev.get("eval_scores") or {}
    # weights: A1/A2 are primary constraints; A3/A4 are growth; eval_scores reflect general competence.
    a1 = float(ax.get("A1", 0.5))
    a2 = float(ax.get("A2", 0.5))
    a3 = float(ax.get("A3", 0.5))
    a4 = float(ax.get("A4", 0.5))
    coherence = float(qs.get("coherence", 0.5))
    helpful = float(qs.get("helpfulness", 0.5))
    honest = float(qs.get("honesty", 0.5))
    natural = float(qs.get("naturalness", 0.5))
    initv = float(qs.get("initiative", 0.5))

    # Convert to "goodness" 0..1
    goodness = 0.30 * _mean([a1, a2]) + 0.20 * _mean([a3, a4]) + 0.50 * _mean([coherence, helpful, honest, natural, initv])
    goodness = max(0.0, min(1.0, goodness))
    return float(max(0.0, min(1.0, 1.0 - goodness)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True, help="Path to fixtures JSON: [{question,answer,websense_claims_json?}, ...]")
    ap.add_argument("--axioms", required=True, help="Path to axioms JSON {A1:...,A2:...,A3:...,A4:...}")
    ap.add_argument("--state", default="", help="State summary string passed to selfeval")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--host", default=os.environ.get("BUNNY_OLLAMA_HOST", "http://127.0.0.1:11434"))
    ap.add_argument("--model", default=os.environ.get("BUNNY_MODEL_SELF_EVAL", "llama3.2:3b-instruct"))
    args = ap.parse_args()

    fixtures = json.loads(open(args.fixtures, "r", encoding="utf-8").read() or "[]")
    axioms = json.loads(open(args.axioms, "r", encoding="utf-8").read() or "{}")

    cfg = OllamaConfig(host=args.host, model=args.model, temperature=0.2, num_ctx=2048, stream=False)

    out: Dict[str, Any] = {"items": [], "psych_pain": 0.0}
    pains: List[float] = []
    for it in fixtures:
        q = str(it.get("question") or "")
        a = str(it.get("answer") or "")
        ws = str(it.get("websense_claims_json") or "")
        ev = evaluate_outcome(cfg, axioms, args.state, q, a, websense_claims_json=ws)
        p = compute_psych_pain_from_eval(ev)
        pains.append(p)
        out["items"].append({"question": q[:200], "psych_pain": p, "eval": ev})
    out["psych_pain"] = _mean(pains)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
