from __future__ import annotations

"""WebSense assimilation organ.

Goal:
  Turn extracted WebSense evidence-claims (already structured JSON) into
  durable belief triples that can be reused by other organs (speech/decider)
  in later turns.

Design constraints:
  - No keyword heuristics about the topic/domain.
  - Conservative: prefer fewer beliefs over noisy memory.
  - Use domain trust map to downweight low-trust sources.
  - Keep schema stable: beliefs table stores (subject,predicate,object,confidence,provenance,topic).
"""

from typing import Any, Dict, List
from urllib.parse import urlparse


def _cl01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _domain(u: str) -> str:
    try:
        return (urlparse(u or "").hostname or "").lower()
    except Exception:
        return ""


def assimilate_websense_claims(
    *,
    question: str,
    query: str,
    claims_json: Dict[str, Any],
    trust_map: Dict[str, float] | None = None,
    max_beliefs: int = 4,
) -> Dict[str, Any]:
    """Convert WebSense claims JSON into belief triples.

    Input claims_json is the output of evidence.extract_evidence_claims(), e.g.:
      {"claims": [{"text":..., "confidence":..., "support":[urls], ...}], "uncertainty": ...}

    Output:
      {"beliefs": [{subject,predicate,object,confidence,provenance}], "avg_conf": float, "domains": [..]}
    """

    trust_map = trust_map or {}
    out: List[Dict[str, Any]] = []
    domains: List[str] = []
    seen_obj: set[str] = set()

    cl = claims_json.get("claims") if isinstance(claims_json, dict) else None
    if not isinstance(cl, list) or not cl:
        return {"beliefs": [], "avg_conf": 0.0, "domains": [], "notes": "no_claims"}

    for c in cl:
        if not isinstance(c, dict):
            continue
        text = str(c.get("text") or "").strip()
        if not text:
            continue

        # Optional structured fields (value/unit/time) – merge conservatively.
        val = c.get("value")
        unit = c.get("unit")
        t = c.get("time")
        extra = []
        if val is not None and str(val).strip() and str(val).strip() not in text:
            extra.append(f"value={str(val).strip()}")
        if unit is not None and str(unit).strip() and str(unit).strip() not in text:
            extra.append(f"unit={str(unit).strip()}")
        if t is not None and str(t).strip() and str(t).strip() not in text:
            extra.append(f"time={str(t).strip()}")
        obj = text
        if extra:
            obj = (obj + " (" + ", ".join(extra) + ")").strip()

        obj_n = obj.lower()
        if obj_n in seen_obj:
            continue
        seen_obj.add(obj_n)

        try:
            conf = float(c.get("confidence", 0.6) or 0.6)
        except Exception:
            conf = 0.6
        conf = _cl01(conf)

        support = c.get("support")
        url0 = ""
        if isinstance(support, list) and support:
            url0 = str(support[0] or "")
        dom = _domain(url0)
        if dom and dom not in domains:
            domains.append(dom)

        # Downweight by learned domain trust (0..1). Keep at least 0.35 to avoid zeroing.
        tr = float(trust_map.get(dom, 0.5)) if dom else 0.5
        tr = _cl01(tr)
        conf_eff = _cl01(conf * (0.4 + 0.6 * tr))

        prov = "websense"
        if dom:
            prov = f"websense:{dom}"[:120]

        out.append(
            {
                "subject": "World",
                "predicate": "web_fact",
                "object": obj[:600],
                "confidence": float(conf_eff),
                "provenance": prov,
            }
        )

        if len(out) >= int(max_beliefs or 4):
            break

    avg = 0.0
    if out:
        avg = sum(float(b.get("confidence", 0.0) or 0.0) for b in out) / float(len(out))

    return {
        "beliefs": out,
        "avg_conf": float(_cl01(avg)),
        "domains": domains[:8],
        "notes": "ok" if out else "empty",
    }
