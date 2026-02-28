from __future__ import annotations
import time
from typing import Dict, List, Tuple
from .db import DB

def now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

DEFAULT_AXES = [
    ("pain","total pain (physical + psychological; must be kept low; derived)")
    ,("pain_physical","physical/system pain (resources/errors/latency; safety)")
    ,("pain_psych","psychological/teleological pain (axiom & outcome violations)"),
    ("energy","ANS-like energy/budget"),
    ("fatigue","sleepiness / accumulated need for rest/consolidation (derived from health/resources)"),
    ("sleep_pressure","pressure to enter idle sleep/consolidation mode (derived, protected)"),
    # Affect subspace (can be shaped by experience; not protected like pain/energy)
    ("valence","affect: positive vs negative tone"),
    ("arousal","affect: activation / arousal"),
    ("security","affect: felt safety / stability"),
    ("frustration","affect: frustration / irritation"),
    ("awe","affect: awe / wonder"),
    ("stress","resource pressure / overload"),
    ("curiosity","exploration drive"),
    ("confidence","epistemic confidence"),
    ("uncertainty","epistemic uncertainty"),
    ("error_signal","recent negative feedback/contradiction signal (decays)"),
    ("freshness_need","need for up-to-date/time-sensitive factual information"),
    ("social_need","social craving / interact urge"),
    ("urge_reply","reply pressure from incoming messages"),
    ("urge_share","share pressure from internal interests"),
    ("pressure_websense","drive to fetch external information via WebSense"),
    ("pressure_daydream","drive to run autonomous daydream/interpretation loops"),
    ("pressure_evolve","drive to propose self-development / mutations"),
    ("capability_gap","perceived gap between goals and current capabilities"),
    ("desire_upgrade","desire for upgrades/organs (A4 operationalization)"),
    ("purpose_a1","teleology: axiom A1 purpose field"),
    ("purpose_a2","teleology: axiom A2 purpose field"),
    ("purpose_a3","teleology: axiom A3 purpose field"),
    ("purpose_a4","teleology: axiom A4 purpose field"),
    ("tension_a1","teleology: axiom A1 tension"),
    ("tension_a2","teleology: axiom A2 tension"),
    ("tension_a3","teleology: axiom A3 tension"),
    ("tension_a4","teleology: axiom A4 tension"),
    ("sat_a1","satiation: short-term satisfaction of serving Oliver (A1); decays"),
    ("sat_a3","satiation: short-term satisfaction with A3 progress; decays"),
    ("sat_a4","satiation: short-term satisfaction with A4 progress; decays"),
]

# Homeostatic baseline for the organism state.
#
# V1 used a scalar decay which otherwise collapses all axes toward 0 over long runs.
# That makes autonomous organs (daydream/websense/evolve) silently stop firing after
# enough ticks. The baseline is a stable initial condition / setpoint: absent new
# events, S(t) relaxes toward these values (not toward 0).
DEFAULT_BASELINE_VALUES: dict[str, float] = {
    # Core budget / affect anchors
    "energy": 0.62,
    "stress": 0.22,
    "curiosity": 0.26,
    "confidence": 0.34,
    "uncertainty": 0.30,
    "freshness_need": 0.18,
    "social_need": 0.18,

    # Autonomy pressures: keep them non-zero so organs can actually run.
    "pressure_daydream": 0.22,
    "pressure_websense": 0.12,
    "pressure_evolve": 0.18,
}


def make_baseline_vector(axis: Dict[str, int], dim: int) -> List[float]:
    """Create a baseline vector aligned to the current axis registry."""
    b = [0.0] * int(dim)
    for name, v in (DEFAULT_BASELINE_VALUES or {}).items():
        idx = axis.get(str(name))
        if idx is None:
            continue
        if 0 <= int(idx) < int(dim):
            b[int(idx)] = float(v)
    return b

def ensure_axes(db: DB, axes: List[Tuple[str,str]] = None) -> Dict[str,int]:
    axes = axes or DEFAULT_AXES
    con = db.connect()
    try:
        for name, desc in axes:
            row = con.execute("SELECT axis_index FROM state_axes WHERE axis_name=?", (name,)).fetchone()
            if row is None:
                idx = con.execute("SELECT COALESCE(MAX(axis_index),-1)+1 AS nx FROM state_axes").fetchone()["nx"]
                con.execute(
                    "INSERT INTO state_axes(axis_index,axis_name,description,created_at) VALUES(?,?,?,?)",
                    (int(idx), name, desc, now_iso())
                )
        con.commit()
        rows = con.execute("SELECT axis_index,axis_name FROM state_axes ORDER BY axis_index").fetchall()
        return {r["axis_name"]: int(r["axis_index"]) for r in rows}
    finally:
        con.close()

def ensure_axes_meta(db: DB, defaults: Dict[str, Dict[str, object]] | None = None) -> None:
    """Ensure state_axes_meta rows exist for all known axes.

    invariant=1 means the axis MUST NOT be influenced by learnable matrices.
    decays=1 means the axis naturally decays toward baseline (handled elsewhere).
    source is informational for UI/debug ('reward_signal','health','decision',...).
    """
    defaults = defaults or {}
    con = db.connect()
    try:
        # table may not exist on older DBs
        con.execute(
            "CREATE TABLE IF NOT EXISTS state_axes_meta(axis_name TEXT PRIMARY KEY, invariant INTEGER NOT NULL DEFAULT 0, decays INTEGER NOT NULL DEFAULT 0, source TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL)"
        )
        rows = con.execute("SELECT axis_name FROM state_axes").fetchall()
        for r in rows:
            name = str(r["axis_name"])
            d = defaults.get(name, {})
            inv = 1 if d.get("invariant", 0) else 0
            dec = 1 if d.get("decays", 0) else 0
            src = str(d.get("source", ""))
            con.execute(
                "INSERT OR IGNORE INTO state_axes_meta(axis_name,invariant,decays,source,updated_at) VALUES(?,?,?,?,?)",
                (name, inv, dec, src, now_iso()),
            )
        # keep metadata fresh-ish for any new defaults
        for name, d in defaults.items():
            con.execute(
                "UPDATE state_axes_meta SET invariant=?, decays=?, source=?, updated_at=? WHERE axis_name=?",
                (1 if d.get("invariant",0) else 0, 1 if d.get("decays",0) else 0, str(d.get("source","")), now_iso(), name),
            )
        con.commit()
    finally:
        con.close()
