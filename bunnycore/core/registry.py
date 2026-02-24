from __future__ import annotations
import time
from typing import Dict, List, Tuple
from .db import DB

def now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

DEFAULT_AXES = [
    ("energy","ANS-like energy/budget"),
    ("stress","resource pressure / overload"),
    ("curiosity","exploration drive"),
    ("confidence","epistemic confidence"),
    ("uncertainty","epistemic uncertainty"),
    ("social_need","social craving / interact urge"),
    ("urge_reply","reply pressure from incoming messages"),
    ("urge_share","share pressure from internal interests"),
    ("pressure_websense","drive to fetch external information via WebSense"),
    ("pressure_daydream","drive to run autonomous daydream/interpretation loops"),
    ("purpose_a1","teleology: axiom A1 purpose field"),
    ("purpose_a2","teleology: axiom A2 purpose field"),
    ("purpose_a3","teleology: axiom A3 purpose field"),
    ("purpose_a4","teleology: axiom A4 purpose field"),
    ("tension_a1","teleology: axiom A1 tension"),
    ("tension_a2","teleology: axiom A2 tension"),
    ("tension_a3","teleology: axiom A3 tension"),
    ("tension_a4","teleology: axiom A4 tension"),
]

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
