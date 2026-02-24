from __future__ import annotations
import json, time
from dataclasses import dataclass
from typing import Any, Dict, Protocol
from .events import Event
from .state import Why
from .db import DB

def now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

class Encoder(Protocol):
    name: str
    def encode(self, state_dim: int, event: Event) -> tuple[list[float], list[Why]]:
        ...

class SimpleTextEncoder:
    name = "simple_text_v1"
    def __init__(self, axis_index_by_name: Dict[str,int]):
        self.axis = axis_index_by_name

    def encode(self, state_dim: int, event: Event):
        x = [0.0] * state_dim
        text = str(event.payload.get("text",""))
        lo = text.lower()
        idx = self.axis.get("urge_reply")
        if idx is not None and idx < state_dim:
            x[idx] += min(1.0, len(text)/400.0)
        idx = self.axis.get("uncertainty")
        if idx is not None and idx < state_dim and "?" in text:
            x[idx] += 0.2
        idx = self.axis.get("social_need")
        if idx is not None and idx < state_dim and any(w in lo for w in ["hi","hallo","hey"]):
            x[idx] += 0.1
        return x, [Why(source="user", note="encoded user text (v1)", data={"len":len(text)})]


class TeleologyHintEncoder:
    """Heuristic teleology encoder.

    Produces purpose/tension nudges per axiom. V1 is intentionally transparent and simple.
    """
    name = "teleology_hint_v1"
    def __init__(self, axis_index_by_name: Dict[str,int]):
        self.axis = axis_index_by_name

    def encode(self, state_dim: int, event: Event):
        x = [0.0] * state_dim
        hint = event.payload or {}
        for k in ["purpose_a1","purpose_a2","purpose_a3","purpose_a4","tension_a1","tension_a2","tension_a3","tension_a4"]:
            v = float(hint.get(k, 0.0) or 0.0)
            idx = self.axis.get(k)
            if idx is not None and idx < state_dim:
                x[idx] += max(-1.0, min(1.0, v))
        # also allow generic drives
        for k in ["curiosity","confidence","uncertainty","urge_share","urge_reply","stress","energy","social_need"]:
            if k in hint:
                idx = self.axis.get(k)
                if idx is not None and idx < state_dim:
                    x[idx] += float(hint.get(k,0.0) or 0.0)
        return x, [Why(source="teleology", note="teleology_hint_v1", data=hint)]

class RatingEncoder:
    """Maps explicit user rating into state adjustments."""
    name = "rating_v1"
    def __init__(self, axis_index_by_name: Dict[str,int]):
        self.axis = axis_index_by_name

    def encode(self, state_dim: int, event: Event):
        x = [0.0] * state_dim
        val = int(event.payload.get("value", 0) or 0)
        # reward/punish: confidence up on +1, uncertainty/stress on -1
        if val > 0:
            for k,v in [("confidence",0.15),("stress",-0.10),("uncertainty",-0.10),("urge_share",0.05)]:
                idx = self.axis.get(k)
                if idx is not None and idx < state_dim:
                    x[idx] += v
        elif val < 0:
            for k,v in [("confidence",-0.15),("stress",0.12),("uncertainty",0.12)]:
                idx = self.axis.get(k)
                if idx is not None and idx < state_dim:
                    x[idx] += v
        return x, [Why(source="user", note="rating_v1", data={"value":val})]


class WebsenseEncoder:
    """Maps WebSense (search/fetch/spider) outcome into state adjustments.

    V1: simple, deterministic, transparent.
    - more pages + more domains -> confidence up, uncertainty down
    - errors -> stress/uncertainty up
    - novelty -> curiosity up
    """
    name = "websense_v1"
    def __init__(self, axis_index_by_name: Dict[str,int]):
        self.axis = axis_index_by_name

    def encode(self, state_dim: int, event: Event):
        x = [0.0] * state_dim
        p = event.payload or {}
        pages = int(p.get("pages", 0) or 0)
        domains = int(p.get("domains", 0) or 0)
        ok = int(p.get("ok", 1) or 0)
        novelty = float(p.get("novelty", 0.0) or 0.0)

        # main effect
        conf = min(0.30, 0.05 * pages + 0.03 * domains)
        unc = -min(0.25, 0.04 * pages)
        cur = min(0.20, 0.05 * pages + novelty)

        if ok <= 0:
            conf *= 0.2
            unc = abs(unc) * 0.8
            # error spikes uncertainty + stress
            for k,v in [("uncertainty", min(0.25, 0.10 + 0.03*pages)), ("stress", 0.10)]:
                idx = self.axis.get(k)
                if idx is not None and idx < state_dim:
                    x[idx] += v
        else:
            for k,v in [("confidence", conf), ("uncertainty", unc), ("curiosity", cur)]:
                idx = self.axis.get(k)
                if idx is not None and idx < state_dim:
                    x[idx] += v

        return x, [Why(source="websense", note="websense_v1", data=p)]


@dataclass
class AdapterBinding:
    event_type: str
    encoder_name: str
    matrix_name: str
    matrix_version: int
    meta: Dict[str,Any]

class AdapterRegistry:
    def __init__(self, db: DB):
        self.db = db

    def upsert(self, binding: AdapterBinding) -> None:
        con = self.db.connect()
        try:
            con.execute(
                """INSERT OR REPLACE INTO adapters(event_type,encoder_name,matrix_name,matrix_version,meta_json,updated_at)
                    VALUES(?,?,?,?,?,?)""",
                (binding.event_type, binding.encoder_name, binding.matrix_name, int(binding.matrix_version),
                 json.dumps(binding.meta or {}), now_iso())
            )
            con.commit()
        finally:
            con.close()

    def get(self, event_type: str) -> AdapterBinding | None:
        con = self.db.connect()
        try:
            row = con.execute(
                "SELECT event_type,encoder_name,matrix_name,matrix_version,meta_json FROM adapters WHERE event_type=?",
                (event_type,)
            ).fetchone()
            if row is None:
                return None
            return AdapterBinding(
                event_type=row["event_type"],
                encoder_name=row["encoder_name"],
                matrix_name=row["matrix_name"],
                matrix_version=int(row["matrix_version"]),
                meta=json.loads(row["meta_json"] or "{}"),
            )
        finally:
            con.close()
