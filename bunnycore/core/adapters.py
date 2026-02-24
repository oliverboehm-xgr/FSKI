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
