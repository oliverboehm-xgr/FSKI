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
        idx = self.axis.get("urge_reply")
        if idx is not None and idx < state_dim:
            x[idx] += min(1.0, len(text)/400.0)
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
            # mild frustration on repeated tool failures
            idx = self.axis.get("frustration")
            if idx is not None and idx < state_dim:
                x[idx] += min(0.15, 0.05 + 0.02 * pages)
        else:
            for k,v in [("confidence", conf), ("uncertainty", unc), ("curiosity", cur)]:
                idx = self.axis.get(k)
                if idx is not None and idx < state_dim:
                    x[idx] += v

            # awe/wonder is a light-weight proxy for novelty exposure
            idx = self.axis.get("awe")
            if idx is not None and idx < state_dim:
                x[idx] += min(0.12, 0.03 * pages + 0.20 * float(novelty or 0.0))

        # Cooldown the trigger axis after a WebSense run.
        # Otherwise pressure_websense can stick above threshold and cause runaway searches.
        idx = self.axis.get("pressure_websense")
        if idx is not None and idx < state_dim:
            if ok > 0:
                # Successful evidence lowers the need to keep searching.
                x[idx] += -min(0.85, 0.25 + 0.08 * pages)
            else:
                # Failures increase the urge to try again (but keep it bounded).
                x[idx] += min(0.60, 0.15 + 0.05 * pages)


        # Also reduce freshness_need after successful evidence retrieval.
        idx = self.axis.get("freshness_need")
        if idx is not None and idx < state_dim:
            if ok > 0:
                x[idx] += -min(0.70, 0.20 + 0.06 * pages)
            else:
                x[idx] += min(0.45, 0.10 + 0.04 * pages)

        return x, [Why(source="websense", note="websense_v1", data=p)]


class DriveFieldEncoder:
    """Generic encoder: event payload contains a dict of axis->delta.

    This is used for LLM-based decision models (no hard-coded keyword triggers).
    Payload schema:
      {"drives": {"uncertainty": 0.1, "pressure_websense": 0.6, ...}}
    """
    name = "drive_field_v1"
    def __init__(self, axis_index_by_name: Dict[str,int]):
        self.axis = axis_index_by_name
        # Axes that should not be written by LLM-driven events.
        # NOTE: These axes are still writable by internal measurement events (health/resources).
        self._protected = {"pain", "pain_physical", "pain_psych", "energy", "fatigue", "sleep_pressure"}
        # V1 invariant: persisted state axes are in [0,1]. We never accept signed target values.
        # Signed updates are only allowed in delta-mode.
        self._signed: set[str] = set()

    def _clamp(self, key: str, v: float, mode: str) -> float:
        """Clamp drive value.

        mode:
          - "target": values represent target levels (most drives are 0..1, affect may be signed)
          - "delta":  values represent deltas (allow signed updates for all axes)
        """
        fv = float(v)
        if str(mode) == "delta":
            return max(-1.0, min(1.0, fv))
        # target-mode (legacy): mostly 0..1, with a small signed affect subspace
        if fv < 0.0:
            return 0.0
        if fv > 1.0:
            return 1.0
        return fv

    def encode(self, state_dim: int, event: Event):
        x = [0.0] * state_dim
        p = event.payload or {}
        mode = str(p.get("_mode") or "target")
        drives = p.get("drives") if isinstance(p.get("drives"), dict) else (p if isinstance(p, dict) else {})
        out: Dict[str, float] = {}
        for k, v in (drives or {}).items():
            try:
                fv = float(v)
            except Exception:
                continue
            kk = str(k)
            idx = self.axis.get(kk)
            if idx is None or idx >= state_dim:
                continue
            # Never allow LLM drives to write measured/protected axes.
            # Internal measurement events are allowed to update them.
            if kk in self._protected and str(event.event_type) not in ("health", "resources"):
                continue
            fv = self._clamp(kk, fv, mode)
            x[idx] += fv
            out[kk] = fv
        return x, [Why(source="decider", note=f"drive_field_v1 mode={mode}", data=out)]


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

    def list_bindings(self) -> list[dict]:
        """Return all adapter bindings (for introspection/debug)."""
        con = self.db.connect()
        try:
            rows = con.execute(
                "SELECT event_type,encoder_name,matrix_name,matrix_version,meta_json,updated_at FROM adapters ORDER BY event_type"
            ).fetchall()
            out: list[dict] = []
            for r in rows:
                try:
                    meta = json.loads(r["meta_json"] or "{}")
                except Exception:
                    meta = {}
                out.append(
                    {
                        "event_type": str(r["event_type"]),
                        "encoder": str(r["encoder_name"]),
                        "matrix": str(r["matrix_name"]),
                        "version": int(r["matrix_version"]),
                        "meta": meta,
                        "updated_at": str(r["updated_at"] or ""),
                    }
                )
            return out
        finally:
            con.close()
