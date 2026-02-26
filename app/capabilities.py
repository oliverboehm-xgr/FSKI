from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from bunnycore.core.db import DB
from bunnycore.core.events import Event
from bunnycore.core.state import now_iso


@dataclass
class Capability:
    name: str
    kind: str = "tool"  # tool|sensor|actor
    meta: Dict[str, Any] | None = None
    handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None


class CapabilityBus:
    """Generic interface to attach optional modules (tools/sensors/actors).

    The core must never hard-depend on any external module.
    Missing/failed capabilities must produce pain via health metrics, but not crash the system.
    """

    def __init__(self, db: DB, event_sink: Callable[[Event], None]):
        self.db = db
        self._event_sink = event_sink
        self._caps: Dict[str, Capability] = {}

    def register(self, cap: Capability) -> None:
        self._caps[cap.name] = cap
        meta_json = json.dumps(cap.meta or {}, ensure_ascii=False)
        con = self.db.connect()
        try:
            con.execute(
                "INSERT OR REPLACE INTO capabilities(name,kind,meta_json,health_json,updated_at) VALUES(?,?,?,?,?)",
                (cap.name, cap.kind, meta_json, "{}", now_iso()),
            )
            con.commit()
        finally:
            con.close()

    def has(self, name: str) -> bool:
        return name in self._caps and self._caps[name].handler is not None

    def publish_sensor_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Sensors push data into the organism exclusively as events."""
        self._event_sink(Event(event_type, payload).with_time())

    def call(self, name: str, args: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Call a capability handler.

        Always returns a dict; failures are encoded as {"ok":0,"error":...}.
        The caller is responsible for logging health and converting failures into pain.
        """
        args = args or {}
        cap = self._caps.get(name)
        if cap is None or cap.handler is None:
            return {"ok": 0, "error": "capability_not_available", "name": name}
        t0 = time.time()
        ok = 0
        err = ""
        result: Dict[str, Any] = {}
        try:
            result = cap.handler(dict(args)) or {}
            ok = 1 if int(result.get("ok", 1)) == 1 else 0
            return {"ok": ok, **result}
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            return {"ok": 0, "error": err, "name": name}
        finally:
            dt_ms = (time.time() - t0) * 1000.0
            # best-effort audit log
            con = self.db.connect()
            try:
                con.execute(
                    "INSERT INTO capability_calls(created_at,name,ok,latency_ms,error,args_json,result_json) VALUES(?,?,?,?,?,?,?)",
                    (
                        now_iso(),
                        name,
                        int(ok),
                        float(dt_ms),
                        err[:300],
                        json.dumps(args, ensure_ascii=False),
                        json.dumps(result or {}, ensure_ascii=False) if ok else "{}",
                    ),
                )

                # also emit into health_log so pain model sees it (organ-level health is universal)
                con.execute(
                    "INSERT INTO health_log(created_at,organ,ok,latency_ms,error,metrics_json) VALUES(?,?,?,?,?,?)",
                    (
                        now_iso(),
                        f"cap:{name}",
                        int(ok),
                        float(dt_ms),
                        err[:300],
                        "{}",
                    ),
                )
                con.commit()
            finally:
                con.close()
