from __future__ import annotations
import json, time
from dataclasses import dataclass
from typing import List
from .events import Event
from .state import StateVector, Why, now_iso
from .db import DB
from .integrator import Integrator

@dataclass
class HeartbeatConfig:
    tick_hz: float = 2.0
    snapshot_every_n_ticks: int = 1

class Heartbeat:
    """The only loop that mutates state_current.

    External code enqueues events; heartbeat integrates them into S(t).
    """
    def __init__(self, db: DB, integrator: Integrator, cfg: HeartbeatConfig):
        self.db = db
        self.integrator = integrator
        self.cfg = cfg
        self._queue: List[Event] = []
        self._tick = 0

    def enqueue(self, ev: Event) -> None:
        ev = ev.with_time()
        self._queue.append(ev)
        con = self.db.connect()
        try:
            con.execute("INSERT INTO event_log(event_type,payload_json,created_at) VALUES(?,?,?)",
                        (ev.event_type, ev.payload_json(), ev.created_at))
            con.commit()
        finally:
            con.close()

    def _axis_dim(self) -> int:
        con = self.db.connect()
        try:
            return int(con.execute("SELECT COUNT(*) AS c FROM state_axes").fetchone()["c"])
        finally:
            con.close()

    def load_state(self) -> StateVector:
        con = self.db.connect()
        try:
            row = con.execute("SELECT vec_json FROM state_current WHERE id=1").fetchone()
            if row is None:
                s = StateVector.zeros(self._axis_dim())
                con.execute("INSERT OR REPLACE INTO state_current(id,vec_json,updated_at) VALUES(1,?,?)",
                            (s.to_json(), now_iso()))
                con.commit()
                return s
            return StateVector.from_json(row["vec_json"])
        finally:
            con.close()

    def save_state(self, s: StateVector, why: List[Why]) -> None:
        con = self.db.connect()
        try:
            con.execute("INSERT OR REPLACE INTO state_current(id,vec_json,updated_at) VALUES(1,?,?)",
                        (s.to_json(), now_iso()))
            if self.cfg.snapshot_every_n_ticks > 0 and (self._tick % self.cfg.snapshot_every_n_ticks == 0):
                con.execute("INSERT INTO state_snapshots(vec_json,why_json,tags_json,created_at) VALUES(?,?,?,?)",
                            (s.to_json(),
                             json.dumps([w.to_dict() for w in why], ensure_ascii=False),
                             "[]",
                             now_iso()))
            con.commit()
        finally:
            con.close()

    def step(self) -> None:
        s = self.load_state()
        events = self._queue
        self._queue = []
        s2, why = self.integrator.tick(s, events)
        self.save_state(s2, why)
        self._tick += 1

    def run_forever(self) -> None:
        period = 1.0 / max(0.1, float(self.cfg.tick_hz))
        while True:
            t0 = time.time()
            self.step()
            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))
