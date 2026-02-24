from __future__ import annotations
import json, time
from dataclasses import dataclass
from typing import Any, Dict

def now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

@dataclass(frozen=True)
class Event:
    event_type: str
    payload: Dict[str, Any]
    created_at: str = ""

    def with_time(self) -> "Event":
        if self.created_at:
            return self
        return Event(self.event_type, self.payload, now_iso())

    def payload_json(self) -> str:
        return json.dumps(self.payload, ensure_ascii=False)
