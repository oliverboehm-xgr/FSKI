from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from bunnycore.core.state import StateVector


@dataclass
class SelfReport:
    pain: float
    energy: float
    fatigue: float
    sleep_pressure: float
    uncertainty: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pain": self.pain,
            "energy": self.energy,
            "fatigue": self.fatigue,
            "sleep_pressure": self.sleep_pressure,
            "uncertainty": self.uncertainty,
        }


def build_self_report(state: StateVector, axis_index: Dict[str, int]) -> SelfReport:
    def g(name: str) -> float:
        i = axis_index.get(name, -1)
        if i < 0:
            return 0.0
        try:
            return float(state.values[i])
        except Exception:
            return 0.0

    return SelfReport(
        pain=g("pain"),
        energy=g("energy"),
        fatigue=g("fatigue"),
        sleep_pressure=g("sleep_pressure"),
        uncertainty=g("uncertainty"),
    )
