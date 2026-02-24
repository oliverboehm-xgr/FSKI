from __future__ import annotations
import json, time
from dataclasses import dataclass
from typing import List, Dict, Any

def now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

@dataclass
class StateVector:
    """Dense state vector (values) with DB-defined axes."""
    values: List[float]

    def dim(self) -> int:
        return len(self.values)

    def copy(self) -> "StateVector":
        return StateVector(self.values.copy())

    def clip(self, lo: float = -1.0, hi: float = 1.0) -> "StateVector":
        self.values = [min(hi, max(lo, float(v))) for v in self.values]
        return self

    def mul_scalar_inplace(self, a: float) -> "StateVector":
        for i in range(self.dim()):
            self.values[i] *= float(a)
        return self

    def to_json(self) -> str:
        return json.dumps(self.values)

    @staticmethod
    def from_json(s: str) -> "StateVector":
        return StateVector(list(map(float, json.loads(s))))

    @staticmethod
    def zeros(n: int) -> "StateVector":
        return StateVector([0.0] * int(n))

@dataclass
class Why:
    source: str
    note: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "note": self.note, "data": self.data}

"""Note on 'state as matrix':
You can represent state as a diagonal matrix diag(S) or a density-matrix-like object for richer dynamics.
V1 keeps state as a vector; operators (matrices) carry most expressiveness and are easy to version/train.
"""
