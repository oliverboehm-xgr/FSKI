from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
from .state import StateVector, Why
from .events import Event
from .matrix_store import MatrixStore
from .adapters import AdapterRegistry, Encoder

@dataclass
class IntegratorConfig:
    clip_lo: float = -1.0
    clip_hi: float = 1.0
    decay: float = 0.995  # scalar decay (V1). Replace with diagonal or sparse matrix later.

class Integrator:
    """Core equation: S' = clip(decay*S + Σ A_k φ(E_k))"""
    def __init__(self, store: MatrixStore, registry: AdapterRegistry, encoders: Dict[str, Encoder], cfg: IntegratorConfig):
        self.store = store
        self.registry = registry
        self.encoders = encoders
        self.cfg = cfg

    def tick(self, state: StateVector, events: List[Event]) -> tuple[StateVector, List[Why]]:
        dim = state.dim()
        why: List[Why] = []
        s2 = state.copy().mul_scalar_inplace(self.cfg.decay)
        why.append(Why(source="core", note="decay", data={"decay": self.cfg.decay}))

        for ev in events:
            binding = self.registry.get(ev.event_type)
            if binding is None:
                continue
            enc = self.encoders.get(binding.encoder_name)
            if enc is None:
                why.append(Why(source="core", note="missing encoder", data={"event_type": ev.event_type, "encoder": binding.encoder_name}))
                continue
            x, wh = enc.encode(dim, ev)
            why.extend(wh)
            A = self.store.get_sparse(binding.matrix_name, binding.matrix_version)
            y = A.apply(x)
            n = min(dim, len(y))
            for i in range(n):
                s2.values[i] += y[i]
            why.append(Why(source="core", note="adapter", data={"event_type": ev.event_type, "matrix": f"{binding.matrix_name}@{binding.matrix_version}"}))

        s2.clip(self.cfg.clip_lo, self.cfg.clip_hi)
        return s2, why
