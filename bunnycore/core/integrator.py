from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
from .state import StateVector, Why
from .events import Event
from .matrix_store import MatrixStore
from .adapters import AdapterRegistry, Encoder

@dataclass
class IntegratorConfig:
    # Indices of state axes that must not be influenced by learnable matrices.
    protect_indices: List[int] = None
    # Event types allowed to directly influence protected axes (besides decay).
    protect_allow_event_types: List[str] = None

    # V1 invariant: state axes live in [0,1]. Signed deltas are allowed in event encodings,
    # but the persisted organism state is always bounded.
    clip_lo: float = 0.0
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
        protect = list(self.cfg.protect_indices or [])
        allow = set(str(x) for x in (self.cfg.protect_allow_event_types or ['health','resources','reward_signal']))
        # baseline values after decay (protected axes can still decay)
        protect_base = {i: s2.values[i] for i in protect if i < dim}
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
            # enforce protected axes invariants for any trainable channel
            if protect and str(ev.event_type) not in allow:
                for pi, pv in protect_base.items():
                    if pi < dim:
                        s2.values[pi] = pv
            why.append(Why(source="core", note="adapter", data={"event_type": ev.event_type, "matrix": f"{binding.matrix_name}@{binding.matrix_version}"}))

        s2.clip(self.cfg.clip_lo, self.cfg.clip_hi)
        return s2, why
