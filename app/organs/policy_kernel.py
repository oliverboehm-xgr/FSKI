from __future__ import annotations

"""Policy kernel: trainable action selection weights.

This is intentionally small and deterministic. It does NOT replace the LLM decider; it provides
(a) a stable action prior (policy_hint) and
(b) a mutatable object that Daydream/Evolve can change.

We store weights as a matrix in the existing MatrixStore (matrices/matrix_entries):
  P_policy@v : shape [n_actions, n_features]

Update rule (policy gradient / REINFORCE-style):
  W <- W + eta * reward * (one_hot(a) - p) \otimes x

Where p = softmax(Wx).

All values are kept small via L2 decay + clamps.
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from bunnycore.core.db import DB
from bunnycore.core.matrix_store import MatrixStore


POLICY_ACTIONS: List[str] = [
    "websense",
    "daydream",
    "evolve",
    "autotalk",
]

# Features are drawn from *bounded* internal state (0..1). Bias term first.
# Keep the list small and stable (architecture, not heuristics).
POLICY_FEATURES: List[str] = [
    "bias",
    "uncertainty",
    "freshness_need",
    "curiosity",
    "stress",
    "social_need",
    "pressure_websense",
    "pressure_daydream",
    "pressure_evolve",
    "capability_gap",
    "desire_upgrade",
    "purpose_a1",
    "purpose_a2",
    "purpose_a3",
    "purpose_a4",
    "tension_a1",
    "tension_a2",
    "tension_a3",
    "tension_a4",
]


def _now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _softmax(logits: List[float]) -> List[float]:
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    if s <= 0.0:
        n = len(logits)
        return [1.0 / n] * n
    return [e / s for e in exps]


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _latest_version(store: MatrixStore, name: str) -> int:
    v = 0
    for m in store.list_matrices():
        if str(m.get("name")) == name:
            try:
                vv = int(m.get("version") or 0)
            except Exception:
                continue
            if vv > v:
                v = vv
    return v


@dataclass
class PolicyKernelConfig:
    enable: bool = True
    eta: float = 0.05
    l2_decay: float = 0.001
    max_abs: float = 3.0
    frob_tau: float = 25.0


class PolicyKernel:
    def __init__(self, db: DB, store: MatrixStore, axis_index: Dict[str, int], *, cfg: PolicyKernelConfig | None = None):
        self.db = db
        self.store = store
        self.axis = axis_index
        self.cfg = cfg or PolicyKernelConfig()
        self.name = "P_policy"

    @property
    def n_actions(self) -> int:
        return len(POLICY_ACTIONS)

    @property
    def n_features(self) -> int:
        return len(POLICY_FEATURES)

    def ensure_seed(self) -> None:
        """Ensure base policy matrix exists."""
        if _latest_version(self.store, self.name) >= 1:
            return
        # start with all-zero weights; softmax -> uniform.
        self.store.put_sparse(
            self.name,
            version=1,
            n_rows=self.n_actions,
            n_cols=self.n_features,
            entries=[],
            meta={
                "desc": "action policy weights (Context->Action prior)",
                "actions": POLICY_ACTIONS,
                "features": POLICY_FEATURES,
                "created_at": _now_iso(),
            },
            parent_version=None,
        )

    def features(self, state_values: List[float]) -> List[float]:
        """Build feature vector x aligned with POLICY_FEATURES."""
        # Map axis name -> value.
        inv: Dict[int, str] = {v: k for k, v in self.axis.items()}
        named: Dict[str, float] = {}
        for i, v in enumerate(state_values):
            nm = inv.get(i)
            if nm is not None:
                try:
                    named[nm] = float(v)
                except Exception:
                    continue

        x: List[float] = []
        for f in POLICY_FEATURES:
            if f == "bias":
                x.append(1.0)
            else:
                x.append(_clamp(float(named.get(f, 0.0) or 0.0), 0.0, 1.0))
        return x

    def predict(self, state_values: List[float]) -> Dict[str, Any]:
        """Return action probabilities + debug info."""
        self.ensure_seed()
        ver = _latest_version(self.store, self.name)
        A = self.store.get_sparse(self.name, ver)
        x = self.features(state_values)
        logits = [0.0] * self.n_actions
        # sparse matvec
        for (i, j, v) in getattr(A, "entries", []) or []:
            ii = int(i)
            jj = int(j)
            if 0 <= ii < self.n_actions and 0 <= jj < self.n_features:
                logits[ii] += float(v) * float(x[jj])
        probs = _softmax(logits)
        return {
            "version": ver,
            "features": x,
            "logits": logits,
            "probs": {POLICY_ACTIONS[i]: float(probs[i]) for i in range(self.n_actions)},
        }

    def apply_update(self, *, from_version: int, x: List[float], action: str, reward: float, note: str = "") -> int:
        """Apply a policy-gradient update and return new version."""
        if not self.cfg.enable:
            return from_version
        if action not in POLICY_ACTIONS:
            return from_version
        a_idx = POLICY_ACTIONS.index(action)

        # Load current
        A = self.store.get_sparse(self.name, int(from_version))
        # Build dense W for small dims
        W = [[0.0 for _ in range(self.n_features)] for _ in range(self.n_actions)]
        for (i, j, v) in getattr(A, "entries", []) or []:
            ii = int(i)
            jj = int(j)
            if 0 <= ii < self.n_actions and 0 <= jj < self.n_features:
                W[ii][jj] = float(v)

        # forward
        logits = [sum(W[i][j] * float(x[j]) for j in range(self.n_features)) for i in range(self.n_actions)]
        probs = _softmax(logits)

        r = _clamp(float(reward), -1.0, 1.0)
        eta = float(self.cfg.eta)

        # update
        delta_frob = 0.0
        for i in range(self.n_actions):
            coeff = (1.0 if i == a_idx else 0.0) - float(probs[i])
            for j in range(self.n_features):
                dv = eta * r * coeff * float(x[j])
                if abs(dv) < 1e-9:
                    continue
                W[i][j] = (1.0 - float(self.cfg.l2_decay)) * float(W[i][j]) + dv
                W[i][j] = _clamp(W[i][j], -float(self.cfg.max_abs), float(self.cfg.max_abs))
                delta_frob += float(dv) * float(dv)

        delta_frob = math.sqrt(delta_frob) if delta_frob > 0 else 0.0

        # Frobenius clamp
        frob = math.sqrt(sum(W[i][j] * W[i][j] for i in range(self.n_actions) for j in range(self.n_features)))
        if frob > float(self.cfg.frob_tau) and frob > 1e-9:
            scale = float(self.cfg.frob_tau) / frob
            for i in range(self.n_actions):
                for j in range(self.n_features):
                    W[i][j] = float(W[i][j]) * scale

        # store sparse (but small: keep all non-zeros)
        entries: List[Tuple[int, int, float]] = []
        for i in range(self.n_actions):
            for j in range(self.n_features):
                v = float(W[i][j])
                if abs(v) > 1e-8:
                    entries.append((i, j, v))

        new_version = int(from_version) + 1
        self.store.put_sparse(
            self.name,
            version=new_version,
            n_rows=self.n_actions,
            n_cols=self.n_features,
            entries=entries,
            meta={
                "desc": "action policy weights (Context->Action prior)",
                "actions": POLICY_ACTIONS,
                "features": POLICY_FEATURES,
                "updated_at": _now_iso(),
                "from_version": int(from_version),
                "reward": float(r),
                "delta_frob": float(delta_frob),
                "note": str(note or ""),
            },
            parent_version=int(from_version),
        )

        # log via matrix_update_log for UI visibility
        try:
            con = self.db.connect()
            con.execute(
                "INSERT INTO matrix_update_log(event_type,matrix_name,from_version,to_version,reward,delta_frob,created_at,notes) VALUES(?,?,?,?,?,?,?,?)",
                (
                    "policy",
                    self.name,
                    int(from_version),
                    int(new_version),
                    float(r),
                    float(delta_frob),
                    _now_iso(),
                    str(note or ""),
                ),
            )
            con.commit()
            con.close()
        except Exception:
            pass

        return new_version
