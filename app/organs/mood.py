from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from bunnycore.core.state import StateVector


@dataclass
class Mood:
    label: str
    factors: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "factors": self.factors}


def _g(state: StateVector, axis_index: Dict[str, int], name: str) -> float:
    i = axis_index.get(name, -1)
    if i < 0:
        return 0.0
    try:
        return float(state.values[i])
    except Exception:
        return 0.0


def project_mood(state: StateVector, axis_index: Dict[str, int]) -> Mood:
    """Map state axes to a compact, AI-like mood label.

    This is *not* a claim of human emotions; it's a readable projection of the organism's internal dynamics.
    """
    energy = _g(state, axis_index, "energy")
    valence = _g(state, axis_index, "valence")
    arousal = _g(state, axis_index, "arousal")
    confidence = _g(state, axis_index, "confidence")
    stress = _g(state, axis_index, "stress")
    uncertainty = _g(state, axis_index, "uncertainty")
    curiosity = _g(state, axis_index, "curiosity")

    # Simple continuous decision: choose the highest-scoring attractor.
    # No keyword heuristics; purely based on state values.
    attractors: Dict[str, float] = {
        "euphoric": 0.6*energy + 0.6*valence + 0.4*arousal + 0.3*confidence - 0.6*stress - 0.4*uncertainty,
        "focused": 0.4*energy + 0.2*valence + 0.2*confidence + 0.5*(1.0-uncertainty) - 0.3*stress,
        "curious": 0.7*curiosity + 0.2*arousal + 0.2*energy - 0.3*stress,
        "strained": 0.7*stress + 0.3*arousal + 0.2*uncertainty - 0.2*valence,
        "uncertain": 0.8*uncertainty + 0.2*stress - 0.2*confidence,
        "low": 0.7*(1.0-energy) + 0.3*(1.0-valence) + 0.2*stress,
    }
    label = max(attractors.items(), key=lambda kv: kv[1])[0]

    factors = {
        "energy": energy,
        "valence": valence,
        "arousal": arousal,
        "confidence": confidence,
        "stress": stress,
        "uncertainty": uncertainty,
        "curiosity": curiosity,
    }
    return Mood(label=label, factors=factors)
