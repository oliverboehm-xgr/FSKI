from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from bunnycore.core.state import now_iso


@dataclass
class SensoryToken:
    modality: str  # vision|audio
    summary: str   # compact semantic summary
    tokens: Dict[str, Any]  # structured attributes
    salience: float
    created_at: str

    def to_row(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "summary": self.summary,
            "tokens_json": json.dumps(self.tokens or {}, ensure_ascii=False),
            "salience": float(self.salience),
            "created_at": self.created_at,
        }


def compute_salience(state_hint: Dict[str, float], novelty: float, relevance: float) -> float:
    """Generic salience score (0..1). No modality-specific heuristics."""
    stress = float(state_hint.get("stress", 0.0))
    curiosity = float(state_hint.get("curiosity", 0.0))
    awe = float(state_hint.get("awe", 0.0))
    base = 0.35*novelty + 0.35*relevance + 0.15*curiosity + 0.15*awe - 0.10*stress
    return max(0.0, min(1.0, base))


def make_vision_token(scene_summary: str, attrs: Dict[str, Any], state_hint: Dict[str, float], novelty: float = 0.5, relevance: float = 0.5) -> SensoryToken:
    sal = compute_salience(state_hint, novelty, relevance)
    return SensoryToken(modality="vision", summary=scene_summary[:280], tokens=attrs or {}, salience=sal, created_at=now_iso())


def make_audio_token(audio_summary: str, attrs: Dict[str, Any], state_hint: Dict[str, float], novelty: float = 0.5, relevance: float = 0.5) -> SensoryToken:
    sal = compute_salience(state_hint, novelty, relevance)
    return SensoryToken(modality="audio", summary=audio_summary[:280], tokens=attrs or {}, salience=sal, created_at=now_iso())
