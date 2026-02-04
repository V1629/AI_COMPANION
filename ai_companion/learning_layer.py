"""
Learning Layer: update phrase meaning maps, evaluate interaction outcomes,
prune outdated patterns and strengthen successful patterns. Implements forgetting curves.
"""
from typing import Dict, Any, List
import time
import math


def evaluate_interaction_outcome(response: str, followups: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Heuristic evaluation: positive follow-up (gratitude, elaboration) increments success; negative terse or ignore lowers it.
    Returns metrics such as 'engagement_delta' and 'satisfaction_estimate'.
    """
    engagement = 0.0
    satisfaction = 0.5
    for f in followups:
        text = f.get("message", "").lower()
        if any(w in text for w in ("thanks", "thank you", "that helps", "appreciate")):
            engagement += 1.0
            satisfaction += 0.2
        elif len(text) < 3:
            engagement -= 0.2
        elif any(w in text for w in ("still", "no", "not really", "yeah")):
            engagement -= 0.1
    satisfaction = max(0.0, min(1.0, satisfaction))
    return {"engagement_delta": engagement, "satisfaction_estimate": satisfaction}


def update_phrase_meaning_map(phrase_map: Dict[str, Dict[str, float]], phrase: str, detected_emotions: Dict[str, float], ts: float = None) -> None:
    """
    Incrementally update phrase_map: phrase -> emotion weights with simple running average.
    phrase_map is mutated in place.
    """
    if ts is None:
        ts = time.time()
    entry = phrase_map.setdefault(phrase.lower(), {"counts": 0.0, "weights": {}})
    counts = entry["counts"]
    for k, v in detected_emotions.items():
        prev = entry["weights"].get(k, 0.0)
        # running average
        new = (prev * counts + v) / (counts + 1.0)
        entry["weights"][k] = new
    entry["counts"] = counts + 1.0
    entry["last_ts"] = ts


def adjust_personality_model(personality_model: dict, feedback_vector: List[float], lr: float = 0.01) -> None:
    """
    Small gradient-like update to personality_model vector from feedback signals.
    personality_model mutated in place.
    """
    if "vector" not in personality_model:
        personality_model["vector"] = [0.5] * len(feedback_vector)
    vec = personality_model["vector"]
    for i in range(len(vec)):
        vec[i] = max(0.0, min(1.0, vec[i] + lr * (feedback_vector[i] - vec[i])))


def prune_outdated_patterns(phrase_map: Dict[str, Any], current_ts: float = None, forget_rate: float = 0.001) -> None:
    """
    Apply exponential decay to counts and remove entries below threshold.
    """
    if current_ts is None:
        current_ts = time.time()
    to_delete = []
    for phrase, entry in list(phrase_map.items()):
        age_days = max(0.0, (current_ts - entry.get("last_ts", current_ts)) / (3600 * 24))
        decay = math.exp(-forget_rate * age_days)
        entry["counts"] = entry.get("counts", 0.0) * decay
        for k in list(entry.get("weights", {}).keys()):
            entry["weights"][k] = entry["weights"][k] * decay
        if entry["counts"] < 0.01:
            to_delete.append(phrase)
    for p in to_delete:
        phrase_map.pop(p, None)


def strengthen_successful_patterns(phrase_map: Dict[str, Any], phrase: str, boost: float = 0.1) -> None:
    """
    Increase weight for emotions associated with a phrase if interaction outcome was positive.
    """
    entry = phrase_map.get(phrase.lower())
    if not entry:
        return
    for k in entry.get("weights", {}):
        entry["weights"][k] = min(1.0, entry["weights"][k] + boost)
    entry["counts"] = entry.get("counts", 0.0) + boost