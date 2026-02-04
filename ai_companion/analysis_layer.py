"""
Analysis Layer: behavioral and semantic analysis functions.
Includes feature extraction for typing/structure, semantic parsing using custom embeddings,
emotion marker detection using rule-based and statistical cues, and baseline comparison.
"""
from typing import Dict, Any, List
import numpy as np
import re
import time
from .embeddings import CharNGramHasher


def extract_behavioral_features(message: str, typing_time_ms: int, sent_ts: float = None) -> Dict[str, Any]:
    """
    Extract behavioral features such as length, punctuation density, emoji usage,
    capitalization ratios and typing speed estimate.
    """
    if sent_ts is None:
        sent_ts = time.time()
    tokens = message.split()
    length = len(message)
    words = len(tokens)
    avg_word_len = sum(len(t) for t in tokens) / (words + 1e-9)
    punctuation = len([c for c in message if c in "!?.,;:"])
    punctuation_density = punctuation / (length + 1e-9)
    emojis = len(re.findall(r'[\U0001F300-\U0001F6FF\U0001F600-\U0001F64F]', message))
    caps = sum(1 for c in message if c.isupper())
    caps_ratio = caps / (length + 1e-9)
    typing_speed_cps = (len(message) / (typing_time_ms / 1000 + 1e-9)) if typing_time_ms > 0 else 0.0
    features = {
        "length": length,
        "words": words,
        "avg_word_len": avg_word_len,
        "punctuation": punctuation,
        "punctuation_density": punctuation_density,
        "emojis": emojis,
        "caps_ratio": caps_ratio,
        "typing_speed_cps": typing_speed_cps,
        "ts": sent_ts,
    }
    return features


def analyze_semantic_content(message: str, embedder: CharNGramHasher) -> Dict[str, Any]:
    """
    Produce embedding and simple lexical signals: negation count, sentiment cues (rule-based),
    topical keywords (freq-based).
    """
    emb = embedder.embed(message)
    negations = len(re.findall(r"\b(no|not|never|n't|none)\b", message.lower()))
    positives = len(re.findall(r"\b(yes|good|great|happy|love|fantastic|awesome)\b", message.lower()))
    negatives = len(re.findall(r"\b(bad|sad|angry|hate|terrible|awful)\b", message.lower()))
    exclam = message.count("!")
    question = message.count("?")
    lexical_density = len(set(message.split())) / (len(message.split()) + 1e-9)
    return {
        "embedding": emb,
        "negations": negations,
        "positives": positives,
        "negatives": negatives,
        "exclam": exclam,
        "question": question,
        "lexical_density": lexical_density,
    }


def detect_emotional_markers(semantic: Dict[str, Any], behavioral: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, float]:
    """
    Combine lexical cues and behavior to return a probability-like score for states:
    calm, stressed, sad, sarcastic, positive.
    Uses a small heuristic ensemble of signals and comparisons to baselines.
    """
    scores = {"calm": 0.1, "stressed": 0.0, "sad": 0.0, "sarcastic": 0.0, "positive": 0.0}
    # lexical cues
    scores["positive"] += 0.3 * min(1.0, semantic.get("positives", 0))
    scores["sad"] += 0.4 * min(1.0, semantic.get("negatives", 0))
    scores["stressed"] += 0.2 * semantic.get("negations", 0)
    # punctuation and behavior
    if behavioral.get("typing_speed_cps", 0) > baseline.get("typing_speed_cps", 0) * 1.5:
        scores["stressed"] += 0.2
    if behavioral.get("punctuation_density", 0) > baseline.get("punctuation_density", 0) + 0.05:
        scores["stressed"] += 0.15
    if behavioral.get("emojis", 0) > 0 and semantic.get("positives", 0) > semantic.get("negatives", 0):
        scores["positive"] += 0.2
    # sarcasm heuristic: positive words + negative punctuation (e.g., "great...") or short "I'm fine" cases are handled elsewhere
    if semantic.get("positives", 0) > 0 and semantic.get("negatives", 0) > 0:
        scores["sarcastic"] += 0.2
    # normalize to probabilistic distribution
    total = sum(max(0.0, v) for v in scores.values()) + 1e-9
    for k in scores:
        scores[k] = float(max(0.0, scores[k]) / total)
    return scores


def build_contextual_embedding(message: str, user_history_embeddings: List[np.ndarray], embedder: CharNGramHasher,
                               recency_weight: float = 0.7) -> np.ndarray:
    """
    Combine current message embedding with a recency-weighted average of recent history embeddings.
    This yields a multi-dimensional contextual vector representing both local and historical context.
    """
    cur = embedder.embed(message)
    if not user_history_embeddings:
        return cur
    # recency-weighted average: assume user_history_embeddings ordered oldest->newest
    weights = np.array([recency_weight ** (len(user_history_embeddings) - 1 - i) for i in range(len(user_history_embeddings))], dtype=float)
    weights = weights / (weights.sum() + 1e-9)
    history_mean = np.sum(np.stack(user_history_embeddings, axis=0) * weights[:, None], axis=0)
    # combine current and history with simple interpolation
    combined = 0.6 * cur + 0.4 * history_mean
    norm = np.linalg.norm(combined) + 1e-9
    return combined / norm

def compare_to_baseline(current_features: Dict[str, Any], baseline: Dict[str, Any], thresholds: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Compare features to baseline and return deviations and flags for significant change.
    thresholds is optional dict controlling sensitivity.
    """
    if thresholds is None:
        thresholds = {"typing_speed_cps": 0.4, "punctuation_density": 0.05, "emojis": 1}
    deviations = {}
    flags = {}
    for k, v in current_features.items():
        base = baseline.get(k, None)
        if base is None:
            deviations[k] = 0.0
            flags[k] = False
            continue
        diff = v - base
        deviations[k] = diff
        if k in thresholds:
            flags[k] = abs(diff) > thresholds[k]
        else:
            flags[k] = abs(diff) > (abs(base) * 0.2 + 1e-9)
    return {"deviations": deviations, "flags": flags}