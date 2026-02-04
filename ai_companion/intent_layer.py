"""
Novel Intent Recognition layer built from primitives.
Implements phrase-to-context graphs, emotional probability estimation,
personality filters, ambiguity resolution and urgency classification.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import math
import time


def create_phrase_history_graph(phrase_records: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
    """
    Build a weighted adjacency map where nodes are phrases and edges denote co-occurrence/context transitions.
    phrase_records: list of (phrase, metadata) - metadata may contain 'ts' and 'context_id'
    Returns adjacency dict: {phrase: {other_phrase: weight}}
    """
    graph = {}
    window_seconds = 3600 * 24  # consider same-day proximity as stronger
    for phrase, meta in phrase_records:
        if phrase not in graph:
            graph[phrase] = {}
    # pairwise co-occurrence weights
    for i, (p1, m1) in enumerate(phrase_records):
        for j in range(i + 1, min(len(phrase_records), i + 20)):  # local window to keep graph sparse
            p2, m2 = phrase_records[j]
            dt = abs((m1.get("ts", 0) - m2.get("ts", 0)))
            weight = 1.0 / (1 + math.log1p(dt / window_seconds))
            graph.setdefault(p1, {})[p2] = graph.get(p1, {}).get(p2, 0.0) + weight
            graph.setdefault(p2, {})[p1] = graph.get(p2, {}).get(p1, 0.0) + weight
    # normalize row-wise
    for p, nbrs in graph.items():
        s = sum(nbrs.values()) + 1e-9
        for q in list(nbrs.keys()):
            nbrs[q] = nbrs[q] / s
    return graph


def calculate_emotional_probability_distribution(phrase: str, context_embedding: np.ndarray, known_mappings: Dict[str, List[Dict[str, Any]]],
                                                 embedder) -> Dict[str, float]:
    """
    For a given phrase (e.g., "I'm fine") compute probability distribution over emotional categories
    by combining historical contexts, embedding similarity and rule-based priors in known_mappings.
    known_mappings: {phrase: [ { 'embedding': np.ndarray, 'emotions': {..}, 'behavior': {...}, 'ts': ... }, ... ] }
    embedder: object providing embed(text) method
    """
    categories = ["genuinely_fine", "deflecting", "overwhelmed", "needing_support"]
    priors = {c: 0.25 for c in categories}
    historical = known_mappings.get(phrase.lower(), [])
    if not historical:
        return priors
    sims = []
    for rec in historical:
        emb = rec.get("embedding")
        if emb is None:
            emb = embedder.embed(rec.get("context", ""))
        sim = float(np.dot(context_embedding, emb) / ((np.linalg.norm(context_embedding) + 1e-9) * (np.linalg.norm(emb) + 1e-9)))
        sims.append((sim, rec))
    sims.sort(key=lambda x: x[0], reverse=True)
    # use top-N ensemble
    topk = sims[:10]
    weights = np.array([s for s, r in topk], dtype=float)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    agg = {c: 0.0 for c in categories}
    for w, (_, rec) in zip(weights, topk):
        em = rec.get("emotions", {})
        for c in categories:
            agg[c] += w * em.get(c, 0.0)
    # blend with priors
    blended = {c: 0.7 * agg.get(c, 0.0) + 0.3 * priors[c] for c in categories}
    # renormalize
    s = sum(blended.values()) + 1e-9
    return {c: float(blended[c] / s) for c in categories}


def detect_personality_filters(behavioral: Dict[str, Any], personality_profile: Dict[str, float]) -> Dict[str, float]:
    """
    Detect communication masks like 'formal', 'sarcastic_tendency', 'short_reply' by combining
    behavioral cues and personality model outputs.
    Returns filter strengths in [0,1].
    """
    filters = {"formal": 0.0, "sarcastic_tendency": 0.0, "short_reply": 0.0}
    if behavioral.get("avg_word_len", 0) > 5.0 and behavioral.get("punctuation_density", 0) > 0.05:
        filters["formal"] += 0.6
    if behavioral.get("emojis", 0) == 0 and personality_profile.get("humor", 0) > 0.6:
        filters["sarcastic_tendency"] += 0.4
    if behavioral.get("words", 0) < 5:
        filters["short_reply"] += 0.8
    # personality influences
    filters["sarcastic_tendency"] = min(1.0, filters["sarcastic_tendency"] + personality_profile.get("neuroticism", 0) * 0.1)
    return filters


def resolve_ambiguity(prob_dist: Dict[str, float], additional_signals: Dict[str, Any]) -> Tuple[str, float]:
    """
    Pick most likely intent from probabilistic distribution but adjust with signals (e.g., urgency).
    Returns (intent_label, confidence).
    """
    # adjust probabilities using signals
    adjusted = prob_dist.copy()
    if additional_signals.get("recent_stress_spike"):
        # boost overwhelmed and needing_support
        adjusted["overwhelmed"] = adjusted.get("overwhelmed", 0) + 0.2
        adjusted["needing_support"] = adjusted.get("needing_support", 0) + 0.15
    # normalize
    s = sum(adjusted.values()) + 1e-9
    for k in adjusted:
        adjusted[k] = adjusted[k] / s
    intent = max(adjusted.items(), key=lambda x: x[1])
    return intent


def classify_urgency_level(behavioral: Dict[str, Any], emotional_probs: Dict[str, float], metadata: Dict[str, Any]) -> str:
    """
    Determine urgency: 'low', 'medium', 'high', 'immediate'. Heuristics based on deviations and emotional probs.
    """
    if emotional_probs.get("needing_support", 0) > 0.6 or emotional_probs.get("overwhelmed", 0) > 0.6:
        if behavioral.get("typing_speed_cps", 0) > 5:
            return "immediate"
        return "high"
    if emotional_probs.get("genuinely_fine", 0) > 0.8:
        return "low"
    if behavioral.get("words", 0) < 3 and metadata.get("mentions") > 0:
        return "medium"
    return "low"