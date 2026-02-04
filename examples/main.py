"""
Example usage showing end-to-end flow:
- Initialize storage
- Process one incoming message
- Analyze, recognize intent, generate response, update learning
"""
from ai_companion.data_layer import initialize_user_profile, store_interaction, retrieve_relevant_history, update_user_model
from ai_companion.embeddings import CharNGramHasher
from ai_companion.analysis_layer import extract_behavioral_features, analyze_semantic_content, detect_emotional_markers, build_contextual_embedding, compare_to_baseline
from ai_companion.intent_layer import calculate_emotional_probability_distribution, resolve_ambiguity, classify_urgency_level
from ai_companion.response_layer import select_response_strategy, generate_contextually_aware_response, incorporate_memory_references, validate_response_appropriateness, adjust_empathy_calibration
from ai_companion.learning_layer import evaluate_interaction_outcome, update_phrase_meaning_map, strengthen_successful_patterns, prune_outdated_patterns, adjust_personality_model
from ai_companion.personality_nn import SimpleMLP
import numpy as np
import time
import yaml

cfg = yaml.safe_load(open("config.yaml"))

DB = cfg["database"]["path"]
KEY = cfg["encryption"]["key_file"]

def run_example():
    # initialize
    profile = initialize_user_profile(DB, KEY)
    user_id = profile["user_id"]
    embedder = CharNGramHasher(dim=cfg["embeddings"]["dim"], ngram_min=cfg["embeddings"]["ngram_min"], ngram_max=cfg["embeddings"]["ngram_max"])
    # simulate incoming message
    message = "I'm fine."
    typing_time_ms = 400  # example
    behavioral = extract_behavioral_features(message, typing_time_ms)
    semantic = analyze_semantic_content(message, embedder)
    # fetch history embeddings
    history = retrieve_relevant_history(DB, KEY, user_id, semantic["embedding"], top_k=5)
    history_embs = [h.get("embedding", semantic["embedding"]) for h in history if h.get("sim", 0) > 0.0]
    contextual = build_contextual_embedding(message, history_embs, embedder)
    # baseline (from profile) - simplified
    baseline = {"typing_speed_cps": 0.05, "punctuation_density": 0.02}
    deviations = compare_to_baseline(behavioral, baseline)
    emotional_probs = detect_emotional_markers(semantic, behavioral, baseline)
    # known mappings simulated (empty here)
    known_mappings = {}
    emo_dist = calculate_emotional_probability_distribution(message, contextual, known_mappings, embedder)
    intent, confidence = resolve_ambiguity(emo_dist, {"recent_stress_spike": deviations["flags"].get("typing_speed_cps")})
    urgency = classify_urgency_level(behavioral, emo_dist, {})
    personality_profile = profile.get("personality", {"optimism": 0.5, "humor": 0.2, "neuroticism": 0.1})
    strategy = select_response_strategy(intent, urgency, personality_profile, {"prefers_direct": False})
    memory_snippets = incorporate_memory_references(history)
    response = generate_contextually_aware_response(message, intent, strategy, memory_snippets)
    # validate
    ok = validate_response_appropriateness(response, sensitive_topics=["medical", "prescription"], user_profile=profile)
    print("Response:", response, "\nValid:", ok)
    # store interaction for learning
    emb = semantic["embedding"]
    iid = store_interaction(DB, KEY, user_id, message, {"intent_guess": intent, "confidence": confidence}, emb, behavioral)
    # simulate follow-up and learning
    followups = [{"message": "thanks, that helps a bit."}]
    outcome = evaluate_interaction_outcome(response, followups)
    update_phrase_meaning_map(profile.setdefault("phrase_map", {}), message, emo_dist)
    strengthen_successful_patterns(profile.setdefault("phrase_map", {}), message, boost=0.05)
    prune_outdated_patterns(profile.setdefault("phrase_map", {}))
    adjust_personality_model(profile.setdefault("personality", {}), [0.6]*8)
    update_user_model(DB, KEY, user_id, {"phrase_map": profile["phrase_map"], "personality": profile["personality"]})
    return response

if __name__ == "__main__":
    run_example()