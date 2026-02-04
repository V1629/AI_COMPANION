"""
Flask API wrapper for AI Companion real-time interactions.

Provides:
- GET /health : health check
- POST /message : process incoming message and return assistant response + debug info
- GET /history?user_id=...&top_k=5 : fetch recent similar history entries

Privacy-first: uses local SQLite storage and encryption key per config.yaml.
"""
from flask import Flask, request, jsonify, current_app
import yaml
import time
from typing import Dict, Any
import numpy as np

from .data_layer import initialize_user_profile, store_interaction, retrieve_relevant_history, update_user_model
from .embeddings import CharNGramHasher
from .analysis_layer import extract_behavioral_features, analyze_semantic_content, detect_emotional_markers, build_contextual_embedding, compare_to_baseline
from .intent_layer import calculate_emotional_probability_distribution, resolve_ambiguity, classify_urgency_level, detect_personality_filters
from .response_layer import select_response_strategy, generate_contextually_aware_response, incorporate_memory_references, validate_response_appropriateness, adjust_empathy_calibration
from .learning_layer import evaluate_interaction_outcome, update_phrase_meaning_map, strengthen_successful_patterns, prune_outdated_patterns, adjust_personality_model

import os

def create_app(config_path: str = "config.yaml") -> Flask:
    """
    Create and configure the Flask application.

    Loads config.yaml, ensures DB and encryption key are initialized, and creates
    a shared embedder instance stored on the Flask app object for reuse.
    """
    app = Flask(__name__)
    cfg = yaml.safe_load(open(config_path))
    app.config["CFG"] = cfg
    db_path = cfg["database"]["path"]
    key_path = cfg["encryption"]["key_file"]

    # Ensure user profile exists
    profile = initialize_user_profile(db_path, key_path)
    app.config["LOCAL_USER_ID"] = profile["user_id"]

    # Shared embedder instance
    embedder = CharNGramHasher(dim=cfg["embeddings"]["dim"],
                               ngram_min=cfg["embeddings"]["ngram_min"],
                               ngram_max=cfg["embeddings"]["ngram_max"])
    app.config["EMBEDDER"] = embedder
    app.config["DB_PATH"] = db_path
    app.config["KEY_PATH"] = key_path

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "time": time.time()})

    @app.route("/message", methods=["POST"])
    def message_endpoint():
        """
        Receive a JSON payload:
        {
          "user_id": "<optional - defaults to local_user>",
          "message": "I'm fine",
          "typing_time_ms": 400,
          "metadata": { ... optional metadata ... }
        }

        Returns assistant response, intent, urgency, confidence, and some debug signals.
        """
        payload = request.get_json(force=True)
        if not payload or "message" not in payload:
            return jsonify({"error": "missing 'message' field"}), 400

        user_id = payload.get("user_id", current_app.config["LOCAL_USER_ID"])
        message = payload["message"]
        typing_time_ms = int(payload.get("typing_time_ms", 0))
        metadata = payload.get("metadata", {})

        cfg = current_app.config["CFG"]
        embedder = current_app.config["EMBEDDER"]
        db = current_app.config["DB_PATH"]
        key = current_app.config["KEY_PATH"]

        # 1) Behavioral and semantic analysis
        behavioral = extract_behavioral_features(message, typing_time_ms)
        semantic = analyze_semantic_content(message, embedder)

        # 2) Retrieve relevant history (top-k by embedding similarity)
        history = retrieve_relevant_history(db, key, user_id, semantic["embedding"], top_k=5)
        history_embs = [h.get("embedding", semantic["embedding"]) for h in history if h.get("sim", 0) > 0.0]
        contextual_emb = build_contextual_embedding(message, history_embs, embedder)

        # 3) Baseline compare (pull from stored profile if available)
        # Attempt to read profile baseline safely; if missing use defaults
        try:
            # read profile via DB (decrypt inside data_layer)
            # simple approach: fetch relevant history and use its average as placeholder baseline
            baseline = {"typing_speed_cps": 0.05, "punctuation_density": 0.02}
        except Exception:
            baseline = {"typing_speed_cps": 0.05, "punctuation_density": 0.02}

        deviations = compare_to_baseline(behavioral, baseline)

        # 4) Emotional detection & intent resolution
        emotional_probs = detect_emotional_markers(semantic, behavioral, baseline)

        # Load known phrase mappings (stored inside user profile in DB). We'll fetch user's profile by retrieving history and reading phrase_map field from profile.
        # For simplicity, known_mappings is built from history messages similar to the phrase
        known_mappings: Dict[str, Any] = {}
        # create simple known_mappings entries for exact message text occurrences
        for h in history:
            ph = h.get("message", "").lower()
            known_mappings.setdefault(ph, []).append({
                "embedding": embedder.embed(h.get("message", "")),
                "emotions": h.get("metadata", {}).get("emotions", {}),
                "context": h.get("message", ""),
                "ts": h.get("ts", 0)
            })

        phrase = message.strip()
        emo_dist = calculate_emotional_probability_distribution(phrase, contextual_emb, known_mappings, embedder)
        intent_label, confidence = resolve_ambiguity(emo_dist, {"recent_stress_spike": deviations["flags"].get("typing_speed_cps", False)})

        urgency = classify_urgency_level(behavioral, emo_dist, metadata)

        # 5) Personality filters (lightweight detection). In a full system personality profile would be loaded; fallback defaults provided.
        # We'll attempt to load personality from the user's stored model later; here we assume defaults.
        personality_profile = {"optimism": 0.5, "humor": 0.2, "neuroticism": 0.1}
        personality_filters = detect_personality_filters(behavioral, personality_profile)

        # 6) Generate response
        strategy = select_response_strategy(intent_label, urgency, personality_profile, {"prefers_direct": metadata.get("prefers_direct", False)})
        memory_snippets = incorporate_memory_references(history)
        response_text = generate_contextually_aware_response(message, intent_label, strategy, memory_snippets)
        empathy_multiplier = adjust_empathy_calibration(emo_dist, {"prefers_direct": metadata.get("prefers_direct", False)})

        # 7) Validate
        valid = validate_response_appropriateness(response_text, sensitive_topics=["medical", "prescription"], user_profile={})

        # 8) Persist interaction (for learning)
        emb = semantic["embedding"]
        try:
            iid = store_interaction(db, key, user_id, message, {"intent_guess": intent_label, "confidence": confidence, "emotions": emo_dist}, emb, behavioral)
        except Exception as e:
            current_app.logger.exception("Failed to store interaction: %s", e)
            iid = None

        # 9) Update short-term learning structures stored in profile; keep lightweight updates here
        try:
            # update in-memory profile structures: phrase map and personality approximations
            # In production more robust merges and locking are needed.
            # Fetch profile and update phrase_map (this is simplified)
            # We'll update phrase_map with the detected emo_dist and adjust personality vector slightly
            # Load existing profile and merge
            # Use update_user_model to persist updates
            updates = {
                "last_interaction_ts": time.time(),
                "phrase_map": {phrase.lower(): {"inferred_emotions": emo_dist}},
                "personality": personality_profile
            }
            update_user_model(db, key, user_id, updates)
        except Exception:
            # best-effort; do not fail the request
            current_app.logger.exception("Non-fatal: failed to update profile")

        # 10) Return structured response including explainability info
        payload_out: Dict[str, Any] = {
            "response": response_text,
            "valid": bool(valid),
            "intent": intent_label,
            "confidence": float(confidence),
            "urgency": urgency,
            "emotional_probs": emo_dist,
            "behavioral": behavioral,
            "deviations": deviations,
            "empathy_multiplier": empathy_multiplier,
            "memory_snippets": memory_snippets,
            "interaction_id": iid,
        }
        return jsonify(payload_out)

    @app.route("/history", methods=["GET"])
    def history_endpoint():
        """
        Retrieve cached similar history for the given user_id.
        Query params: user_id (optional), top_k (optional)
        """
        user_id = request.args.get("user_id", current_app.config["LOCAL_USER_ID"])
        top_k = int(request.args.get("top_k", 5))
        # For demonstration, we'll compute an empty query embedding to fetch most recent
        emb = np.zeros(cfg["embeddings"]["dim"], dtype=np.float32)
        items = retrieve_relevant_history(db, key, user_id, emb, top_k=top_k)
        return jsonify({"items": items})

    return app