"""
Response generation layer that adapts tone, empathy, references memories and validates appropriateness.
This uses templates combined with dynamic inserts; it is lightweight and explainable.
"""
from typing import Dict, Any, List
import random
import time


def select_response_strategy(intent: str, urgency: str, personality_profile: Dict[str, float], history_flags: Dict[str, bool]) -> Dict[str, Any]:
    """
    Select a strategy dict that controls tone, verbosity, memory usage, and action suggestions.
    """
    strategy = {"tone": "neutral", "verbosity": "short", "use_memory": True, "suggest_action": False}
    if intent in ("needing_support", "overwhelmed"):
        strategy["tone"] = "empathetic"
        strategy["verbosity"] = "medium"
        strategy["suggest_action"] = True
    if urgency in ("immediate", "high"):
        strategy["verbosity"] = "concise"
    # adapt to personality: mirror user's optimism/humor if safe
    if personality_profile.get("optimism", 0) > 0.6:
        strategy["tone"] = "upbeat" if strategy["tone"] == "neutral" else strategy["tone"]
    if history_flags.get("prefers_direct"):
        strategy["verbosity"] = "short"
    return strategy


def generate_contextually_aware_response(message: str, intent: str, strategy: Dict[str, Any], memory_snippets: List[str]) -> str:
    """
    Build a textual response that references memory snippets where relevant and follows the chosen strategy.
    """
    intro = ""
    if strategy["tone"] == "empathetic":
        intro = random.choice(["I'm sorry you're going through that.", "That sounds tough — I'm here for you."])
    elif strategy["tone"] == "upbeat":
        intro = random.choice(["Hey — good to hear from you!", "Nice to see your message."])
    else:
        intro = random.choice(["Okay.", "I hear you."])
    body = ""
    if intent == "genuinely_fine":
        body = random.choice(["Glad to hear that. Want to share more?", "That's good — anything you'd like to chat about?"])
    elif intent == "deflecting":
        body = "I notice you might be deflecting—I'm here if you want to dig deeper."
    elif intent == "overwhelmed":
        body = "It sounds like things are a lot right now. Would you like a breathing exercise or to talk it through?"
    elif intent == "needing_support":
        body = "I'm here with you. Tell me what's most pressing right now."
    # integrate memory
    mem = ""
    if strategy["use_memory"] and memory_snippets:
        mem = f"Previously you mentioned: \"{memory_snippets[0]}\". "
    # verbosity control
    if strategy["verbosity"] == "short":
        resp = f"{intro} {body}"
    elif strategy["verbosity"] == "concise":
        resp = f"{intro} {body}"
    else:
        resp = f"{intro} {mem}{body} Here to help as much or as little as you prefer."
    return resp.strip()


def adjust_empathy_calibration(emotional_probs: Dict[str, float], user_preference: Dict[str, Any]) -> float:
    """
    Return an empathy multiplier in [0,2] controlling how strong empathetic phrasing should be.
    """
    base = 1.0
    support = emotional_probs.get("needing_support", 0) + emotional_probs.get("overwhelmed", 0)
    multiplier = base + support * 1.0
    # respect user preference to avoid over-empathizing
    if user_preference.get("prefers_direct"):
        multiplier *= 0.6
    return float(min(2.0, multiplier))


def incorporate_memory_references(relevant_history: List[Dict[str, Any]]) -> List[str]:
    """
    Return a short list of memory snippets (strings) prioritized by recency and similarity.
    """
    snippets = []
    for h in relevant_history[:3]:
        snippets.append(h.get("message", "")[:200])
    return snippets


def validate_response_appropriateness(response: str, sensitive_topics: List[str], user_profile: Dict[str, Any]) -> bool:
    """
    Simple validators: avoid medical/legal definitive advice, detect risky suggestions, and respect user blocks.
    """
    lower = response.lower()
    for t in sensitive_topics:
        if t in lower:
            return False
    # check for user-specific bans
    if user_profile.get("no_phrases"):
        for p in user_profile.get("no_phrases", []):
            if p.lower() in lower:
                return False
    return True