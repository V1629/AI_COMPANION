# core/config.py

from typing import Dict, List
import os

# ==================== STATE THRESHOLDS ====================

# PRISM score thresholds for state classification
ST_THRESHOLD = 15.0
MT_THRESHOLD = 75.0

# Confidence threshold for clarification
CONFIDENCE_THRESHOLD = 0.65

# User baseline calibration
COLD_START_MESSAGE_THRESHOLD = 5  # Need 5 messages before calibration


# ==================== DECAY CONSTANTS ====================

# Short-Term (Exponential Decay)
ST_DECAY_LAMBDA = 0.3              # ~70% reduction every 2 days
ST_MAX_LIFETIME_DAYS = 14          # Auto-delete after 14 days

# Mid-Term (S-Curve Decay)
MT_HALF_LIFE_DAYS = 60             # Default half-life
MT_DECAY_STEEPNESS_MIN = 0.1       # k parameter range
MT_DECAY_STEEPNESS_MAX = 0.5
MT_MAX_LIFETIME_DAYS = 120         # Demote to ST threshold after this

# Long-Term (Asymptotic Decay)
LT_DECAY_MU = 0.001                # Very slow decay
LT_BASELINE_FLOOR = 30.0           # Never decays below this relevance
LT_CHRONIC_BASELINE = 50.0         # For chronic conditions


# ==================== COMPOUNDING RULES ====================

# ST → MT escalation
ST_COMPOUND_THRESHOLD = 3          # Number of ST incidents needed
ST_COMPOUND_WINDOW_DAYS = 7        # Within this time window
ST_COMPOUND_SAME_DOMAIN = True     # Must be same domain?

# MT → LT escalation
MT_TO_LT_MENTION_COUNT = 5         # Mentioned 5+ times
MT_TO_LT_MIN_DURATION_DAYS = 60    # Over at least 60 days


# ==================== RESURGENCE PARAMETERS ====================

# LT trauma resurgence
RESURGENCE_ANNIVERSARY_WINDOW_DAYS = 7  # ±7 days around anniversary
RESURGENCE_SPIKE_MULTIPLIER = 1.5       # 50% relevance boost
RESURGENCE_SPIKE_DECAY_DAYS = 14        # Spike lasts 2 weeks


# ==================== LIFE DOMAINS ====================

LIFE_DOMAINS: Dict[str, List[str]] = {
    'work': [
        'job', 'work', 'career', 'boss', 'coworker', 'colleague',
        'fired', 'laid off', 'quit', 'resign', 'promotion', 'demotion',
        'deadline', 'project', 'office', 'meeting', 'performance review',
        'salary', 'pay', 'wage', 'contract', 'unemployment', 'interview'
    ],
    
    'relationships': [
        'partner', 'spouse', 'husband', 'wife', 'boyfriend', 'girlfriend',
        'breakup', 'divorce', 'separated', 'ex', 'dating', 'marriage',
        'friend', 'friendship', 'family', 'parent', 'sibling', 'child',
        'alone', 'lonely', 'isolated', 'abandoned', 'rejected',
        'fight', 'argument', 'conflict', 'cheating', 'betrayal'
    ],
    
    'health': [
        'sick', 'illness', 'disease', 'pain', 'hurt', 'ache',
        'doctor', 'hospital', 'clinic', 'emergency', 'surgery',
        'diagnosis', 'diagnosed', 'symptoms', 'treatment', 'medication',
        'injury', 'accident', 'chronic', 'condition', 'disability',
        'sleep', 'insomnia', 'tired', 'exhausted', 'fatigue',
        'appetite', 'eating', 'weight', 'exercise', 'fitness'
    ],
    
    'identity': [
        'who i am', 'myself', 'my identity', 'lost myself', 'not myself',
        'failure', 'failed', 'worthless', 'useless', 'inadequate',
        'not good enough', 'disappointing', 'shame', 'embarrassed',
        'my purpose', 'my value', 'my worth', 'define me', 'defines me',
        'proud', 'ashamed', 'confident', 'insecure', 'self-esteem'
    ],
    
    'safety': [
        'danger', 'dangerous', 'threat', 'threatened', 'unsafe',
        'scared', 'afraid', 'fear', 'terrified', 'frightened',
        'trauma', 'traumatic', 'ptsd', 'flashback', 'nightmare',
        'abuse', 'abused', 'violence', 'violent', 'attacked',
        'assault', 'harassed', 'stalked', 'manipulated',
        'worried about safety', 'fear for', 'in danger'
    ]
}


# ==================== PRISM COMPONENT RANGES ====================

# Persistence (P): Expected duration
PERSISTENCE_RANGE = (0.1, 10.0)
PERSISTENCE_DEFAULT = 0.1

# Resonance (R): Emotional intensity
RESONANCE_RANGE = (1.0, 10.0)
RESONANCE_DEFAULT = 5.0

# Impact (I): Life domain breadth
IMPACT_RANGE = (1, 5)
IMPACT_DEFAULT = 1

# Severity (S): Functional impairment
SEVERITY_RANGE = (0.1, 3.0)
SEVERITY_DEFAULT = 0.1

# Malleability (M): Perceived control
MALLEABILITY_RANGE = (0.5, 2.0)
MALLEABILITY_DEFAULT = 1.0


# ==================== CONFIDENCE WEIGHTS ====================

CONFIDENCE_WEIGHTS = {
    'signal_agreement': 0.25,
    'data_completeness': 0.20,
    'temporal_certainty': 0.15,
    'emotional_clarity': 0.15,
    'functional_clarity': 0.10,
    'historical_depth': 0.10,
    'ambiguity_penalty': 0.05
}


# ==================== CALIBRATION FACTORS ====================

# Expression style adjustments
EXPRESSION_CALIBRATION = {
    'stoic': 1.3,        # Inflate scores by 30%
    'neutral': 1.0,      # No adjustment
    'expressive': 0.8    # Deflate scores by 20%
}


# ==================== STORAGE CONFIGURATION ====================

# Redis (ST cache)
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)
REDIS_ST_TTL_SECONDS = ST_MAX_LIFETIME_DAYS * 24 * 60 * 60  # 14 days

# MongoDB (MT/LT storage)
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'temporal_state_engine')

# Collections
MONGODB_COLLECTIONS = {
    'incidents': 'incidents',
    'state_transitions': 'state_transitions',
    'user_baselines': 'user_baselines',
    'event_graph': 'event_graph',
    'embeddings': 'embeddings',
    'compounding_events': 'compounding_events',
    'resurgence_events': 'resurgence_events'
}


# ==================== EMPATHY MAPPING ====================

# Map state distributions to empathy levels
EMPATHY_LEVEL_THRESHOLDS = {
    'crisis': {'crisis_weight': 1.0},      # Any crisis incident
    'high': {'LT_weight': 0.5},            # LT represents 50%+ of total
    'moderate': {'MT_weight': 0.4},        # MT represents 40%+ of total
    'light': {'ST_weight': 0.6}            # ST represents 60%+ of total
}

TONE_RECOMMENDATIONS = {
    'crisis': 'immediate_safety_protocol',
    'high': 'deeply_empathetic_cautious',
    'moderate': 'attentive_validating',
    'light': 'casual_supportive'
}


# ==================== CRISIS DETECTION ====================

# Suicide risk patterns (case-insensitive regex)
CRISIS_PATTERNS = [
    r'\b(want to die|end it all|no reason to live)\b',
    r'\b(suicide plan|kill myself|end my life)\b',
    r'\b(saying goodbye|giving away|final goodbye)\b',
    r'\b(better off dead|burden to everyone)\b',
    r'\b(can\'?t go on|can\'?t take it anymore)\b'
]

# Crisis override - always classify as highest priority
CRISIS_OVERRIDE_SIGNIFICANCE = 1000.0


# ==================== QUERY OPTIMIZATION ====================

# Maximum incidents to retrieve per query
MAX_INCIDENTS_PER_QUERY = 100

# Minimum relevance threshold for active incidents
MIN_ACTIVE_RELEVANCE = 1.0

# Vector similarity threshold
MIN_SIMILARITY_SCORE = 0.7


# ==================== BACKGROUND JOBS ====================

# Decay recalculation frequency
DECAY_RECALC_INTERVAL_HOURS = 24  # Daily

# State transition check frequency
TRANSITION_CHECK_INTERVAL_HOURS = 12  # Twice daily

# Cleanup frequency for expired ST incidents
CLEANUP_INTERVAL_HOURS = 6  # Every 6 hours


# ==================== LOGGING ====================

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# PII fields to scrub from logs
PII_FIELDS = [
    'original_message',
    'description',
    'user_id',  # Consider hashing instead
    'clarification_question'
]


# ==================== FEATURE FLAGS ====================

ENABLE_CRISIS_DETECTION = True
ENABLE_COMPOUNDING = True
ENABLE_RESURGENCE_TRACKING = True
ENABLE_VECTOR_SIMILARITY = True
ENABLE_USER_SUPPRESSION = True


# ==================== VALIDATION ====================

def validate_config():
    """
    Validate configuration on module load
    Raises ValueError if configuration is invalid
    """
    assert ST_THRESHOLD > 0, "ST_THRESHOLD must be positive"
    assert MT_THRESHOLD > ST_THRESHOLD, "MT_THRESHOLD must be greater than ST_THRESHOLD"
    assert 0.0 < CONFIDENCE_THRESHOLD < 1.0, "CONFIDENCE_THRESHOLD must be between 0 and 1"
    assert ST_DECAY_LAMBDA > 0, "ST_DECAY_LAMBDA must be positive"
    assert MT_HALF_LIFE_DAYS > 0, "MT_HALF_LIFE_DAYS must be positive"
    assert LT_DECAY_MU > 0, "LT_DECAY_MU must be positive"
    assert ST_COMPOUND_THRESHOLD >= 2, "Need at least 2 incidents to compound"
    
    # Validate domain keywords exist
    for domain, keywords in LIFE_DOMAINS.items():
        assert len(keywords) > 0, f"Domain {domain} has no keywords"
    
    print("✅ Configuration validated successfully")


# Run validation on import
validate_config()