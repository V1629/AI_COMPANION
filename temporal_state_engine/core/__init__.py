# core/__init__.py

"""
Core module for Temporal State Engine
Provides foundational models, enums, and configuration
"""

from .enums import (
    StateLayer,
    LifeDomain,
    ImpairmentLevel,
    EmotionalValence,
    TransitionReason,
    ExpressionStyle,
    ConfidenceLevel,
    ClarificationDimension,
    EmpathyLevel,
    DecayModel
)

from .models import (
    # Extraction models
    RawSignals,
    ConfidenceMetrics,
    ExtractionResult,
    
    # Scoring models
    PRISMScore,
    StateClassification,
    
    # Incident models
    Incident,
    
    # State management models
    StateTransition,
    CompoundingEvent,
    ResurgenceEvent,
    DecaySnapshot,
    
    # User baseline
    UserBaseline,
    
    # Context export
    TemporalContext,
    EmpathyFlags,
    
    # Query models
    IncidentQuery,
    SimilarityQuery
)

from .config import (
    # Thresholds
    ST_THRESHOLD,
    MT_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    
    # Decay constants
    ST_DECAY_LAMBDA,
    MT_HALF_LIFE_DAYS,
    LT_DECAY_MU,
    
    # Compounding
    ST_COMPOUND_THRESHOLD,
    ST_COMPOUND_WINDOW_DAYS,
    
    # Life domains
    LIFE_DOMAINS,
    
    # PRISM ranges
    PERSISTENCE_RANGE,
    RESONANCE_RANGE,
    IMPACT_RANGE,
    SEVERITY_RANGE,
    MALLEABILITY_RANGE,
    
    # Storage
    MONGODB_URI,
    MONGODB_DATABASE,
    MONGODB_COLLECTIONS,
    REDIS_HOST,
    REDIS_PORT,
    
    # Crisis
    CRISIS_PATTERNS,
    ENABLE_CRISIS_DETECTION,
    
    # Empathy
    EMPATHY_LEVEL_THRESHOLDS,
    TONE_RECOMMENDATIONS
)

__version__ = '0.1.0'

__all__ = [
    # Enums
    'StateLayer',
    'LifeDomain',
    'ImpairmentLevel',
    'EmotionalValence',
    'TransitionReason',
    'ExpressionStyle',
    'ConfidenceLevel',
    'ClarificationDimension',
    'EmpathyLevel',
    'DecayModel',
    
    # Models
    'RawSignals',
    'ConfidenceMetrics',
    'ExtractionResult',
    'PRISMScore',
    'StateClassification',
    'Incident',
    'StateTransition',
    'CompoundingEvent',
    'ResurgenceEvent',
    'DecaySnapshot',
    'UserBaseline',
    'TemporalContext',
    'EmpathyFlags',
    'IncidentQuery',
    'SimilarityQuery',
    
    # Config values
    'ST_THRESHOLD',
    'MT_THRESHOLD',
    'CONFIDENCE_THRESHOLD',
    'ST_DECAY_LAMBDA',
    'MT_HALF_LIFE_DAYS',
    'LT_DECAY_MU',
    'ST_COMPOUND_THRESHOLD',
    'ST_COMPOUND_WINDOW_DAYS',
    'LIFE_DOMAINS',
    'PERSISTENCE_RANGE',
    'RESONANCE_RANGE',
    'IMPACT_RANGE',
    'SEVERITY_RANGE',
    'MALLEABILITY_RANGE',
    'MONGODB_URI',
    'MONGODB_DATABASE',
    'MONGODB_COLLECTIONS',
    'REDIS_HOST',
    'REDIS_PORT',
    'CRISIS_PATTERNS',
    'ENABLE_CRISIS_DETECTION',
    'EMPATHY_LEVEL_THRESHOLDS',
    'TONE_RECOMMENDATIONS',
]