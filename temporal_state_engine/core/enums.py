# core/enums.py

from enum import Enum


class StateLayer(Enum):
    """
    Temporal state classification for incidents
    """
    ST = "short_term"      # 0-14 days, transient
    MT = "mid_term"        # 2 weeks - 4 months, persistent
    LT = "long_term"       # Permanent baseline impact
    CRISIS = "crisis"      # Special override for safety concerns


class LifeDomain(Enum):
    """
    Five core life domains for impact assessment
    """
    WORK = "work"
    RELATIONSHIPS = "relationships"
    HEALTH = "health"
    IDENTITY = "identity"
    SAFETY = "safety"


class ImpairmentLevel(Enum):
    """
    Functional impairment severity categories
    """
    MINIMAL = "minimal"      # 0.1-1.0: Minor disruption
    MODERATE = "moderate"    # 1.0-2.0: Significant impact
    SEVERE = "severe"        # 2.0-3.0: Major dysfunction


class EmotionalValence(Enum):
    """
    Emotional tone classification
    """
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class TransitionReason(Enum):
    """
    Why an incident changed state
    """
    DECAY = "decay"                    # Natural time-based reduction
    ESCALATION = "escalation"          # User re-mentioned or worsened
    COMPOUNDING = "compounding"        # Multiple ST → MT
    RESURGENCE = "resurgence"          # LT trauma trigger
    USER_SUPPRESSION = "user_suppression"  # User requested deletion
    MANUAL_OVERRIDE = "manual_override"    # Admin intervention


class ExpressionStyle(Enum):
    """
    User's emotional expression pattern for calibration
    """
    STOIC = "stoic"            # Understated (inflate scores by 30%)
    NEUTRAL = "neutral"        # Balanced (no adjustment)
    EXPRESSIVE = "expressive"  # Hyperbolic (deflate scores by 20%)


class ConfidenceLevel(Enum):
    """
    Categorized confidence levels
    """
    VERY_LOW = "very_low"      # 0.0-0.40
    LOW = "low"                # 0.40-0.65 (needs clarification)
    MEDIUM = "medium"          # 0.65-0.80
    HIGH = "high"              # 0.80-0.90
    VERY_HIGH = "very_high"    # 0.90-1.00


class ClarificationDimension(Enum):
    """
    Which extraction dimension needs clarification
    """
    TEMPORAL = "temporal"
    EMOTIONAL = "emotional"
    FUNCTIONAL = "functional"
    SIGNAL_AGREEMENT = "signal_agreement"


class EmpathyLevel(Enum):
    """
    Response tone recommendation for main pipeline
    """
    LIGHT = "light"                    # Casual, supportive
    MODERATE = "moderate"              # Attentive, validating
    HIGH = "high"                      # Deeply empathetic, cautious
    CRISIS = "crisis"                  # Immediate safety protocol


class DecayModel(Enum):
    """
    Mathematical decay function type
    """
    EXPONENTIAL = "exponential"        # ST layer
    SIGMOID = "sigmoid"                # MT layer
    ASYMPTOTIC = "asymptotic"          # LT layer