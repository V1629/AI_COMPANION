# core/models.py

from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, validator, root_validator
from .enums import (
    StateLayer, LifeDomain, ImpairmentLevel, EmotionalValence,
    TransitionReason, ExpressionStyle, ConfidenceLevel,
    ClarificationDimension, EmpathyLevel, DecayModel
)


# ==================== EXTRACTION MODELS ====================

class RawSignals(BaseModel):
    """
    Raw extracted signals before fusion
    Used for debugging and audit trails
    """
    lexical_score: float = Field(..., ge=0.0, le=10.0)
    temporal_score: float = Field(..., ge=0.1, le=10.0)
    functional_indicators: Dict[str, Any]
    calibrated_baseline: Dict[str, Any]
    extraction_metadata: Dict[str, Any]


class ConfidenceMetrics(BaseModel):
    """
    Detailed confidence breakdown from ConfidenceScorer
    """
    signal_agreement: float = Field(..., ge=0.0, le=1.0)
    data_completeness: float = Field(..., ge=0.0, le=1.0)
    temporal_certainty: float = Field(..., ge=0.0, le=1.0)
    emotional_clarity: float = Field(..., ge=0.0, le=1.0)
    functional_clarity: float = Field(..., ge=0.0, le=1.0)
    historical_depth: float = Field(..., ge=0.0, le=1.0)
    ambiguity_penalty: float = Field(..., ge=0.0, le=1.0)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Categorize overall confidence"""
        if self.overall_confidence >= 0.90:
            return ConfidenceLevel.VERY_HIGH
        elif self.overall_confidence >= 0.80:
            return ConfidenceLevel.HIGH
        elif self.overall_confidence >= 0.65:
            return ConfidenceLevel.MEDIUM
        elif self.overall_confidence >= 0.40:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class ExtractionResult(BaseModel):
    """
    Output from extraction/signal_extractor.py
    Contains PRISM scores and confidence metrics
    """
    # PRISM components
    persistence: float = Field(..., ge=0.1, le=10.0)
    resonance: float = Field(..., ge=1.0, le=10.0)
    impact: int = Field(..., ge=1, le=5)
    severity: float = Field(..., ge=0.1, le=3.0)
    malleability: float = Field(..., ge=0.5, le=2.0)
    
    # Confidence tracking
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_metrics: Optional[ConfidenceMetrics] = None
    
    # Clarification handling
    requires_clarification: bool = False
    clarification_question: Optional[str] = None
    clarification_dimension: Optional[ClarificationDimension] = None
    
    # Raw data for audit
    raw_signals: Optional[RawSignals] = None
    partial_signals: Optional[Dict[str, float]] = None
    
    # Metadata
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc)
)
    
    @validator('confidence')
    def validate_confidence_threshold(cls, v, values):
        """Ensure clarification flag matches confidence"""
        from .config import CONFIDENCE_THRESHOLD
        if v < CONFIDENCE_THRESHOLD and not values.get('requires_clarification'):
            raise ValueError(f"Confidence {v} below threshold but clarification not flagged")
        return v


# ==================== SCORING MODELS ====================

class PRISMScore(BaseModel):
    """
    Calculated PRISM significance score
    """
    persistence: float
    resonance: float
    impact: int
    severity: float
    malleability: float
    significance_score: float
    calculated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @validator('significance_score')
    def validate_score_calculation(cls, v, values):
        """Verify score matches formula"""
        expected = (
            values.get('persistence', 0) *
            values.get('resonance', 0) *
            values.get('impact', 0) *
            values.get('severity', 0)
        ) / values.get('malleability', 1)
        
        if abs(v - expected) > 0.01:  # Allow small floating point error
            raise ValueError(f"Score {v} doesn't match formula result {expected}")
        return v


class StateClassification(BaseModel):
    """
    State classification result
    """
    state_layer: StateLayer
    significance_score: float
    confidence: float
    classified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    classification_reason: Optional[str] = None


# ==================== INCIDENT MODELS ====================

class Incident(BaseModel):
    """
    Core incident model - represents a single life event
    Stored in MongoDB incidents collection
    """
    # Identifiers
    incident_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    
    # Timestamps
    from datetime import datetime, timezone

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_mentioned_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    
    # State
    state_layer: StateLayer
    previous_state: Optional[StateLayer] = None
    
    # PRISM scores
    persistence: float = Field(..., ge=0.1, le=10.0)
    resonance: float = Field(..., ge=1.0, le=10.0)
    impact: int = Field(..., ge=1, le=5)
    severity: float = Field(..., ge=0.1, le=3.0)
    malleability: float = Field(..., ge=0.5, le=2.0)
    significance_score: float = Field(..., ge=0.0)
    
    # Current relevance (decay-adjusted)
    initial_significance: float  # Never changes
    current_relevance: float     # Recalculated by decay_engine
    
    # Decay parameters
    decay_model: DecayModel
    decay_params: Dict[str, float]  # Model-specific parameters
    
    # Content
    description: str = Field(..., min_length=1)
    original_message: str
    affected_domains: List[LifeDomain]
    impairment_level: ImpairmentLevel
    emotional_valence: EmotionalValence
    
    # Metadata
    mention_count: int = Field(default=1, ge=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    user_suppressed: bool = False
    active_weight_multiplier: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Relationships
    related_incident_ids: List[str] = Field(default_factory=list)
    triggered_by_incident_id: Optional[str] = None
    
    class Config:
        use_enum_values = True
    
    @root_validator
    def set_decay_model(cls, values):
        """Auto-assign decay model based on state"""
        state = values.get('state_layer')
        if state == StateLayer.ST:
            values['decay_model'] = DecayModel.EXPONENTIAL
        elif state == StateLayer.MT:
            values['decay_model'] = DecayModel.SIGMOID
        elif state == StateLayer.LT:
            values['decay_model'] = DecayModel.ASYMPTOTIC
        return values
    
    def is_active(self) -> bool:
        """Check if incident is currently active"""
        if self.user_suppressed:
            return False
        if self.state_layer == StateLayer.LT:
            return True  # Always active
        if self.current_relevance < 1.0:
            return False
        return True
    
    def days_since_creation(self) -> int:
        """Calculate days since incident was created"""
        return (datetime.now(timezone.utc) - self.created_at).days
    
    def days_since_last_mention(self) -> int:
        """Calculate days since last user mention"""
        return (datetime.now(timezone.utc) - self.last_mentioned_at).days


# ==================== STATE MANAGEMENT MODELS ====================

class StateTransition(BaseModel):
    """
    Records when an incident changes state
    Stored for audit trail
    """
    transition_id: str = Field(..., min_length=1)
    incident_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    
    from_state: StateLayer
    to_state: StateLayer
    transition_reason: TransitionReason
    
    timestamp: datetime = Field(default_factory = lambda: datetime.now(timezone.utc))
    
    # Scores at transition time
    significance_before: float
    significance_after: float
    
    # Context
    triggered_by_mention: bool = False
    manual_override: bool = False
    notes: Optional[str] = None
    
    class Config:
        use_enum_values = True


class CompoundingEvent(BaseModel):
    """
    Tracks when multiple ST incidents compound into MT
    """
    compounding_id: str
    user_id: str
    
    source_incident_ids: List[str] = Field(..., min_items=2)
    resulting_incident_id: str
    
    compound_window_days: int  # Usually 7
    affected_domain: LifeDomain
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc)
)


class ResurgenceEvent(BaseModel):
    """
    Tracks when LT incident resurges due to trigger
    """
    resurgence_id: str
    incident_id: str
    user_id: str
    
    trigger_type: str  # "anniversary", "similar_incident", "user_mention"
    trigger_description: Optional[str]
    
    relevance_before: float
    relevance_after: float
    spike_magnitude: float
    
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc)
)


class DecaySnapshot(BaseModel):
    """
    Point-in-time relevance calculation
    Used for historical tracking
    """
    snapshot_id: str
    incident_id: str
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc)
)
    relevance: float
    days_elapsed: int
    
    decay_parameters_used: Dict[str, float]


# ==================== USER BASELINE MODELS ====================

class UserBaseline(BaseModel):
    """
    User's emotional expression baseline for calibration
    Stored in MongoDB user_baselines collection
    """
    user_id: str = Field(..., min_length=1)
    
    # Expression style
    expression_style: ExpressionStyle
    avg_intensity: float = Field(..., ge=0.0, le=10.0)
    intensity_stddev: float = Field(..., ge=0.0)
    
    # Historical data
    message_count: int = Field(default=0, ge=0)
    incident_count: int = Field(default=0, ge=0)
    
    # Past incidents for similarity matching
    past_incidents: List[str] = Field(default_factory=list)  # incident_ids
    
    # Calibration factors
    calibration_factor: float = Field(default=1.0, ge=0.5, le=1.5)
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_conversation_at: Optional[datetime] = None
    
    class Config:
        use_enum_values = True
    
    def is_cold_start(self) -> bool:
        """Check if we have enough data for calibration"""
        from .config import COLD_START_MESSAGE_THRESHOLD
        return self.message_count < COLD_START_MESSAGE_THRESHOLD


# ==================== CONTEXT EXPORT MODELS ====================

class TemporalContext(BaseModel):
    """
    Final output from context_export/ to main pipeline
    """
    user_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc)
)
    
    # State distribution
    dominant_state: StateLayer
    state_distribution: Dict[str, float]  # {"ST": 0.15, "MT": 0.60, "LT": 0.25}
    
    # Response guidance
    empathy_level: EmpathyLevel
    tone_recommendation: str
    
    # Active incidents
    active_incidents: List[Dict[str, Any]]
    total_active_incidents: int
    
    # Special flags
    special_flags: Dict[str, bool] = Field(default_factory=dict)
    # Examples: "acknowledge_ongoing_struggles", "extra_sensitivity_required"
    
    # Warnings
    crisis_detected: bool = False
    crisis_incident_id: Optional[str] = None
    
    class Config:
        use_enum_values = True


class EmpathyFlags(BaseModel):
    """
    Specific guidance flags for response generation
    """
    acknowledge_ongoing_struggles: bool = False
    avoid_toxic_positivity: bool = False
    extra_sensitivity_required: bool = False
    avoid_deep_probing: bool = False
    validate_emotions: bool = True
    
    crisis_protocol: bool = False
    crisis_resources_needed: bool = False


# ==================== QUERY MODELS ====================

class IncidentQuery(BaseModel):
    """
    Query parameters for retrieving incidents
    """
    user_id: str
    state_layers: Optional[List[StateLayer]] = None
    min_relevance: float = 0.0
    include_suppressed: bool = False
    limit: int = Field(default=10, ge=1, le=100)
    
    # Time filters
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    last_mentioned_after: Optional[datetime] = None
    
    # Domain filters
    affected_domains: Optional[List[LifeDomain]] = None


class SimilarityQuery(BaseModel):
    """
    Query for finding similar past incidents
    """
    user_id: str
    query_text: str
    min_similarity_score: float = Field(default=0.7, ge=0.0, le=1.0)
    limit: int = Field(default=5, ge=1, le=20)
    exclude_incident_ids: List[str] = Field(default_factory=list)