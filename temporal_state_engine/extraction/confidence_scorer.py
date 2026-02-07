

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ConfidenceMetrics:
    """Container for all confidence sub-scores"""
    signal_agreement: float  # 0-1: Do all signals point same direction?
    data_completeness: float  # 0-1: How much info do we have?
    temporal_certainty: float  # 0-1: Clear time markers?
    emotional_clarity: float  # 0-1: Clear emotion words?
    functional_clarity: float  # 0-1: Clear life impact?
    historical_depth: float  # 0-1: Enough user history?
    ambiguity_penalty: float  # 0-1: Conflicting/vague language?
    overall_confidence: float  # Final weighted score


class ConfidenceScorer:
    """
    Calculates confidence in the extracted PRISM scores.
    
    High confidence = All signals agree + Clear language + Sufficient data
    Low confidence = Signals conflict + Vague language + Missing info
    
    Threshold: 0.65
    - Below 0.65 → Ask clarifying question
    - Above 0.65 → Proceed with classification
    """
    
    def __init__(self):
        # Weights for each confidence dimension
        self.weights = {
            'signal_agreement': 0.25,      # Most important
            'data_completeness': 0.20,
            'temporal_certainty': 0.15,
            'emotional_clarity': 0.15,
            'functional_clarity': 0.10,
            'historical_depth': 0.10,
            'ambiguity_penalty': 0.05
        }
    
    def calculate(self,lexical_signals: Dict,temporal_signals: Dict, functional_signals: Dict,calibrated_signals: Dict,message: str,user_history_depth: int) -> ConfidenceMetrics:
        """
        Main entry point: Calculate overall confidence score
        
        Args:
            lexical_signals: Output from LexicalAnalyzer
            temporal_signals: Output from TemporalParser
            functional_signals: Output from FunctionalDetector
            calibrated_signals: Output from EmotionalCalibrator
            message: Original user message
            user_history_depth: Number of past conversations
            
        Returns:
            ConfidenceMetrics with all sub-scores and overall score
        """
        
        # 1. Signal Agreement Score
        signal_agreement = self._calculate_signal_agreement(lexical_signals, temporal_signals,functional_signals)
        
        # 2. Data Completeness Score
        data_completeness = self._calculate_data_completeness(lexical_signals,temporal_signals,functional_signals)
        
        # 3. Temporal Certainty
        temporal_certainty = temporal_signals.get('temporal_certainty', 0.3)
        
        # 4. Emotional Clarity
        emotional_clarity = self._calculate_emotional_clarity(lexical_signals)
        
        # 5. Functional Clarity
        functional_clarity = self._calculate_functional_clarity(functional_signals)
        
        # 6. Historical Depth Score
        historical_depth = self._calculate_historical_depth(user_history_depth)
        
        # 7. Ambiguity Penalty
        ambiguity_penalty = self._calculate_ambiguity_penalty(message)
        
        # 8. Compute weighted overall confidence
        overall_confidence = (
            signal_agreement * self.weights['signal_agreement'] +
            data_completeness * self.weights['data_completeness'] +
            temporal_certainty * self.weights['temporal_certainty'] +
            emotional_clarity * self.weights['emotional_clarity'] +
            functional_clarity * self.weights['functional_clarity'] +
            historical_depth * self.weights['historical_depth'] +
            ambiguity_penalty * self.weights['ambiguity_penalty']
        )
        
        return ConfidenceMetrics(
            signal_agreement=signal_agreement,
            data_completeness=data_completeness,
            temporal_certainty=temporal_certainty,
            emotional_clarity=emotional_clarity,
            functional_clarity=functional_clarity,
            historical_depth=historical_depth,
            ambiguity_penalty=ambiguity_penalty,
            overall_confidence=overall_confidence
        )
    
    def _calculate_signal_agreement(self, lexical: Dict , temporal: Dict , functional: Dict) -> float:
        """
        Check if all signals point in the same direction
        
        Logic:
        - High lexical intensity + long temporal + severe functional = HIGH agreement
        - High lexical intensity + short temporal + mild functional = LOW agreement
        
        Method: Calculate variance - low variance = high agreement
        """
        
        # Normalize all scores to 0-1 scale
        normalized_scores = []
        
        # 1. Normalize lexical intensity (0-10 → 0-1)
        lexical_intensity = lexical.get('intensity_level', 5.0) / 10.0
        normalized_scores.append(lexical_intensity)
        
        # 2. Normalize temporal persistence (0-10 → 0-1)
        temporal_persistence = temporal.get('persistence_score', 0.1) / 10.0
        normalized_scores.append(temporal_persistence)
        
        # 3. Normalize functional severity (0-3 → 0-1)
        functional_severity = functional.get('impairment_severity', 0.1) / 3.0
        normalized_scores.append(functional_severity)
        
        # Calculate mean
        mean = sum(normalized_scores) / len(normalized_scores)
        
        # Calculate variance
        variance = sum((score - mean) ** 2 for score in normalized_scores) / len(normalized_scores)
        
        # Convert variance to agreement score
        # Low variance (0.0) → High agreement (1.0)
        # High variance (0.33+) → Low agreement (0.0)
        agreement = max(0.0, 1.0 - (variance * 3))  # Scale variance
        
        return agreement
    
    def _calculate_data_completeness(self,  lexical: Dict, temporal: Dict, functional : Dict) -> float:
        """
        Check how much information we have across all dimensions
        
        Score breakdown:
        - Temporal info present: +0.25
        - Emotional info present: +0.25
        - Functional info present: +0.25
        - Multiple indicators per dimension: +0.25
        """
        
        completeness = 0.0
        
        # 1. Temporal completeness (0.25)
        if temporal.get('time_references'):
            completeness += 0.15
        if temporal.get('is_ongoing'):
            completeness += 0.10
        
        # 2. Emotional completeness (0.25)
        if lexical.get('intensity_keywords'):
            completeness += 0.15
        if lexical.get('sentiment_score') and abs(lexical['sentiment_score']) > 0.3:
            completeness += 0.10
        
        # 3. Functional completeness (0.25)
        if functional.get('affected_domains') and len(functional['affected_domains']) > 0:
            completeness += 0.15
        if functional.get('functional_indicators') and len(functional['functional_indicators']) > 0:
            completeness += 0.10
        
        # 4. Richness bonus (0.25)
        total_indicators = (
            len(lexical.get('intensity_keywords', [])) +
            len(temporal.get('time_references', [])) +
            len(functional.get('functional_indicators', []))
        )
        
        if total_indicators >= 5:
            completeness += 0.25
        elif total_indicators >= 3:
            completeness += 0.15
        elif total_indicators >= 1:
            completeness += 0.05
        
        return min(completeness, 1.0)
    
    def _calculate_emotional_clarity(self, lexical: Dict) -> float:
        """
        How clear is the emotional content?
        
        High clarity: Strong emotion words + clear sentiment
        Low clarity: Vague words like "okay", "fine", "whatever"
        """
        
        clarity = 0.0
        
        # 1. Check if emotion words are present
        intensity_keywords = lexical.get('intensity_keywords', [])
        if len(intensity_keywords) >= 2:
            clarity += 0.4
        elif len(intensity_keywords) == 1:
            clarity += 0.2
        
        # 2. Check intensity level
        intensity_level = lexical.get('intensity_level', 5.0)
        if intensity_level >= 7.0:  # Strong emotion
            clarity += 0.3
        elif intensity_level >= 4.0:  # Moderate emotion
            clarity += 0.15
        
        # 3. Check sentiment clarity (not neutral)
        sentiment = lexical.get('sentiment_score', 0.0)
        if abs(sentiment) > 0.5:  # Clear positive or negative
            clarity += 0.3
        elif abs(sentiment) > 0.2:  # Somewhat clear
            clarity += 0.15
        
        return min(clarity, 1.0)
    
    def _calculate_functional_clarity(self, functional: Dict) -> float:
        """
        How clear is the life impact?
        
        High clarity: Multiple domains + specific impairments
        Low clarity: No domains mentioned or vague impact
        """
        
        clarity = 0.0
        
        # 1. Domain clarity
        domain_count = functional.get('domain_count', 0)
        if domain_count >= 3:
            clarity += 0.4
        elif domain_count == 2:
            clarity += 0.25
        elif domain_count == 1:
            clarity += 0.1
        
        # 2. Impairment specificity
        functional_indicators = functional.get('functional_indicators', [])
        if len(functional_indicators) >= 2:
            clarity += 0.35
        elif len(functional_indicators) == 1:
            clarity += 0.2
        
        # 3. Severity clarity
        severity = functional.get('impairment_severity', 0.1)
        if severity >= 2.0:  # Clear severe impact
            clarity += 0.25
        elif severity >= 1.0:  # Clear moderate impact
            clarity += 0.15
        
        return min(clarity, 1.0)
    
    def _calculate_historical_depth(self, user_history_depth: int) -> float:
        """
        How much user history do we have for calibration?
        
        More history = Better calibration = Higher confidence
        """
        
        if user_history_depth >= 20:
            return 1.0  # Excellent calibration data
        elif user_history_depth >= 10:
            return 0.8  # Good calibration data
        elif user_history_depth >= 5:
            return 0.6  # Some calibration data
        elif user_history_depth >= 2:
            return 0.4  # Minimal calibration data
        else:
            return 0.2  # Cold start - no calibration
    
    def _calculate_ambiguity_penalty(self, message: str) -> float:
        """
        Detect ambiguous/hedging language that reduces confidence
        
        Penalties for:
        - Hedging words: "maybe", "kind of", "sort of"
        - Uncertainty: "I don't know", "not sure"
        - Contradictions: "but", "although", "however" (when excessive)
        
        Returns: 0.0 (high ambiguity) to 1.0 (no ambiguity)
        """
        
        message_lower = message.lower()
        penalty = 1.0  # Start at full confidence
        
        # Hedging patterns
        hedging_patterns = [
            'maybe', 'perhaps', 'might', 'could be',
            'kind of', 'sort of', 'kinda', 'sorta',
            'i guess', 'i suppose', 'probably'
        ]
        
        hedging_count = sum(1 for pattern in hedging_patterns if pattern in message_lower)
        penalty -= hedging_count * 0.15
        
        # Uncertainty patterns
        uncertainty_patterns = [
            "don't know", "not sure", "unclear", "uncertain",
            "confused", "can't tell", "hard to say"
        ]
        
        uncertainty_count = sum(1 for pattern in uncertainty_patterns if pattern in message_lower)
        penalty -= uncertainty_count * 0.20
        
        # Contradiction markers (excessive use)
        contradiction_markers = ['but', 'although', 'however', 'though', 'yet']
        contradiction_count = sum(1 for marker in contradiction_markers if marker in message_lower)
        
        if contradiction_count >= 3:
            penalty -= 0.25  # Many contradictions = unclear state
        elif contradiction_count == 2:
            penalty -= 0.10
        
        # Vague time references
        ##Have to add more phrases in this list
        vague_time = ['sometime', 'at some point', 'eventually', 'whenever']
        if any(vague in message_lower for vague in vague_time):
            penalty -= 0.15
        
        return max(penalty, 0.0)  # Can't go below 0
    
    def requires_clarification(self, confidence: float) -> bool:
        """
        Simple threshold check
        
        Args:
            confidence: Overall confidence score (0-1)
            
        Returns:
            True if confidence < 0.65 (need clarification)
            False if confidence >= 0.65 (proceed)
        """
        return confidence < 0.65
    
    def get_clarification_priority(self,metrics: ConfidenceMetrics) -> List[Tuple[str, float]]:
        """
        Identify which dimensions are weakest (need clarification most)
        
        Returns:
            List of (dimension_name, score) sorted by priority (lowest first)
        """
        
        dimensions = [
            ('temporal', metrics.temporal_certainty),
            ('emotional', metrics.emotional_clarity),
            ('functional', metrics.functional_clarity),
            ('signal_agreement', metrics.signal_agreement)
        ]
        
        # Sort by score (lowest = highest priority for clarification)
        sorted_dimensions = sorted(dimensions, key=lambda x: x[1])
        
        return sorted_dimensions