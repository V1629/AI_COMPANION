# extraction/signal_extractor.py
from typing import Dict, List, Optional
from dataclasses import dataclass
from .lexical_analyzer import LexicalAnalyzer
from .temporal_parser import TemporalParser
from .functional_detector import FunctionalDetector
from .emotional_calibrator import EmotionalCalibrator
from .confidence_scorer import ConfidenceScorer, ConfidenceMetrics
from core.models import ExtractionResult, SignalConfidence
from core.enums import StateLayer, LifeDomain

@dataclass
class RawSignals:
    """Container for raw extracted signals before fusion"""
    lexical_score: float
    temporal_score: float
    functional_indicators: Dict[str, any]
    calibrated_baseline: float
    extraction_metadata: Dict


class SignalExtractor:
    """
    Orchestrator for multi-layer signal extraction from user messages.
    Coordinates 4 specialized analyzers and fuses their outputs.
    """
    
    def __init__(self, nlp_model, user_history_service):
        self.nlp_model = nlp_model  # spaCy or transformer model
        self.user_history = user_history_service
        
        # Initialize sub-analyzers
        self.lexical = LexicalAnalyzer(nlp_model)
        self.temporal = TemporalParser()
        self.functional = FunctionalDetector()
        self.calibrator = EmotionalCalibrator(user_history_service)
        self.confidence = ConfidenceScorer()
        
    def extract(self, message: str, user_id: str) -> ExtractionResult:
        """
        Main extraction pipeline - coordinates all analyzers
        
        Process:
        1. Run all 4 analyzers in parallel
        2. Fuse signals with weighted combination
        3. Calculate confidence score
        4. Decide if clarification needed
        5. Return structured result
        """
        
        # === LAYER 1: Lexical Analysis ===
        lexical_signals = self.lexical.analyze(message)
        # Returns: {
        #   'intensity_keywords': ['devastated', 'unbearable'],
        #   'sentiment_score': -0.85,
        #   'emotional_valence': 'negative',
        #   'intensity_level': 8.5
        # }
        
        # === LAYER 2: Temporal Parsing ===
        temporal_signals = self.temporal.parse(message)
        # Returns: {
        #   'time_references': ['3 months ago', 'since then'],
        #   'persistence_score': 8.0,
        #   'future_projection': None,
        #   'temporal_certainty': 0.9
        # }
        
        # === LAYER 3: Functional Impact Detection ===
        functional_signals = self.functional.detect(message)
        # Returns: {
        #   'affected_domains': ['health', 'work', 'relationships'],
        #   'impairment_severity': 2.5,
        #   'functional_indicators': ['can't sleep', 'can't work'],
        #   'domain_count': 3
        # }
        
        # === LAYER 4: Emotional Calibration ===
        user_baseline = self.calibrator.get_baseline(user_id)
        
        calibrated_signals = self.calibrator.calibrate(
            raw_intensity=lexical_signals['intensity_level'],
            user_baseline=user_baseline,
            message=message
        )
        # Returns: {
        #   'calibrated_intensity': 7.2,  # Adjusted from 8.5 for hyperbolic user
        #   'user_expression_style': 'expressive',
        #   'baseline_deviation': +1.8
        # }
        
        # === CONFIDENCE SCORING ===
        confidence_metrics = self.confidence.calculate(
            lexical_signals=lexical_signals,
            temporal_signals=temporal_signals,
            functional_signals=functional_signals,
            calibrated_signals=calibrated_signals,
            message=message,
            user_history_depth=len(user_baseline.get('past_incidents', []))
        )
        
        # === CLARIFICATION LOGIC ===
        if self.confidence.requires_clarification(confidence_metrics.overall_confidence):
            clarification = self._generate_clarification_probe(
                confidence_metrics=confidence_metrics,
                message=message
            )
            return ExtractionResult(
                requires_clarification=True,
                clarification_question=clarification,
                confidence=confidence_metrics.overall_confidence,
                confidence_metrics=confidence_metrics,
                partial_signals=self._create_partial_signals(
                    lexical_signals,
                    temporal_signals,
                    functional_signals,
                    calibrated_signals
                )
            )
        
        # === SIGNAL FUSION ===
        fused_signals = self._fuse_signals(
            lexical_signals,
            temporal_signals,
            functional_signals,
            calibrated_signals
        )
        
        # === FINAL EXTRACTION RESULT ===
        return ExtractionResult(
            persistence=fused_signals['P'],
            resonance=fused_signals['R'],
            impact=fused_signals['I'],
            severity=fused_signals['S'],
            malleability=fused_signals['M'],
            confidence=confidence_metrics.overall_confidence,
            confidence_metrics=confidence_metrics,
            requires_clarification=False,
            raw_signals=RawSignals(
                lexical_score=lexical_signals['intensity_level'],
                temporal_score=temporal_signals['persistence_score'],
                functional_indicators=functional_signals,
                calibrated_baseline=user_baseline,
                extraction_metadata={
                    'message_length': len(message),
                    'detected_keywords': lexical_signals['intensity_keywords'],
                    'temporal_markers': temporal_signals['time_references']
                }
            )
        )
    
    def _fuse_signals(self, lexical, temporal, functional, calibrated) -> Dict:
        """
        Weighted fusion of all signals into PRISM components
        """
        return {
            'P': temporal['persistence_score'],
            'R': calibrated['calibrated_intensity'],
            'I': functional['domain_count'],
            'S': functional['impairment_severity'],
            'M': self._extract_malleability(lexical, functional)
        }
    
    def _extract_malleability(self, lexical, functional) -> float:
        """Extract control attribution from lexical patterns"""
        # Check for control-related language
        message_lower = lexical.get('original_message', '').lower()

        ###We have to add more phrases here 
        LOW_CONTROL = [
            'nothing i can do', 'helpless', 'out of my control',
            'happened to me', 'victim', 'can\'t change'
        ]
        HIGH_CONTROL = [
            'i can fix', 'i\'ll handle', 'my fault', 
            'i should have', 'my responsibility'
        ]
        
        for pattern in LOW_CONTROL:
            if pattern in message_lower:
                return 0.6  # Low control → amplifies significance
        
        for pattern in HIGH_CONTROL:
            if pattern in message_lower:
                return 1.7  # High control → reduces significance
        
        return 1.0  # Neutral default
    
    def _create_partial_signals(self, lexical, temporal, functional, calibrated) -> Dict:
        """
        Create partial signals object when confidence is low
        Used for debugging and potential re-extraction
        """
        return {
            'P': temporal.get('persistence_score', 0.1),
            'R': calibrated.get('calibrated_intensity', 5.0),
            'I': functional.get('domain_count', 1),
            'S': functional.get('impairment_severity', 0.1),
            'M': 1.0  # Neutral default when uncertain
        }
    
    def _generate_clarification_probe(self,confidence_metrics: ConfidenceMetrics, message: str) -> str:
        """
        Generate targeted follow-up question based on weakest dimension
        """
        # Get prioritized list of weak dimensions
        priorities = self.confidence.get_clarification_priority(confidence_metrics)
        weakest_dimension = priorities[0][0]  # First item = weakest
        
        # Generate appropriate question based on weakest dimension
        clarification_questions = {
            'temporal': "How long has this been affecting you?",
            'emotional': "Can you tell me more about how this is making you feel?",
            'functional': "How is this impacting your daily life (work, sleep, relationships)?",
            'signal_agreement': "I want to understand this better - can you share a bit more about what's going on?"
        }
        
        return clarification_questions.get(
            weakest_dimension,
            "Could you tell me a bit more about this?"  # Fallback
        )