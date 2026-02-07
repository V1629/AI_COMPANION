###






from typing import Dict, List


class EmotionalCalibrator:
    """Adjusts scores based on user's historical expression patterns"""
    
    def __init__(self, user_history_service):
        self.user_history = user_history_service
        
    def get_baseline(self, user_id: str) -> Dict:
        """
        Retrieve user's emotional baseline profile
        """
        history = self.user_history.get_user_data(user_id)
        
        if not history or len(history.get('messages', [])) < 5:
            # Cold start: return neutral baseline
            return {
                'expression_style': 'neutral',
                'avg_intensity': 5.0,
                'intensity_stddev': 2.0,
                'message_count': len(history.get('messages', [])),
                'past_incidents': []
            }
        
        # Calculate historical averages
        intensity_scores = [msg['intensity'] for msg in history['messages'] if 'intensity' in msg]
        
        return {
            'expression_style': self._classify_style(intensity_scores),
            'avg_intensity': sum(intensity_scores) / len(intensity_scores),
            'intensity_stddev': self._calculate_stddev(intensity_scores),
            'message_count': len(history['messages']),
            'past_incidents': history.get('incidents', [])
        }
    
    def calibrate(self, raw_intensity: float, user_baseline: Dict, message: str) -> Dict:
        """
        Adjust raw intensity based on user's baseline
        
        Logic:
        - Expressive users (always dramatic) → deflate scores
        - Stoic users (understated) → inflate scores
        """
        style = user_baseline['expression_style']
        avg = user_baseline['avg_intensity']
        
        if style == 'expressive':
            # User tends to exaggerate - deflate by 20%
            calibration_factor = 0.8
        elif style == 'stoic':
            # User understates - inflate by 30%
            calibration_factor = 1.3
        else:
            calibration_factor = 1.0
        
        calibrated_intensity = raw_intensity * calibration_factor
        
        # Additional check: Is this significantly above/below user's normal?
        deviation = raw_intensity - avg
        
        return {
            'calibrated_intensity': min(calibrated_intensity, 10.0),
            'user_expression_style': style,
            'baseline_deviation': deviation,
            'calibration_factor_applied': calibration_factor
        }
    
    def _classify_style(self, intensity_scores: List[float]) -> str:
        """Classify user as expressive, stoic, or neutral"""
        if not intensity_scores:
            return 'neutral'
        
        avg = sum(intensity_scores) / len(intensity_scores)
        
        if avg > 7.0:
            return 'expressive'  # Tends to use strong language
        elif avg < 4.0:
            return 'stoic'  # Tends to understate
        else:
            return 'neutral'
    
    def _calculate_stddev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 1.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5