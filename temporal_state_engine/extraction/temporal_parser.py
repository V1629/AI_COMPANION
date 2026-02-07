import re
from datetime import datetime, timedelta
from dateparser import parse as date_parse
from typing import Dict,List,Optional

class TemporalParser:
    """Extracts time-related information from text"""
    
    def __init__(self):
        self.temporal_patterns = self._build_patterns()
        
    def parse(self, message: str) -> Dict:
        """
        Extract temporal signals and calculate persistence score
        """
        message_lower = message.lower()
        
        # 1. Detect explicit time references
        detected_times = []
        for pattern, persistence in self.temporal_patterns.items():
            if re.search(pattern, message_lower):
                detected_times.append({
                    'pattern': pattern,
                    'persistence': persistence
                })
        
        # 2. Use dateparser for natural language dates
        parsed_dates = self._extract_dates(message)
        
        # 3. Calculate persistence score
        if detected_times:
            # Take maximum persistence (most significant timeframe)
            max_persistence = max(t['persistence'] for t in detected_times)
        elif parsed_dates:
            # Calculate from date difference
            max_persistence = self._calculate_persistence_from_date(parsed_dates[0])
        else:
            # No temporal info - assume very recent
            max_persistence = 0.1
        
        # 4. Check for ongoing markers
        ongoing_markers = ['still', 'always', 'constantly', 'every day', 'since']
        is_ongoing = any(marker in message_lower for marker in ongoing_markers)
        
        if is_ongoing:
            max_persistence = min(max_persistence * 1.5, 10.0)  # Amplify persistence
        
        # 5. Temporal certainty
        certainty = 1.0 if detected_times or parsed_dates else 0.3
        
        return {
            'time_references': [t['pattern'] for t in detected_times],
            'parsed_dates': parsed_dates,
            'persistence_score': max_persistence,
            'is_ongoing': is_ongoing,
            'future_projection': self._check_future_projection(message_lower),
            'temporal_certainty': certainty
        }
    
    def _build_patterns(self) -> Dict[str, float]:
        """Map temporal phrases to persistence scores"""
        return {
            r'\b(just now|right now|this moment)\b': 0.1,
            r'\b(today|this morning|this afternoon)\b': 0.2,
            r'\b(yesterday|last night)\b': 0.3,
            r'\b(this week|past few days|couple days)\b': 1.0,
            r'\b(last week|week ago)\b': 2.0,
            r'\b(two weeks|couple weeks)\b': 3.0,
            r'\b(this month|few weeks)\b': 4.0,
            r'\b(last month|month ago)\b': 5.0,
            r'\b(few months|several months|couple months)\b': 6.0,
            r'\b(half year|six months)\b': 7.0,
            r'\b(this year|year ago|last year)\b': 8.0,
            r'\b(years|multiple years|long time)\b': 9.0,
            r'\b(forever|always|entire life|since childhood|permanent|chronic)\b': 10.0,
        }
    
    def _extract_dates(self, message: str) -> List[datetime]:
        """Use dateparser to extract natural language dates"""
        # Examples: "three months ago", "since January", "last summer"
        try:
            parsed = date_parse(message, settings={'PREFER_DATES_FROM': 'past'})
            return [parsed] if parsed else []
        except:
            return []
    
    def _calculate_persistence_from_date(self, date: datetime) -> float:
        """Convert date difference to persistence score"""
        days_ago = (datetime.now() - date).days
        
        if days_ago < 1:
            return 0.1
        elif days_ago < 7:
            return 1.0
        elif days_ago < 30:
            return 3.0
        elif days_ago < 90:
            return 5.0
        elif days_ago < 180:
            return 7.0
        elif days_ago < 365:
            return 8.0
        else:
            return 9.0
    
    def _check_future_projection(self, message: str) -> Optional[str]:
        """Detect if user mentions future impact"""
        future_patterns = [
            'will affect me', 'will never', 'going to be', 
            'for the rest of', 'will always'
        ]
        for pattern in future_patterns:
            if pattern in message:
                return pattern
        return None