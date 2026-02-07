import re
from typing import Dict, List


class FunctionalDetector:
    """Detects life domain impact and functional impairment"""
    
    def __init__(self):
        self.domain_keywords = self._load_domain_keywords()
        self.impairment_patterns = self._load_impairment_patterns()


    def detect(self, message: str) -> Dict:
        """
        Identify affected life domains and severity of impairment
        """
        message_lower = message.lower()
        
        # 1. Detect affected domains
        affected_domains = []
        for domain, keywords in self.domain_keywords.items():
            if any(kw in message_lower for kw in keywords):
                affected_domains.append(domain)
        
        # 2. Detect functional impairment indicators
        detected_impairments = []
        max_severity = 0.1  # Default minimal
        
        for pattern, severity in self.impairment_patterns.items():
            if re.search(pattern, message_lower):
                detected_impairments.append(pattern)
                max_severity = max(max_severity, severity)
        
        # 3. Multi-symptom amplification
        if len(detected_impairments) >= 3:
            max_severity = min(max_severity * 1.3, 3.0)
        
        return {
            'affected_domains': affected_domains,
            'domain_count': max(len(affected_domains), 1),  # Minimum 1
            'impairment_severity': max_severity,
            'functional_indicators': detected_impairments,
            'has_multiple_symptoms': len(detected_impairments) >= 3
        }
    
    def _load_domain_keywords(self) -> Dict[str, List[str]]:
        """Keywords for each life domain"""
        return {
            'work': [
                'job', 'work', 'career', 'boss', 'coworker', 'fired',
                'promotion', 'deadline', 'project', 'office', 'resign'
            ],
            'relationships': [
                'partner', 'spouse', 'boyfriend', 'girlfriend', 'breakup',
                'divorce', 'friend', 'family', 'alone', 'lonely', 'isolated'
            ],
            'health': [
                'sick', 'pain', 'doctor', 'hospital', 'diagnosis', 'disease',
                'injury', 'medical', 'surgery', 'chronic', 'symptom'
            ],
            'identity': [
                'who i am', 'lost myself', 'failure', 'worthless', 'useless',
                'not myself', 'define me', 'my purpose', 'my value'
            ],
            'safety': [
                'danger', 'threat', 'scared', 'unsafe', 'attacked', 'trauma',
                'abuse', 'violence', 'fear for', 'worried about safety'
            ]
        }
    
    def _load_impairment_patterns(self) -> Dict[str, float]:
        """Functional impairment patterns with severity scores"""
        return {
            # Severe impairment (2.0-3.0)
            r"can'?t get out of bed": 3.0,
            r"suicidal|want to die|end it all": 3.0,
            r"can'?t function": 2.8,
            r"can'?t work|unable to work": 2.5,
            r"can'?t eat|not eating": 2.3,
            r"can'?t sleep for (weeks|days)": 2.5,
            r"crying (every day|all the time)": 2.0,
            r"panic attack": 2.0,
            
            # Moderate impairment (1.0-2.0)
            r"hard to (focus|concentrate)": 1.5,
            r"losing sleep|not sleeping well": 1.4,
            r"avoiding (people|everyone|social)": 1.3,
            r"not eating well|lost appetite": 1.2,
            r"tired all the time|exhausted": 1.2,
            r"can'?t (enjoy|find joy)": 1.5,
            
            # Mild impairment (0.1-1.0)
            r"distracted|hard to think": 0.8,
            r"a bit tired|slightly off": 0.5,
            r"not (feeling like )?myself": 0.7,
            r"bothering me": 0.4
        }