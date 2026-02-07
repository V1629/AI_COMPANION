from typing import Dict, List

class LexicalAnalyzer:
    """Analyzes word-level patterns for emotional intensity"""
    
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        self.intensity_lexicon = self._load_intensity_lexicon()
        
    def analyze(self, message: str) -> Dict:
        """
        Extract emotional signals from text
        """
        doc = self.nlp(message)
        
        # 1. Keyword matching
        detected_keywords = []
        intensity_sum = 0
        
        for token in doc:
            lemma = token.lemma_.lower()
            if lemma in self.intensity_lexicon:
                keyword_intensity = self.intensity_lexicon[lemma]
                detected_keywords.append({
                    'word': token.text,
                    'intensity': keyword_intensity
                })
                intensity_sum += keyword_intensity
        
        # 2. Sentiment analysis (using nlp model)
        sentiment_score = self._calculate_sentiment(doc)
        
        # 3. Intensifier detection ("very sad" > "sad")
        intensifiers = ['very', 'extremely', 'really', 'so', 'completely']
        has_intensifier = any(token.text.lower() in intensifiers for token in doc)
        
        # 4. Negation handling ("not terrible" vs "terrible")
        negations = ['not', 'no', 'never', "n't"]
        has_negation = any(token.text.lower() in negations for token in doc)
        
        # Calculate final intensity
        base_intensity = (intensity_sum / max(len(detected_keywords), 1)) if detected_keywords else 5.0
        
        if has_intensifier:
            base_intensity *= 1.3
        if has_negation:
            base_intensity *= 0.6
        
        base_intensity = min(base_intensity, 10.0)  # Cap at 10
        
        return {
            'intensity_keywords': [kw['word'] for kw in detected_keywords],
            'sentiment_score': sentiment_score,
            'emotional_valence': 'negative' if sentiment_score < -0.2 else 'positive' if sentiment_score > 0.2 else 'neutral',
            'intensity_level': base_intensity,
            'has_intensifier': has_intensifier,
            'has_negation': has_negation,
            'original_message': message
        }
    
    def _load_intensity_lexicon(self) -> Dict[str, float]:
        """Load pre-defined emotion intensity mappings"""
        return {
            # Negative HIGH INTENSITY (7-10) - Extreme emotions
            'devastated': 9.5, 'destroyed': 9.5, 'shattered': 9.5, 'obliterated': 9.5,
            'suicidal': 10.0, 'hopeless': 9.0, 'despairing': 9.0, 'anguished': 9.0,
            'heartbroken': 8.5, 'terrified': 8.5, 'agonizing': 8.5, 'traumatized': 8.5,
            'panicking': 8.5, 'enraged': 8.5, 'furious': 8.5, 'livid': 8.5,
            'awful': 7.5, 'terrible': 7.5, 'horrible': 7.5, 'miserable': 7.5,
            'excruciating': 8.0, 'unbearable': 8.0, 'overwhelming': 7.5, 'crushing': 8.0,
            'distraught': 8.0, 'hysterical': 8.5, 'petrified': 8.5, 'horrified': 8.0,
            
            # Negative MEDIUM-HIGH INTENSITY (6-7)
            'depressed': 7.0, 'anxious': 6.5, 'angry': 6.5, 'scared': 6.5,
            'upset': 6.0, 'distressed': 6.5, 'troubled': 6.0, 'disturbed': 6.5,
            'hurt': 6.0, 'painful': 6.5, 'suffering': 7.0, 'tormented': 7.5,
            
            # Negative MEDIUM INTENSITY (4-6)
            'frustrated': 5.5, 'worried': 5.5, 'nervous': 5.0, 'uneasy': 4.5,
            'stressed': 5.0, 'disappointed': 4.5, 'uncomfortable': 4.0, 'sad': 5.0,
            'unhappy': 4.5, 'down': 4.5, 'blue': 4.0, 'gloomy': 4.5,
            'irritated': 4.5, 'agitated': 5.0, 'restless': 4.5, 'tense': 5.0,
            'afraid': 5.5, 'fearful': 5.5, 'apprehensive': 5.0, 'insecure': 4.5,
            
            # Negative LOW-MEDIUM INTENSITY (2.5-4)
            'annoyed': 3.0, 'bothered': 3.0, 'concerned': 3.5, 'confused': 3.0,
            'uncertain': 3.0, 'doubtful': 3.0, 'hesitant': 2.5, 'reluctant': 2.5,
            'tired': 3.5, 'drained': 4.0, 'exhausted': 5.0, 'weary': 4.0,
            'lonely': 5.5, 'isolated': 5.0, 'alone': 4.5, 'abandoned': 7.0,
            
            # LOW INTENSITY (1-2.5)
            'slightly': 2.0, 'mildly': 2.0, 'a bit': 2.0, 'somewhat': 2.5,
            'okay': 1.5, 'fine': 1.0, 'alright': 1.5, 'meh': 2.0,
            'bored': 2.5, 'indifferent': 1.0, 'apathetic': 2.0, 'numb': 3.5,
            
            # POSITIVE HIGH INTENSITY (for contrast)
            'ecstatic': 9.0, 'euphoric': 9.5, 'elated': 8.5, 'thrilled': 8.0,
            'overjoyed': 8.5, 'delighted': 7.5, 'blissful': 8.5, 'exhilarated': 8.0,
            
            # POSITIVE MEDIUM INTENSITY
            'happy': 6.0, 'joyful': 7.0, 'glad': 5.5, 'pleased': 5.0,
            'content': 4.0, 'satisfied': 4.5, 'cheerful': 6.0, 'excited': 7.0,
            'hopeful': 5.5, 'optimistic': 5.0, 'encouraged': 5.0, 'relieved': 5.5,
            
            # POSITIVE LOW INTENSITY
            'calm': 2.0, 'peaceful': 2.5, 'relaxed': 2.0, 'comfortable': 2.0,
            'grateful': 4.0, 'thankful': 4.0, 'appreciative': 3.5,
            
            # INTENSIFIERS (multiply base emotion)
            'extremely': 1.5, 'very': 1.3, 'really': 1.2, 'quite': 1.1,
            'absolutely': 1.5, 'completely': 1.4, 'totally': 1.4, 'utterly': 1.5,
            'incredibly': 1.4, 'immensely': 1.4, 'deeply': 1.3, 'profoundly': 1.4,
            
            # DE-INTENSIFIERS (reduce base emotion)
            'kind of': 0.7, 'sort of': 0.7, 'pretty': 0.9, 'fairly': 0.8,
            'rather': 0.85, 'moderately': 0.75, 'reasonably': 0.8,
        }
    
    def _calculate_sentiment(self, doc) -> float:
        """Calculate sentiment using NLP model (-1 to +1)"""
        # Use spaCy's sentiment or transformer model
        if hasattr(doc, 'sentiment'):
            return doc.sentiment
        
        # Fallback: Use TextBlob or VADER
        from textblob import TextBlob
        blob = TextBlob(doc.text)
        return blob.sentiment.polarity