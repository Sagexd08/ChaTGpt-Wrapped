"""
Sentiment and tone analysis module
Provides emotional analysis and user persona classification
"""

from typing import Dict, Any, List
from datetime import datetime

class SentimentAnalyzer:
    """Analyze sentiment and emotional tone of messages"""
    
    def __init__(self):
        self.sentiment_scores = []
        self.emotional_tones = []
    
    def analyze(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze overall sentiment and tone
        Returns emotional profile and persona classification
        """
        user_messages = [m for m in messages if m.get('role') == 'user']
        
        if not user_messages:
            return {'error': 'No user messages found'}
        
        # Placeholder sentiment analysis
        sentiments = [self._analyze_message(msg.get('content', '')) for msg in user_messages]
        valid_sentiments = [s for s in sentiments if s is not None]
        
        if not valid_sentiments:
            avg_sentiment = 0.5
        else:
            avg_sentiment = sum(valid_sentiments) / len(valid_sentiments)
        
        persona = self._classify_persona(avg_sentiment, user_messages)
        
        return {
            'status': 'success',
            'overall_sentiment': avg_sentiment,
            'sentiment_label': self._sentiment_to_label(avg_sentiment),
            'persona': persona,
            'emotion_breakdown': self._get_emotion_breakdown(valid_sentiments),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _analyze_message(self, text: str) -> float:
        """
        Analyze single message sentiment (0.0 to 1.0)
        Placeholder: basic keyword matching
        """
        if not text:
            return 0.5
        
        positive_words = ['love', 'great', 'excellent', 'happy', 'thanks', 'awesome', 'good']
        negative_words = ['hate', 'bad', 'terrible', 'sad', 'angry', 'frustrated', 'awful']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.5
        
        return positive_count / total
    
    def _sentiment_to_label(self, sentiment: float) -> str:
        """Convert sentiment score to label"""
        if sentiment < 0.3:
            return 'negative'
        elif sentiment < 0.6:
            return 'neutral'
        else:
            return 'positive'
    
    def _classify_persona(self, sentiment: float, messages: List[Dict[str, Any]]) -> str:
        """Classify user personality type"""
        personas = ['Academic Warrior', 'Curious Learner', 'Problem Solver', 'Creative Mind']
        
        # Simple persona classification based on message count
        msg_count = len(messages)
        
        if msg_count > 100:
            return personas[0]
        elif msg_count > 50:
            return personas[1]
        elif msg_count > 20:
            return personas[2]
        else:
            return personas[3]
    
    def _get_emotion_breakdown(self, sentiments: List[float]) -> Dict[str, float]:
        """Get breakdown of emotions"""
        if not sentiments:
            return {}
        
        positive = sum(1 for s in sentiments if s > 0.6)
        neutral = sum(1 for s in sentiments if 0.3 <= s <= 0.6)
        negative = sum(1 for s in sentiments if s < 0.3)
        
        total = len(sentiments)
        
        return {
            'positive': positive / total if total > 0 else 0,
            'neutral': neutral / total if total > 0 else 0,
            'negative': negative / total if total > 0 else 0,
        }
