"""
Topic modeling and categorization
Identifies main topics and subjects discussed
"""

from typing import Dict, Any, List
from collections import Counter
from datetime import datetime

class TopicAnalyzer:
    """Identify and categorize conversation topics"""
    
    # Default topic keywords
    TOPIC_KEYWORDS = {
        'tech': ['code', 'programming', 'software', 'python', 'javascript', 'api', 'database'],
        'learning': ['study', 'learn', 'education', 'course', 'tutorial', 'guide', 'help'],
        'career': ['job', 'resume', 'interview', 'career', 'company', 'work', 'salary'],
        'creative': ['write', 'story', 'poem', 'design', 'art', 'creative', 'imagine'],
        'casual': ['chat', 'talk', 'hello', 'hey', 'hi', 'how are you', 'thanks'],
    }
    
    def __init__(self):
        self.topics: List[Dict[str, Any]] = []
        self.topic_distribution: Dict[str, float] = {}
    
    def analyze(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze topics from conversation
        Returns top topics and categorization
        """
        user_messages = [m.get('content', '') for m in messages if m.get('role') == 'user']
        
        if not user_messages:
            return {'error': 'No user messages found'}
        
        # Combine all text and categorize
        combined_text = ' '.join(user_messages).lower()
        
        topic_counts = self._categorize_topics(combined_text)
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        total_words = sum(topic_counts.values())
        topic_distribution = {
            topic: count / total_words if total_words > 0 else 0
            for topic, count in top_topics
        }
        
        return {
            'status': 'success',
            'top_topics': [{'name': topic, 'count': count} for topic, count in top_topics],
            'topic_distribution': topic_distribution,
            'primary_topic': top_topics[0][0] if top_topics else 'General',
            'topic_diversity': len(set([t[0] for t in top_topics])),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _categorize_topics(self, text: str) -> Dict[str, int]:
        """Categorize text into topic counts"""
        topic_counts: Dict[str, int] = {}
        
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            count = 0
            for keyword in keywords:
                count += text.count(keyword)
            
            if count > 0:
                topic_counts[topic] = count
        
        # If no keywords matched, return generic categorization
        if not topic_counts:
            topic_counts['general'] = len(text.split())
        
        return topic_counts
    
    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        words = text.lower().split()
        
        # Simple phrase extraction (3-word phrases)
        phrases = []
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            if any(keyword in phrase for keywords in self.TOPIC_KEYWORDS.values() for keyword in keywords):
                phrases.append(phrase)
        
        # Return top 5 unique phrases
        phrase_counts = Counter(phrases)
        return [phrase for phrase, _ in phrase_counts.most_common(5)]
