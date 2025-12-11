"""
Metrics calculation module
Computes usage statistics and trends
"""

from typing import Dict, Any, List
from datetime import datetime
from collections import Counter

class MetricsCalculator:
    """Calculate usage metrics from conversation data"""
    
    def __init__(self):
        self.total_messages = 0
        self.total_words = 0
        self.word_counts: Dict[str, int] = {}
    
    def calculate(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics
        Returns usage statistics, streaks, and trends
        """
        user_messages = [m for m in messages if m.get('role') == 'user']
        assistant_messages = [m for m in messages if m.get('role') == 'assistant']
        
        if not user_messages:
            return {'error': 'No user messages found'}
        
        user_text = ' '.join([m.get('content', '') for m in user_messages])
        
        metrics = {
            'status': 'success',
            'message_counts': {
                'total': len(messages),
                'user': len(user_messages),
                'assistant': len(assistant_messages),
            },
            'word_counts': {
                'total_words': len(user_text.split()),
                'unique_words': len(set(user_text.lower().split())),
                'average_message_length': len(user_text.split()) / len(user_messages) if user_messages else 0,
            },
            'streaks': self._calculate_streaks(messages),
            'activity': self._analyze_activity(messages),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return metrics
    
    def _calculate_streaks(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate message streaks and patterns"""
        if not messages:
            return {}
        
        # Count consecutive user messages
        max_streak = 0
        current_streak = 0
        
        for msg in messages:
            if msg.get('role') == 'user':
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return {
            'longest_user_streak': max_streak,
            'average_exchange_length': len(messages) / 2 if messages else 0,
        }
    
    def _analyze_activity(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze activity patterns"""
        user_messages = [m for m in messages if m.get('role') == 'user']
        
        if not user_messages:
            return {}
        
        # Placeholder activity analysis
        message_lengths = [len(m.get('content', '').split()) for m in user_messages]
        
        return {
            'most_active_hour': 'evening',  # Placeholder
            'peak_activity': len(user_messages),
            'message_length_stats': {
                'min': min(message_lengths) if message_lengths else 0,
                'max': max(message_lengths) if message_lengths else 0,
                'average': sum(message_lengths) / len(message_lengths) if message_lengths else 0,
            }
        }
