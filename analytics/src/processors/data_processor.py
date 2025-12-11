"""
Data processor for ChatGPT conversation exports
Handles .json, .txt, and raw text imports
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

class DataProcessor:
    """Process and normalize ChatGPT conversation data"""
    
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
    
    def process_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process ChatGPT JSON export format
        Handles both direct data and nested structures
        """
        try:
            # Handle different JSON export formats
            messages = data.get('conversations', data.get('messages', data.get('data', [])))
            
            if not isinstance(messages, list):
                return {'error': 'Invalid JSON structure', 'status': 'failed'}
            
            processed_messages = []
            for msg in messages:
                processed = self._normalize_message(msg)
                if processed:
                    processed_messages.append(processed)
            
            return {
                'status': 'success',
                'message_count': len(processed_messages),
                'messages': processed_messages,
                'metadata': self._extract_metadata(processed_messages)
            }
        except Exception as e:
            return {'error': str(e), 'status': 'failed'}
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process plain text chat export
        Assumes simple message format
        """
        try:
            lines = text.strip().split('\n')
            messages = []
            
            for line in lines:
                if line.strip():
                    messages.append({
                        'content': line,
                        'role': 'user',
                        'timestamp': datetime.utcnow().isoformat()
                    })
            
            return {
                'status': 'success',
                'message_count': len(messages),
                'messages': messages,
                'metadata': self._extract_metadata(messages)
            }
        except Exception as e:
            return {'error': str(e), 'status': 'failed'}
    
    def _normalize_message(self, msg: Any) -> Optional[Dict[str, Any]]:
        """Normalize message format"""
        if isinstance(msg, str):
            return {'content': msg, 'role': 'user', 'timestamp': datetime.utcnow().isoformat()}
        
        if isinstance(msg, dict):
            return {
                'content': msg.get('content', msg.get('text', '')),
                'role': msg.get('role', msg.get('author', 'user')),
                'timestamp': msg.get('timestamp', msg.get('created_at', datetime.utcnow().isoformat()))
            }
        
        return None
    
    def _extract_metadata(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract metadata from messages"""
        if not messages:
            return {}
        
        timestamps = [msg.get('timestamp') for msg in messages if msg.get('timestamp')]
        
        return {
            'total_messages': len(messages),
            'date_range': {
                'start': timestamps[0] if timestamps else None,
                'end': timestamps[-1] if timestamps else None,
            },
            'message_types': {
                'user': len([m for m in messages if m.get('role') == 'user']),
                'assistant': len([m for m in messages if m.get('role') == 'assistant']),
            }
        }
