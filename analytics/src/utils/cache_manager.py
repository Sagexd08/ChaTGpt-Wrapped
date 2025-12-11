"""
Advanced Caching and Performance Layer
Redis-based caching and result persistence
"""

import redis
import json
import hashlib
import logging
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import pickle
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheManager:
    """Manage caching with Redis fallback to memory"""
    
    def __init__(self, 
                 redis_host: str = os.getenv('REDIS_HOST', 'localhost'),
                 redis_port: int = int(os.getenv('REDIS_PORT', 6379)),
                 redis_db: int = 0,
                 use_redis: bool = True):
        
        self.use_redis = use_redis
        self.redis_client = None
        self.memory_cache = {}
        
        if use_redis:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=False,
                    socket_connect_timeout=5
                )
                self.redis_client.ping()
                logger.info("Connected to Redis")
            except Exception as e:
                logger.warning(f"Could not connect to Redis: {e}, using in-memory cache")
                self.use_redis = False
    
    def _get_cache_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from prefix and data hash"""
        
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True, default=str)
        else:
            data_str = str(data)
        
        hash_suffix = hashlib.md5(data_str.encode()).hexdigest()[:8]
        return f"{prefix}:{hash_suffix}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        
        try:
            if self.redis_client:
                cached = self.redis_client.get(key)
                if cached:
                    return pickle.loads(cached)
            else:
                return self.memory_cache.get(key)
        except Exception as e:
            logger.warning(f"Error retrieving from cache: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL"""
        
        try:
            if self.redis_client:
                serialized = pickle.dumps(value)
                self.redis_client.setex(key, ttl, serialized)
            else:
                self.memory_cache[key] = {
                    'value': value,
                    'expires': datetime.now() + timedelta(seconds=ttl)
                }
            
            return True
        except Exception as e:
            logger.warning(f"Error setting cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        
        try:
            if self.redis_client:
                self.redis_client.delete(key)
            else:
                self.memory_cache.pop(key, None)
            
            return True
        except Exception as e:
            logger.warning(f"Error deleting from cache: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache"""
        
        try:
            if self.redis_client:
                self.redis_client.flushdb()
            else:
                self.memory_cache.clear()
            
            return True
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        stats = {
            'backend': 'redis' if self.redis_client else 'memory',
            'connected': self.redis_client is not None,
        }
        
        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats['memory_usage'] = info.get('used_memory_human')
                stats['keys'] = self.redis_client.dbsize()
            except Exception as e:
                logger.warning(f"Error getting Redis stats: {e}")
        else:
            stats['keys'] = len(self.memory_cache)
        
        return stats

class ResultCache:
    """Cache analysis results for quick retrieval"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
    
    def cache_analysis_result(self, 
                             messages: List[Dict],
                             analysis: Dict,
                             session_id: str = None) -> str:
        """Cache complete analysis result"""
        
        if not session_id:
            session_id = hashlib.md5(
                json.dumps(messages, sort_keys=True, default=str).encode()
            ).hexdigest()
        
        cache_key = f"analysis:{session_id}"
        result = {
            'session_id': session_id,
            'analysis': analysis,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        self.cache.set(cache_key, result, ttl=86400)  # 24 hours
        
        return session_id
    
    def get_cached_analysis(self, session_id: str) -> Optional[Dict]:
        """Retrieve cached analysis"""
        
        cache_key = f"analysis:{session_id}"
        return self.cache.get(cache_key)
    
    def cache_generated_content(self,
                               session_id: str,
                               content_type: str,
                               content: Any) -> bool:
        """Cache generated content (text, images, etc.)"""
        
        cache_key = f"content:{session_id}:{content_type}"
        return self.cache.set(cache_key, content, ttl=86400)
    
    def get_cached_content(self, session_id: str, content_type: str) -> Optional[Any]:
        """Retrieve cached generated content"""
        
        cache_key = f"content:{session_id}:{content_type}"
        return self.cache.get(cache_key)

class QueryOptimizer:
    """Optimize repeated queries and inference calls"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.query_history = {}
    
    def optimize_text_generation(self, prompt: str, generation_params: Dict) -> Dict:
        """Optimize text generation with caching"""
        
        cache_key = self.cache._get_cache_key("text_gen", {
            'prompt': prompt,
            'params': generation_params
        })
        
        # Check cache
        cached = self.cache.get(cache_key)
        if cached:
            return {'cached': True, 'result': cached}
        
        return {'cached': False, 'cache_key': cache_key}
    
    def optimize_sentiment_analysis(self, text: str) -> Dict:
        """Optimize sentiment analysis with caching"""
        
        cache_key = self.cache._get_cache_key("sentiment", {'text': text})
        
        cached = self.cache.get(cache_key)
        if cached:
            return {'cached': True, 'result': cached}
        
        return {'cached': False, 'cache_key': cache_key}
    
    def optimize_image_generation(self, prompt: str, style: str) -> Dict:
        """Optimize image generation with caching"""
        
        cache_key = self.cache._get_cache_key("image_gen", {
            'prompt': prompt,
            'style': style
        })
        
        cached = self.cache.get(cache_key)
        if cached:
            return {'cached': True, 'result': cached}
        
        return {'cached': False, 'cache_key': cache_key}

class PerformanceMonitor:
    """Monitor ML model performance and inference times"""
    
    def __init__(self):
        self.metrics = {
            'inference_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'model_errors': 0,
        }
    
    def record_inference(self, model_name: str, duration_ms: float, success: bool = True):
        """Record inference timing"""
        
        self.metrics['inference_times'].append({
            'model': model_name,
            'duration_ms': duration_ms,
            'success': success,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def record_cache_hit(self):
        """Record cache hit"""
        self.metrics['cache_hits'] += 1
    
    def record_cache_miss(self):
        """Record cache miss"""
        self.metrics['cache_misses'] += 1
    
    def record_error(self, model_name: str, error: str):
        """Record model error"""
        self.metrics['model_errors'] += 1
        logger.error(f"Model {model_name} error: {error}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        
        avg_inference_time = 0
        if self.metrics['inference_times']:
            avg_inference_time = sum(
                m['duration_ms'] for m in self.metrics['inference_times']
            ) / len(self.metrics['inference_times'])
        
        total_requests = self.metrics['cache_hits'] + self.metrics['cache_misses']
        hit_rate = 0
        if total_requests > 0:
            hit_rate = self.metrics['cache_hits'] / total_requests
        
        return {
            'average_inference_time_ms': avg_inference_time,
            'cache_hit_rate': hit_rate,
            'total_cache_hits': self.metrics['cache_hits'],
            'total_cache_misses': self.metrics['cache_misses'],
            'total_errors': self.metrics['model_errors'],
        }
