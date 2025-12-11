"""
Unified ML Inference Engine
Orchestrate all models for wrapped generation
"""

import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional
import logging
from functools import lru_cache
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLInferenceEngine:
    """Unified inference engine for all ML models"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize models (lazy loading)
        self._models = {}
        self._initialized = False
        
        # Caching
        self.inference_cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    def initialize(self):
        """Initialize all models"""
        
        logger.info("Initializing ML models...")
        
        try:
            from models.text_generation.advanced_text_generator import AdvancedTextGenerator, TextStyleTransfer
            from models.image_generation.advanced_image_generator import AdvancedImageGenerator
            from models.embeddings.semantic_analyzer import SemanticAnalyzer, AdvancedNLPAnalyzer, DeepSemanticModel
            
            self._models = {
                'text_generator': AdvancedTextGenerator(device=self.device),
                'text_style_transfer': None,  # Will be initialized after text_generator
                'image_generator': AdvancedImageGenerator(device=self.device),
                'semantic_analyzer': SemanticAnalyzer(),
                'nlp_analyzer': AdvancedNLPAnalyzer(),
                'deep_semantic_model': DeepSemanticModel().to(self.device),
            }
            
            # Initialize text style transfer with text generator
            self._models['text_style_transfer'] = TextStyleTransfer(self._models['text_generator'])
            
            self._initialized = True
            logger.info("ML models initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def is_initialized(self) -> bool:
        """Check if models are initialized"""
        return self._initialized
    
    async def generate_complete_wrapped(self, 
                                       analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete wrapped experience asynchronously"""
        
        if not self.is_initialized():
            self.initialize()
        
        logger.info("Generating complete wrapped experience...")
        
        start_time = time.time()
        
        # Run all generation tasks in parallel
        tasks = [
            self._generate_text_content(analysis_data),
            self._generate_cover_art(analysis_data),
            self._generate_insights(analysis_data),
            self._generate_achievements(analysis_data),
        ]
        
        results = await asyncio.gather(*tasks)
        
        wrapped_result = {
            'text_content': results[0],
            'cover_art': results[1],
            'insights': results[2],
            'achievements': results[3],
            'generation_time': time.time() - start_time,
            'model_info': {
                'device': self.device,
                'models_used': list(self._models.keys()),
            }
        }
        
        return wrapped_result
    
    async def _generate_text_content(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Generate text content asynchronously"""
        
        loop = asyncio.get_event_loop()
        
        return await loop.run_in_executor(
            self.executor,
            self._sync_generate_text_content,
            data
        )
    
    def _sync_generate_text_content(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Synchronous text generation"""
        
        generator = self._models['text_generator']
        style_transfer = self._models['text_style_transfer']
        
        primary_topic = data.get('topics', {}).get('primary_topic', 'ChatGPT')
        sentiment = data.get('sentiment', {}).get('sentiment_label', 'positive')
        persona = data.get('sentiment', {}).get('persona', 'Curious Learner')
        
        result = {
            'tagline': generator.generate_tagline(primary_topic, sentiment),
            'persona_description': generator.generate_persona_description(persona, data),
            'insights': generator.generate_insights(data),
            'poetic_summary': style_transfer.transfer(
                primary_topic,
                style='poetic'
            ) if style_transfer else "Your Year in AI",
            'casual_summary': style_transfer.transfer(
                primary_topic,
                style='casual'
            ) if style_transfer else f"You explored {primary_topic}",
        }
        
        return result
    
    async def _generate_cover_art(self, data: Dict[str, Any]) -> Optional[str]:
        """Generate cover art asynchronously"""
        
        loop = asyncio.get_event_loop()
        
        return await loop.run_in_executor(
            self.executor,
            self._sync_generate_cover_art,
            data
        )
    
    def _sync_generate_cover_art(self, data: Dict[str, Any]) -> Optional[str]:
        """Synchronous cover art generation"""
        
        try:
            generator = self._models['image_generator']
            
            persona = data.get('sentiment', {}).get('persona', 'Curious Explorer')
            sentiment = data.get('sentiment', {}).get('sentiment_label', 'positive')
            primary_topic = data.get('topics', {}).get('primary_topic', 'ChatGPT')
            
            image = generator.generate_cover_art(
                title="ChatGPT Wrapped 2024",
                subtitle="Your Year in Conversations",
                persona=persona,
                primary_emotion=sentiment
            )
            
            # Save image temporarily
            image_path = f"/tmp/wrapped_cover_{int(time.time())}.png"
            image.save(image_path)
            
            return image_path
        
        except Exception as e:
            logger.error(f"Error generating cover art: {e}")
            return None
    
    async def _generate_insights(self, data: Dict[str, Any]) -> List[str]:
        """Generate insights asynchronously"""
        
        loop = asyncio.get_event_loop()
        
        return await loop.run_in_executor(
            self.executor,
            self._sync_generate_insights,
            data
        )
    
    def _sync_generate_insights(self, data: Dict[str, Any]) -> List[str]:
        """Synchronous insight generation"""
        
        generator = self._models['text_generator']
        semantic = self._models['semantic_analyzer']
        
        # Generate text insights
        text_insights = generator.generate_insights(data, num_insights=3)
        
        # Generate semantic insights
        messages = data.get('messages', [])
        if messages:
            key_concepts = semantic.extract_key_concepts(
                ' '.join([m.get('content', '') for m in messages[:100]]),
                top_k=3
            )
            semantic_insights = [f"Key concept: {concept}" for concept in key_concepts]
        else:
            semantic_insights = []
        
        return text_insights + semantic_insights
    
    async def _generate_achievements(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate achievement badges asynchronously"""
        
        loop = asyncio.get_event_loop()
        
        return await loop.run_in_executor(
            self.executor,
            self._sync_generate_achievements,
            data
        )
    
    def _sync_generate_achievements(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Synchronous achievement generation"""
        
        achievements = []
        
        metrics = data.get('message_counts', {})
        total_messages = metrics.get('total', 0)
        user_messages = metrics.get('user', 0)
        
        # Achievement definitions
        achievement_rules = [
            {
                'id': 'conversation_starter',
                'name': 'Conversation Starter',
                'description': 'Initiated conversations',
                'condition': user_messages > 10,
                'icon': '💬'
            },
            {
                'id': 'marathon_runner',
                'name': 'Marathon Runner',
                'description': 'Over 50 total messages',
                'condition': total_messages > 50,
                'icon': '🏃'
            },
            {
                'id': 'knowledge_seeker',
                'name': 'Knowledge Seeker',
                'description': 'Diverse topic exploration',
                'condition': data.get('topics', {}).get('topic_diversity', 0) > 3,
                'icon': '📚'
            },
            {
                'id': 'night_owl',
                'name': 'Night Owl',
                'description': 'Most active in evening hours',
                'condition': True,  # Placeholder
                'icon': '🌙'
            },
            {
                'id': 'sentiment_explorer',
                'name': 'Sentiment Explorer',
                'description': 'Expressed diverse emotions',
                'condition': True,  # Placeholder
                'icon': '🎭'
            }
        ]
        
        for achievement in achievement_rules:
            if achievement['condition']:
                achievements.append({
                    'id': achievement['id'],
                    'name': achievement['name'],
                    'description': achievement['description'],
                    'icon': achievement['icon'],
                    'unlocked': True,
                })
        
        return achievements
    
    @lru_cache(maxsize=128)
    def _get_cached_inference(self, cache_key: str) -> Optional[Any]:
        """Get cached inference result"""
        
        if cache_key in self.inference_cache:
            cached_item = self.inference_cache[cache_key]
            if time.time() - cached_item['timestamp'] < self.cache_ttl:
                return cached_item['result']
            else:
                del self.inference_cache[cache_key]
        
        return None
    
    def _cache_inference(self, cache_key: str, result: Any):
        """Cache inference result"""
        
        self.inference_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
    
    def clear_cache(self):
        """Clear inference cache"""
        self.inference_cache.clear()
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded models"""
        
        stats = {
            'device': self.device,
            'initialized': self._initialized,
            'models_loaded': list(self._models.keys()),
            'cache_size': len(self.inference_cache),
        }
        
        # Model-specific stats
        if self._initialized:
            for name, model in self._models.items():
                if isinstance(model, torch.nn.Module):
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    
                    stats[f'{name}_params'] = {
                        'total': total_params,
                        'trainable': trainable_params,
                    }
        
        return stats
