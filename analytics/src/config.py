"""
Advanced Configuration & Constants
Central place for all configuration
"""

import os
from typing import Dict, Any

class MLConfig:
    """Machine Learning Configuration"""
    
    # Device settings
    DEVICE = os.getenv('ML_DEVICE', 'cuda' if __import__('torch').cuda.is_available() else 'cpu')
    
    # Text Generation
    TEXT_GEN_MODEL = "gpt2"
    TEXT_GEN_MAX_LENGTH = 150
    TEXT_GEN_TEMPERATURE = 0.7
    TEXT_GEN_TOP_P = 0.9
    TEXT_GEN_NUM_BEAMS = 3
    
    # Image Generation
    IMAGE_GEN_MODEL = "runwayml/stable-diffusion-v1-5"
    IMAGE_GEN_HEIGHT = 1024
    IMAGE_GEN_WIDTH = 1024
    IMAGE_GEN_STEPS = 50
    IMAGE_GEN_GUIDANCE_SCALE = 7.5
    
    # Embeddings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 768
    
    # NLP
    SPACY_MODEL = "en_core_web_md"
    
    # Optimization
    QUANTIZATION_BITS = 8
    PRUNING_AMOUNT = 0.3
    
    # Inference
    BATCH_SIZE = 32
    MAX_INFERENCE_TIME_SECONDS = 4
    
    # Caching
    CACHE_TTL = 3600  # 1 hour
    REDIS_ENABLED = os.getenv('REDIS_ENABLED', 'true').lower() == 'true'
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))

class AnalyticsConfig:
    """Analytics Engine Configuration"""
    
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    TEMP_DIR = os.getenv('TEMP_DIR', './tmp')
    PROCESSING_TIMEOUT = int(os.getenv('PROCESSING_TIMEOUT_SECONDS', 4))
    
    # Analytics parameters
    SENTIMENT_CLASSES = ['negative', 'neutral', 'positive']
    EMOTION_CLASSES = ['anger', 'fear', 'joy', 'sadness', 'surprise', 'disgust', 'neutral', 'anticipation']
    TOPIC_COUNT = 10
    ACHIEVEMENT_THRESHOLD = {
        'conversation_starter': 10,
        'marathon_runner': 50,
        'knowledge_seeker': 3,
    }

class AppConfig:
    """Application Configuration"""
    
    # Ports
    BACKEND_PORT = int(os.getenv('PORT', 3001))
    ANALYTICS_PORT = int(os.getenv('ANALYTICS_PORT', 5000))
    
    # Hosts
    BACKEND_HOST = os.getenv('BACKEND_HOST', 'localhost')
    ANALYTICS_HOST = os.getenv('ANALYTICS_HOST', 'localhost')
    
    # CORS
    CORS_ORIGIN = os.getenv('CORS_ORIGIN', 'http://localhost:3000')
    
    # Environment
    ENV = os.getenv('NODE_ENV', 'development')
    DEBUG = ENV == 'development'
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Feature flags
FEATURES = {
    'ml_text_generation': True,
    'ml_image_generation': True,
    'semantic_analysis': True,
    'model_optimization': True,
    'redis_caching': MLConfig.REDIS_ENABLED,
    'async_processing': True,
    'knowledge_distillation': True,
}

# Model paths
MODEL_PATHS = {
    'text_generator': os.getenv('TEXT_GEN_PATH', './models/text_generator'),
    'image_generator': os.getenv('IMAGE_GEN_PATH', './models/image_generator'),
    'sentiment_classifier': os.getenv('SENTIMENT_PATH', './models/sentiment'),
    'semantic_embeddings': os.getenv('EMBEDDINGS_PATH', './models/embeddings'),
}

# Achievement definitions
ACHIEVEMENTS = [
    {
        'id': 'conversation_starter',
        'name': 'Conversation Starter',
        'description': 'Initiated conversations with ChatGPT',
        'icon': '💬',
        'condition': lambda m: m.get('user', 0) > 10
    },
    {
        'id': 'marathon_runner',
        'name': 'Marathon Runner',
        'description': 'Over 50 total messages exchanged',
        'icon': '🏃',
        'condition': lambda m: m.get('total', 0) > 50
    },
    {
        'id': 'knowledge_seeker',
        'name': 'Knowledge Seeker',
        'description': 'Explored diverse topics',
        'icon': '📚',
        'condition': lambda m: True
    },
    {
        'id': 'night_owl',
        'name': 'Night Owl',
        'description': 'Most active during evening hours',
        'icon': '🌙',
        'condition': lambda m: True
    },
    {
        'id': 'sentiment_explorer',
        'name': 'Sentiment Explorer',
        'description': 'Expressed diverse emotional range',
        'icon': '🎭',
        'condition': lambda m: True
    },
    {
        'id': 'ai_whisperer',
        'name': 'AI Whisperer',
        'description': 'Crafted highly detailed prompts',
        'icon': '✨',
        'condition': lambda m: True
    },
    {
        'id': 'code_wizard',
        'name': 'Code Wizard',
        'description': 'Asked coding-related questions',
        'icon': '🧙',
        'condition': lambda m: True
    },
]

# Writing styles
WRITING_STYLES = {
    'poetic': {
        'temperature': 0.9,
        'top_p': 0.95,
        'keywords': ['beautiful', 'profound', 'lyrical']
    },
    'technical': {
        'temperature': 0.5,
        'top_p': 0.8,
        'keywords': ['precise', 'accurate', 'detailed']
    },
    'casual': {
        'temperature': 0.8,
        'top_p': 0.9,
        'keywords': ['friendly', 'relaxed', 'conversational']
    },
    'professional': {
        'temperature': 0.6,
        'top_p': 0.85,
        'keywords': ['formal', 'structured', 'clear']
    },
    'humorous': {
        'temperature': 0.95,
        'top_p': 0.98,
        'keywords': ['witty', 'playful', 'funny']
    },
}

# Image styles
IMAGE_STYLES = {
    'neon': 'high contrast, vibrant neon colors, glow effects',
    'hologram': 'holographic iridescent effect, color shifting',
    'glitch': 'digital glitch effect, scanlines, distortion',
    'plasma': 'plasma energy, electric effects, dynamic motion',
    'ethereal': 'soft glow, dreamy atmosphere, smooth gradients',
}

# Color palette (neon cyberpunk)
COLORS = {
    'neon_pink': '#FF006E',
    'neon_blue': '#00D9FF',
    'neon_green': '#39FF14',
    'neon_purple': '#B100FF',
    'cyber_dark': '#0a0e27',
    'cyber_darker': '#050612',
    'cyber_accent': '#1a1f3a',
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary"""
    return {
        'ml': {
            'device': MLConfig.DEVICE,
            'text_gen_model': MLConfig.TEXT_GEN_MODEL,
            'image_gen_model': MLConfig.IMAGE_GEN_MODEL,
            'embedding_model': MLConfig.EMBEDDING_MODEL,
            'quantization_bits': MLConfig.QUANTIZATION_BITS,
            'cache_ttl': MLConfig.CACHE_TTL,
        },
        'analytics': {
            'max_file_size': AnalyticsConfig.MAX_FILE_SIZE,
            'processing_timeout': AnalyticsConfig.PROCESSING_TIMEOUT,
            'sentiment_classes': AnalyticsConfig.SENTIMENT_CLASSES,
            'emotion_classes': AnalyticsConfig.EMOTION_CLASSES,
        },
        'app': {
            'backend_port': AppConfig.BACKEND_PORT,
            'analytics_port': AppConfig.ANALYTICS_PORT,
            'environment': AppConfig.ENV,
            'debug': AppConfig.DEBUG,
        },
        'features': FEATURES,
        'colors': COLORS,
    }
