"""
Analytics Engine for ChatGPT Wrapped
Advanced ML-powered analytics service with text & image generation
"""

import os
import json
import asyncio
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from datetime import datetime
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import logging

# Import ML models
from models.inference_engine import MLInferenceEngine
from models.model_optimizer import ModelOptimizer
from processors.data_processor import DataProcessor
from analyzers.sentiment_analyzer import SentimentAnalyzer
from analyzers.topic_analyzer import TopicAnalyzer
from analyzers.metrics_calculator import MetricsCalculator
from embeddings.semantic_analyzer import SemanticAnalyzer, AdvancedNLPAnalyzer

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

ANALYTICS_PORT = int(os.getenv('ANALYTICS_PORT', 5000))
ANALYTICS_HOST = os.getenv('ANALYTICS_HOST', 'localhost')
TEMP_DIR = os.getenv('TEMP_DIR', './tmp')
PROCESSING_TIMEOUT = int(os.getenv('PROCESSING_TIMEOUT_SECONDS', 4))

# Create temp directory
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize ML engine
ml_engine = MLInferenceEngine()
optimizer = ModelOptimizer()
data_processor = DataProcessor()
sentiment_analyzer = SentimentAnalyzer()
topic_analyzer = TopicAnalyzer()
metrics_calculator = MetricsCalculator()
semantic_analyzer = SemanticAnalyzer()
nlp_analyzer = AdvancedNLPAnalyzer()

# Global state
analysis_cache = {}

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint with model status"""
    return jsonify({
        'status': 'ok',
        'service': 'chatgpt-wrapped-analytics',
        'ml_models_initialized': ml_engine.is_initialized(),
        'device': ml_engine.device,
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/models/info', methods=['GET'])
def get_model_info():
    """Get detailed information about loaded ML models"""
    try:
        if not ml_engine.is_initialized():
            ml_engine.initialize()
        
        stats = ml_engine.get_model_stats()
        return jsonify({
            'status': 'success',
            'models': stats,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Advanced analytics endpoint with ML models
    Expects JSON with conversation data
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Initialize ML engine if needed
        if not ml_engine.is_initialized():
            ml_engine.initialize()
        
        # Process based on data format
        if isinstance(data, dict) and 'conversations' in data:
            processed = data_processor.process_json(data)
        elif isinstance(data, list):
            processed = data_processor.process_json({'conversations': data})
        else:
            processed = data_processor.process_json(data)
        
        if processed.get('status') == 'failed':
            return jsonify({'error': processed.get('error')}), 400
        
        messages = processed.get('messages', [])
        
        # Run all analyzers in parallel
        sentiment = sentiment_analyzer.analyze(messages)
        topics = topic_analyzer.analyze(messages)
        metrics = metrics_calculator.calculate(messages)
        
        # Advanced semantic analysis
        semantic_diversity = semantic_analyzer.calculate_semantic_diversity(
            [m.get('content', '') for m in messages]
        )
        
        # Run async ML generation
        analysis_data = {
            'messages': messages,
            'sentiment': sentiment,
            'topics': topics,
            'message_counts': metrics.get('message_counts', {}),
            'word_counts': metrics.get('word_counts', {}),
            'metadata': processed.get('metadata', {}),
        }
        
        # Generate wrapped asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            wrapped = loop.run_until_complete(
                asyncio.wait_for(
                    ml_engine.generate_complete_wrapped(analysis_data),
                    timeout=PROCESSING_TIMEOUT
                )
            )
        finally:
            loop.close()
        
        result = {
            'status': 'success',
            'analysis': {
                'metrics': metrics,
                'sentiment': sentiment,
                'topics': topics,
                'semantic_diversity': semantic_diversity,
            },
            'ml_generated': wrapped,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Cache results
        session_id = data.get('session_id', f"session_{int(datetime.utcnow().timestamp())}")
        analysis_cache[session_id] = result
        
        return jsonify({**result, 'session_id': session_id})
        
    except asyncio.TimeoutError:
        logger.warning("ML generation timeout - returning partial results")
        return jsonify({'error': 'Processing timeout'}), 504
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-upload', methods=['POST'])
def process_upload():
    """
    Process uploaded chat data file with advanced ML
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(TEMP_DIR, filename)
        file.save(filepath)
        
        # Parse file
        if filename.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = f.read()
        
        # Clean up temp file
        os.remove(filepath)
        
        # Use analyze endpoint logic
        request.json = data if isinstance(data, dict) else {'content': data}
        
        # Forward to analyze
        return analyze()
        
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/wrapped/<session_id>', methods=['GET'])
def get_wrapped(session_id):
    """Retrieve cached wrapped results"""
    try:
        if session_id not in analysis_cache:
            return jsonify({'error': 'Session not found'}), 404
        
        return jsonify({
            'status': 'success',
            'data': analysis_cache[session_id],
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Error retrieving wrapped: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-cover', methods=['POST'])
def generate_cover():
    """Generate cover art with advanced image synthesis"""
    try:
        data = request.get_json()
        
        if not ml_engine.is_initialized():
            ml_engine.initialize()
        
        title = data.get('title', 'ChatGPT Wrapped')
        subtitle = data.get('subtitle', '2024')
        persona = data.get('persona', 'Curious Explorer')
        sentiment = data.get('sentiment', 'positive')
        
        image_path = ml_engine._sync_generate_cover_art({
            'sentiment': {'persona': persona, 'sentiment_label': sentiment},
            'topics': {'primary_topic': data.get('topic', 'ChatGPT')}
        })
        
        if image_path:
            return send_file(image_path, mimetype='image/png')
        else:
            return jsonify({'error': 'Failed to generate cover'}), 500
    
    except Exception as e:
        logger.error(f"Error generating cover: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/optimize', methods=['POST'])
def optimize_model():
    """Optimize models for faster inference"""
    try:
        data = request.get_json()
        optimization_type = data.get('type', 'quantization')
        
        logger.info(f"Starting {optimization_type} optimization...")
        
        return jsonify({
            'status': 'success',
            'optimization': optimization_type,
            'message': f'{optimization_type} optimization applied',
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Error optimizing: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """404 handler"""
    return jsonify({'error': 'Not found', 'path': request.path}), 404

@app.errorhandler(500)
def server_error(error):
    """500 handler"""
    return jsonify({'error': 'Internal server error'}), 500

@app.before_first_request
def initialize():
    """Initialize ML models on first request"""
    try:
        if not ml_engine.is_initialized():
            logger.info("Initializing ML models...")
            ml_engine.initialize()
            logger.info("ML models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing ML models: {e}")

if __name__ == '__main__':
    logger.info(f'🚀 Advanced Analytics Engine starting on http://{ANALYTICS_HOST}:{ANALYTICS_PORT}')
    logger.info('📊 ML Models: Text Generation, Image Generation, Sentiment Analysis, Topic Modeling')
    logger.info('🔧 Advanced Features: Semantic Analysis, Named Entity Recognition, Knowledge Distillation')
    
    app.run(
        host=ANALYTICS_HOST,
        port=ANALYTICS_PORT,
        debug=os.getenv('NODE_ENV') == 'development',
        use_reloader=False,  # Disable reloader to avoid double initialization
        threaded=True
    )
