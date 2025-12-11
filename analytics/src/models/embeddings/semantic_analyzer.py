"""
Advanced NLP and Embedding Models
Semantic analysis, clustering, and understanding
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from typing import List, Dict, Tuple, Any
import spacy

class SemanticAnalyzer:
    """Advanced semantic understanding using embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings_cache = {}
    
    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        """Generate embeddings for multiple texts"""
        return self.model.encode(texts, convert_to_tensor=True)
    
    def find_similar_messages(self,
                             query: str,
                             messages: List[str],
                             top_k: int = 5) -> List[Tuple[str, float]]:
        """Find semantically similar messages"""
        
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        message_embeddings = self.embed_texts(messages)
        
        # Calculate cosine similarities
        similarities = util.pytorch_cos_sim(query_embedding, message_embeddings)[0]
        
        # Get top K
        top_results = torch.topk(similarities, k=min(top_k, len(messages)))
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append((messages[idx], float(score)))
        
        return results
    
    def cluster_messages(self,
                        messages: List[str],
                        min_samples: int = 2,
                        eps: float = 0.5) -> Dict[int, List[str]]:
        """Cluster messages by semantic similarity"""
        
        embeddings = self.embed_texts(messages).cpu().numpy()
        
        # Use DBSCAN for density-based clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings)
        
        clusters = {}
        for label, message in zip(labels, messages):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(message)
        
        return clusters
    
    def calculate_semantic_diversity(self, messages: List[str]) -> float:
        """Calculate semantic diversity of messages (0-1)"""
        
        if len(messages) < 2:
            return 0.0
        
        embeddings = self.embed_texts(messages).cpu().numpy()
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                distances.append(dist)
        
        # Normalize to 0-1
        mean_distance = np.mean(distances) if distances else 0
        return min(mean_distance, 1.0)
    
    def extract_key_concepts(self, text: str, top_k: int = 5) -> List[str]:
        """Extract key concepts using semantic analysis"""
        
        # Split into sentences
        sentences = text.split('. ')
        if len(sentences) < 3:
            sentences = text.split(', ')
        
        # Get embeddings
        embeddings = self.embed_texts(sentences)
        
        # Calculate diversity
        center = torch.mean(embeddings, dim=0)
        distances = [float(util.pytorch_cos_sim(center, emb)) for emb in embeddings]
        
        # Get top diverse sentences
        top_indices = np.argsort(distances)[::-1][:top_k]
        concepts = [sentences[i] for i in top_indices]
        
        return concepts

class AdvancedNLPAnalyzer:
    """Advanced NLP using spaCy and custom models"""
    
    def __init__(self, model_name: str = "en_core_web_md"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Downloading {model_name}...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities and important terms"""
        
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            label = ent.label_
            if label not in entities:
                entities[label] = []
            entities[label].append(ent.text)
        
        return entities
    
    def analyze_syntax(self, text: str) -> Dict[str, Any]:
        """Analyze syntactic structure"""
        
        doc = self.nlp(text)
        
        return {
            'tokens': [(token.text, token.pos_) for token in doc],
            'noun_phrases': [chunk.text for chunk in doc.noun_chunks],
            'dependencies': [(token.text, token.dep_, token.head.text) for token in doc],
            'similarity_to_average': np.mean([token.vector_norm for token in doc]) if doc else 0,
        }
    
    def detect_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Detect linguistic features and writing style"""
        
        doc = self.nlp(text)
        tokens = [token for token in doc if not token.is_punct]
        
        # Calculate metrics
        avg_token_length = np.mean([len(token.text) for token in tokens]) if tokens else 0
        unique_words = len(set(token.text.lower() for token in tokens))
        
        # POS tag distribution
        pos_counts = {}
        for token in tokens:
            pos = token.pos_
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        return {
            'avg_token_length': avg_token_length,
            'lexical_diversity': unique_words / len(tokens) if tokens else 0,
            'pos_distribution': pos_counts,
            'complexity_score': (avg_token_length / 10) * (unique_words / len(tokens) if tokens else 0)
        }

class DeepSemanticModel(nn.Module):
    """Deep neural network for semantic understanding"""
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Task-specific heads
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # negative, neutral, positive
        )
        
        self.topic_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 10 topic categories
        )
        
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 8)  # 8 emotion categories
        )
    
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning multiple predictions"""
        
        encoded = self.encoder(embeddings)
        
        return {
            'sentiment': torch.softmax(self.sentiment_head(encoded), dim=-1),
            'topics': torch.softmax(self.topic_head(encoded), dim=-1),
            'emotions': torch.softmax(self.emotion_head(encoded), dim=-1),
        }
