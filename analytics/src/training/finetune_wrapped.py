"""
Custom Fine-tuning Scripts
Train models specifically for ChatGPT Wrapped
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatGPTWrappedFinetuner:
    """Fine-tune models specifically for Wrapped experience"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_dir = Path("./models/checkpoints")
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def fine_tune_text_generator(self,
                                training_data: List[str],
                                epochs: int = 5,
                                batch_size: int = 8) -> str:
        """Fine-tune text generation on Wrapped-specific data"""
        
        logger.info("Fine-tuning text generator for ChatGPT Wrapped...")
        
        from models.text_generation.advanced_text_generator import AdvancedTextGenerator
        from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
        
        generator = AdvancedTextGenerator(device=self.device)
        
        # Create training data
        self._save_training_data(training_data, "text_training.txt")
        
        # Prepare dataset
        train_dataset = TextDataset(
            tokenizer=generator.tokenizer,
            file_path="text_training.txt",
            block_size=128,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=generator.tokenizer,
            mlm=False
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.model_dir / "text_generator"),
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=100,
            save_total_limit=2,
            logging_steps=50,
            learning_rate=5e-5,
        )
        
        # Trainer
        trainer = Trainer(
            model=generator.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )
        
        trainer.train()
        
        checkpoint_path = str(self.model_dir / "text_generator" / "final")
        generator.model.save_pretrained(checkpoint_path)
        generator.tokenizer.save_pretrained(checkpoint_path)
        
        logger.info(f"Text generator fine-tuned and saved to {checkpoint_path}")
        return checkpoint_path
    
    def fine_tune_sentiment_classifier(self,
                                      labeled_data: List[Dict[str, Any]],
                                      epochs: int = 5) -> str:
        """Fine-tune sentiment classifier"""
        
        logger.info("Fine-tuning sentiment classifier...")
        
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            TextClassificationDataset
        )
        
        model_name = "distilbert-base-uncased"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3  # negative, neutral, positive
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Prepare data
        self._save_classification_data(labeled_data, "sentiment_training.json")
        
        # Create dataset
        class SentimentDataset(torch.utils.data.Dataset):
            def __init__(self, data, tokenizer):
                self.tokenizer = tokenizer
                self.texts = [d['text'] for d in data]
                self.labels = [d['label'] for d in data]
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                encoding = self.tokenizer(
                    self.texts[idx],
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors='pt'
                )
                return {
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze(),
                    'labels': torch.tensor(self.labels[idx])
                }
        
        dataset = SentimentDataset(labeled_data, tokenizer)
        
        training_args = TrainingArguments(
            output_dir=str(self.model_dir / "sentiment_classifier"),
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            learning_rate=2e-5,
            save_steps=100,
            save_total_limit=2,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        
        trainer.train()
        
        checkpoint_path = str(self.model_dir / "sentiment_classifier" / "final")
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        
        logger.info(f"Sentiment classifier fine-tuned and saved to {checkpoint_path}")
        return checkpoint_path
    
    def fine_tune_image_generator(self,
                                 image_prompts: List[Dict[str, str]],
                                 epochs: int = 5) -> str:
        """Fine-tune image generator on Wrapped aesthetic"""
        
        logger.info("Fine-tuning image generator for Wrapped aesthetic...")
        
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
        from diffusers.optimization import get_cosine_schedule_with_warmup
        
        pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipeline.to(self.device)
        
        # This is a simplified approach - actual fine-tuning would require
        # custom training loop with image data
        
        checkpoint_path = str(self.model_dir / "image_generator" / "final")
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config = {
            'model': 'stable-diffusion-v1-5',
            'custom_aesthetic': 'cyberpunk-neon',
            'training_prompts': image_prompts,
            'epochs': epochs
        }
        
        with open(Path(checkpoint_path) / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Image generator configured and saved to {checkpoint_path}")
        return checkpoint_path
    
    def _save_training_data(self, texts: List[str], filename: str):
        """Save texts for training"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(texts))
    
    def _save_classification_data(self, data: List[Dict[str, Any]], filename: str):
        """Save labeled data for classification"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

class ModelEnsemble:
    """Combine multiple models for better predictions"""
    
    def __init__(self, models: List[torch.nn.Module], weights: List[float] = None):
        self.models = models
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def predict(self, inputs):
        """Make ensemble prediction"""
        
        predictions = []
        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                pred = model(inputs)
            predictions.append(pred * weight)
        
        ensemble_pred = torch.stack(predictions).sum(dim=0)
        return ensemble_pred
    
    def add_model(self, model: torch.nn.Module, weight: float = 1.0):
        """Add model to ensemble"""
        self.models.append(model)
        total_weight = sum(self.weights) + weight
        self.weights = [w * sum(self.weights) / total_weight for w in self.weights]
        self.weights.append(weight / total_weight)
