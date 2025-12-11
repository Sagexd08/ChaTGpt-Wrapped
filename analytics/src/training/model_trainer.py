"""
Advanced Model Training Pipeline
Fine-tune models on custom datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from typing import List, Dict, Tuple, Any
import json
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Advanced trainer for fine-tuning models"""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01):
        
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.training_history = {
            'loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def train_epoch(self, 
                   train_loader: DataLoader,
                   criterion: nn.Module,
                   scheduler=None) -> float:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            else:
                batch = tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t 
                            for t in batch)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch)
            loss = criterion(outputs, batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        self.training_history['loss'].append(avg_loss)
        
        return avg_loss
    
    def validate(self,
                val_loader: DataLoader,
                criterion: nn.Module) -> float:
        """Validate model"""
        
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                else:
                    batch = tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t 
                                for t in batch)
                
                outputs = self.model(batch)
                loss = criterion(outputs, batch)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.training_history['val_loss'].append(avg_loss)
        
        return avg_loss
    
    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             num_epochs: int = 10,
             criterion: nn.Module = None,
             save_path: str = None) -> Dict[str, Any]:
        """Complete training loop"""
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, criterion, scheduler)
            logger.info(f"Training Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate(val_loader, criterion)
            logger.info(f"Validation Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if save_path:
                    self.save_checkpoint(save_path, epoch, best_val_loss)
                    logger.info(f"Model saved to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break
        
        return {
            'best_loss': best_val_loss,
            'history': self.training_history,
            'epochs_trained': epoch + 1
        }
    
    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """Save model checkpoint"""
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'training_history': self.training_history
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Loaded checkpoint from {path}")
        logger.info(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")

class DataAugmentation:
    """Data augmentation techniques for training"""
    
    @staticmethod
    def back_translate(text: str, forward_lang: str = "de", back_lang: str = "en") -> str:
        """Back-translation augmentation"""
        try:
            from transformers import MarianMTModel, MarianTokenizer
            
            # Forward translation
            model_name = f'Helsinki-NLP/Tatoeba-MT-models::{forward_lang}-en'
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            translated = model.generate(**inputs)
            forward = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
            
            # Back translation
            model_name = f'Helsinki-NLP/Tatoeba-MT-models::{back_lang}-{forward_lang}'
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            
            inputs = tokenizer(forward, return_tensors="pt", padding=True)
            translated = model.generate(**inputs)
            back = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
            
            return back
        except Exception as e:
            logger.warning(f"Back-translation failed: {e}")
            return text
    
    @staticmethod
    def paraphrase(text: str, num_paraphrases: int = 3) -> List[str]:
        """Generate paraphrases of text"""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            tokenizer = AutoTokenizer.from_pretrained("t5-base")
            model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
            
            input_ids = tokenizer.encode(f"paraphrase: {text} </s>", return_tensors="pt")
            outputs = model.generate(
                input_ids=input_ids,
                max_length=256,
                num_beams=5,
                num_return_sequences=num_paraphrases,
                temperature=1.5
            )
            
            return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        except Exception as e:
            logger.warning(f"Paraphrase generation failed: {e}")
            return [text]
    
    @staticmethod
    def mixup(text1: str, text2: str) -> str:
        """Mixup augmentation combining two texts"""
        words1 = text1.split()
        words2 = text2.split()
        
        mixed = []
        for w1, w2 in zip(words1, words2):
            mixed.extend([w1, w2])
        
        return ' '.join(mixed[:max(len(words1), len(words2))])
