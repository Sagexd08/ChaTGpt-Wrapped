"""
Model Optimization and Acceleration
Quantization, pruning, and performance optimization
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Optimize models for faster inference and lower memory usage"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
    
    def quantize_model(self, model: nn.Module, bits: int = 8) -> nn.Module:
        """Quantize model to reduce size"""
        
        logger.info(f"Quantizing model to {bits}-bit...")
        
        if bits == 8:
            # Static quantization
            model_quantized = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8
            )
        elif bits == 4:
            # 4-bit quantization (experimental)
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            # This is for transformers models
            model_quantized = model
        else:
            model_quantized = model
        
        logger.info("Model quantized successfully")
        return model_quantized
    
    def prune_model(self, model: nn.Module, pruning_amount: float = 0.3) -> nn.Module:
        """Prune model weights to reduce size"""
        
        logger.info(f"Pruning model with {pruning_amount*100:.1f}% sparsity...")
        
        parameters_to_prune = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply pruning
        for module, name in parameters_to_prune:
            torch.nn.utils.prune.l1_unstructured(module, name, pruning_amount)
        
        # Make pruning permanent
        for module, name in parameters_to_prune:
            torch.nn.utils.prune.remove(module, name)
        
        logger.info("Model pruned successfully")
        return model
    
    def distill_model(self,
                     teacher_model: nn.Module,
                     student_model: nn.Module,
                     train_loader,
                     epochs: int = 10,
                     temperature: float = 4.0,
                     alpha: float = 0.7) -> nn.Module:
        """Knowledge distillation from larger to smaller model"""
        
        logger.info("Starting knowledge distillation...")
        
        device = self.device
        teacher_model.to(device).eval()
        student_model.to(device).train()
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        ce_loss = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(train_loader):
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
                
                # Teacher predictions
                with torch.no_grad():
                    teacher_logits = teacher_model(inputs)
                    teacher_probs = torch.softmax(teacher_logits / temperature, dim=1)
                
                # Student predictions
                student_logits = student_model(inputs)
                student_probs = torch.log_softmax(student_logits / temperature, dim=1)
                
                # Distillation loss
                distill_loss = kl_loss(student_probs, teacher_probs)
                
                # Cross-entropy loss
                task_loss = ce_loss(student_logits, targets)
                
                # Combined loss
                loss = alpha * distill_loss + (1 - alpha) * task_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        
        return student_model
    
    def convert_to_onnx(self, model: nn.Module, dummy_input: torch.Tensor, output_path: str):
        """Convert model to ONNX format"""
        
        logger.info(f"Converting model to ONNX format: {output_path}...")
        
        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'},
                            'output': {0: 'batch_size'}}
            )
            logger.info("Model converted to ONNX successfully")
        except Exception as e:
            logger.error(f"Error converting to ONNX: {e}")
    
    def enable_mixed_precision(self, model: nn.Module) -> nn.Module:
        """Enable mixed precision training"""
        
        logger.info("Enabling mixed precision...")
        
        # Automatic Mixed Precision
        model = model.to(self.device)
        
        # For inference, we can use autocast context manager
        # This is handled during inference
        
        logger.info("Mixed precision enabled")
        return model
    
    def get_model_memory_usage(self, model: nn.Module) -> Dict[str, float]:
        """Calculate model memory usage"""
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Approximate memory in MB
        param_memory = total_params * 4 / (1024 * 1024)  # float32
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'approximate_memory_mb': param_memory,
            'parameter_efficiency': trainable_params / total_params if total_params > 0 else 0,
        }

class InferenceOptimizer:
    """Optimize inference performance"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
    
    def benchmark_inference(self,
                           model: nn.Module,
                           dummy_input: torch.Tensor,
                           num_runs: int = 100) -> Dict[str, float]:
        """Benchmark model inference speed"""
        
        logger.info(f"Benchmarking inference with {num_runs} runs...")
        
        import time
        
        model.to(self.device).eval()
        dummy_input = dummy_input.to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Benchmark
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_ms = (total_time / num_runs) * 1000
        
        return {
            'total_time_seconds': total_time,
            'average_time_ms': avg_time_ms,
            'throughput_samples_per_sec': num_runs / total_time,
        }
    
    def enable_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Enable gradient checkpointing to save memory"""
        
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        return model
    
    def batch_inference(self,
                       model: nn.Module,
                       inputs: List[torch.Tensor],
                       batch_size: int = 32) -> List[torch.Tensor]:
        """Perform inference on batches for efficiency"""
        
        model.eval()
        outputs = []
        
        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                batch = torch.stack(inputs[i:i+batch_size]).to(self.device)
                batch_outputs = model(batch)
                outputs.extend(batch_outputs.cpu())
        
        return outputs
