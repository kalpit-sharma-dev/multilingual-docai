#!/usr/bin/env python3
"""
GPU-Optimized Training Service for PS-05 Challenge
Optimized for A100 GPU with PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import json
import time
from dataclasses import dataclass
from transformers import (
    LayoutLMv3Processor, 
    LayoutLMv3ForSequenceClassification,
    TrainingArguments,
    Trainer
)
from ultralytics import YOLO
import easyocr
from PIL import Image
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for GPU-optimized training."""
    batch_size: int = 16  # Optimized for A100 memory
    learning_rate: float = 1e-4
    epochs: int = 50
    gpu_acceleration: bool = True
    mixed_precision: bool = True  # FP16 for A100
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100

class DocumentDataset(Dataset):
    """Custom dataset for document understanding."""
    
    def __init__(self, data_dir: str, processor=None, transform=None):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.transform = transform
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load dataset samples."""
        samples = []
        image_dir = self.data_dir / "images"
        label_dir = self.data_dir / "labels"
        
        if image_dir.exists() and label_dir.exists():
            for img_file in image_dir.glob("*.jpg"):
                label_file = label_dir / f"{img_file.stem}.json"
                if label_file.exists():
                    samples.append({
                        "image_path": str(img_file),
                        "label_path": str(label_file)
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Load labels
        with open(sample["label_path"], 'r') as f:
            labels = json.load(f)
        
        if self.processor:
            # Process with LayoutLMv3 processor
            encoding = self.processor(
                image,
                text=labels.get("text", ""),
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            return {
                "pixel_values": encoding["pixel_values"].squeeze(),
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": torch.tensor(labels.get("class_id", 0))
            }
        else:
            return {
                "image": image,
                "labels": labels
            }

class GPUTrainingService:
    """
    GPU-optimized training service for document understanding models.
    Optimized for A100 GPU with PyTorch.
    """
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.device = self._setup_device()
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        
    def _setup_device(self) -> str:
        """Setup GPU device with optimization."""
        if self.config.gpu_acceleration and torch.cuda.is_available():
            device = "cuda"
            
            # A100-specific optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable mixed precision for A100
            if self.config.mixed_precision:
                torch.backends.cuda.matmul.allow_tf16 = True
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.9)
            
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            logger.info(f"Mixed Precision: {self.config.mixed_precision}")
        else:
            device = "cpu"
            logger.info("Using CPU training")
        
        return device
    
    def setup_layout_training(self, model_name: str = "microsoft/layoutlmv3-base"):
        """Setup LayoutLMv3 training for document layout classification."""
        try:
            logger.info("Setting up LayoutLMv3 training...")
            
            # Load model and processor
            self.models['layoutlmv3'] = LayoutLMv3ForSequenceClassification.from_pretrained(
                model_name,
                num_labels=6  # Background, Text, Title, List, Table, Figure
            )
            self.processor = LayoutLMv3Processor.from_pretrained(model_name)
            
            # Move to GPU
            if self.device == "cuda":
                self.models['layoutlmv3'].to(self.device)
            
            # Setup optimizer with weight decay
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.models['layoutlmv3'].named_parameters() 
                              if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.config.weight_decay,
                },
                {
                    "params": [p for n, p in self.models['layoutlmv3'].named_parameters() 
                              if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            
            self.optimizers['layoutlmv3'] = optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                eps=1e-8
            )
            
            # Setup learning rate scheduler
            self.schedulers['layoutlmv3'] = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizers['layoutlmv3'],
                T_0=10,
                T_mult=2
            )
            
            logger.info("LayoutLMv3 training setup completed!")
            
        except Exception as e:
            logger.error(f"Error setting up LayoutLMv3 training: {e}")
            raise
    
    def setup_yolo_training(self, model_path: str = "yolov8x.pt"):
        """Setup YOLOv8 training for object detection."""
        try:
            logger.info("Setting up YOLOv8 training...")
            
            # Load YOLO model
            self.models['yolo'] = YOLO(model_path)
            
            # Configure training parameters
            self.yolo_config = {
                'data': 'data.yaml',
                'epochs': self.config.epochs,
                'batch': self.config.batch_size,
                'imgsz': 640,
                'device': self.device,
                'workers': 8,
                'patience': 50,
                'save': True,
                'save_period': 10,
                'cache': True,
                'optimizer': 'AdamW',
                'lr0': self.config.learning_rate,
                'weight_decay': self.config.weight_decay,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'fl_gamma': 0.0,
                'label_smoothing': 0.0,
                'nbs': 64,
                'overlap_mask': True,
                'mask_ratio': 4,
                'dropout': 0.0,
                'val': True,
                'plots': True
            }
            
            logger.info("YOLOv8 training setup completed!")
            
        except Exception as e:
            logger.error(f"Error setting up YOLOv8 training: {e}")
            raise
    
    def train_layout_model(self, train_data_dir: str, val_data_dir: str, output_dir: str):
        """Train LayoutLMv3 model for document layout classification."""
        try:
            logger.info("Starting LayoutLMv3 training...")
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=self.config.epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                warmup_steps=self.config.warmup_steps,
                weight_decay=self.config.weight_decay,
                logging_dir=f"{output_dir}/logs",
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps,
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                fp16=self.config.mixed_precision and self.device == "cuda",
                dataloader_num_workers=4,
                remove_unused_columns=False,
                push_to_hub=False,
                report_to=None
            )
            
            # Create datasets
            train_dataset = DocumentDataset(train_data_dir, self.processor)
            val_dataset = DocumentDataset(val_data_dir, self.processor)
            
            # Create trainer
            trainer = Trainer(
                model=self.models['layoutlmv3'],
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.processor,
                data_collator=self._collate_fn
            )
            
            # Start training
            start_time = time.time()
            trainer.train()
            training_time = time.time() - start_time
            
            # Save final model
            final_model_path = f"{output_dir}/final_model"
            trainer.save_model(final_model_path)
            self.processor.save_pretrained(final_model_path)
            
            logger.info(f"LayoutLMv3 training completed in {training_time:.2f}s")
            
            return {
                "status": "completed",
                "training_time": training_time,
                "model_path": final_model_path,
                "config": self.config.__dict__
            }
            
        except Exception as e:
            logger.error(f"Error in LayoutLMv3 training: {e}")
            raise
    
    def train_yolo_model(self, data_yaml_path: str, output_dir: str):
        """Train YOLOv8 model for document object detection."""
        try:
            logger.info("Starting YOLOv8 training...")
            
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Start training
            start_time = time.time()
            results = self.models['yolo'].train(
                data=data_yaml_path,
                **self.yolo_config
            )
            training_time = time.time() - start_time
            
            # Get best model path
            best_model_path = results.save_dir / "weights" / "best.pt"
            
            logger.info(f"YOLOv8 training completed in {training_time:.2f}s")
            
            return {
                "status": "completed",
                "training_time": training_time,
                "model_path": str(best_model_path),
                "metrics": {
                    "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                    "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
                    "precision": results.results_dict.get("metrics/precision(B)", 0),
                    "recall": results.results_dict.get("metrics/recall(B)", 0)
                },
                "config": self.config.__dict__
            }
            
        except Exception as e:
            logger.error(f"Error in YOLOv8 training: {e}")
            raise
    
    def _collate_fn(self, batch):
        """Custom collate function for LayoutLMv3 training."""
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics and GPU usage."""
        stats = {
            "device": self.device,
            "models_loaded": list(self.models.keys()),
            "config": self.config.__dict__
        }
        
        if self.device == "cuda" and torch.cuda.is_available():
            stats.update({
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "gpu_memory_cached_gb": torch.cuda.memory_reserved() / 1e9
            })
        
        return stats
    
    def cleanup(self):
        """Cleanup GPU memory and resources."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared")
        
        # Clear models
        self.models.clear()
        self.optimizers.clear()
        self.schedulers.clear()
        logger.info("Training service cleaned up")
