#!/usr/bin/env python3
"""
YOLOv8 Training Script for PS-05 Layout Detection

Simple training script using YOLOv8 for layout detection.
"""

import argparse
import logging
import yaml
import json
from pathlib import Path
from typing import Dict, List
import torch
import shutil
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """Load training configuration."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def train_yolo_model(config: Dict, dataset_yaml: str, output_dir: str, epochs: int = 50) -> bool:
    """Train YOLOv8 model for layout detection.
    
    Args:
        config: Training configuration
        dataset_yaml: Path to dataset YAML file
        output_dir: Output directory for model
        epochs: Number of training epochs
        
    Returns:
        True if training successful, False otherwise
    """
    try:
        # Check GPU availability
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"🚀 GPU Available: {gpu_name}")
            logger.info(f"💾 GPU Memory: {gpu_memory:.2f} GB")
            logger.info(f"🔧 CUDA Version: {torch.version.cuda}")
        else:
            device = torch.device('cpu')
            logger.warning("⚠️  CUDA not available, using CPU for training")
        
        # Get training parameters
        batch_size = config.get('training', {}).get('batch_size', 4)
        learning_rate = config.get('training', {}).get('learning_rate', 0.001)
        
        # Adjust batch size based on GPU memory
        if torch.cuda.is_available():
            if gpu_memory < 6:  # RTX 2070 has 8GB, but we'll be conservative
                batch_size = min(batch_size, 4)
                logger.info(f"📊 Adjusted batch size to {batch_size} for GPU memory constraints")
        
        logger.info(f"🎯 Starting YOLOv8 training for {epochs} epochs")
        logger.info(f"📁 Dataset: {dataset_yaml}")
        logger.info(f"⚙️  Batch size: {batch_size}, Learning rate: {learning_rate}")
        logger.info(f"💻 Output directory: {output_dir}")
        logger.info(f"🖥️  Device: {device}")
        
        # Import YOLOv8
        try:
            from ultralytics import YOLO
            logger.info("✅ YOLOv8 imported successfully")
        except ImportError:
            logger.error("❌ YOLOv8 not available. Install with: pip install ultralytics")
            return False
        
        # Load YOLOv8 model
        model = YOLO('yolov8x.pt')  # Load pre-trained YOLOv8x
        
        # Train the model
        logger.info("🚀 Starting training...")
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            device=device,
            project=output_dir,
            name='layout_detection',
            save=True,
            save_period=10,
            patience=20,
            lr0=learning_rate,
            weight_decay=0.0005,
            warmup_epochs=5,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            pose=12.0,
            kobj=2.0,
            label_smoothing=0.0,
            nbs=64,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            plots=True
        )
        
        logger.info("✅ YOLOv8 training completed successfully!")
        logger.info(f"📁 Model saved to: {output_dir}/layout_detection")
        
        # Save training results
        results_path = Path(output_dir) / "layout_detection" / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                "status": "completed",
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "device": str(device),
                "dataset": dataset_yaml
            }, f, indent=2)
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="YOLOv8 Training for PS-05")
    parser.add_argument("--config", default="configs/ps05_config.yaml", help="Config file path")
    parser.add_argument("--data", required=True, help="Dataset YAML file path")
    parser.add_argument("--output", default="models", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start training
    success = train_yolo_model(config, args.data, str(output_dir), args.epochs)
    
    if success:
        logger.info("🎉 Training completed successfully!")
        logger.info(f"📁 Model saved to: {output_dir}/layout_detection")
    else:
        logger.error("❌ Training failed!")

if __name__ == "__main__":
    main()
