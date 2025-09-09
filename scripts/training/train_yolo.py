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
import platform
import glob
import cv2

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

def train_yolo_model(
    config: Dict,
    dataset_yaml: str,
    output_dir: str,
    epochs: int = 50,
    weights: str = "",
    imgsz: int = 640,
    batch_override: int = 0,
    lr0_override: float = 0.0,
    workers: int = 2,
    device_str: str = "0"
) -> bool:
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
        
        # Get training parameters (CLI overrides > config > defaults)
        batch_size = batch_override or config.get('training', {}).get('batch_size', 8)
        learning_rate = lr0_override or config.get('training', {}).get('learning_rate', 0.001)
        # On Windows, set workers=0 to avoid DataLoader pickling issues ("Ran out of input")
        if platform.system().lower().startswith('win'):
            workers = 0
        
        # Adjust batch size based on GPU memory
        if torch.cuda.is_available():
            if gpu_memory < 6:  # RTX 2070 has 8GB, but we'll be conservative
                batch_size = min(batch_size, 4)
                logger.info(f"📊 Adjusted batch size to {batch_size} for GPU memory constraints")
        
        logger.info(f"🎯 Starting YOLOv8 training for {epochs} epochs")
        logger.info(f"📁 Dataset: {dataset_yaml}")
        logger.info(f"⚙️  Batch size: {batch_size}, Learning rate: {learning_rate}, ImgSz: {imgsz}, Workers: {workers}")
        logger.info(f"💻 Output directory: {output_dir}")
        logger.info(f"🖥️  Device: {device}")
        
        # Import YOLOv8
        try:
            from ultralytics import YOLO
            logger.info("✅ YOLOv8 imported successfully")
        except ImportError:
            logger.error("❌ YOLOv8 not available. Install with: pip install ultralytics")
            return False
        
        # Resolve starting weights
        default_layout_weights = Path("models/layout_yolo_6class.pt")
        if not weights:
            if default_layout_weights.exists():
                weights = str(default_layout_weights)
                logger.info(f"🏁 Using default 6-class layout weights: {weights}")
            else:
                weights = 'yolov8x.pt'
                logger.info("🏁 Defaulting to COCO weights 'yolov8x.pt' (consider providing 6-class weights)")

        # Load YOLOv8 model from weights
        model = YOLO(weights)
        
        # Clean any stale cache files that can cause DataLoader issues on Windows
        try:
            ds_root = Path(dataset_yaml).parent
            for cache_path in ds_root.rglob("*.cache"):
                try:
                    cache_path.unlink()
                except Exception:
                    pass
        except Exception:
            pass

        # Build explicit image manifests to avoid iterator issues (Windows safety)
        ds_root = Path(dataset_yaml).parent
        classes_cfg = (
            config.get('models', {})
                  .get('layout', {})
                  .get('classes', ['Background', 'Text', 'Title', 'List', 'Table', 'Figure'])
        )

        def _gather(split: str):
            img_dir = ds_root / split / 'images'
            lbl_dir = ds_root / split / 'labels'
            manifest = ds_root / f"{split}.txt"
            img_paths = []
            if img_dir.exists():
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
                    img_paths.extend(sorted(img_dir.rglob(ext)))
            kept = 0
            list_paths = []
            with open(manifest, 'w', encoding='utf-8') as mf:
                for ip in img_paths:
                    try:
                        im = cv2.imread(str(ip))
                        if im is None or im.size == 0:
                            continue
                        # Ensure label file exists (empty allowed)
                        lbl = lbl_dir / f"{ip.stem}.txt"
                        lbl.parent.mkdir(parents=True, exist_ok=True)
                        if not lbl.exists():
                            lbl.write_text("", encoding='utf-8')
                        norm = str(ip.resolve()).replace('\\', '/')
                        mf.write(norm + "\n")
                        list_paths.append(norm)
                        kept += 1
                    except Exception:
                        continue
            logger.info(f"Manifest {split}: {kept} files")
            return manifest, list_paths

        train_manifest = _gather('train')[0]
        val_manifest = _gather('val')[0]
        # Keep using YAML for Ultralytics compatibility

        # Train the model
        logger.info("🚀 Starting training...")
        # Ultralytics accepts device as int/str; prefer '0' for first GPU
        device_arg = device_str if torch.cuda.is_available() else 'cpu'
        def _run_train(wk: int):
            return model.train(
                data=dataset_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=imgsz,
                device=device_arg,
                workers=wk,
                cache=False,
                rect=False,
                amp=True,
                val=False,  # Disable validation if no val set
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
                plots=True
            )

        try:
            results = _run_train(workers)
        except Exception as e:
            # Auto-retry with workers=0 if DataLoader errors (common on Windows)
            if 'Ran out of input' in str(e) or 'EOFError' in str(e):
                logger.warning("DataLoader error detected; retrying with workers=0")
                results = _run_train(0)
            else:
                raise
        
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
    parser.add_argument("--weights", type=str, default="", help="Starting weights path (default: models/layout_yolo_6class.pt if exists, else yolov8x.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=0, help="Override batch size (0 = auto/config)")
    parser.add_argument("--lr0", type=float, default=0.0, help="Override initial learning rate (0 = config default)")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers (set low on Windows)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device string/index (e.g., '0' or 'cpu')")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start training
    success = train_yolo_model(
        config,
        args.data,
        str(output_dir),
        args.epochs,
        args.weights,
        imgsz=args.imgsz,
        batch_override=args.batch,
        lr0_override=args.lr0,
        workers=args.workers,
        device_str=args.device
    )
    
    if success:
        logger.info("🎉 Training completed successfully!")
        logger.info(f"📁 Model saved to: {output_dir}/layout_detection")
    else:
        logger.error("❌ Training failed!")

if __name__ == "__main__":
    main()
