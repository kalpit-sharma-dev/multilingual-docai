#!/usr/bin/env python3
"""
Stage 1 Training Script for PS-05 Layout Detection

Complete training pipeline for Stage 1:
1. Prepare dataset in YOLO format
2. Train layout detection model
3. Validate model performance
4. Save trained model
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

from scripts.prepare_dataset import prepare_dataset
from src.models.layout_detector import LayoutDetector

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

def train_stage1_model(config: Dict, dataset_yaml: str, output_dir: str) -> bool:
    """Train the Stage 1 layout detection model.
    
    Args:
        config: Training configuration
        dataset_yaml: Path to dataset YAML file
        output_dir: Output directory for model
        
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
        
        # Initialize layout detector
        detector = LayoutDetector()
        
        # Get training parameters
        epochs = config.get('training', {}).get('epochs', 100)
        batch_size = config.get('training', {}).get('batch_size', 8)
        learning_rate = config.get('training', {}).get('learning_rate', 0.001)
        
        # Adjust batch size based on GPU memory
        if torch.cuda.is_available():
            if gpu_memory < 6:  # RTX 2070 has 8GB, but we'll be conservative
                batch_size = min(batch_size, 4)
                logger.info(f"📊 Adjusted batch size to {batch_size} for GPU memory constraints")
        
        logger.info(f"🎯 Starting Stage 1 training for {epochs} epochs")
        logger.info(f"📁 Dataset: {dataset_yaml}")
        logger.info(f"⚙️  Batch size: {batch_size}, Learning rate: {learning_rate}")
        logger.info(f"💻 Output directory: {output_dir}")
        logger.info(f"🖥️  Device: {device}")
        
        # Train the model
        results = detector.train(
            data_yaml=dataset_yaml,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=output_dir
        )
        
        if results is not None:
            logger.info("✅ Stage 1 training completed successfully!")
            return True
        else:
            logger.error("❌ Stage 1 training failed!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Stage 1 training failed: {e}")
        return False

def validate_stage1_model(config: Dict, dataset_yaml: str, model_path: str) -> Dict:
    """Validate the trained Stage 1 model.
    
    Args:
        config: Configuration
        dataset_yaml: Path to dataset YAML file
        model_path: Path to trained model
        
    Returns:
        Validation metrics dictionary
    """
    try:
        # Load the trained model
        detector = LayoutDetector()
        detector.load_model(model_path)
        
        logger.info("Starting Stage 1 model validation...")
        
        # Run validation
        metrics = detector.validate(dataset_yaml)
        
        if metrics:
            logger.info("Validation completed successfully!")
            logger.info(f"mAP50: {metrics.get('mAP50', 0):.4f}")
            logger.info(f"mAP50-95: {metrics.get('mAP50-95', 0):.4f}")
        else:
            logger.warning("Validation completed but no metrics returned")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {}

def create_stage1_submission(model_path: str, output_dir: str) -> str:
    """Create Stage 1 submission package.
    
    Args:
        model_path: Path to trained model
        output_dir: Output directory
        
    Returns:
        Path to submission package
    """
    try:
        submission_dir = Path(output_dir) / "stage1_submission"
        submission_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy trained model
        model_name = Path(model_path).name
        submission_model_path = submission_dir / model_name
        shutil.copy2(model_path, submission_model_path)
        
        # Create submission info
        submission_info = {
            "stage": 1,
            "description": "PS-05 Stage 1 Layout Detection Model",
            "classes": ["Background", "Text", "Title", "List", "Table", "Figure"],
            "model_file": model_name,
            "evaluation_metric": "mAP at IoU threshold >= 0.5",
            "output_format": "JSON with bbox [x,y,w,h] and class classification"
        }
        
        info_path = submission_dir / "submission_info.json"
        with open(info_path, 'w') as f:
            json.dump(submission_info, f, indent=2)
        
        # Create README
        readme_content = """# PS-05 Stage 1 Submission

## Model Information
- **Stage**: 1 (Layout Detection Only)
- **Classes**: Background, Text, Title, List, Table, Figure
- **Output Format**: JSON with bounding box coordinates [x,y,w,h] and class labels
- **Evaluation Metric**: mAP at IoU threshold >= 0.5

## Usage
```python
from src.models.layout_detector import LayoutDetector

# Load model
detector = LayoutDetector()
detector.load_model("layout_detector.pt")

# Predict layout
image = cv2.imread("document.png")
results = detector.predict(image)

# Results format
# [
#   {
#     "bbox": [x, y, w, h],
#     "cls": "Text",
#     "score": 0.95,
#     "class_id": 1
#   },
#   ...
# ]
```

## Model Performance
- Training completed on: {training_date}
- Dataset: Custom document layout dataset
- Framework: YOLOv8
- Input size: 640x640
"""
        
        readme_path = submission_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content.format(training_date=Path(model_path).stat().st_mtime))
        
        logger.info(f"Stage 1 submission package created at {submission_dir}")
        return str(submission_dir)
        
    except Exception as e:
        logger.error(f"Failed to create submission package: {e}")
        return ""

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Stage 1 layout detection model")
    parser.add_argument('--data', required=True, help='Input data directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--config', default='configs/ps05_config.yaml', help='Configuration file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--skip-dataset-prep', action='store_true', help='Skip dataset preparation')
    parser.add_argument('--skip-validation', action='store_true', help='Skip model validation')
    parser.add_argument('--create-submission', action='store_true', help='Create submission package')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        if not config:
            logger.error("Failed to load configuration")
            return
        
        # Override config with command line arguments
        if 'training' not in config:
            config['training'] = {}
        config['training']['epochs'] = args.epochs
        config['training']['batch_size'] = args.batch_size
        config['training']['learning_rate'] = args.learning_rate
        
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Prepare dataset
        dataset_yaml = None
        if not args.skip_dataset_prep:
            logger.info("Step 1: Preparing dataset...")
            dataset_yaml = prepare_dataset(args.data, str(output_path / "dataset"))
            
            if not dataset_yaml:
                logger.error("Dataset preparation failed!")
                return
        else:
            # Look for existing dataset.yaml
            dataset_yaml = str(output_path / "dataset" / "dataset.yaml")
            if not Path(dataset_yaml).exists():
                logger.error(f"Dataset YAML not found at {dataset_yaml}")
                return
        
        # Step 2: Train model
        logger.info("Step 2: Training Stage 1 model...")
        training_success = train_stage1_model(
            config, dataset_yaml, str(output_path / "training")
        )
        
        if not training_success:
            logger.error("Training failed!")
            return
        
        # Step 3: Validate model
        model_path = "models/layout_detector.pt"
        if not args.skip_validation and Path(model_path).exists():
            logger.info("Step 3: Validating model...")
            metrics = validate_stage1_model(config, dataset_yaml, model_path)
            
            if metrics:
                # Save validation results
                metrics_path = output_path / "validation_metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                logger.info(f"Validation metrics saved to {metrics_path}")
        
        # Step 4: Create submission package
        if args.create_submission and Path(model_path).exists():
            logger.info("Step 4: Creating submission package...")
            submission_path = create_stage1_submission(model_path, str(output_path))
            
            if submission_path:
                logger.info(f"Submission package created at {submission_path}")
        
        logger.info("Stage 1 training pipeline completed successfully!")
        logger.info(f"Trained model: {model_path}")
        logger.info(f"Output directory: {output_path}")
        
        # Print next steps
        print("\n" + "="*60)
        print("STAGE 1 TRAINING COMPLETED!")
        print("="*60)
        print(f"Model saved to: {model_path}")
        print(f"Dataset prepared at: {dataset_yaml}")
        print(f"Output directory: {output_path}")
        print("\nNext steps:")
        print("1. Test the model on sample images:")
        print(f"   python ps05.py infer --input test_image.png --output results/ --stage 1")
        print("2. Evaluate on validation data:")
        print(f"   python src/evaluation/stage1_evaluator.py --predictions preds.json --ground-truth gt.json")
        print("3. Prepare for submission (due by 5 Nov 2025)")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
