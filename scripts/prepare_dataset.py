#!/usr/bin/env python3
"""
Dataset Preparation Script for PS-05 Layout Detection

Converts existing JSON annotations to YOLO format for training.
Creates train/val/test splits and generates dataset.yaml file.
"""

import argparse
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_annotations(json_path: str) -> Dict:
    """Load annotations from JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {json_path}: {e}")
        return {}

def convert_bbox_to_yolo(bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """Convert [x, y, w, h] bbox to YOLO format [x_center, y_center, width, height] (normalized)."""
    x, y, w, h = bbox
    
    # Calculate center coordinates
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    
    # Normalize width and height
    width = w / img_width
    height = h / img_height
    
    return [x_center, y_center, width, height]

def process_image_and_annotations(image_path: str, annotations: Dict, output_dir: Path, 
                                split: str, class_mapping: Dict[str, int]) -> bool:
    """Process a single image and its annotations.
    
    Args:
        image_path: Path to the image file
        annotations: Annotations dictionary
        output_dir: Output directory
        split: Data split (train/val/test)
        class_mapping: Mapping from class names to IDs
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not load image: {image_path}")
            return False
        
        img_height, img_width = image.shape[:2]
        
        # Create output directories
        images_dir = output_dir / split / "images"
        labels_dir = output_dir / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy image
        image_name = Path(image_path).name
        output_image_path = images_dir / image_name
        shutil.copy2(image_path, output_image_path)
        
        # Create label file
        label_name = Path(image_path).stem + ".txt"
        label_path = labels_dir / label_name
        
        # Process annotations
        with open(label_path, 'w') as f:
            for ann in annotations.get('annotations', []):
                category_id = ann.get('category_id', 1)  # Default to Text if not specified
                
                # Map category_id to our 6 classes (0-5)
                if category_id == 1:  # Text
                    class_id = 1
                elif category_id == 2:  # Title
                    class_id = 2
                else:  # Default to Text for now
                    class_id = 1
                
                # Convert bbox to YOLO format
                bbox = ann.get('bbox', [0, 0, 1, 1])
                yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)
                
                # Write label line: class_id x_center y_center width height
                line = f"{class_id} {' '.join([f'{coord:.6f}' for coord in yolo_bbox])}\n"
                f.write(line)
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return False

def create_dataset_yaml(output_dir: Path, class_names: List[str]) -> str:
    """Create dataset.yaml file for YOLO training.
    
    Args:
        output_dir: Output directory
        class_names: List of class names
        
    Returns:
        Path to the created dataset.yaml file
    """
    yaml_path = output_dir / "dataset.yaml"
    
    dataset_config = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    try:
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"Dataset YAML created at {yaml_path}")
        return str(yaml_path)
        
    except ImportError:
        # Fallback to JSON if yaml not available
        json_path = output_dir / "dataset.json"
        with open(json_path, 'w') as f:
            json.dump(dataset_config, f, indent=2)
        
        logger.info(f"Dataset JSON created at {json_path}")
        return str(json_path)

def prepare_dataset(data_dir: str, output_dir: str, train_ratio: float = 0.7, 
                   val_ratio: float = 0.2, test_ratio: float = 0.1):
    """Prepare dataset for YOLO training.
    
    Args:
        data_dir: Input data directory containing images and JSON files
        output_dir: Output directory for processed dataset
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
    """
    try:
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        # Check if we have enhanced processed structure (images/ subdirectory)
        images_subdir = data_path / "images"
        if images_subdir.exists():
            # Enhanced processed structure
            for ext in image_extensions:
                image_files.extend(images_subdir.glob(f"*{ext}"))
                image_files.extend(images_subdir.glob(f"*{ext.upper()}"))
        else:
            # Direct structure
            for ext in image_extensions:
                image_files.extend(data_path.glob(f"*{ext}"))
                image_files.extend(data_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.error(f"No image files found in {data_dir}")
            return None
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Class mapping for PS-05 Stage 1
        class_names = ['Background', 'Text', 'Title', 'List', 'Table', 'Figure']
        class_mapping = {name: i for i, name in enumerate(class_names)}
        
        # Shuffle and split data
        random.shuffle(image_files)
        total_files = len(image_files)
        
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)
        
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        logger.info(f"Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        # Process each split
        splits = [
            ('train', train_files),
            ('val', val_files),
            ('test', test_files)
        ]
        
        processed_count = 0
        for split_name, files in splits:
            logger.info(f"Processing {split_name} split...")
            
            for image_file in files:
                # Find corresponding JSON file
                # Check if we have enhanced processed structure (annotations/ subdirectory)
                annotations_subdir = data_path / "annotations"
                if annotations_subdir.exists():
                    # Enhanced processed structure
                    json_file = annotations_subdir / f"{image_file.stem}.json"
                else:
                    # Direct structure
                    json_file = image_file.with_suffix('.json')
                
                if json_file.exists():
                    annotations = load_annotations(str(json_file))
                    if process_image_and_annotations(str(image_file), annotations, output_path, 
                                                   split_name, class_mapping):
                        processed_count += 1
                else:
                    logger.warning(f"No annotations found for {image_file}")
        
        logger.info(f"Successfully processed {processed_count} images")
        
        # Create dataset configuration file
        yaml_path = create_dataset_yaml(output_path, class_names)
        
        return yaml_path
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        return None

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Prepare dataset for YOLO training")
    parser.add_argument('--data', required=True, help='Input data directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training data ratio')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation data ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test data ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        logger.error("Ratios must sum to 1.0")
        return
    
    # Prepare dataset
    logger.info("Starting dataset preparation...")
    yaml_path = prepare_dataset(
        args.data, 
        args.output, 
        args.train_ratio, 
        args.val_ratio, 
        args.test_ratio
    )
    
    if yaml_path:
        logger.info(f"Dataset preparation completed successfully!")
        logger.info(f"Dataset configuration: {yaml_path}")
        logger.info(f"Output directory: {args.output}")
    else:
        logger.error("Dataset preparation failed!")

if __name__ == "__main__":
    main()
