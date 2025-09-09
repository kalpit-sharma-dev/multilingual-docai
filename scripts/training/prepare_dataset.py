#!/usr/bin/env python3
"""
Dataset Preparation Script for PS-05 Layout Detection

Converts existing JSON annotations to YOLO format for training.
Creates train/val/test splits and generates dataset.yaml file.
"""

import argparse
import json
import logging
import yaml
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any
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

def iter_annotation_entries(annotations: Dict) -> Iterable[Tuple[str, List[float]]]:
    """Yield (class_name, bbox_xywh) from various supported JSON formats.
    Supports keys: 'annotations' (list of {category_name/category_id, bbox}),
    'elements' or 'layout_elements' (list of {type, bbox}).
    """
    # Format A: { annotations: [ {category_name|category_id, bbox:[x,y,w,h]}, ... ] }
    if isinstance(annotations.get('annotations'), list):
        for ann in annotations['annotations']:
            cname = ann.get('category_name')
            cid = ann.get('category_id')
            bbox = ann.get('bbox')
            if bbox and (cname is not None or cid is not None):
                yield (cname if cname is not None else cid, bbox)
        return
    # Format B: { elements: [ {type, bbox:[x,y,w,h]}, ... ] }
    if isinstance(annotations.get('elements'), list):
        for el in annotations['elements']:
            cname = el.get('type')
            bbox = el.get('bbox')
            if cname and bbox:
                yield (cname, bbox)
        return
    # Format C: { layout_elements: [ {type, bbox:[x,y,w,h]}, ... ] }
    if isinstance(annotations.get('layout_elements'), list):
        for el in annotations['layout_elements']:
            cname = el.get('type')
            bbox = el.get('bbox')
            if cname and bbox:
                yield (cname, bbox)
        return

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
            wrote_any = False
            for cname_or_id, bbox in iter_annotation_entries(annotations):
                # Map class
                if isinstance(cname_or_id, str):
                    mapped = class_mapping.get(cname_or_id, None)
                else:
                    id_to_name = {
                        0: 'Background', 1: 'Text', 2: 'Title', 3: 'List', 4: 'Table', 5: 'Figure'
                    }
                    mapped = class_mapping.get(id_to_name.get(int(cname_or_id), 'Text'), None)
                if mapped is None:
                    continue
                if mapped == 0:  # Skip Background for labels
                    continue
                # Convert bbox to YOLO format
                try:
                    yolo_bbox = convert_bbox_to_yolo([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])], img_width, img_height)
                except Exception:
                    continue
                # Write label line: class_id x_center y_center width height
                line = f"{mapped} {' '.join([f'{coord:.6f}' for coord in yolo_bbox])}\n"
                f.write(line)
                wrote_any = True
            if not wrote_any:
                # Ensure empty file exists
                f.write("")

        # Also emit Stage-1 JSON output alongside YOLO labels
        try:
            annotations_dir = output_dir / split / "annotations"
            annotations_dir.mkdir(parents=True, exist_ok=True)

            # Build JSON elements from original absolute bboxes
            id_to_name = {0: 'Background', 1: 'Text', 2: 'Title', 3: 'List', 4: 'Table', 5: 'Figure'}
            elements = []
            for cname_or_id, bbox in iter_annotation_entries(annotations):
                if isinstance(cname_or_id, str):
                    class_name = cname_or_id
                    class_id = class_mapping.get(class_name, 0)
                else:
                    class_id = int(cname_or_id)
                    class_name = id_to_name.get(class_id, 'Background')
                # Expect bbox as [x, y, w, h]; convert to [x, y, h, w]
                try:
                    x, y, w, h = [int(float(bbox[0])), int(float(bbox[1])), int(float(bbox[2])), int(float(bbox[3]))]
                except Exception:
                    continue
                elements.append({
                    "class": class_name,
                    "class_id": class_id,
                    "bbox": [x, y, h, w],
                    "score": 1.0
                })

            json_payload = {
                "metadata": {
                    "source": image_name,
                    "language": "en"
                },
                "elements": elements
            }

            json_out = annotations_dir / f"{Path(image_path).stem}.json"
            with open(json_out, 'w', encoding='utf-8') as jf:
                json.dump(json_payload, jf, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to write Stage-1 JSON for {image_path}: {e}")

        return True
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return False

def copy_image_with_empty_label(image_path: str, output_dir: Path, split: str) -> bool:
    """Copy image and create an empty YOLO label file when no annotations are present.
    This ensures Ultralytics can load datasets where some images have no objects.
    """
    try:
        # Create output directories
        images_dir = output_dir / split / "images"
        labels_dir = output_dir / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Copy image
        image_name = Path(image_path).name
        output_image_path = images_dir / image_name
        shutil.copy2(image_path, output_image_path)

        # Create empty label
        label_name = Path(image_path).stem + ".txt"
        label_path = labels_dir / label_name
        with open(label_path, 'w') as f:
            f.write("")
        return True
    except Exception as e:
        logger.error(f"Error copying image or creating empty label for {image_path}: {e}")
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
        'nc': len(class_names),
        'names': class_names
    }

    # Only include splits that actually exist
    if (output_dir / 'train' / 'images').exists():
        dataset_config['train'] = 'train/images'
    if (output_dir / 'val' / 'images').exists():
        dataset_config['val'] = 'val/images'
    if (output_dir / 'test' / 'images').exists():
        dataset_config['test'] = 'test/images'
    
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
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        
        # Check if we have enhanced processed structure (images/ subdirectory)
        images_subdir = data_path / "images"
        if images_subdir.exists():
            # Enhanced processed structure
            for ext in image_extensions:
                image_files.extend(images_subdir.rglob(f"*{ext}"))
                image_files.extend(images_subdir.rglob(f"*{ext.upper()}"))
        else:
            # Direct structure
            for ext in image_extensions:
                image_files.extend(data_path.rglob(f"*{ext}"))
                image_files.extend(data_path.rglob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.error(f"No image files found in {data_dir}")
            return None
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Class mapping for PS-05 Stage 1 loaded from config (fallback to default)
        class_mapping = {}
        class_names = []
        try:
            cfg_path = Path("configs/ps05_config.yaml")
            if cfg_path.exists():
                with open(cfg_path, 'r', encoding='utf-8') as cf:
                    cfg = yaml.safe_load(cf) or {}
                classes = (
                    cfg.get('models', {})
                       .get('layout', {})
                       .get('classes', ['Background', 'Text', 'Title', 'List', 'Table', 'Figure'])
                )
                class_names = list(classes)
                class_mapping = {name: idx for idx, name in enumerate(class_names)}
            else:
                raise FileNotFoundError("ps05_config.yaml not found")
        except Exception:
            class_mapping = {
                'Background': 0,
                'Text': 1,
                'Title': 2,
                'List': 3,
                'Table': 4,
                'Figure': 5
            }
            class_names = ['Background', 'Text', 'Title', 'List', 'Table', 'Figure']
        
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
                    # Still include the image with an empty label so the loader doesn't fail
                    logger.warning(f"No annotations found for {image_file}; creating empty label.")
                    if copy_image_with_empty_label(str(image_file), output_path, split_name):
                        processed_count += 1
        
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
