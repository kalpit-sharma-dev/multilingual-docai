#!/usr/bin/env python3
"""
Prepare Current Dataset for Training

Converts existing dataset format to YOLO format for training.
"""

import json
import os
from pathlib import Path
import shutil
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPreparer:
    """Prepares current dataset for training."""
    
    def __init__(self, input_dir: str = "data/train", output_dir: str = "data/yolo_dataset"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.classes = ["Background", "Text", "Title", "List", "Table", "Figure"]
        
    def prepare_dataset(self) -> bool:
        """Prepare the dataset in YOLO format."""
        try:
            logger.info("🚀 Preparing dataset for training...")
            
            # Create output directories
            self._create_directories()
            
            # Convert annotations
            self._convert_annotations()
            
            # Create dataset.yaml
            self._create_dataset_yaml()
            
            logger.info("✅ Dataset preparation completed!")
            return True
            
        except Exception as e:
            logger.error(f"Dataset preparation failed: {e}")
            return False
    
    def _create_directories(self):
        """Create YOLO dataset directory structure."""
        dirs = [
            self.output_dir / "images" / "train",
            self.output_dir / "images" / "val",
            self.output_dir / "labels" / "train",
            self.output_dir / "labels" / "val"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created: {dir_path}")
    
    def _convert_annotations(self):
        """Convert annotations to YOLO format."""
        # Get all JSON files
        json_files = list(self.input_dir.glob("*.json"))
        
        # Split into train/val (80/20)
        train_count = int(len(json_files) * 0.8)
        train_files = json_files[:train_count]
        val_files = json_files[train_count:]
        
        logger.info(f"📊 Total files: {len(json_files)}")
        logger.info(f"🚂 Train files: {len(train_files)}")
        logger.info(f"✅ Val files: {len(val_files)}")
        
        # Process train files
        for json_file in train_files:
            self._process_file(json_file, "train")
        
        # Process val files
        for json_file in val_files:
            self._process_file(json_file, "val")
    
    def _process_file(self, json_file: Path, split: str):
        """Process a single JSON file."""
        try:
            # Load annotations
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Get image file
            image_file = json_file.with_suffix('.png')
            if not image_file.exists():
                logger.warning(f"Image not found: {image_file}")
                return
            
            # Copy image
            dest_image = self.output_dir / "images" / split / image_file.name
            shutil.copy2(image_file, dest_image)
            
            # Convert annotations to YOLO format
            yolo_annotations = self._convert_to_yolo(data, image_file)
            
            # Save YOLO annotations
            label_file = self.output_dir / "labels" / split / f"{json_file.stem}.txt"
            with open(label_file, 'w') as f:
                for annotation in yolo_annotations:
                    f.write(f"{annotation}\n")
                    
        except Exception as e:
            logger.error(f"Failed to process {json_file}: {e}")
    
    def _convert_to_yolo(self, data: Dict, image_file: Path) -> List[str]:
        """Convert annotations to YOLO format."""
        yolo_annotations = []
        
        # Get image dimensions (you might need to adjust this)
        # For now, using a default size
        img_width, img_height = 800, 1000  # Default size
        
        for annotation in data.get("annotations", []):
            bbox = annotation.get("bbox", [])
            category_id = annotation.get("category_id", 0)
            
            if len(bbox) == 4:
                x, y, w, h = bbox
                
                # Convert to YOLO format (center_x, center_y, width, height) - normalized
                center_x = (x + w/2) / img_width
                center_y = (y + h/2) / img_height
                norm_width = w / img_width
                norm_height = h / img_height
                
                # Ensure values are between 0 and 1
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                norm_width = max(0, min(1, norm_width))
                norm_height = max(0, min(1, norm_height))
                
                yolo_line = f"{category_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
                yolo_annotations.append(yolo_line)
        
        return yolo_annotations
    
    def _create_dataset_yaml(self):
        """Create dataset.yaml file."""
        dataset_yaml = {
            "path": str(self.output_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(self.classes),
            "names": self.classes
        }
        
        yaml_path = self.output_dir / "dataset.yaml"
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        logger.info(f"📄 Created: {yaml_path}")

def main():
    """Main function."""
    preparer = DatasetPreparer()
    success = preparer.prepare_dataset()
    
    if success:
        logger.info("🎉 Dataset is ready for training!")
        logger.info("📁 Use: data/yolo_dataset/dataset.yaml")
    else:
        logger.error("❌ Dataset preparation failed!")

if __name__ == "__main__":
    main()
