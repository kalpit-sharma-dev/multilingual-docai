#!/usr/bin/env python3
"""
Training Data Preparation Script for PS-05

This script prepares your small English training dataset with:
1. Image cleaning and augmentation
2. YOLO format conversion
3. Train/validation split
4. Dataset configuration generation
5. Quality checks and validation
"""

import logging
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import yaml
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingDataPreparator:
    """Comprehensive training data preparation for PS-05."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.output_dir = None
        self.dataset_info = {}
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for training data preparation."""
        return {
            "input_dir": "data/train",  # Your existing training data
            "output_dir": "data/training_prepared",
            "train_split": 0.8,  # 80% training, 20% validation
            "target_size": (800, 1000),  # Standard document size
            "augmentation": {
                "enabled": True,
                "rotation_angles": [-15, -10, -5, 5, 10, 15],  # Skew angles
                "brightness_factors": [0.7, 0.85, 1.15, 1.3],  # Brightness variations
                "contrast_factors": [0.8, 0.9, 1.1, 1.2],      # Contrast variations
                "noise_levels": [5, 10, 15],                    # Noise addition
                "blur_levels": [1, 2],                          # Blur variations
                "augmentations_per_image": 3                     # Number of augmented versions per image
            },
            "yolo_classes": [
                "Background", "Text", "Title", "List", "Table", "Figure"
            ],
            "quality_checks": {
                "min_image_size": (100, 100),
                "min_annotation_count": 1,
                "max_aspect_ratio": 10.0,
                "validate_bbox_coordinates": True
            }
        }
    
    def prepare_training_dataset(self) -> Dict:
        """
        Main method to prepare the complete training dataset.
        
        Returns:
            Dictionary with preparation results and statistics
        """
        try:
            logger.info("Starting comprehensive training data preparation...")
            
            # Setup directories
            self._setup_directories()
            
            # Load and validate input data
            input_data = self._load_input_data()
            if not input_data:
                raise ValueError("No valid input data found")
            
            # Clean and preprocess images
            cleaned_data = self._clean_and_preprocess_images(input_data)
            
            # Apply augmentation
            if self.config["augmentation"]["enabled"]:
                augmented_data = self._apply_augmentation(cleaned_data)
                cleaned_data.extend(augmented_data)
            
            # Convert to YOLO format
            yolo_data = self._convert_to_yolo_format(cleaned_data)
            
            # Split into train/validation
            train_data, val_data = self._split_train_validation(yolo_data)
            
            # Save datasets
            self._save_datasets(train_data, val_data)
            
            # Generate dataset configuration
            dataset_config = self._generate_dataset_config()
            
            # Perform quality checks
            quality_report = self._perform_quality_checks(train_data, val_data)
            
            # Generate final report
            final_report = self._generate_final_report(
                input_data, cleaned_data, train_data, val_data, quality_report
            )
            
            logger.info("Training data preparation completed successfully!")
            return final_report
            
        except Exception as e:
            logger.error(f"Training data preparation failed: {e}")
            raise
    
    def _setup_directories(self):
        """Setup output directory structure."""
        self.output_dir = Path(self.config["output_dir"])
        
        # Create main directories
        (self.output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directories created: {self.output_dir}")
    
    def _load_input_data(self) -> List[Dict]:
        """Load and validate input training data."""
        input_dir = Path(self.config["input_dir"])
        if not input_dir.exists():
            raise ValueError(f"Input directory not found: {input_dir}")
        
        input_data = []
        
        # Look for image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Load annotations if available
        annotations_file = input_dir / "annotations.json"
        annotations = {}
        if annotations_file.exists():
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
            logger.info(f"Loaded annotations for {len(annotations)} images")
        
        # Process each image
        for img_path in image_files:
            try:
                # Load image
                with Image.open(img_path) as img:
                    img_array = np.array(img)
                
                # Get annotations for this image
                img_annotations = annotations.get(img_path.name, [])
                
                # Validate image
                if self._validate_image(img_array, img_annotations):
                    input_data.append({
                        "image_path": img_path,
                        "image_array": img_array,
                        "annotations": img_annotations,
                        "filename": img_path.name
                    })
                
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
        
        logger.info(f"Successfully loaded {len(input_data)} valid images")
        return input_data
    
    def _validate_image(self, img_array: np.ndarray, annotations: List) -> bool:
        """Validate image and annotations."""
        try:
            # Check image size
            height, width = img_array.shape[:2]
            min_height, min_width = self.config["quality_checks"]["min_image_size"]
            
            if height < min_height or width < min_width:
                logger.warning(f"Image too small: {width}x{height}")
                return False
            
            # Check aspect ratio
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > self.config["quality_checks"]["max_aspect_ratio"]:
                logger.warning(f"Image aspect ratio too extreme: {aspect_ratio:.2f}")
                return False
            
            # Check annotations
            if len(annotations) < self.config["quality_checks"]["min_annotation_count"]:
                logger.warning(f"Too few annotations: {len(annotations)}")
                return False
            
            # Validate bounding boxes if enabled
            if self.config["quality_checks"]["validate_bbox_coordinates"]:
                for ann in annotations:
                    if "bbox" in ann:
                        bbox = ann["bbox"]
                        if len(bbox) != 4:
                            logger.warning(f"Invalid bbox format: {bbox}")
                            return False
                        
                        x, y, w, h = bbox
                        if x < 0 or y < 0 or w <= 0 or h <= 0:
                            logger.warning(f"Invalid bbox coordinates: {bbox}")
                            return False
                        
                        if x + w > width or y + h > height:
                            logger.warning(f"Bbox outside image bounds: {bbox}")
                            return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Image validation failed: {e}")
            return False
    
    def _clean_and_preprocess_images(self, input_data: List[Dict]) -> List[Dict]:
        """Clean and preprocess images for training."""
        logger.info("Cleaning and preprocessing images...")
        
        cleaned_data = []
        
        for data in input_data:
            try:
                # Clean image
                cleaned_img = self._clean_image(data["image_array"])
                
                # Resize to target size
                resized_img = self._resize_image(cleaned_img)
                
                # Update data
                cleaned_data.append({
                    **data,
                    "image_array": resized_img,
                    "original_size": data["image_array"].shape[:2],
                    "cleaned_size": resized_img.shape[:2]
                })
                
            except Exception as e:
                logger.warning(f"Failed to clean {data['filename']}: {e}")
        
        logger.info(f"Cleaned {len(cleaned_data)} images")
        return cleaned_data
    
    def _clean_image(self, img_array: np.ndarray) -> np.ndarray:
        """Clean image using basic preprocessing."""
        try:
            # Convert to RGB if needed
            if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            elif len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
            
            # Enhance contrast
            lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Image cleaning failed: {e}")
            return img_array
    
    def _resize_image(self, img_array: np.ndarray) -> np.ndarray:
        """Resize image to target size while maintaining aspect ratio."""
        target_width, target_height = self.config["target_size"]
        
        # Calculate aspect ratio
        height, width = img_array.shape[:2]
        aspect_ratio = width / height
        target_aspect = target_width / target_height
        
        if aspect_ratio > target_aspect:
            # Image is wider than target
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            # Image is taller than target
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        # Resize image
        resized = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Create new image with target size and paste resized image
        new_img = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        new_img[paste_y:paste_y + new_height, paste_x:paste_x + new_width] = resized
        
        return new_img
    
    def _apply_augmentation(self, cleaned_data: List[Dict]) -> List[Dict]:
        """Apply data augmentation to create additional training samples."""
        logger.info("Applying data augmentation...")
        
        augmented_data = []
        aug_config = self.config["augmentation"]
        
        for data in cleaned_data:
            # Create multiple augmented versions
            for i in range(aug_config["augmentations_per_image"]):
                try:
                    # Apply random augmentation
                    augmented_img = self._apply_random_augmentation(data["image_array"])
                    
                    # Create augmented data entry
                    aug_data = {
                        **data,
                        "image_array": augmented_img,
                        "filename": f"aug_{i+1}_{data['filename']}",
                        "is_augmented": True,
                        "augmentation_type": f"augmentation_{i+1}"
                    }
                    
                    augmented_data.append(aug_data)
                    
                except Exception as e:
                    logger.warning(f"Failed to augment {data['filename']}: {e}")
        
        logger.info(f"Created {len(augmented_data)} augmented images")
        return augmented_data
    
    def _apply_random_augmentation(self, img_array: np.ndarray) -> np.ndarray:
        """Apply random augmentation to an image."""
        aug_config = self.config["augmentation"]
        
        # Convert to PIL for easier augmentation
        pil_img = Image.fromarray(img_array)
        
        # Random rotation (skew)
        if aug_config["rotation_angles"]:
            angle = random.choice(aug_config["rotation_angles"])
            pil_img = pil_img.rotate(angle, fillcolor=(255, 255, 255))
        
        # Random brightness adjustment
        if aug_config["brightness_factors"]:
            factor = random.choice(aug_config["brightness_factors"])
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(factor)
        
        # Random contrast adjustment
        if aug_config["contrast_factors"]:
            factor = random.choice(aug_config["contrast_factors"])
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(factor)
        
        # Random noise addition
        if aug_config["noise_levels"]:
            noise_level = random.choice(aug_config["noise_levels"])
            img_array = np.array(pil_img)
            noise = np.random.normal(0, noise_level, img_array.shape).astype(np.uint8)
            img_array = np.clip(img_array + noise, 0, 255)
            pil_img = Image.fromarray(img_array)
        
        # Random blur
        if aug_config["blur_levels"]:
            blur_level = random.choice(aug_config["blur_levels"])
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_level))
        
        return np.array(pil_img)
    
    def _convert_to_yolo_format(self, cleaned_data: List[Dict]) -> List[Dict]:
        """Convert annotations to YOLO format."""
        logger.info("Converting annotations to YOLO format...")
        
        yolo_data = []
        
        for data in cleaned_data:
            try:
                # Convert annotations to YOLO format
                yolo_annotations = self._convert_annotations_to_yolo(
                    data["annotations"], 
                    data["original_size"], 
                    data["cleaned_size"]
                )
                
                yolo_data.append({
                    **data,
                    "yolo_annotations": yolo_annotations
                })
                
            except Exception as e:
                logger.warning(f"Failed to convert {data['filename']}: {e}")
        
        logger.info(f"Converted {len(yolo_data)} images to YOLO format")
        return yolo_data
    
    def _convert_annotations_to_yolo(
        self, 
        annotations: List[Dict], 
        original_size: Tuple[int, int], 
        cleaned_size: Tuple[int, int]
    ) -> List[str]:
        """Convert COCO-style annotations to YOLO format."""
        yolo_lines = []
        
        orig_height, orig_width = original_size
        clean_height, clean_width = cleaned_size
        
        for ann in annotations:
            if "bbox" in ann and "category_id" in ann:
                # Get bbox coordinates
                x, y, w, h = ann["bbox"]
                
                # Convert to YOLO format (normalized coordinates)
                x_center = (x + w/2) / orig_width
                y_center = (y + h/2) / orig_height
                width_norm = w / orig_width
                height_norm = h / orig_height
                
                # Get class ID
                class_id = ann["category_id"]
                
                # Create YOLO line
                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
                yolo_lines.append(yolo_line)
        
        return yolo_lines
    
    def _split_train_validation(self, yolo_data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Split data into training and validation sets."""
        logger.info("Splitting data into train/validation sets...")
        
        # Shuffle data
        random.shuffle(yolo_data)
        
        # Calculate split
        total_samples = len(yolo_data)
        train_size = int(total_samples * self.config["train_split"])
        
        train_data = yolo_data[:train_size]
        val_data = yolo_data[train_size:]
        
        logger.info(f"Train: {len(train_data)}, Validation: {len(val_data)}")
        return train_data, val_data
    
    def _save_datasets(self, train_data: List[Dict], val_data: List[Dict]):
        """Save train and validation datasets."""
        logger.info("Saving datasets...")
        
        # Save training data
        self._save_dataset_split(train_data, "train")
        
        # Save validation data
        self._save_dataset_split(val_data, "val")
        
        logger.info("Datasets saved successfully")
    
    def _save_dataset_split(self, data: List[Dict], split_name: str):
        """Save a dataset split (train or validation)."""
        images_dir = self.output_dir / "images" / split_name
        labels_dir = self.output_dir / "labels" / split_name
        
        for item in data:
            try:
                # Save image
                img_path = images_dir / item["filename"]
                cv2.imwrite(str(img_path), cv2.cvtColor(item["image_array"], cv2.COLOR_RGB2BGR))
                
                # Save YOLO labels
                label_path = labels_dir / f"{Path(item['filename']).stem}.txt"
                with open(label_path, 'w') as f:
                    for line in item["yolo_annotations"]:
                        f.write(line + '\n')
                
            except Exception as e:
                logger.warning(f"Failed to save {item['filename']}: {e}")
    
    def _generate_dataset_config(self) -> Dict:
        """Generate YOLO dataset configuration file."""
        config = {
            "path": str(self.output_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(self.config["yolo_classes"]),
            "names": self.config["yolo_classes"]
        }
        
        # Save config file
        config_path = self.output_dir / "dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Dataset configuration saved: {config_path}")
        return config
    
    def _perform_quality_checks(self, train_data: List[Dict], val_data: List[Dict]) -> Dict:
        """Perform quality checks on the prepared dataset."""
        logger.info("Performing quality checks...")
        
        quality_report = {
            "total_images": len(train_data) + len(val_data),
            "train_images": len(train_data),
            "val_images": len(val_data),
            "total_annotations": 0,
            "class_distribution": {},
            "bbox_validation": {
                "valid_bboxes": 0,
                "invalid_bboxes": 0,
                "issues": []
            }
        }
        
        # Check all data
        all_data = train_data + val_data
        
        for item in all_data:
            # Count annotations
            annotation_count = len(item["yolo_annotations"])
            quality_report["total_annotations"] += annotation_count
            
            # Validate bounding boxes
            for line in item["yolo_annotations"]:
                try:
                    parts = line.split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Validate coordinates
                        if (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                            0 < width <= 1 and 0 < height <= 1):
                            quality_report["bbox_validation"]["valid_bboxes"] += 1
                            
                            # Update class distribution
                            class_name = self.config["yolo_classes"][class_id]
                            quality_report["class_distribution"][class_name] = \
                                quality_report["class_distribution"].get(class_name, 0) + 1
                        else:
                            quality_report["bbox_validation"]["invalid_bboxes"] += 1
                            quality_report["bbox_validation"]["issues"].append(
                                f"Invalid bbox in {item['filename']}: {line}"
                            )
                    else:
                        quality_report["bbox_validation"]["invalid_bboxes"] += 1
                        quality_report["bbox_validation"]["issues"].append(
                            f"Malformed line in {item['filename']}: {line}"
                        )
                        
                except Exception as e:
                    quality_report["bbox_validation"]["invalid_bboxes"] += 1
                    quality_report["bbox_validation"]["issues"].append(
                        f"Error parsing {item['filename']}: {e}"
                    )
        
        # Save quality report
        quality_path = self.output_dir / "quality_report.json"
        with open(quality_path, 'w') as f:
            json.dump(quality_report, f, indent=2)
        
        logger.info("Quality checks completed")
        return quality_report
    
    def _generate_final_report(
        self, 
        input_data: List[Dict], 
        cleaned_data: List[Dict], 
        train_data: List[Dict], 
        val_data: List[Dict], 
        quality_report: Dict
    ) -> Dict:
        """Generate final preparation report."""
        final_report = {
            "preparation_summary": {
                "input_images": len(input_data),
                "cleaned_images": len(cleaned_data),
                "train_images": len(train_data),
                "val_images": len(val_data),
                "augmentation_enabled": self.config["augmentation"]["enabled"],
                "augmentations_per_image": self.config["augmentation"]["augmentations_per_image"]
            },
            "quality_report": quality_report,
            "output_directory": str(self.output_dir),
            "dataset_config_file": "dataset.yaml",
            "quality_report_file": "quality_report.json",
            "preparation_timestamp": str(datetime.now()),
            "configuration_used": self.config
        }
        
        # Save final report
        report_path = self.output_dir / "preparation_report.json"
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        logger.info(f"Final report saved: {report_path}")
        return final_report

def main():
    """Main function to run training data preparation."""
    try:
        # Initialize preparator
        preparator = TrainingDataPreparator()
        
        # Run preparation
        report = preparator.prepare_training_dataset()
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING DATA PREPARATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Input images: {report['preparation_summary']['input_images']}")
        print(f"Cleaned images: {report['preparation_summary']['cleaned_images']}")
        print(f"Training images: {report['preparation_summary']['train_images']}")
        print(f"Validation images: {report['preparation_summary']['val_images']}")
        print(f"Total annotations: {report['quality_report']['total_annotations']}")
        print(f"Output directory: {report['output_directory']}")
        print(f"Dataset config: {report['dataset_config_file']}")
        print("="*60)
        
        print("\nNext steps:")
        print("1. Review quality_report.json for any issues")
        print("2. Use dataset.yaml for YOLO training")
        print("3. Start training with: python scripts/train_stage1.py")
        
    except Exception as e:
        logger.error(f"Training data preparation failed: {e}")
        raise

if __name__ == "__main__":
    main()
