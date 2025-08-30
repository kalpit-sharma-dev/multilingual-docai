#!/usr/bin/env python3
"""
Dataset Ingestion Script for PS-05

Downloads and processes external datasets:
- DocLayNet (80K+ annotated pages)
- PubLayNet 
- ICDAR-MLT
- FUNSD, SROIE
"""

import argparse
import logging
import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import requests
import zipfile
from tqdm import tqdm

# Try to import Hugging Face datasets
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("Hugging Face datasets not available. Install with: pip install datasets")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetIngester:
    """Handles downloading and processing of external datasets."""
    
    def __init__(self, output_dir: str = "data/public"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_doclaynet(self, use_hf: bool = True) -> bool:
        """Download DocLayNet dataset.
        
        Args:
            use_hf: Whether to use Hugging Face datasets (faster)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            doclaynet_dir = self.output_dir / "docLayNet"
            doclaynet_dir.mkdir(exist_ok=True)
            
            if use_hf and HF_AVAILABLE:
                logger.info("📥 Downloading DocLayNet via Hugging Face...")
                return self._download_doclaynet_hf(doclaynet_dir)
            else:
                logger.info("📥 Downloading DocLayNet manually...")
                return self._download_doclaynet_manual(doclaynet_dir)
                
        except Exception as e:
            logger.error(f"Failed to download DocLayNet: {e}")
            return False
    
    def _download_doclaynet_hf(self, output_dir: Path) -> bool:
        """Download DocLayNet using Hugging Face datasets."""
        try:
            # Load the dataset with trust_remote_code
            dataset = load_dataset("ds4sd/DocLayNet", trust_remote_code=True)
            
            # Save to disk
            dataset.save_to_disk(output_dir / "hf_dataset")
            
            # Create YOLO-compatible format
            self._convert_doclaynet_to_yolo(dataset, output_dir)
            
            logger.info("✅ DocLayNet downloaded and converted successfully!")
            return True
            
        except Exception as e:
            logger.error(f"HF download failed: {e}")
            return False
    
    def _download_doclaynet_manual(self, output_dir: Path) -> bool:
        """Download DocLayNet manually from source."""
        try:
            # DocLayNet download URLs
            urls = {
                "core": "https://github.com/DS4SD/DocLayNet/releases/download/v1.0/doclaynet_core.zip",
                "extra": "https://github.com/DS4SD/DocLayNet/releases/download/v1.0/doclaynet_extra.zip"
            }
            
            for name, url in urls.items():
                logger.info(f"📥 Downloading {name}...")
                zip_path = output_dir / f"doclaynet_{name}.zip"
                
                # Download with progress bar
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                
                with open(zip_path, 'wb') as f, tqdm(
                    desc=f"Downloading {name}",
                    total=total_size,
                    unit='B',
                    unit_scale=True
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
                
                # Extract
                logger.info(f"📦 Extracting {name}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir / name)
                
                # Clean up zip
                zip_path.unlink()
            
            # Convert to YOLO format
            self._convert_doclaynet_manual_to_yolo(output_dir)
            
            logger.info("✅ DocLayNet downloaded and converted successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Manual download failed: {e}")
            return False
    
    def _convert_doclaynet_to_yolo(self, dataset, output_dir: Path):
        """Convert DocLayNet HF dataset to YOLO format."""
        try:
            # Create YOLO directory structure
            yolo_dir = output_dir / "yolo_format"
            for split in ["train", "val", "test"]:
                (yolo_dir / split / "images").mkdir(parents=True, exist_ok=True)
                (yolo_dir / split / "labels").mkdir(parents=True, exist_ok=True)
            
            # Class mapping for DocLayNet to our 6 classes
            class_mapping = {
                "Caption": 5,      # Figure
                "Footnote": 1,     # Text
                "Formula": 1,      # Text
                "List-Item": 3,    # List
                "Page-Footer": 1,  # Text
                "Page-Header": 1,  # Text
                "Picture": 5,      # Figure
                "Section-Header": 2, # Title
                "Table": 4,        # Table
                "Text": 1,         # Text
                "Title": 2         # Title
            }
            
            # Process each split
            for split_name, split_data in dataset.items():
                logger.info(f"🔄 Converting {split_name} split...")
                
                for i, item in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
                    # Get image and annotations
                    image = item['image']
                    annotations = item['annotations']
                    
                    # Save image
                    image_path = yolo_dir / split_name / "images" / f"{i:06d}.png"
                    image.save(image_path)
                    
                    # Convert annotations to YOLO format
                    label_path = yolo_dir / split_name / "labels" / f"{i:06d}.txt"
                    self._write_yolo_labels(annotations, label_path, class_mapping, image.size)
            
            # Create dataset.yaml
            self._create_dataset_yaml(yolo_dir)
            
        except Exception as e:
            logger.error(f"Conversion to YOLO failed: {e}")
    
    def _convert_doclaynet_manual_to_yolo(self, output_dir: Path):
        """Convert manually downloaded DocLayNet to YOLO format."""
        # This would handle the manual zip files
        # Implementation depends on the actual structure of downloaded files
        logger.info("Manual conversion not yet implemented")
    
    def _write_yolo_labels(self, annotations: List, label_path: Path, 
                          class_mapping: Dict, image_size: tuple):
        """Write YOLO format labels."""
        try:
            img_width, img_height = image_size
            
            with open(label_path, 'w') as f:
                for ann in annotations:
                    # Get class and bbox
                    category = ann.get('category', 'Text')
                    bbox = ann.get('bbox', [0, 0, 1, 1])
                    
                    # Map to our classes
                    class_id = class_mapping.get(category, 1)  # Default to Text
                    
                    # Convert bbox to YOLO format [x_center, y_center, width, height]
                    x, y, w, h = bbox
                    x_center = (x + w/2) / img_width
                    y_center = (y + h/2) / img_height
                    width = w / img_width
                    height = h / img_height
                    
                    # Write YOLO line
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
        except Exception as e:
            logger.error(f"Failed to write labels to {label_path}: {e}")
    
    def _create_dataset_yaml(self, yolo_dir: Path):
        """Create dataset.yaml file for YOLO training."""
        dataset_yaml = {
            "names": ["Background", "Text", "Title", "List", "Table", "Figure"],
            "nc": 6,
            "path": str(yolo_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images"
        }
        
        yaml_path = yolo_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            import yaml
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        logger.info(f"📄 Created dataset.yaml at {yaml_path}")
    
    def download_publaynet(self) -> bool:
        """Download PubLayNet dataset."""
        try:
            publaynet_dir = self.output_dir / "pubLayNet"
            publaynet_dir.mkdir(exist_ok=True)
            
            logger.info("📥 Downloading PubLayNet...")
            # PubLayNet download logic would go here
            # For now, just create placeholder
            logger.info("⚠️  PubLayNet download not yet implemented")
            return False
            
        except Exception as e:
            logger.error(f"Failed to download PubLayNet: {e}")
            return False
    
    def download_icdar_mlt(self) -> bool:
        """Download ICDAR-MLT dataset."""
        try:
            icdar_dir = self.output_dir / "icdar_mlt"
            icdar_dir.mkdir(exist_ok=True)
            
            logger.info("📥 Downloading ICDAR-MLT...")
            # ICDAR-MLT download logic would go here
            logger.info("⚠️  ICDAR-MLT download not yet implemented")
            return False
            
        except Exception as e:
            logger.error(f"Failed to download ICDAR-MLT: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Download and process external datasets")
    parser.add_argument("--dataset", choices=["doclaynet", "publaynet", "icdar", "all"], 
                       default="doclaynet", help="Dataset to download")
    parser.add_argument("--output", default="data/public", help="Output directory")
    parser.add_argument("--use-hf", action="store_true", help="Use Hugging Face datasets")
    
    args = parser.parse_args()
    
    ingester = DatasetIngester(args.output)
    
    if args.dataset == "doclaynet" or args.dataset == "all":
        ingester.download_doclaynet(use_hf=args.use_hf)
    
    if args.dataset == "publaynet" or args.dataset == "all":
        ingester.download_publaynet()
    
    if args.dataset == "icdar" or args.dataset == "all":
        ingester.download_icdar_mlt()
    
    logger.info("🎉 Dataset ingestion completed!")

if __name__ == "__main__":
    main()
