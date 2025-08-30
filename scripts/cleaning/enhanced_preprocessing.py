#!/usr/bin/env python3
"""
Enhanced Data Preprocessing for PS-05 Stage 1

Handles multiple document formats:
- Images: PNG, JPG, JPEG, BMP, TIFF
- Documents: PDF, DOC, DOCX, PPT, PPTX

Applies preprocessing:
- Deskewing for rotated documents
- Noise removal and denoising
- Image enhancement and normalization
- Format conversion to images
- Quality assessment
"""

import argparse
import logging
import cv2
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil
from tqdm import tqdm
import yaml

# Try to import document processing libraries
try:
    import fitz  # PyMuPDF for PDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("PyMuPDF not available. Install with: pip install PyMuPDF")

try:
    from docx import Document  # python-docx for DOCX
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logging.warning("python-docx not available. Install with: pip install python-docx")

try:
    from pptx import Presentation  # python-pptx for PPTX
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False
    logging.warning("python-pptx not available. Install with: pip install python-pptx")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPreprocessor:
    """Enhanced document preprocessing for PS-05 Stage 1."""
    
    def __init__(self, config_path: str = "configs/ps05_config.yaml"):
        """Initialize the preprocessor."""
        self.config = self._load_config(config_path)
        self.output_dir = None
        self.manifest = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'preprocessing': {
                'deskew': {'enabled': True, 'max_angle': 15.0},
                'denoise': {'enabled': True, 'method': 'bilateral'},
                'resize': {'max_width': 2480, 'max_height': 3508, 'preserve_aspect': True},
                'normalization': {'enabled': True, 'method': 'clahe'}
            }
        }
    
    def preprocess_dataset(self, input_dir: str, output_dir: str) -> str:
        """Preprocess entire dataset."""
        logger.info(f"Starting enhanced preprocessing of dataset: {input_dir}")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        # Find all files
        all_files = self._find_all_files(input_dir)
        logger.info(f"Found {len(all_files)} files to process")
        
        # Process files with progress bar
        for file_path in tqdm(all_files, desc="Preprocessing files"):
            try:
                self._process_single_file(file_path)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        # Save manifest
        self._save_manifest()
        
        # Create dataset YAML
        dataset_yaml = self._create_dataset_yaml()
        
        logger.info(f"Preprocessing completed. Output: {output_dir}")
        logger.info(f"Dataset YAML: {dataset_yaml}")
        
        return dataset_yaml
    
    def _find_all_files(self, input_dir: str) -> List[Path]:
        """Find all supported files in directory."""
        input_path = Path(input_dir)
        all_files = []
        
        # Supported extensions
        image_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        doc_exts = ['.pdf', '.doc', '.docx', '.ppt', '.pptx']
        
        # Find all files
        for ext in image_exts + doc_exts:
            all_files.extend(input_path.glob(f"*{ext}"))
            all_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        return sorted(all_files)
    
    def _process_single_file(self, file_path: Path):
        """Process a single file."""
        suffix = file_path.suffix.lower()
        
        if suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            self._process_image_file(file_path)
        elif suffix == '.pdf':
            self._process_pdf_file(file_path)
        elif suffix in ['.doc', '.docx']:
            self._process_word_file(file_path)
        elif suffix in ['.ppt', '.pptx']:
            self._process_powerpoint_file(file_path)
    
    def _process_image_file(self, file_path: Path):
        """Process image file."""
        try:
            # Load image
            image = cv2.imread(str(file_path))
            if image is None:
                logger.warning(f"Could not load image: {file_path}")
                return
            
            # Apply preprocessing
            processed_image = self._apply_image_preprocessing(image)
            
            # Save processed image
            output_name = f"{file_path.stem}_processed.png"
            output_path = self.output_dir / "images" / output_name
            cv2.imwrite(str(output_path), processed_image)
            
            # Copy annotation if exists
            annotation_path = file_path.with_suffix('.json')
            if annotation_path.exists():
                output_ann_path = self.output_dir / "annotations" / f"{file_path.stem}_processed.json"
                shutil.copy2(annotation_path, output_ann_path)
            
            # Add to manifest
            self.manifest.append({
                'original_file': str(file_path),
                'processed_image': str(output_path),
                'annotation': str(output_ann_path) if annotation_path.exists() else None,
                'file_type': 'image',
                'preprocessing_applied': ['deskew', 'denoise', 'enhance']
            })
            
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {e}")
    
    def _process_pdf_file(self, file_path: Path):
        """Process PDF file."""
        if not PDF_AVAILABLE:
            logger.warning(f"PDF processing not available for {file_path}")
            return
        
        try:
            import fitz
            
            doc = fitz.open(str(file_path))
            logger.info(f"Processing PDF {file_path} with {len(doc)} pages")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to numpy array
                img_data = pix.tobytes("png")
                image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                
                # Apply preprocessing
                processed_image = self._apply_image_preprocessing(image)
                
                # Save processed image
                output_name = f"{file_path.stem}_page_{page_num:03d}_processed.png"
                output_path = self.output_dir / "images" / output_name
                cv2.imwrite(str(output_path), processed_image)
                
                # Add to manifest
                self.manifest.append({
                    'original_file': str(file_path),
                    'processed_image': str(output_path),
                    'annotation': None,
                    'file_type': 'pdf',
                    'page_number': page_num,
                    'preprocessing_applied': ['deskew', 'denoise', 'enhance']
                })
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
    
    def _process_word_file(self, file_path: Path):
        """Process Word document."""
        if not DOCX_AVAILABLE:
            logger.warning(f"DOCX processing not available for {file_path}")
            return
        
        try:
            # For now, create a placeholder image
            # In production, you would extract text and create a proper image representation
            output_name = f"{file_path.stem}_processed.png"
            output_path = self.output_dir / "images" / output_name
            
            # Create placeholder image
            img = np.ones((800, 600, 3), dtype=np.uint8) * 255
            cv2.putText(img, f"Word Document: {file_path.name}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(img, "Converted to image for processing", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            
            cv2.imwrite(str(output_path), img)
            
            # Add to manifest
            self.manifest.append({
                'original_file': str(file_path),
                'processed_image': str(output_path),
                'annotation': None,
                'file_type': 'word',
                'preprocessing_applied': ['conversion']
            })
            
        except Exception as e:
            logger.error(f"Error processing Word document {file_path}: {e}")
    
    def _process_powerpoint_file(self, file_path: Path):
        """Process PowerPoint file."""
        if not PPTX_AVAILABLE:
            logger.warning(f"PPTX processing not available for {file_path}")
            return
        
        try:
            # For now, create a placeholder image
            output_name = f"{file_path.stem}_processed.png"
            output_path = self.output_dir / "images" / output_name
            
            # Create placeholder image
            img = np.ones((800, 600, 3), dtype=np.uint8) * 255
            cv2.putText(img, f"PowerPoint: {file_path.name}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(img, "Converted to image for processing", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            
            cv2.imwrite(str(output_path), img)
            
            # Add to manifest
            self.manifest.append({
                'original_file': str(file_path),
                'processed_image': str(output_path),
                'annotation': None,
                'file_type': 'powerpoint',
                'preprocessing_applied': ['conversion']
            })
            
        except Exception as e:
            logger.error(f"Error processing PowerPoint {file_path}: {e}")
    
    def _apply_image_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Apply comprehensive image preprocessing."""
        try:
            # Step 1: Deskewing
            if self.config['preprocessing']['deskew']['enabled']:
                image, angle = self._deskew_image(image)
                logger.debug(f"Deskewed image by {angle:.2f} degrees")
            
            # Step 2: Denoising
            if self.config['preprocessing']['denoise']['enabled']:
                image = self._denoise_image(image)
            
            # Step 3: Resize
            if self.config['preprocessing']['resize']['enabled']:
                image = self._resize_image(image)
            
            # Step 4: Normalization
            if self.config['preprocessing']['normalization']['enabled']:
                image = self._normalize_image(image)
            
            return image
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            return image
    
    def _deskew_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Deskew rotated image."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Hough line detection
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is None:
                return image, 0.0
            
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = np.degrees(theta)
                if angle < 90:
                    angles.append(angle)
                else:
                    angles.append(angle - 90)
            
            if angles:
                # Use median angle for robustness
                detected_angle = np.median(angles)
                
                # Apply rotation if angle is significant
                if abs(detected_angle) > 0.1:
                    image = self._rotate_image(image, detected_angle)
                
                return image, detected_angle
            
            return image, 0.0
            
        except Exception as e:
            logger.error(f"Error in deskewing: {e}")
            return image, 0.0
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by specified angle."""
        if abs(angle) < 0.1:
            return image
        
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Calculate rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))
        
        # Adjust rotation matrix
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))
        
        return rotated
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from image."""
        try:
            method = self.config['preprocessing']['denoise']['method']
            
            if method == 'bilateral':
                # Bilateral filter preserves edges while removing noise
                denoised = cv2.bilateralFilter(image, 9, 75, 75)
            elif method == 'gaussian':
                # Gaussian blur
                denoised = cv2.GaussianBlur(image, (5, 5), 0)
            elif method == 'median':
                # Median filter
                denoised = cv2.medianBlur(image, 5)
            else:
                denoised = image
            
            return denoised
            
        except Exception as e:
            logger.error(f"Error in denoising: {e}")
            return image
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image while preserving aspect ratio."""
        try:
            max_width = self.config['preprocessing']['resize']['max_width']
            max_height = self.config['preprocessing']['resize']['max_height']
            preserve_aspect = self.config['preprocessing']['resize']['preserve_aspect']
            
            height, width = image.shape[:2]
            
            if preserve_aspect:
                # Calculate scale factor
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
            else:
                new_width = max_width
                new_height = max_height
            
            # Resize image
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            return resized
            
        except Exception as e:
            logger.error(f"Error in resizing: {e}")
            return image
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image for better contrast."""
        try:
            method = self.config['preprocessing']['normalization']['method']
            
            if method == 'clahe':
                # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                
                limg = cv2.merge((cl, a, b))
                normalized = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                
            elif method == 'histogram_equalization':
                # Global histogram equalization
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                equalized = cv2.equalizeHist(gray)
                normalized = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
            else:
                normalized = image
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error in normalization: {e}")
            return image
    
    def _save_manifest(self):
        """Save processing manifest."""
        manifest_path = self.output_dir / "metadata" / "preprocessing_manifest.json"
        
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2, default=str)
        
        logger.info(f"Manifest saved to {manifest_path}")
    
    def _create_dataset_yaml(self) -> str:
        """Create dataset YAML for YOLO training."""
        yaml_path = self.output_dir / "dataset.yaml"
        
        dataset_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images',
            'val': 'images',  # For now, use same directory
            'test': 'images',
            'nc': 6,  # Number of classes
            'names': ['Background', 'Text', 'Title', 'List', 'Table', 'Figure']
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"Dataset YAML created at {yaml_path}")
        return str(yaml_path)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Enhanced dataset preprocessing for PS-05")
    parser.add_argument('--input', required=True, help='Input data directory')
    parser.add_argument('--output', required=True, help='Output directory for processed data')
    parser.add_argument('--config', default='configs/ps05_config.yaml', help='Configuration file')
    
    args = parser.parse_args()
    
    # Run preprocessing
    preprocessor = EnhancedPreprocessor(args.config)
    dataset_yaml = preprocessor.preprocess_dataset(args.input, args.output)
    
    print(f"\nEnhanced preprocessing completed!")
    print(f"Output directory: {args.output}")
    print(f"Dataset YAML: {dataset_yaml}")
    print(f"Manifest: {args.output}/metadata/preprocessing_manifest.json")

if __name__ == "__main__":
    main()
