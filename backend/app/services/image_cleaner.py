"""
Image Cleaning Service for PS-05

Implements all 10 image cleaning tasks:
1. Removing Corrupt/Unreadable Files
2. Deduplication
3. Resizing & Rescaling
4. Color Space Conversion
5. Handling Low-Resolution Images
6. Noise Reduction
7. Augmentation (Pre-processing)
8. Annotation Cleaning
9. EXIF Data Normalization
10. Outlier Detection
"""

import logging
import os
import hashlib
import cv2
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import imagehash
from skimage.metrics import structural_similarity as ssim
from skimage import filters, restoration
import json
import shutil

logger = logging.getLogger(__name__)

class ImageCleaningService:
    """Comprehensive image cleaning service for document datasets."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.cleaned_dir = None
        self.cleaning_log = []
        
    def _get_default_config(self) -> Dict:
        """Get default cleaning configuration."""
        return {
            "target_size": (800, 1000),  # Standard document size
            "min_resolution": (100, 100),  # Minimum acceptable resolution
            "max_file_size_mb": 50,  # Maximum file size
            "similarity_threshold": 0.95,  # For deduplication
            "noise_reduction": True,
            "exif_normalization": True,
            "augmentation": True,
            "quality_threshold": 0.7,  # Minimum quality score
            "supported_formats": ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        }
    
    def clean_dataset(
        self, 
        input_dir: Path, 
        output_dir: Path,
        annotations_file: Optional[Path] = None
    ) -> Dict:
        """
        Clean entire image dataset with all 10 cleaning tasks.
        
        Args:
            input_dir: Directory containing raw images
            output_dir: Directory for cleaned images
            annotations_file: Optional annotations file for cleaning
            
        Returns:
            Dictionary with cleaning results and statistics
        """
        try:
            logger.info(f"Starting comprehensive image cleaning for {input_dir}")
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            self.cleaned_dir = output_dir
            
            # Initialize cleaning log
            self.cleaning_log = []
            
            # Get all image files
            image_files = self._get_image_files(input_dir)
            logger.info(f"Found {len(image_files)} images to process")
            
            # Initialize statistics
            stats = {
                "total_images": len(image_files),
                "cleaned_images": 0,
                "removed_corrupt": 0,
                "removed_duplicates": 0,
                "removed_low_res": 0,
                "removed_outliers": 0,
                "augmented_images": 0,
                "cleaning_errors": 0
            }
            
            # Task 1: Remove corrupt/unreadable files
            valid_images = self._remove_corrupt_files(image_files, stats)
            
            # Task 2: Deduplication
            unique_images = self._remove_duplicates(valid_images, stats)
            
            # Task 3-6: Clean individual images
            cleaned_images = []
            for img_path in unique_images:
                try:
                    cleaned_path = self._clean_single_image(img_path, output_dir)
                    if cleaned_path:
                        cleaned_images.append(cleaned_path)
                        stats["cleaned_images"] += 1
                except Exception as e:
                    logger.error(f"Failed to clean {img_path}: {e}")
                    stats["cleaning_errors"] += 1
            
            # Task 7: Augmentation
            if self.config["augmentation"]:
                augmented_images = self._apply_augmentation(cleaned_images, output_dir)
                stats["augmented_images"] = len(augmented_images)
            
            # Task 8: Annotation cleaning (if provided)
            if annotations_file and annotations_file.exists():
                self._clean_annotations(annotations_file, output_dir, cleaned_images)
            
            # Task 10: Outlier detection
            final_images = self._detect_outliers(cleaned_images, stats)
            
            # Save cleaning log
            self._save_cleaning_log(output_dir)
            
            # Generate final statistics
            final_stats = self._generate_final_stats(stats, final_images)
            
            logger.info(f"Image cleaning completed. Final images: {len(final_images)}")
            return final_stats
            
        except Exception as e:
            logger.error(f"Image cleaning failed: {e}")
            raise
    
    def _get_image_files(self, input_dir: Path) -> List[Path]:
        """Get all supported image files from directory."""
        image_files = []
        for format_ext in self.config["supported_formats"]:
            image_files.extend(input_dir.glob(f"*{format_ext}"))
            image_files.extend(input_dir.glob(f"*{format_ext.upper()}"))
        return sorted(image_files)
    
    def _remove_corrupt_files(self, image_files: List[Path], stats: Dict) -> List[Path]:
        """Task 1: Remove corrupt/unreadable files."""
        logger.info("Removing corrupt and unreadable files...")
        valid_images = []
        
        for img_path in image_files:
            try:
                # Check file size
                file_size_mb = img_path.stat().st_size / (1024 * 1024)
                if file_size_mb > self.config["max_file_size_mb"]:
                    logger.warning(f"File too large: {img_path} ({file_size_mb:.2f} MB)")
                    stats["removed_corrupt"] += 1
                    continue
                
                # Try to open with PIL
                with Image.open(img_path) as img:
                    img.verify()
                
                # Try to open with OpenCV
                img_cv = cv2.imread(str(img_path))
                if img_cv is None:
                    logger.warning(f"OpenCV cannot read: {img_path}")
                    stats["removed_corrupt"] += 1
                    continue
                
                valid_images.append(img_path)
                
            except Exception as e:
                logger.warning(f"Corrupt file removed: {img_path} - {e}")
                stats["removed_corrupt"] += 1
        
        logger.info(f"Valid images after corruption check: {len(valid_images)}")
        return valid_images
    
    def _remove_duplicates(self, image_files: List[Path], stats: Dict) -> List[Path]:
        """Task 2: Remove exact and near-duplicate images."""
        logger.info("Removing duplicate images...")
        
        # Calculate perceptual hashes for all images
        image_hashes = {}
        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Calculate perceptual hash
                    hash_value = imagehash.average_hash(img)
                    image_hashes[img_path] = hash_value
                    
            except Exception as e:
                logger.warning(f"Failed to hash {img_path}: {e}")
                continue
        
        # Remove duplicates based on hash similarity
        unique_images = []
        processed_hashes = set()
        
        for img_path, hash_value in image_hashes.items():
            is_duplicate = False
            
            for processed_hash in processed_hashes:
                # Calculate hash difference (0 = identical, higher = more different)
                hash_diff = hash_value - processed_hash
                if hash_diff <= 5:  # Threshold for similarity
                    is_duplicate = True
                    stats["removed_duplicates"] += 1
                    break
            
            if not is_duplicate:
                unique_images.append(img_path)
                processed_hashes.add(hash_value)
        
        logger.info(f"Unique images after deduplication: {len(unique_images)}")
        return unique_images
    
    def _clean_single_image(self, img_path: Path, output_dir: Path) -> Optional[Path]:
        """Tasks 3-6: Clean individual image (resize, convert, denoise, normalize)."""
        try:
            # Load image
            with Image.open(img_path) as pil_img:
                # Task 9: EXIF normalization
                if self.config["exif_normalization"]:
                    pil_img = ImageOps.exif_transpose(pil_img)
                
                # Convert to RGB
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                # Task 3: Resize & rescale
                pil_img = self._resize_image(pil_img)
                
                # Convert to numpy for OpenCV operations
                img_array = np.array(pil_img)
                
                # Task 4: Color space conversion (RGB to BGR for OpenCV)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Task 5: Handle low-resolution images
                if not self._check_resolution(img_bgr):
                    logger.warning(f"Image resolution too low: {img_path}")
                    return None
                
                # Task 6: Noise reduction
                if self.config["noise_reduction"]:
                    img_bgr = self._reduce_noise(img_bgr)
                
                # Convert back to RGB
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # Save cleaned image
                output_path = output_dir / f"cleaned_{img_path.stem}.png"
                cleaned_img = Image.fromarray(img_rgb)
                cleaned_img.save(output_path, 'PNG', quality=95)
                
                # Log cleaning action
                self.cleaning_log.append({
                    "action": "image_cleaned",
                    "original": str(img_path),
                    "cleaned": str(output_path),
                    "operations": ["resize", "color_convert", "noise_reduction", "exif_normalize"]
                })
                
                return output_path
                
        except Exception as e:
            logger.error(f"Failed to clean {img_path}: {e}")
            return None
    
    def _resize_image(self, img: Image.Image) -> Image.Image:
        """Resize image to target size while maintaining aspect ratio."""
        target_width, target_height = self.config["target_size"]
        
        # Calculate aspect ratio
        img_width, img_height = img.size
        aspect_ratio = img_width / img_height
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
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with target size and paste resized image
        new_img = Image.new('RGB', (target_width, target_height), (255, 255, 255))
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        new_img.paste(img_resized, (paste_x, paste_y))
        
        return new_img
    
    def _check_resolution(self, img: np.ndarray) -> bool:
        """Check if image meets minimum resolution requirements."""
        height, width = img.shape[:2]
        min_width, min_height = self.config["min_resolution"]
        return width >= min_width and height >= min_height
    
    def _reduce_noise(self, img: np.ndarray) -> np.ndarray:
        """Apply noise reduction filters."""
        # Convert to grayscale for noise reduction
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter (preserves edges while reducing noise)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Convert back to BGR
        denoised_bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        
        return denoised_bgr
    
    def _apply_augmentation(self, cleaned_images: List[Path], output_dir: Path) -> List[Path]:
        """Task 7: Apply data augmentation for training robustness."""
        logger.info("Applying data augmentation...")
        augmented_images = []
        
        for img_path in cleaned_images:
            try:
                with Image.open(img_path) as img:
                    # Create augmented versions
                    augmented_versions = self._create_augmented_versions(img)
                    
                    for i, aug_img in enumerate(augmented_versions):
                        aug_path = output_dir / f"aug_{img_path.stem}_v{i+1}.png"
                        aug_img.save(aug_path, 'PNG', quality=95)
                        augmented_images.append(aug_path)
                        
                        # Log augmentation
                        self.cleaning_log.append({
                            "action": "augmentation",
                            "original": str(img_path),
                            "augmented": str(aug_path),
                            "type": f"augmentation_v{i+1}"
                        })
                
            except Exception as e:
                logger.error(f"Failed to augment {img_path}: {e}")
        
        logger.info(f"Created {len(augmented_images)} augmented images")
        return augmented_images
    
    def _create_augmented_versions(self, img: Image.Image) -> List[Image.Image]:
        """Create multiple augmented versions of an image."""
        augmented = []
        
        # Rotation (slight skew)
        for angle in [-5, -2, 2, 5]:
            rotated = img.rotate(angle, fillcolor=(255, 255, 255))
            augmented.append(rotated)
        
        # Brightness adjustment
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(img)
        bright = enhancer.enhance(1.2)
        dark = enhancer.enhance(0.8)
        augmented.extend([bright, dark])
        
        # Contrast adjustment
        enhancer = ImageEnhance.Contrast(img)
        high_contrast = enhancer.enhance(1.3)
        low_contrast = enhancer.enhance(0.7)
        augmented.extend([high_contrast, low_contrast])
        
        # Add noise
        noisy = self._add_noise(img)
        augmented.append(noisy)
        
        return augmented
    
    def _add_noise(self, img: Image.Image) -> Image.Image:
        """Add random noise to image."""
        img_array = np.array(img)
        noise = np.random.normal(0, 25, img_array.shape).astype(np.uint8)
        noisy_array = np.clip(img_array + noise, 0, 255)
        return Image.fromarray(noisy_array.astype(np.uint8))
    
    def _clean_annotations(self, annotations_file: Path, output_dir: Path, cleaned_images: List[Path]):
        """Task 8: Clean annotations to match cleaned images."""
        try:
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
            
            # Create mapping from original to cleaned image names
            image_mapping = {}
            for cleaned_path in cleaned_images:
                if cleaned_path.name.startswith("cleaned_"):
                    original_name = cleaned_path.name[8:]  # Remove "cleaned_" prefix
                    image_mapping[original_name] = cleaned_path.name
            
            # Update annotations
            cleaned_annotations = []
            for annotation in annotations:
                if "filename" in annotation:
                    original_filename = annotation["filename"]
                    if original_filename in image_mapping:
                        # Update filename to cleaned version
                        annotation["filename"] = image_mapping[original_filename]
                        cleaned_annotations.append(annotation)
            
            # Save cleaned annotations
            cleaned_annotations_file = output_dir / "cleaned_annotations.json"
            with open(cleaned_annotations_file, 'w') as f:
                json.dump(cleaned_annotations, f, indent=2)
            
            logger.info(f"Cleaned annotations saved: {cleaned_annotations_file}")
            
        except Exception as e:
            logger.error(f"Failed to clean annotations: {e}")
    
    def _detect_outliers(self, cleaned_images: List[Path], stats: Dict) -> List[Path]:
        """Task 10: Detect and remove outlier images."""
        logger.info("Detecting outlier images...")
        
        if len(cleaned_images) < 3:
            return cleaned_images  # Need at least 3 images for outlier detection
        
        try:
            # Calculate average image characteristics
            characteristics = []
            for img_path in cleaned_images:
                try:
                    with Image.open(img_path) as img:
                        img_array = np.array(img)
                        # Calculate brightness, contrast, and edge density
                        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                        brightness = np.mean(gray)
                        contrast = np.std(gray)
                        edges = cv2.Canny(gray, 50, 150)
                        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                        
                        characteristics.append({
                            'path': img_path,
                            'brightness': brightness,
                            'contrast': contrast,
                            'edge_density': edge_density
                        })
                except Exception as e:
                    logger.warning(f"Failed to analyze {img_path}: {e}")
                    continue
            
            if len(characteristics) < 3:
                return cleaned_images
            
            # Calculate statistics
            brightness_values = [c['brightness'] for c in characteristics]
            contrast_values = [c['contrast'] for c in characteristics]
            edge_density_values = [c['edge_density'] for c in characteristics]
            
            # Remove outliers (beyond 2 standard deviations)
            non_outliers = []
            for char in characteristics:
                is_outlier = False
                
                # Check brightness
                if abs(char['brightness'] - np.mean(brightness_values)) > 2 * np.std(brightness_values):
                    is_outlier = True
                
                # Check contrast
                if abs(char['contrast'] - np.mean(contrast_values)) > 2 * np.std(contrast_values):
                    is_outlier = True
                
                # Check edge density
                if abs(char['edge_density'] - np.mean(edge_density_values)) > 2 * np.std(edge_density_values):
                    is_outlier = True
                
                if not is_outlier:
                    non_outliers.append(char['path'])
                else:
                    stats["removed_outliers"] += 1
                    logger.info(f"Outlier removed: {char['path']}")
            
            logger.info(f"Images after outlier removal: {len(non_outliers)}")
            return non_outliers
            
        except Exception as e:
            logger.error(f"Outlier detection failed: {e}")
            return cleaned_images
    
    def _save_cleaning_log(self, output_dir: Path):
        """Save detailed cleaning log."""
        log_file = output_dir / "cleaning_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.cleaning_log, f, indent=2)
        logger.info(f"Cleaning log saved: {log_file}")
    
    def _generate_final_stats(self, stats: Dict, final_images: List[Path]) -> Dict:
        """Generate final cleaning statistics."""
        final_stats = {
            "cleaning_summary": stats,
            "final_image_count": len(final_images),
            "cleaning_efficiency": (len(final_images) / stats["total_images"]) * 100,
            "output_directory": str(self.cleaned_dir),
            "cleaning_log_file": "cleaning_log.json"
        }
        
        return final_stats
