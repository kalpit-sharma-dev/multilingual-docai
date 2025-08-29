"""
Image Deskewing Module for PS-05

Detects and corrects rotated/skewed images using multiple methods:
1. Hough Line Transform
2. Contour-based detection
3. Text orientation detection
4. Manual angle specification
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import math

logger = logging.getLogger(__name__)

class ImageDeskewer:
    """Robust image deskewing for document images."""
    
    def __init__(self, config: Dict = None):
        """Initialize deskewer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.methods = {
            'hough': self._deskew_hough,
            'contour': self._deskew_contour,
            'text': self._deskew_text,
            'manual': self._deskew_manual
        }
        
        # Default parameters
        self.default_params = {
            'hough_threshold': 100,
            'hough_angle_resolution': 1.0,
            'min_line_length': 100,
            'max_line_gap': 10,
            'contour_min_area': 1000,
            'text_detection_confidence': 0.5
        }
        
        # Update with config
        if config:
            self.default_params.update(config.get('deskew', {}))
    
    def deskew_image(self, image: np.ndarray, method: str = 'auto', 
                     angle: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """Deskew image using specified method.
        
        Args:
            image: Input image (BGR format)
            method: Deskewing method ('auto', 'hough', 'contour', 'text', 'manual')
            angle: Manual rotation angle (degrees, positive = clockwise)
            
        Returns:
            Tuple of (deskewed_image, detected_angle)
        """
        if image is None:
            raise ValueError("Input image is None")
        
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        detected_angle = 0.0
        
        if method == 'auto':
            # Try multiple methods and use the best result
            detected_angle = self._auto_deskew(gray)
        elif method == 'manual' and angle is not None:
            detected_angle = angle
        elif method in self.methods:
            detected_angle = self.methods[method](gray)
        else:
            raise ValueError(f"Unknown deskewing method: {method}")
        
        # Apply rotation if angle is significant
        if abs(detected_angle) > 0.1:  # Threshold of 0.1 degrees
            deskewed_image = self._rotate_image(image, detected_angle)
            logger.info(f"Deskewed image by {detected_angle:.2f} degrees")
        else:
            deskewed_image = image.copy()
            logger.info("No significant skew detected")
        
        return deskewed_image, detected_angle
    
    def _auto_deskew(self, gray: np.ndarray) -> float:
        """Automatically detect skew using multiple methods.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Detected skew angle
        """
        angles = []
        
        # Try Hough line method
        try:
            angle = self._deskew_hough(gray)
            if abs(angle) < 45:  # Reasonable angle range
                angles.append(angle)
        except Exception as e:
            logger.debug(f"Hough method failed: {e}")
        
        # Try contour method
        try:
            angle = self._deskew_contour(gray)
            if abs(angle) < 45:
                angles.append(angle)
        except Exception as e:
            logger.debug(f"Contour method failed: {e}")
        
        # Try text method
        try:
            angle = self._deskew_text(gray)
            if abs(angle) < 45:
                angles.append(angle)
        except Exception as e:
            logger.debug(f"Text method failed: {e}")
        
        if angles:
            # Use median angle for robustness
            detected_angle = np.median(angles)
            logger.info(f"Auto deskew: detected angles {angles}, using {detected_angle:.2f}")
            return detected_angle
        else:
            logger.warning("All deskewing methods failed, returning 0")
            return 0.0
    
    def _deskew_hough(self, gray: np.ndarray) -> float:
        """Detect skew using Hough Line Transform.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Detected skew angle
        """
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.default_params['hough_threshold'],
            minLineLength=self.default_params['min_line_length'],
            maxLineGap=self.default_params['max_line_gap']
        )
        
        if lines is None:
            return 0.0
        
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            if x2 - x1 != 0:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                # Normalize to -90 to 90 degrees
                if angle > 90:
                    angle -= 180
                elif angle < -90:
                    angle += 180
                angles.append(angle)
        
        if angles:
            # Use median for robustness
            detected_angle = np.median(angles)
            return detected_angle
        else:
            return 0.0
    
    def _deskew_contour(self, gray: np.ndarray) -> float:
        """Detect skew using contour analysis.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Detected skew angle
        """
        # Threshold image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        angles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.default_params['contour_min_area']:
                continue
            
            # Fit rotated rectangle
            rect = cv2.minAreaRect(contour)
            angle = rect[2]
            
            # Normalize angle
            if angle < -45:
                angle += 90
            angles.append(angle)
        
        if angles:
            detected_angle = np.median(angles)
            return detected_angle
        else:
            return 0.0
    
    def _deskew_text(self, gray: np.ndarray) -> float:
        """Detect skew using text orientation detection.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Detected skew angle
        """
        # This is a simplified text orientation detection
        # In production, you might use more sophisticated OCR-based methods
        
        # Use morphological operations to connect text lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of text lines
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        angles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Small text line threshold
                continue
            
            # Fit line to contour
            if len(contour) >= 2:
                [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                angle = np.arctan2(vy, vx) * 180 / np.pi
                
                # Normalize angle
                if angle > 90:
                    angle -= 180
                elif angle < -90:
                    angle += 180
                
                angles.append(angle)
        
        if angles:
            detected_angle = np.median(angles)
            return detected_angle
        else:
            return 0.0
    
    def _deskew_manual(self, gray: np.ndarray) -> float:
        """Manual deskewing (placeholder for user-specified angles).
        
        Args:
            gray: Grayscale image (not used in manual mode)
            
        Returns:
            Manual angle (0 for now, should be set by user)
        """
        return 0.0
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by specified angle.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees (positive = clockwise)
            
        Returns:
            Rotated image
        """
        if abs(angle) < 0.1:
            return image.copy()
        
        # Get image dimensions
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Calculate rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))
        
        # Adjust rotation matrix for new dimensions
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Apply rotation
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(255, 255, 255))
        
        return rotated_image
    
    def detect_rotation_quality(self, image: np.ndarray, angle: float) -> Dict:
        """Assess the quality of rotation correction.
        
        Args:
            image: Original image
            angle: Applied rotation angle
            
        Returns:
            Quality metrics dictionary
        """
        if abs(angle) < 0.1:
            return {'quality_score': 1.0, 'improvement': 0.0, 'recommendation': 'No rotation needed'}
        
        # Apply rotation
        rotated = self._rotate_image(image, angle)
        
        # Calculate quality metrics
        # 1. Line straightness (using Hough lines)
        gray_orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray_rot = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY) if len(rotated.shape) == 3 else rotated
        
        # Edge detection
        edges_orig = cv2.Canny(gray_orig, 50, 150)
        edges_rot = cv2.Canny(gray_rot, 50, 150)
        
        # Line detection
        lines_orig = cv2.HoughLines(edges_orig, 1, np.pi/180, threshold=100)
        lines_rot = cv2.HoughLines(edges_rot, 1, np.pi/180, threshold=100)
        
        # Calculate line angle variance (lower is better)
        orig_angles = []
        if lines_orig is not None:
            for line in lines_orig:
                theta = line[0][1]
                angle_deg = theta * 180 / np.pi
                if angle_deg > 90:
                    angle_deg -= 90
                orig_angles.append(abs(angle_deg))
        
        rot_angles = []
        if lines_rot is not None:
            for line in lines_rot:
                theta = line[0][1]
                angle_deg = theta * 180 / np.pi
                if angle_deg > 90:
                    angle_deg -= 90
                rot_angles.append(abs(angle_deg))
        
        # Calculate improvement
        if orig_angles and rot_angles:
            orig_variance = np.var(orig_angles) if len(orig_angles) > 1 else 0
            rot_variance = np.var(rot_angles) if len(rot_angles) > 1 else 0
            
            improvement = max(0, (orig_variance - rot_variance) / max(orig_variance, 1e-6))
            quality_score = min(1.0, improvement + 0.5)  # Base score of 0.5
        else:
            improvement = 0.0
            quality_score = 0.5
        
        # Generate recommendation
        if quality_score > 0.8:
            recommendation = "Excellent rotation correction"
        elif quality_score > 0.6:
            recommendation = "Good rotation correction"
        elif quality_score > 0.4:
            recommendation = "Moderate rotation correction"
        else:
            recommendation = "Poor rotation correction - consider manual adjustment"
        
        return {
            'quality_score': quality_score,
            'improvement': improvement,
            'original_variance': np.var(orig_angles) if orig_angles else 0,
            'rotated_variance': np.var(rot_angles) if rot_angles else 0,
            'recommendation': recommendation
        }

def deskew_image(image: np.ndarray, method: str = 'auto', 
                 angle: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """Convenience function for deskewing single image.
    
    Args:
        image: Input image
        method: Deskewing method
        angle: Manual rotation angle
        
    Returns:
        Tuple of (deskewed_image, detected_angle)
    """
    deskewer = ImageDeskewer()
    return deskewer.deskew_image(image, method, angle)

def batch_deskew_images(image_paths: List[str], output_dir: str, 
                       method: str = 'auto') -> Dict[str, Dict]:
    """Deskew multiple images in batch.
    
    Args:
        image_paths: List of image file paths
        output_dir: Output directory for deskewed images
        method: Deskewing method
        
    Returns:
        Dictionary with results for each image
    """
    deskewer = ImageDeskewer()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for img_path in image_paths:
        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                results[img_path] = {'error': 'Could not load image'}
                continue
            
            # Deskew image
            deskewed_image, detected_angle = deskewer.deskew_image(image, method)
            
            # Save deskewed image
            img_name = Path(img_path).stem
            output_path_img = output_path / f"{img_name}_deskewed.png"
            cv2.imwrite(str(output_path_img), deskewed_image)
            
            # Assess quality
            quality_metrics = deskewer.detect_rotation_quality(image, detected_angle)
            
            results[img_path] = {
                'detected_angle': detected_angle,
                'output_path': str(output_path_img),
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            results[img_path] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deskew images")
    parser.add_argument('--input', required=True, help='Input image or directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--method', default='auto', 
                       choices=['auto', 'hough', 'contour', 'text', 'manual'],
                       help='Deskewing method')
    parser.add_argument('--angle', type=float, help='Manual rotation angle (degrees)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single image
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"Could not load image: {input_path}")
            exit(1)
        
        deskewed, angle = deskew_image(image, args.method, args.angle)
        
        # Save result
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / f"{input_path.stem}_deskewed.png"
        cv2.imwrite(str(output_file), deskewed)
        
        print(f"Deskewed image by {angle:.2f} degrees")
        print(f"Saved to: {output_file}")
        
    elif input_path.is_dir():
        # Process directory
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if image_files:
            results = batch_deskew_images([str(p) for p in image_files], args.output, args.method)
            
            print(f"Processed {len(image_files)} images:")
            for img_path, result in results.items():
                if 'error' in result:
                    print(f"  {Path(img_path).name}: ERROR - {result['error']}")
                else:
                    print(f"  {Path(img_path).name}: {result['detected_angle']:.2f}° "
                          f"(Quality: {result['quality_metrics']['quality_score']:.2f})")
        else:
            print("No image files found in directory")
    
    else:
        print(f"Input path does not exist: {input_path}")
