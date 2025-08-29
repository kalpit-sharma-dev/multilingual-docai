import cv2
import numpy as np
from PIL import Image
import easyocr
import pytesseract
from typing import List, Dict, Any, Optional, Tuple
import time
import uuid
import os
from pathlib import Path
import logging
from app.models.schemas import (
    ProcessingResult, LayoutElement, TextLine, TableResult, 
    ChartResult, FigureResult, MapResult, PreprocessingInfo
)
from app.config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Main document processing service that handles all processing stages."""
    
    def __init__(self):
        self.easyocr_reader = None
        self.supported_languages = settings.SUPPORTED_LANGUAGES
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize OCR and other AI models."""
        try:
            # Initialize EasyOCR for multilingual support
            self.easyocr_reader = easyocr.Reader(
                self.supported_languages,
                model_storage_directory=settings.EASYOCR_MODEL_PATH,
                download_enabled=True
            )
            logger.info("EasyOCR models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.easyocr_reader = None
    
    def process_document(
        self, 
        image_path: str, 
        stage: int = 3,
        high_quality: bool = False,
        language_hint: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process document through specified stages.
        
        Args:
            image_path: Path to the image file
            stage: Processing stage (1, 2, or 3)
            high_quality: Use high quality processing
            language_hint: Language hint for OCR
            
        Returns:
            ProcessingResult with all detected elements
        """
        start_time = time.time()
        
        try:
            # Load and preprocess image
            image = self._load_image(image_path)
            if image is None:
                raise ValueError("Failed to load image")
            
            # Stage 1: Layout Detection
            layout_elements = self._detect_layout(image, high_quality)
            
            # Initialize result
            result = ProcessingResult(
                page=1,
                size={'w': image.shape[1], 'h': image.shape[0]},
                elements=layout_elements,
                preprocess=self._get_preprocessing_info(image),
                processing_time=0.0
            )
            
            # Stage 2: OCR and Language Detection
            if stage >= 2:
                text_lines = self._extract_text_and_language(
                    image, layout_elements, language_hint, high_quality
                )
                result.text_lines = text_lines
            
            # Stage 3: Full Analysis
            if stage >= 3:
                tables = self._detect_tables(image, layout_elements, high_quality)
                charts = self._detect_charts(image, layout_elements, high_quality)
                figures = self._detect_figures(image, layout_elements, high_quality)
                maps = self._detect_maps(image, layout_elements, high_quality)
                
                result.tables = tables
                result.charts = charts
                result.figures = figures
                result.maps = maps
            
            # Calculate processing time
            result.processing_time = time.time() - start_time
            
            logger.info(f"Document processed successfully in {result.processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and validate image."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Load image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to load image with OpenCV")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except Exception as e:
            logger.error(f"Image loading failed: {e}")
            return None
    
    def _detect_layout(self, image: np.ndarray, high_quality: bool) -> List[LayoutElement]:
        """Stage 1: Detect layout elements (text, titles, lists, tables, figures)."""
        try:
            elements = []
            
            # Convert to grayscale for processing
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply preprocessing
            if high_quality:
                # High quality preprocessing
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                gray = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
            else:
                # Standard preprocessing
                gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Find contours
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and classify contours
            for i, contour in enumerate(contours):
                # Filter small contours
                area = cv2.contourArea(contour)
                if area < 100:  # Minimum area threshold
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Classify element type based on shape and size
                element_type = self._classify_element(contour, area, w, h)
                
                # Calculate confidence score
                confidence = self._calculate_confidence(contour, area, w, h)
                
                # Create layout element
                element = LayoutElement(
                    id=str(uuid.uuid4()),
                    cls=element_type,
                    bbox=[float(x), float(y), float(x + w), float(y + h)],
                    score=confidence,
                    area=float(area)
                )
                
                elements.append(element)
            
            # Sort elements by area (largest first)
            elements.sort(key=lambda x: x.area, reverse=True)
            
            logger.info(f"Detected {len(elements)} layout elements")
            return elements
            
        except Exception as e:
            logger.error(f"Layout detection failed: {e}")
            return []
    
    def _classify_element(self, contour: np.ndarray, area: float, width: int, height: int) -> str:
        """Classify element type based on contour properties."""
        try:
            # Calculate aspect ratio
            aspect_ratio = width / height if height > 0 else 0
            
            # Calculate contour properties
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Classification logic
            if circularity > 0.8:
                return "figure"  # Circular/oval shapes
            elif aspect_ratio > 3 or aspect_ratio < 0.33:
                return "text"  # Very wide or tall elements
            elif area > 10000:  # Large elements
                if aspect_ratio > 1.5:
                    return "table"  # Wide tables
                else:
                    return "figure"  # Large figures
            elif 0.5 <= aspect_ratio <= 2.0 and area < 5000:
                return "title"  # Medium-sized rectangular elements
            else:
                return "text"  # Default to text
                
        except Exception as e:
            logger.error(f"Element classification failed: {e}")
            return "text"
    
    def _calculate_confidence(self, contour: np.ndarray, area: float, width: int, height: int) -> float:
        """Calculate confidence score for element detection."""
        try:
            # Base confidence on area and shape regularity
            area_score = min(area / 10000, 1.0)  # Normalize area score
            
            # Calculate shape regularity
            perimeter = cv2.arcLength(contour, True)
            shape_score = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Combine scores
            confidence = (area_score + shape_score) / 2
            
            # Ensure confidence is between 0 and 1
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _extract_text_and_language(
        self, 
        image: np.ndarray, 
        layout_elements: List[LayoutElement],
        language_hint: Optional[str],
        high_quality: bool
    ) -> List[TextLine]:
        """Stage 2: Extract text and detect language for each text element."""
        try:
            text_lines = []
            
            # Filter text elements
            text_elements = [elem for elem in layout_elements if elem.cls in ["text", "title"]]
            
            for element in text_elements:
                try:
                    # Extract region of interest
                    x1, y1, x2, y2 = map(int, element.bbox)
                    roi = image[y1:y2, x1:x2]
                    
                    if roi.size == 0:
                        continue
                    
                    # Extract text using EasyOCR
                    if self.easyocr_reader:
                        ocr_result = self.easyocr_reader.readtext(roi)
                        
                        for (bbox, text, confidence) in ocr_result:
                            if text.strip() and confidence > 0.3:  # Confidence threshold
                                # Detect language
                                detected_lang = self._detect_language(text, language_hint)
                                
                                # Create text line
                                text_line = TextLine(
                                    id=str(uuid.uuid4()),
                                    bbox=element.bbox,  # Use element bbox for now
                                    text=text.strip(),
                                    lang=detected_lang,
                                    score=confidence,
                                    lang_confidence=0.8  # Default confidence
                                )
                                
                                text_lines.append(text_line)
                    
                    # Fallback to Tesseract if EasyOCR fails
                    if not text_lines and pytesseract:
                        try:
                            text = pytesseract.image_to_string(roi, lang='eng')
                            if text.strip():
                                text_line = TextLine(
                                    id=str(uuid.uuid4()),
                                    bbox=element.bbox,
                                    text=text.strip(),
                                    lang="en",
                                    score=0.6,
                                    lang_confidence=0.7
                                )
                                text_lines.append(text_line)
                        except Exception as e:
                            logger.warning(f"Tesseract fallback failed: {e}")
                            
                except Exception as e:
                    logger.warning(f"Text extraction failed for element {element.id}: {e}")
                    continue
            
            logger.info(f"Extracted {len(text_lines)} text lines")
            return text_lines
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return []
    
    def _detect_language(self, text: str, language_hint: Optional[str]) -> str:
        """Detect language of text."""
        try:
            # Simple language detection based on character sets
            # This is a basic implementation - in production, use a proper language detection library
            
            # Check for Arabic/Persian characters
            if any('\u0600' <= char <= '\u06FF' for char in text):
                return "ar" if language_hint == "ar" else "fa"
            
            # Check for Devanagari (Hindi, Nepali)
            if any('\u0900' <= char <= '\u097F' for char in text):
                return "hi" if language_hint == "hi" else "ne"
            
            # Check for Urdu
            if any('\u0600' <= char <= '\u06FF' for char in text) and any(char in 'ے' for char in text):
                return "ur"
            
            # Default to English
            return "en"
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "en"
    
    def _detect_tables(self, image: np.ndarray, layout_elements: List[LayoutElement], high_quality: bool) -> List[TableResult]:
        """Detect tables in the image."""
        try:
            tables = []
            
            # Filter table elements
            table_elements = [elem for elem in layout_elements if elem.cls == "table"]
            
            for element in table_elements:
                try:
                    # Extract table region
                    x1, y1, x2, y2 = map(int, element.bbox)
                    roi = image[y1:y2, x1:x2]
                    
                    # Simple table detection (in production, use advanced table detection models)
                    table_result = TableResult(
                        bbox=element.bbox,
                        summary="Table containing structured data",
                        confidence=element.score,
                        rows=3,  # Placeholder
                        columns=4,  # Placeholder
                        content=None
                    )
                    
                    tables.append(table_result)
                    
                except Exception as e:
                    logger.warning(f"Table detection failed for element {element.id}: {e}")
                    continue
            
            logger.info(f"Detected {len(tables)} tables")
            return tables
            
        except Exception as e:
            logger.error(f"Table detection failed: {e}")
            return []
    
    def _detect_charts(self, image: np.ndarray, layout_elements: List[LayoutElement], high_quality: bool) -> List[ChartResult]:
        """Detect charts in the image."""
        try:
            charts = []
            
            # Filter figure elements that might be charts
            chart_elements = [elem for elem in layout_elements if elem.cls == "figure" and elem.area > 5000]
            
            for element in chart_elements:
                try:
                    chart_result = ChartResult(
                        bbox=element.bbox,
                        type="chart",  # Placeholder
                        summary="Chart or graph visualization",
                        confidence=element.score,
                        data_points=None
                    )
                    
                    charts.append(chart_result)
                    
                except Exception as e:
                    logger.warning(f"Chart detection failed for element {element.id}: {e}")
                    continue
            
            logger.info(f"Detected {len(charts)} charts")
            return charts
            
        except Exception as e:
            logger.error(f"Chart detection failed: {e}")
            return []
    
    def _detect_figures(self, image: np.ndarray, layout_elements: List[LayoutElement], high_quality: bool) -> List[FigureResult]:
        """Detect figures in the image."""
        try:
            figures = []
            
            # Filter figure elements
            figure_elements = [elem for elem in layout_elements if elem.cls == "figure"]
            
            for element in figure_elements:
                try:
                    figure_result = FigureResult(
                        bbox=element.bbox,
                        summary="Image or diagram",
                        confidence=element.score,
                        figure_type="image"
                    )
                    
                    figures.append(figure_result)
                    
                except Exception as e:
                    logger.warning(f"Figure detection failed for element {element.id}: {e}")
                    continue
            
            logger.info(f"Detected {len(figures)} figures")
            return figures
            
        except Exception as e:
            logger.error(f"Figure detection failed: {e}")
            return []
    
    def _detect_maps(self, image: np.ndarray, layout_elements: List[LayoutElement], high_quality: bool) -> List[MapResult]:
        """Detect maps in the image."""
        try:
            maps = []
            
            # Simple map detection (in production, use specialized map detection models)
            # For now, we'll identify potential maps based on size and shape
            
            for element in layout_elements:
                try:
                    # Large rectangular elements might be maps
                    if element.area > 15000 and element.cls in ["figure", "table"]:
                        map_result = MapResult(
                            bbox=element.bbox,
                            summary="Map or geographical visualization",
                            confidence=element.score * 0.8,  # Lower confidence for maps
                            map_type="geographical"
                        )
                        
                        maps.append(map_result)
                        
                except Exception as e:
                    logger.warning(f"Map detection failed for element {element.id}: {e}")
                    continue
            
            logger.info(f"Detected {len(maps)} maps")
            return maps
            
        except Exception as e:
            logger.error(f"Map detection failed: {e}")
            return []
    
    def _get_preprocessing_info(self, image: np.ndarray) -> PreprocessingInfo:
        """Get preprocessing information including deskew angle."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Calculate deskew angle using Hough transform
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            angles = []
            if lines is not None:
                for rho, theta in lines[:10]:  # Use first 10 lines
                    angle = theta * 180 / np.pi
                    if angle < 45:
                        angles.append(angle)
                    elif angle > 135:
                        angles.append(angle - 180)
            
            # Calculate average deskew angle
            deskew_angle = np.mean(angles) if angles else 0.0
            
            # Calculate image quality score
            quality_score = self._calculate_image_quality(gray)
            
            return PreprocessingInfo(
                deskew_angle=float(deskew_angle),
                resolution=(image.shape[1], image.shape[0]),
                quality_score=float(quality_score)
            )
            
        except Exception as e:
            logger.error(f"Preprocessing info calculation failed: {e}")
            return PreprocessingInfo(
                deskew_angle=0.0,
                resolution=None,
                quality_score=0.5
            )
    
    def _calculate_image_quality(self, gray_image: np.ndarray) -> float:
        """Calculate image quality score based on sharpness and contrast."""
        try:
            # Calculate Laplacian variance (sharpness)
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Calculate contrast
            contrast = gray_image.std()
            
            # Normalize scores
            sharpness_score = min(sharpness / 1000, 1.0)
            contrast_score = min(contrast / 100, 1.0)
            
            # Combine scores
            quality_score = (sharpness_score + contrast_score) / 2
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Image quality calculation failed: {e}")
            return 0.5
