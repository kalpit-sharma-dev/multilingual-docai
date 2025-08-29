"""PS-05 Document Understanding Pipeline

Complete end-to-end inference flow for multilingual document understanding:
Stage 1: Layout detection (Background, Text, Title, List, Table, Figure)
Stage 2: OCR + Language identification
Stage 3: Natural language generation for tables, charts, maps, images
"""

import cv2
import json
import logging
import time
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np

# Import our models
from src.models.layout_detector import LayoutDetector
from src.models.ocr_engine import OCREngine
from src.models.langid_classifier import LanguageClassifier
from src.models.nl_generator import NLGenerator
from src.data.preprocess import deskew, preprocess_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PS05Pipeline:
    """PS-05 Document Understanding Pipeline."""
    
    def __init__(self, config_path: str = "configs/ps05_config.yaml"):
        """Initialize the PS-05 pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        
        # Initialize models
        self.layout_detector = LayoutDetector(config_path)
        
        # Load the trained model if available
        trained_model_path = "outputs/stage1_enhanced/training/layout_detector3/weights/best.pt"
        if Path(trained_model_path).exists():
            self.layout_detector.load_model(trained_model_path)
            logger.info(f"Loaded trained model from {trained_model_path}")
        
        self.ocr_engine = OCREngine(config_path)
        self.lang_classifier = LanguageClassifier(config_path)
        self.nl_generator = NLGenerator(config_path)
        
        logger.info("PS-05 Pipeline initialized successfully")
    
    def process_image(self, image_path: str, stage: int = 1) -> Dict:
        """Process a single document image.
        
        Args:
            image_path: Path to the input image
            stage: Processing stage (1: Layout, 2: +OCR, 3: +NL)
            
        Returns:
            Dictionary with complete document analysis results
        """
        start_time = time.time()
        
        try:
            # Load and preprocess image
            image = self._load_image(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Preprocessing
            image, deskew_angle = deskew(image)
            logger.info(f"Image deskewed by {deskew_angle:.2f} degrees")
            
            # Stage 1: Layout Detection
            layout_results = self._detect_layout(image)
            
            # Initialize result structure
            result = {
            "page": 1,
                "size": {"w": image.shape[1], "h": image.shape[0]},
                "elements": layout_results,
                "preprocess": {"deskew_angle": deskew_angle},
                "processing_time": time.time() - start_time
            }
            
            # Stage 2: OCR and Language Identification
            if stage >= 2:
                ocr_results = self._process_ocr(image, layout_results)
                result["text_lines"] = ocr_results
            
            # Stage 3: Natural Language Generation
            if stage >= 3:
                nl_results = self._generate_nl_descriptions(image, layout_results)
                result.update(nl_results)
            
            logger.info(f"Processing completed in {time.time() - start_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            return {
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _load_image(self, image_path: str):
        """Load image from path."""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def _detect_layout(self, image) -> List[Dict]:
        """Detect layout elements in the image."""
        try:
            detections = self.layout_detector.predict(image)
            
            # Convert to required format
            elements = []
            for i, detection in enumerate(detections):
                element = {
                    "id": f"e{i+1}",
                    "cls": detection.get("cls", "Unknown"),
                    "bbox": detection.get("bbox", []),
                    "score": detection.get("score", 0.0)
                }
                elements.append(element)
            
            logger.info(f"Detected {len(elements)} layout elements")
            return elements
            
        except Exception as e:
            logger.error(f"Error in layout detection: {e}")
            return []
    
    def _process_ocr(self, image, layout_elements: List[Dict]) -> List[Dict]:
        """Process OCR and language identification."""
        try:
            # Extract text regions from layout
            text_regions = [elem for elem in layout_elements if elem.get("cls") == "Text"]
            
            # Run OCR on text regions
            text_lines = self.ocr_engine.detect_text_lines(image, text_regions)
            
            # Process each text line
            processed_lines = []
            for i, line in enumerate(text_lines):
                # Detect language for the text
                lang_result = self.lang_classifier.classify_text(line.get("text", ""))
                
                processed_line = {
                    "id": f"line_{i+1}",
                    "bbox": line.get("bbox", []),
                    "text": line.get("text", ""),
                    "lang": lang_result.get("lang", "unknown"),
                    "score": line.get("score", 0.0),
                    "lang_confidence": lang_result.get("confidence", 0.0)
                }
                
                # Add direction for RTL languages
                if lang_result.get("lang") in ["ar", "ur", "fa"]:
                    processed_line["dir"] = "rtl"
                
                processed_lines.append(processed_line)
            
            logger.info(f"Processed {len(processed_lines)} text lines")
            return processed_lines
            
        except Exception as e:
            logger.error(f"Error in OCR processing: {e}")
            return []
    
    def _generate_nl_descriptions(self, image, layout_elements: List[Dict]) -> Dict:
        """Generate natural language descriptions for special elements."""
        try:
            nl_results = {
                "tables": [],
                "figures": [],
                "charts": [],
                "maps": []
            }
            
            for element in layout_elements:
                element_type = element.get("cls", "").lower()
                bbox = element.get("bbox", [])
                
                if not bbox or len(bbox) != 4:
                    continue
                
                # Extract element region
                x, y, w, h = bbox
                element_img = image[y:y+h, x:x+w]
                
                if element_img.size == 0:
                    continue
                
                if element_type == "table":
                    # Generate table summary
                    table_summary = self.nl_generator.generate_table_summary({}, element_img)
                    nl_results["tables"].append({
                        "bbox": bbox,
                        "summary": table_summary.get("summary", ""),
                        "confidence": table_summary.get("confidence", 0.0)
                    })
                
                elif element_type == "figure":
                    # Determine if it's a chart, map, or general image
                    chart_type = self._classify_figure_type(element_img)
                    
                    if chart_type == "chart":
                        chart_summary = self.nl_generator.generate_chart_summary({}, element_img)
                        nl_results["charts"].append({
                            "bbox": bbox,
                            "type": chart_summary.get("chart_type", "unknown"),
                            "summary": chart_summary.get("summary", ""),
                            "confidence": chart_summary.get("confidence", 0.0)
                        })
                    elif chart_type == "map":
                        map_summary = self.nl_generator.generate_map_summary({}, element_img)
                        nl_results["maps"].append({
                            "bbox": bbox,
                            "summary": map_summary.get("summary", ""),
                            "confidence": map_summary.get("confidence", 0.0)
                        })
                    else:
                        # General image
                        image_caption = self.nl_generator.generate_image_caption(element_img)
                        nl_results["figures"].append({
                            "bbox": bbox,
                            "summary": image_caption.get("caption", ""),
                            "confidence": image_caption.get("confidence", 0.0)
                        })
            
            logger.info(f"Generated NL descriptions: {len(nl_results['tables'])} tables, "
                       f"{len(nl_results['figures'])} figures, {len(nl_results['charts'])} charts, "
                       f"{len(nl_results['maps'])} maps")
            
            return nl_results
            
        except Exception as e:
            logger.error(f"Error in NL generation: {e}")
            return {"tables": [], "figures": [], "charts": [], "maps": []}
    
    def _classify_figure_type(self, image) -> str:
        """Classify figure type (chart, map, or general image)."""
        try:
            # Simple heuristics for figure classification
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Check for chart-like patterns
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / edges.size
            
            if edge_density > 0.02:  # High edge density suggests charts
                return "chart"
            
            # Check for map-like patterns (more complex analysis needed)
            # For now, use simple heuristics
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation_std = np.std(hsv[:, :, 1])
            
            if saturation_std > 30:  # High saturation variance might indicate maps
                return "map"
            
            return "image"
            
        except Exception as e:
            logger.error(f"Error classifying figure type: {e}")
            return "image"

def infer_page(image_path: str, config_path: str = "configs/ps05_config.yaml", stage: int = 1) -> Dict:
    """Convenience function for single page inference.
    
    Args:
        image_path: Path to the input image
        config_path: Path to configuration file
        stage: Processing stage (1: Layout, 2: +OCR, 3: +NL)
        
    Returns:
        Dictionary with document analysis results
    """
    pipeline = PS05Pipeline(config_path)
    return pipeline.process_image(image_path, stage)

def process_batch(image_paths: List[str], config_path: str = "configs/ps05_config.yaml", 
                 stage: int = 1, output_dir: str = "outputs") -> List[Dict]:
    """Process a batch of images.
    
    Args:
        image_paths: List of image paths
        config_path: Path to configuration file
        stage: Processing stage
        output_dir: Output directory for results
        
    Returns:
        List of processing results
    """
    pipeline = PS05Pipeline(config_path)
    results = []
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for i, image_path in enumerate(image_paths):
        logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        
        try:
            result = pipeline.process_image(image_path, stage)
            
            # Save individual result
            output_path = Path(output_dir) / f"result_{i+1}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            results.append({"error": str(e), "file": image_path})
    
    # Save batch summary
    summary_path = Path(output_dir) / "batch_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total_images": len(image_paths),
            "successful": len([r for r in results if "error" not in r]),
            "failed": len([r for r in results if "error" in r]),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PS-05 Document Understanding Pipeline")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--config", default="configs/ps05_config.yaml", help="Configuration file path")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3], help="Processing stage")
    parser.add_argument("--output", default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Process single image
    result = infer_page(args.image_path, args.config, args.stage)
    
    # Save result
    Path(args.output).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) / "result.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Processing completed. Result saved to {output_path}")
    print(f"Processing time: {result.get('processing_time', 0):.2f}s")
