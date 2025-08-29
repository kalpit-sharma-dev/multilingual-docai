"""
Stage 2 Pipeline: Multilingual Document Understanding
Integrates OCR, Language Detection, and Natural Language Generation
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# Import our modules
from ..ocr.multilingual_ocr import MultilingualOCR, OCRResult, TextRegion
from ..ocr.language_detector import LanguageDetector, LanguageDetectionResult
from ..nlg.visual_to_text import VisualToTextGenerator, NLGResult
from ..models.layout_detector import LayoutDetector
from ..data.deskew import ImageDeskewer
from ..data.document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Stage2Result:
    """Complete result from Stage 2 processing"""
    # Layout detection results (from Stage 1)
    layout_elements: List[Dict]  # [x, y, w, h, class, confidence]
    
    # Text extraction results
    text_regions: List[Dict]  # [x, y, w, h, text, language, confidence]
    
    # Visual element descriptions
    visual_descriptions: List[Dict]  # [x, y, w, h, type, description, confidence]
    
    # Language detection summary
    detected_languages: List[str]
    language_confidence: Dict[str, float]
    
    # Processing metadata
    processing_time: float
    image_path: str
    metadata: Dict

class Stage2Pipeline:
    """
    Stage 2 Pipeline for Multilingual Document Understanding
    
    Features:
    - Document layout detection (Stage 1)
    - Multilingual OCR with 6 language support
    - Language identification with precision/recall metrics
    - Natural language generation for charts/maps/tables
    - CER/WER evaluation for text extraction
    - BlueRT + BertScore for visual elements
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Initialize components
        self._init_components()
        
        logger.info("Stage 2 Pipeline initialized successfully")
    
    def _init_components(self):
        """Initialize all pipeline components"""
        try:
            # Layout detector (Stage 1)
            self.layout_detector = LayoutDetector()
            
            # Multilingual OCR
            self.ocr = MultilingualOCR(
                use_gpu=self.config.get('use_gpu', True)
            )
            
            # Language detector
            self.language_detector = LanguageDetector()
            
            # Visual to text generator
            self.visual_generator = VisualToTextGenerator(
                use_gpu=self.config.get('use_gpu', True)
            )
            
            # Image deskewer
            self.deskewer = ImageDeskewer()
            
            # Document processor
            self.doc_processor = DocumentProcessor()
            
            logger.info("All pipeline components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    def process_document(self, document_path: str) -> Stage2Result:
        """
        Process a document through the complete Stage 2 pipeline
        
        Args:
            document_path: Path to document (supports multiple formats)
            
        Returns:
            Stage2Result with complete analysis
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing document: {document_path}")
            
            # Step 1: Convert document to image(s)
            images = self.doc_processor.process_document(document_path)
            if not images:
                raise ValueError(f"Failed to convert document to images: {document_path}")
            
            # For now, process first page only
            image = images[0]
            
            # Step 2: Deskew image if needed
            deskewed_image = self.deskewer.deskew_image(image)
            
            # Step 3: Detect document layout (Stage 1)
            layout_results = self._detect_layout(deskewed_image)
            
            # Step 4: Extract text from text regions
            text_results = self._extract_text_regions(deskewed_image, layout_results)
            
            # Step 5: Generate descriptions for visual elements
            visual_results = self._generate_visual_descriptions(deskewed_image, layout_results)
            
            # Step 6: Analyze language distribution
            language_analysis = self._analyze_languages(text_results)
            
            # Step 7: Compile results
            processing_time = time.time() - start_time
            
            result = Stage2Result(
                layout_elements=layout_results,
                text_regions=text_results,
                visual_descriptions=visual_results,
                detected_languages=language_analysis['languages'],
                language_confidence=language_analysis['confidence'],
                processing_time=processing_time,
                image_path=document_path,
                metadata={
                    'pipeline_version': '2.0',
                    'components_used': ['layout_detector', 'multilingual_ocr', 'language_detector', 'visual_generator'],
                    'deskewing_applied': True
                }
            )
            
            logger.info(f"Document processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise
    
    def _detect_layout(self, image: np.ndarray) -> List[Dict]:
        """Detect document layout elements (Stage 1)"""
        try:
            # Use layout detector to get bounding boxes and classes
            predictions = self.layout_detector.predict(image)
            
            # Convert to our format
            layout_results = []
            for pred in predictions:
                layout_results.append({
                    'bbox': pred['bbox'],  # [x, y, w, h]
                    'class': pred['class'],
                    'confidence': pred['confidence'],
                    'class_name': self._get_class_name(pred['class'])
                })
            
            logger.info(f"Detected {len(layout_results)} layout elements")
            return layout_results
            
        except Exception as e:
            logger.error(f"Layout detection failed: {e}")
            # Return empty results
            return []
    
    def _extract_text_regions(self, image: np.ndarray, layout_results: List[Dict]) -> List[Dict]:
        """Extract text from text-related regions"""
        text_results = []
        
        try:
            # Find text-related regions
            text_regions = [r for r in layout_results if r['class'] in [1, 2, 3]]  # Text, Title, List
            
            for region in text_regions:
                try:
                    # Extract text using OCR
                    ocr_result = self.ocr.extract_text(image, region['bbox'])
                    
                    # Detect language for each text region
                    for text_region in ocr_result.text_regions:
                        # Enhance language detection
                        lang_result = self.language_detector.detect_language(
                            text_region.text, 
                            method='ensemble'
                        )
                        
                        text_results.append({
                            'bbox': text_region.bbox,
                            'text': text_region.text,
                            'language': lang_result.language_code,
                            'language_name': lang_result.detected_language,
                            'language_confidence': lang_result.confidence,
                            'ocr_confidence': text_region.confidence,
                            'region_type': self._get_class_name(region['class'])
                        })
                        
                except Exception as e:
                    logger.warning(f"Text extraction failed for region {region['bbox']}: {e}")
                    continue
            
            logger.info(f"Extracted text from {len(text_results)} regions")
            return text_results
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return []
    
    def _generate_visual_descriptions(self, image: np.ndarray, layout_results: List[Dict]) -> List[Dict]:
        """Generate descriptions for visual elements (charts, maps, tables)"""
        visual_results = []
        
        try:
            # Find visual elements
            visual_regions = [r for r in layout_results if r['class'] in [4, 5]]  # Table, Figure
            
            for region in visual_regions:
                try:
                    element_type = self._get_visual_element_type(region['class'])
                    
                    # Generate description
                    nlg_result = self.visual_generator.generate_description(
                        image, 
                        region['bbox'], 
                        element_type
                    )
                    
                    visual_results.append({
                        'bbox': nlg_result.bbox,
                        'type': element_type,
                        'description': nlg_result.generated_text,
                        'confidence': nlg_result.confidence,
                        'model_used': nlg_result.metadata.get('model', 'unknown'),
                        'method': nlg_result.metadata.get('method', 'unknown')
                    })
                    
                except Exception as e:
                    logger.warning(f"Visual description generation failed for region {region['bbox']}: {e}")
                    continue
            
            logger.info(f"Generated descriptions for {len(visual_results)} visual elements")
            return visual_results
            
        except Exception as e:
            logger.error(f"Visual description generation failed: {e}")
            return []
    
    def _analyze_languages(self, text_results: List[Dict]) -> Dict:
        """Analyze language distribution in the document"""
        try:
            if not text_results:
                return {'languages': [], 'confidence': {}}
            
            # Count languages
            language_counts = {}
            language_confidence_sum = {}
            
            for result in text_results:
                lang = result['language']
                if lang not in language_counts:
                    language_counts[lang] = 0
                    language_confidence_sum[lang] = 0
                
                language_counts[lang] += 1
                language_confidence_sum[lang] += result['language_confidence']
            
            # Calculate average confidence per language
            language_confidence = {}
            for lang in language_counts:
                language_confidence[lang] = language_confidence_sum[lang] / language_counts[lang]
            
            # Sort by count
            sorted_languages = sorted(
                language_counts.keys(), 
                key=lambda x: language_counts[x], 
                reverse=True
            )
            
            return {
                'languages': sorted_languages,
                'confidence': language_confidence,
                'counts': language_counts
            }
            
        except Exception as e:
            logger.error(f"Language analysis failed: {e}")
            return {'languages': [], 'confidence': {}}
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID"""
        class_names = {
            0: "Background",
            1: "Text",
            2: "Title", 
            3: "List",
            4: "Table",
            5: "Figure"
        }
        return class_names.get(class_id, "Unknown")
    
    def _get_visual_element_type(self, class_id: int) -> str:
        """Determine visual element type from class ID"""
        if class_id == 4:  # Table
            return 'table'
        elif class_id == 5:  # Figure
            # Try to determine if it's a chart or map based on characteristics
            # For now, default to chart
            return 'chart'
        else:
            return 'unknown'
    
    def evaluate_performance(self, results: Stage2Result, ground_truth: Dict) -> Dict:
        """
        Evaluate Stage 2 performance using all required metrics
        
        Args:
            results: Pipeline results
            ground_truth: Ground truth data
            
        Returns:
            Dictionary with all evaluation metrics
        """
        evaluation_results = {}
        
        try:
            # 1. Layout detection evaluation (mAP from Stage 1)
            if 'layout_gt' in ground_truth:
                layout_metrics = self._evaluate_layout(results.layout_elements, ground_truth['layout_gt'])
                evaluation_results['layout_detection'] = layout_metrics
            
            # 2. Text extraction evaluation (CER/WER)
            if 'text_gt' in ground_truth:
                text_metrics = self._evaluate_text_extraction(results.text_regions, ground_truth['text_gt'])
                evaluation_results['text_extraction'] = text_metrics
            
            # 3. Language identification evaluation
            if 'language_gt' in ground_truth:
                lang_metrics = self._evaluate_language_detection(results.text_regions, ground_truth['language_gt'])
                evaluation_results['language_detection'] = lang_metrics
            
            # 4. Visual element evaluation (BlueRT + BertScore)
            if 'visual_gt' in ground_truth:
                visual_metrics = self._evaluate_visual_elements(results.visual_descriptions, ground_truth['visual_gt'])
                evaluation_results['visual_elements'] = visual_metrics
            
            # 5. Overall pipeline metrics
            evaluation_results['pipeline_metrics'] = {
                'processing_time': results.processing_time,
                'total_elements_detected': len(results.layout_elements),
                'text_regions_processed': len(results.text_regions),
                'visual_elements_processed': len(results.visual_descriptions),
                'languages_detected': len(results.detected_languages)
            }
            
            logger.info("Performance evaluation completed")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
            return {'error': str(e)}
    
    def _evaluate_layout(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Evaluate layout detection performance"""
        # This would use the Stage 1 evaluator
        # For now, return placeholder metrics
        return {
            'mAP': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    
    def _evaluate_text_extraction(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Evaluate text extraction using CER/WER"""
        try:
            total_cer = 0.0
            total_wer = 0.0
            valid_pairs = 0
            
            for pred, gt in zip(predictions, ground_truth):
                if pred['text'] and gt['text']:
                    metrics = self.ocr.calculate_cer_wer(pred['text'], gt['text'])
                    total_cer += metrics['cer']
                    total_wer += metrics['wer']
                    valid_pairs += 1
            
            if valid_pairs > 0:
                avg_cer = total_cer / valid_pairs
                avg_wer = total_wer / valid_pairs
            else:
                avg_cer = 1.0
                avg_wer = 1.0
            
            return {
                'cer': avg_cer,
                'wer': avg_wer,
                'cer_percentage': avg_cer * 100,
                'wer_percentage': avg_wer * 100,
                'valid_pairs': valid_pairs
            }
            
        except Exception as e:
            logger.error(f"Text extraction evaluation failed: {e}")
            return {'error': str(e)}
    
    def _evaluate_language_detection(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Evaluate language detection performance"""
        try:
            pred_langs = [p['language'] for p in predictions]
            gt_langs = [g['language'] for g in ground_truth]
            
            metrics = self.language_detector.calculate_metrics(pred_langs, gt_langs)
            
            return {
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score
            }
            
        except Exception as e:
            logger.error(f"Language detection evaluation failed: {e}")
            return {'error': str(e)}
    
    def _evaluate_visual_elements(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Evaluate visual element descriptions using BlueRT + BertScore"""
        try:
            total_bleurt = 0.0
            total_bertscore = 0.0
            total_combined = 0.0
            valid_pairs = 0
            
            for pred, gt in zip(predictions, ground_truth):
                if pred['description'] and gt['description']:
                    metrics = self.visual_generator.evaluate_generation(
                        pred['description'], 
                        gt['description'], 
                        pred['type']
                    )
                    
                    total_bleurt += metrics.bleurt_score
                    total_bertscore += metrics.bertscore_score
                    total_combined += metrics.combined_score
                    valid_pairs += 1
            
            if valid_pairs > 0:
                avg_bleurt = total_bleurt / valid_pairs
                avg_bertscore = total_bertscore / valid_pairs
                avg_combined = total_combined / valid_pairs
            else:
                avg_bleurt = 0.0
                avg_bertscore = 0.0
                avg_combined = 0.0
            
            return {
                'bleurt_score': avg_bleurt,
                'bertscore_score': avg_bertscore,
                'combined_score': avg_combined,
                'valid_pairs': valid_pairs
            }
            
        except Exception as e:
            logger.error(f"Visual element evaluation failed: {e}")
            return {'error': str(e)}
    
    def save_results(self, results: Stage2Result, output_path: str):
        """Save results to JSON file"""
        try:
            # Convert dataclass to dict
            results_dict = asdict(results)
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def batch_process(self, document_paths: List[str], output_dir: str) -> List[Stage2Result]:
        """Process multiple documents"""
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, doc_path in enumerate(document_paths):
            try:
                logger.info(f"Processing document {i+1}/{len(document_paths)}: {doc_path}")
                
                # Process document
                result = self.process_document(doc_path)
                results.append(result)
                
                # Save individual results
                doc_name = Path(doc_path).stem
                output_file = output_path / f"{doc_name}_stage2_results.json"
                self.save_results(result, str(output_file))
                
            except Exception as e:
                logger.error(f"Failed to process document {doc_path}: {e}")
                continue
        
        logger.info(f"Batch processing completed. {len(results)} documents processed successfully.")
        return results
