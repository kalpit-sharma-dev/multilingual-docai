"""
Stage 3 Pipeline: Advanced Natural Language Generation
Extends Stage 2 with enhanced NLG capabilities for complex document understanding
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
from .stage2_pipeline import Stage2Pipeline, Stage2Result
from ..nlg.visual_to_text import VisualToTextGenerator, NLGResult
from ..models.layout_detector import LayoutDetector
from ..data.deskew import ImageDeskewer
from ..data.document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Stage3Result:
    """Complete result from Stage 3 processing"""
    # Inherit from Stage 2
    stage2_result: Stage2Result
    
    # Enhanced NLG results
    enhanced_descriptions: List[Dict]  # Enhanced descriptions with context
    semantic_analysis: Dict  # Semantic understanding of document
    cross_references: List[Dict]  # Cross-references between elements
    summary_generation: Dict  # Document summary and key insights
    
    # Advanced processing metadata
    processing_time: float
    enhancement_applied: List[str]
    metadata: Dict

class Stage3Pipeline(Stage2Pipeline):
    """
    Stage 3 Pipeline: Advanced Natural Language Generation
    
    Extends Stage 2 with:
    - Enhanced visual element descriptions with context
    - Semantic analysis and cross-referencing
    - Document summarization and key insight extraction
    - Advanced language understanding for complex elements
    - Multi-modal content integration
    """
    
    def __init__(self, config: Optional[Dict] = None):
        # Initialize Stage 2 components
        super().__init__(config)
        
        # Initialize Stage 3 specific components
        self._init_stage3_components()
        
        logger.info("Stage 3 Pipeline initialized successfully")
    
    def _init_stage3_components(self):
        """Initialize Stage 3 specific components"""
        try:
            # Enhanced visual generator with context awareness
            self.enhanced_visual_generator = self._create_enhanced_visual_generator()
            
            # Semantic analyzer
            self.semantic_analyzer = self._create_semantic_analyzer()
            
            # Cross-reference detector
            self.cross_reference_detector = self._create_cross_reference_detector()
            
            # Summary generator
            self.summary_generator = self._create_summary_generator()
            
            logger.info("Stage 3 components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Stage 3 components: {e}")
            raise
    
    def _create_enhanced_visual_generator(self):
        """Create enhanced visual generator with context awareness"""
        # For now, return the existing visual generator
        # In a full implementation, this would be enhanced with context models
        return self.visual_generator
    
    def _create_semantic_analyzer(self):
        """Create semantic analyzer for document understanding"""
        # Placeholder for semantic analysis component
        class SemanticAnalyzer:
            def analyze_document(self, text_regions, visual_elements, layout_elements):
                return {
                    'document_type': 'unknown',
                    'main_topics': [],
                    'semantic_structure': {},
                    'key_entities': [],
                    'sentiment': 'neutral'
                }
        
        return SemanticAnalyzer()
    
    def _create_cross_reference_detector(self):
        """Create cross-reference detector"""
        # Placeholder for cross-reference detection
        class CrossReferenceDetector:
            def detect_references(self, text_regions, visual_elements):
                return []
        
        return CrossReferenceDetector()
    
    def _create_summary_generator(self):
        """Create summary generator"""
        # Placeholder for summary generation
        class SummaryGenerator:
            def generate_summary(self, text_regions, visual_elements, semantic_analysis):
                return {
                    'executive_summary': 'Document summary not available',
                    'key_points': [],
                    'insights': [],
                    'recommendations': []
                }
        
        return SummaryGenerator()
    
    def process_document(self, document_path: str) -> Stage3Result:
        """
        Process a document through the complete Stage 3 pipeline
        
        Args:
            document_path: Path to document (supports multiple formats)
            
        Returns:
            Stage3Result with complete analysis including enhanced NLG
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing document with Stage 3 pipeline: {document_path}")
            
            # Step 1: Run Stage 2 processing
            stage2_result = super().process_document(document_path)
            
            # Step 2: Enhance visual descriptions with context
            enhanced_descriptions = self._enhance_visual_descriptions(
                stage2_result, 
                document_path
            )
            
            # Step 3: Perform semantic analysis
            semantic_analysis = self._perform_semantic_analysis(stage2_result)
            
            # Step 4: Detect cross-references
            cross_references = self._detect_cross_references(stage2_result)
            
            # Step 5: Generate document summary
            summary = self._generate_document_summary(
                stage2_result, 
                semantic_analysis
            )
            
            # Step 6: Compile Stage 3 results
            processing_time = time.time() - start_time
            
            result = Stage3Result(
                stage2_result=stage2_result,
                enhanced_descriptions=enhanced_descriptions,
                semantic_analysis=semantic_analysis,
                cross_references=cross_references,
                summary_generation=summary,
                processing_time=processing_time,
                enhancement_applied=[
                    'enhanced_visual_descriptions',
                    'semantic_analysis',
                    'cross_reference_detection',
                    'summary_generation'
                ],
                metadata={
                    'pipeline_version': '3.0',
                    'stage2_integrated': True,
                    'enhancements': 'advanced_nlg'
                }
            )
            
            logger.info(f"Stage 3 document processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Stage 3 document processing failed: {e}")
            raise
    
    def _enhance_visual_descriptions(self, stage2_result: Stage2Result, document_path: str) -> List[Dict]:
        """Enhance visual descriptions with context and better understanding"""
        enhanced_descriptions = []
        
        try:
            for visual_element in stage2_result.visual_descriptions:
                try:
                    # Get context from surrounding text regions
                    context = self._get_visual_element_context(
                        visual_element, 
                        stage2_result.text_regions
                    )
                    
                    # Enhance description based on context
                    enhanced_description = self._create_contextual_description(
                        visual_element, 
                        context
                    )
                    
                    enhanced_descriptions.append({
                        'original': visual_element,
                        'enhanced': enhanced_description,
                        'context': context,
                        'enhancement_method': 'contextual_analysis'
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to enhance description for {visual_element['type']}: {e}")
                    # Keep original description
                    enhanced_descriptions.append({
                        'original': visual_element,
                        'enhanced': visual_element,
                        'context': {},
                        'enhancement_method': 'none'
                    })
            
            logger.info(f"Enhanced {len(enhanced_descriptions)} visual descriptions")
            return enhanced_descriptions
            
        except Exception as e:
            logger.error(f"Visual description enhancement failed: {e}")
            return []
    
    def _get_visual_element_context(self, visual_element: Dict, text_regions: List[Dict]) -> Dict:
        """Get contextual information for a visual element"""
        context = {
            'nearby_text': [],
            'related_titles': [],
            'captions': [],
            'references': []
        }
        
        try:
            visual_bbox = visual_element['bbox']
            visual_center_x = visual_bbox[0] + visual_bbox[2] / 2
            visual_center_y = visual_bbox[1] + visual_bbox[3] / 2
            
            for text_region in text_regions:
                text_bbox = text_region['bbox']
                text_center_x = text_bbox[0] + text_bbox[2] / 2
                text_center_y = text_bbox[1] + text_bbox[3] / 2
                
                # Calculate distance
                distance = np.sqrt(
                    (visual_center_x - text_center_x) ** 2 + 
                    (visual_center_y - text_center_y) ** 2
                )
                
                # Consider text regions within reasonable distance
                if distance < 200:  # Threshold distance
                    if text_region['region_type'] == 'Title':
                        context['related_titles'].append({
                            'text': text_region['text'],
                            'distance': distance,
                            'language': text_region['language']
                        })
                    elif 'caption' in text_region['text'].lower() or 'figure' in text_region['text'].lower():
                        context['captions'].append({
                            'text': text_region['text'],
                            'distance': distance,
                            'language': text_region['language']
                        })
                    else:
                        context['nearby_text'].append({
                            'text': text_region['text'],
                            'distance': distance,
                            'language': text_region['language']
                        })
            
            # Sort by distance
            for key in ['related_titles', 'captions', 'nearby_text']:
                context[key].sort(key=lambda x: x['distance'])
            
        except Exception as e:
            logger.warning(f"Context extraction failed: {e}")
        
        return context
    
    def _create_contextual_description(self, visual_element: Dict, context: Dict) -> Dict:
        """Create enhanced description using context"""
        enhanced = visual_element.copy()
        
        try:
            # Enhance based on element type
            if visual_element['type'] == 'table':
                enhanced['description'] = self._enhance_table_description_with_context(
                    visual_element, context
                )
            elif visual_element['type'] == 'chart':
                enhanced['description'] = self._enhance_chart_description_with_context(
                    visual_element, context
                )
            elif visual_element['type'] == 'map':
                enhanced['description'] = self._enhance_map_description_with_context(
                    visual_element, context
                )
            
            # Add context information
            enhanced['contextual_info'] = {
                'has_caption': len(context['captions']) > 0,
                'has_title': len(context['related_titles']) > 0,
                'nearby_text_count': len(context['nearby_text']),
                'primary_language': self._get_primary_language(context)
            }
            
        except Exception as e:
            logger.warning(f"Contextual description creation failed: {e}")
        
        return enhanced
    
    def _enhance_table_description_with_context(self, table_element: Dict, context: Dict) -> str:
        """Enhance table description using context"""
        base_description = table_element['description']
        enhanced = base_description
        
        try:
            # Add title context
            if context['related_titles']:
                title = context['related_titles'][0]['text']
                enhanced = f"Table titled '{title}': {base_description}"
            
            # Add caption context
            if context['captions']:
                caption = context['captions'][0]['text']
                enhanced += f" Caption: {caption}"
            
            # Add nearby text context
            if context['nearby_text']:
                nearby_text = context['nearby_text'][0]['text']
                if len(nearby_text) < 100:  # Only add if not too long
                    enhanced += f" Related text: {nearby_text}"
            
        except Exception as e:
            logger.warning(f"Table enhancement failed: {e}")
        
        return enhanced
    
    def _enhance_chart_description_with_context(self, chart_element: Dict, context: Dict) -> str:
        """Enhance chart description using context"""
        base_description = chart_element['description']
        enhanced = base_description
        
        try:
            # Add title context
            if context['related_titles']:
                title = context['related_titles'][0]['text']
                enhanced = f"Chart titled '{title}': {base_description}"
            
            # Add caption context
            if context['captions']:
                caption = context['captions'][0]['text']
                enhanced += f" Caption: {caption}"
            
            # Enhance with chart-specific language
            if 'data' in base_description.lower() or 'trend' in base_description.lower():
                enhanced += " The chart provides visual representation of the data for easy interpretation."
            
        except Exception as e:
            logger.warning(f"Chart enhancement failed: {e}")
        
        return enhanced
    
    def _enhance_map_description_with_context(self, map_element: Dict, context: Dict) -> str:
        """Enhance map description using context"""
        base_description = map_element['description']
        enhanced = base_description
        
        try:
            # Add title context
            if context['related_titles']:
                title = context['related_titles'][0]['text']
                enhanced = f"Map titled '{title}': {base_description}"
            
            # Add caption context
            if context['captions']:
                caption = context['captions'][0]['text']
                enhanced += f" Caption: {caption}"
            
            # Enhance with map-specific language
            if 'geographic' in base_description.lower() or 'location' in base_description.lower():
                enhanced += " The map provides spatial context and geographic information."
            
        except Exception as e:
            logger.warning(f"Map enhancement failed: {e}")
        
        return enhanced
    
    def _get_primary_language(self, context: Dict) -> str:
        """Get primary language from context"""
        all_languages = []
        
        for key in ['related_titles', 'captions', 'nearby_text']:
            for item in context[key]:
                if 'language' in item:
                    all_languages.append(item['language'])
        
        if all_languages:
            # Return most common language
            from collections import Counter
            return Counter(all_languages).most_common(1)[0][0]
        
        return 'en'  # Default to English
    
    def _perform_semantic_analysis(self, stage2_result: Stage2Result) -> Dict:
        """Perform semantic analysis of the document"""
        try:
            semantic_analysis = self.semantic_analyzer.analyze_document(
                stage2_result.text_regions,
                stage2_result.visual_descriptions,
                stage2_result.layout_elements
            )
            
            # Enhance with additional analysis
            semantic_analysis.update({
                'document_structure': self._analyze_document_structure(stage2_result),
                'content_complexity': self._assess_content_complexity(stage2_result),
                'multilingual_analysis': self._analyze_multilingual_aspects(stage2_result)
            })
            
            logger.info("Semantic analysis completed")
            return semantic_analysis
            
        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_document_structure(self, stage2_result: Stage2Result) -> Dict:
        """Analyze document structure and organization"""
        try:
            # Count different element types
            element_counts = {}
            for element in stage2_result.layout_elements:
                class_name = element['class_name']
                element_counts[class_name] = element_counts.get(class_name, 0) + 1
            
            # Analyze spatial distribution
            spatial_analysis = {
                'has_header': any(e['class_name'] == 'Title' and e['bbox'][1] < 100 for e in stage2_result.layout_elements),
                'has_footer': any(e['bbox'][1] > 500 for e in stage2_result.layout_elements),  # Assuming page height > 600
                'element_distribution': element_counts
            }
            
            return spatial_analysis
            
        except Exception as e:
            logger.warning(f"Document structure analysis failed: {e}")
            return {}
    
    def _assess_content_complexity(self, stage2_result: Stage2Result) -> Dict:
        """Assess content complexity"""
        try:
            # Analyze text complexity
            total_text_length = sum(len(tr['text']) for tr in stage2_result.text_regions)
            avg_text_length = total_text_length / len(stage2_result.text_regions) if stage2_result.text_regions else 0
            
            # Analyze visual complexity
            visual_complexity = len(stage2_result.visual_descriptions)
            
            # Language complexity
            language_complexity = len(stage2_result.detected_languages)
            
            return {
                'text_complexity': {
                    'total_length': total_text_length,
                    'average_length': avg_text_length,
                    'text_regions': len(stage2_result.text_regions)
                },
                'visual_complexity': {
                    'visual_elements': visual_complexity,
                    'has_tables': any(v['type'] == 'table' for v in stage2_result.visual_descriptions),
                    'has_charts': any(v['type'] == 'chart' for v in stage2_result.visual_descriptions)
                },
                'language_complexity': {
                    'languages_detected': language_complexity,
                    'is_multilingual': language_complexity > 1
                }
            }
            
        except Exception as e:
            logger.warning(f"Content complexity assessment failed: {e}")
            return {}
    
    def _analyze_multilingual_aspects(self, stage2_result: Stage2Result) -> Dict:
        """Analyze multilingual aspects of the document"""
        try:
            language_analysis = {
                'primary_language': stage2_result.detected_languages[0] if stage2_result.detected_languages else 'unknown',
                'language_distribution': stage2_result.language_confidence,
                'mixed_script_detected': len(stage2_result.detected_languages) > 1,
                'script_types': self._categorize_scripts(stage2_result.detected_languages)
            }
            
            return language_analysis
            
        except Exception as e:
            logger.warning(f"Multilingual analysis failed: {e}")
            return {}
    
    def _categorize_scripts(self, languages: List[str]) -> List[str]:
        """Categorize languages by script type"""
        script_categories = {
            'latin': ['en'],
            'devanagari': ['hi', 'ne'],
            'arabic': ['ur', 'ar', 'fa']
        }
        
        detected_scripts = set()
        for lang in languages:
            for script, lang_list in script_categories.items():
                if lang in lang_list:
                    detected_scripts.add(script)
        
        return list(detected_scripts)
    
    def _detect_cross_references(self, stage2_result: Stage2Result) -> List[Dict]:
        """Detect cross-references between document elements"""
        try:
            cross_references = self.cross_reference_detector.detect_references(
                stage2_result.text_regions,
                stage2_result.visual_descriptions
            )
            
            # Enhanced cross-reference detection
            enhanced_references = self._enhance_cross_references(
                cross_references, 
                stage2_result
            )
            
            logger.info(f"Detected {len(enhanced_references)} cross-references")
            return enhanced_references
            
        except Exception as e:
            logger.error(f"Cross-reference detection failed: {e}")
            return []
    
    def _enhance_cross_references(self, base_references: List[Dict], stage2_result: Stage2Result) -> List[Dict]:
        """Enhance cross-references with additional context"""
        enhanced_references = []
        
        try:
            for ref in base_references:
                enhanced_ref = ref.copy()
                
                # Add context information
                enhanced_ref['context'] = {
                    'source_type': 'unknown',
                    'target_type': 'unknown',
                    'reference_strength': 'weak'
                }
                
                enhanced_references.append(enhanced_ref)
            
        except Exception as e:
            logger.warning(f"Cross-reference enhancement failed: {e}")
        
        return enhanced_references
    
    def _generate_document_summary(self, stage2_result: Stage2Result, semantic_analysis: Dict) -> Dict:
        """Generate comprehensive document summary"""
        try:
            summary = self.summary_generator.generate_summary(
                stage2_result.text_regions,
                stage2_result.visual_descriptions,
                semantic_analysis
            )
            
            # Enhance summary with additional insights
            enhanced_summary = self._enhance_summary(summary, stage2_result, semantic_analysis)
            
            logger.info("Document summary generated")
            return enhanced_summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {'error': str(e)}
    
    def _enhance_summary(self, base_summary: Dict, stage2_result: Stage2Result, semantic_analysis: Dict) -> Dict:
        """Enhance summary with additional insights"""
        enhanced_summary = base_summary.copy()
        
        try:
            # Add document statistics
            enhanced_summary['document_statistics'] = {
                'total_elements': len(stage2_result.layout_elements),
                'text_regions': len(stage2_result.text_regions),
                'visual_elements': len(stage2_result.visual_descriptions),
                'languages_detected': len(stage2_result.detected_languages),
                'processing_time': stage2_result.processing_time
            }
            
            # Add key insights
            enhanced_summary['key_insights'] = self._extract_key_insights(
                stage2_result, 
                semantic_analysis
            )
            
            # Add recommendations
            enhanced_summary['recommendations'] = self._generate_recommendations(
                stage2_result, 
                semantic_analysis
            )
            
        except Exception as e:
            logger.warning(f"Summary enhancement failed: {e}")
        
        return enhanced_summary
    
    def _extract_key_insights(self, stage2_result: Stage2Result, semantic_analysis: Dict) -> List[str]:
        """Extract key insights from the document"""
        insights = []
        
        try:
            # Language insights
            if len(stage2_result.detected_languages) > 1:
                insights.append("Document contains multiple languages, indicating international or multicultural content")
            
            # Visual content insights
            if stage2_result.visual_descriptions:
                insights.append(f"Document includes {len(stage2_result.visual_descriptions)} visual elements for enhanced understanding")
            
            # Structure insights
            if semantic_analysis.get('document_structure', {}).get('has_header'):
                insights.append("Document has a clear header structure for better organization")
            
        except Exception as e:
            logger.warning(f"Key insights extraction failed: {e}")
        
        return insights
    
    def _generate_recommendations(self, stage2_result: Stage2Result, semantic_analysis: Dict) -> List[str]:
        """Generate recommendations based on document analysis"""
        recommendations = []
        
        try:
            # Language recommendations
            if len(stage2_result.detected_languages) > 1:
                recommendations.append("Consider providing translations for key content in multiple languages")
            
            # Visual content recommendations
            if not stage2_result.visual_descriptions:
                recommendations.append("Consider adding visual elements to improve document comprehension")
            
            # Structure recommendations
            if not semantic_analysis.get('document_structure', {}).get('has_header'):
                recommendations.append("Consider adding clear headers for better document organization")
            
        except Exception as e:
            logger.warning(f"Recommendations generation failed: {e}")
        
        return recommendations
    
    def evaluate_stage3_performance(self, results: Stage3Result, ground_truth: Dict) -> Dict:
        """Evaluate Stage 3 performance with enhanced metrics"""
        try:
            # Get Stage 2 evaluation
            stage2_eval = self.evaluate_performance(results.stage2_result, ground_truth)
            
            # Add Stage 3 specific evaluation
            stage3_eval = {
                'enhancement_quality': self._evaluate_enhancement_quality(results),
                'semantic_analysis_quality': self._evaluate_semantic_analysis(results),
                'cross_reference_accuracy': self._evaluate_cross_references(results),
                'summary_quality': self._evaluate_summary_quality(results)
            }
            
            # Combine evaluations
            complete_eval = {
                'stage2_metrics': stage2_eval,
                'stage3_metrics': stage3_eval,
                'overall_pipeline_score': self._calculate_overall_score(stage2_eval, stage3_eval)
            }
            
            return complete_eval
            
        except Exception as e:
            logger.error(f"Stage 3 performance evaluation failed: {e}")
            return {'error': str(e)}
    
    def _evaluate_enhancement_quality(self, results: Stage3Result) -> Dict:
        """Evaluate quality of visual description enhancements"""
        try:
            enhanced_count = sum(1 for d in results.enhanced_descriptions if d['enhancement_method'] != 'none')
            total_count = len(results.enhanced_descriptions)
            
            enhancement_ratio = enhanced_count / total_count if total_count > 0 else 0
            
            return {
                'enhancement_ratio': enhancement_ratio,
                'enhanced_descriptions': enhanced_count,
                'total_descriptions': total_count,
                'quality_score': enhancement_ratio * 100
            }
            
        except Exception as e:
            logger.warning(f"Enhancement quality evaluation failed: {e}")
            return {'error': str(e)}
    
    def _evaluate_semantic_analysis(self, results: Stage3Result) -> Dict:
        """Evaluate quality of semantic analysis"""
        try:
            semantic_analysis = results.semantic_analysis
            
            # Check if analysis was successful
            if 'error' in semantic_analysis:
                return {'quality_score': 0, 'error': semantic_analysis['error']}
            
            # Calculate quality based on completeness
            analysis_fields = ['document_type', 'main_topics', 'semantic_structure', 'key_entities']
            completed_fields = sum(1 for field in analysis_fields if field in semantic_analysis and semantic_analysis[field])
            
            quality_score = (completed_fields / len(analysis_fields)) * 100
            
            return {
                'quality_score': quality_score,
                'completed_fields': completed_fields,
                'total_fields': len(analysis_fields),
                'analysis_completeness': quality_score
            }
            
        except Exception as e:
            logger.warning(f"Semantic analysis evaluation failed: {e}")
            return {'error': str(e)}
    
    def _evaluate_cross_references(self, results: Stage3Result) -> Dict:
        """Evaluate accuracy of cross-reference detection"""
        try:
            cross_refs = results.cross_references
            
            # For now, return basic metrics
            # In a full implementation, this would compare with ground truth
            return {
                'detected_references': len(cross_refs),
                'quality_score': 50.0,  # Placeholder
                'evaluation_method': 'basic_count'
            }
            
        except Exception as e:
            logger.warning(f"Cross-reference evaluation failed: {e}")
            return {'error': str(e)}
    
    def _evaluate_summary_quality(self, results: Stage3Result) -> Dict:
        """Evaluate quality of generated summary"""
        try:
            summary = results.summary_generation
            
            # Check if summary was successful
            if 'error' in summary:
                return {'quality_score': 0, 'error': summary['error']}
            
            # Calculate quality based on completeness
            summary_fields = ['executive_summary', 'key_points', 'insights', 'recommendations']
            completed_fields = sum(1 for field in summary_fields if field in summary and summary[field])
            
            quality_score = (completed_fields / len(summary_fields)) * 100
            
            return {
                'quality_score': quality_score,
                'completed_fields': completed_fields,
                'total_fields': len(summary_fields),
                'summary_completeness': quality_score
            }
            
        except Exception as e:
            logger.warning(f"Summary quality evaluation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_score(self, stage2_eval: Dict, stage3_eval: Dict) -> float:
        """Calculate overall pipeline score"""
        try:
            # Weight Stage 2 and Stage 3 scores
            stage2_weight = 0.6
            stage3_weight = 0.4
            
            # Calculate Stage 2 score (average of available metrics)
            stage2_scores = []
            for metric_name, metric_data in stage2_eval.items():
                if isinstance(metric_data, dict) and 'mAP' in metric_data:
                    stage2_scores.append(metric_data['mAP'])
                elif isinstance(metric_data, dict) and 'accuracy' in metric_data:
                    stage2_scores.append(metric_data['accuracy'])
            
            stage2_avg = np.mean(stage2_scores) if stage2_scores else 0.0
            
            # Calculate Stage 3 score (average of quality scores)
            stage3_scores = []
            for metric_name, metric_data in stage3_eval.items():
                if isinstance(metric_data, dict) and 'quality_score' in metric_data:
                    stage3_scores.append(metric_data['quality_score'])
            
            stage3_avg = np.mean(stage3_scores) if stage3_scores else 0.0
            
            # Calculate weighted overall score
            overall_score = (stage2_avg * stage2_weight) + (stage3_avg * stage3_weight)
            
            return overall_score
            
        except Exception as e:
            logger.warning(f"Overall score calculation failed: {e}")
            return 0.0
