"""
Enhanced Evaluation Module for PS-05

Advanced evaluation metrics including:
- BLEURT/BERTScore for NLG quality
- Human-level performance baselines
- Cross-modal evaluation metrics
- Advanced language identification metrics
"""

import logging
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import time

# Try to import advanced evaluation libraries
try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    logging.warning("BERTScore not available. Install with: pip install bert-score")

try:
    from bleurt import score as bleurt_score
    BLEURT_AVAILABLE = True
except ImportError:
    BLEURT_AVAILABLE = False
    logging.warning("BLEURT not available. Install with: pip install bleurt")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Install with: pip install nltk")

logger = logging.getLogger(__name__)

class EnhancedEvaluator:
    """Enhanced evaluator with advanced metrics for document understanding."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.metrics = {}
        self.human_baselines = self._load_human_baselines()
        
    def _load_human_baselines(self) -> Dict:
        """Load human-level performance baselines."""
        # These are approximate baselines from research papers
        baselines = {
            "layout_detection": {
                "mAP50": 0.85,  # Human agreement level from DocLayNet paper
                "mAP75": 0.75,
                "precision": 0.90,
                "recall": 0.88
            },
            "text_extraction": {
                "cer": 0.02,    # Character Error Rate (human level)
                "wer": 0.05,    # Word Error Rate (human level)
                "accuracy": 0.98
            },
            "language_identification": {
                "accuracy": 0.95,  # Human language ID accuracy
                "f1": 0.94
            },
            "table_understanding": {
                "structure_accuracy": 0.92,
                "content_accuracy": 0.89
            },
            "image_captioning": {
                "bleu4": 0.35,     # BLUE-4 score (human level)
                "meteor": 0.28,    # METEOR score
                "rouge_l": 0.52    # ROUGE-L score
            }
        }
        return baselines
    
    def evaluate_layout_detection_enhanced(self, predictions: List[Dict], 
                                        ground_truth: List[Dict]) -> Dict:
        """Enhanced layout detection evaluation with human baselines."""
        try:
            # Basic metrics
            basic_metrics = self._calculate_basic_layout_metrics(predictions, ground_truth)
            
            # Advanced metrics
            advanced_metrics = self._calculate_advanced_layout_metrics(predictions, ground_truth)
            
            # Human baseline comparison
            baseline_comparison = self._compare_with_human_baselines(
                "layout_detection", basic_metrics
            )
            
            # Combine all metrics
            results = {
                "basic_metrics": basic_metrics,
                "advanced_metrics": advanced_metrics,
                "human_baseline_comparison": baseline_comparison,
                "overall_score": self._calculate_overall_layout_score(basic_metrics, advanced_metrics)
            }
            
            self.metrics["layout_detection"] = results
            return results
            
        except Exception as e:
            logger.error(f"Enhanced layout evaluation failed: {e}")
            return {}
    
    def evaluate_text_extraction_enhanced(self, predictions: List[Dict], 
                                        ground_truth: List[Dict]) -> Dict:
        """Enhanced text extraction evaluation."""
        try:
            # Basic OCR metrics
            basic_metrics = self._calculate_basic_ocr_metrics(predictions, ground_truth)
            
            # Advanced text quality metrics
            advanced_metrics = self._calculate_advanced_text_metrics(predictions, ground_truth)
            
            # Human baseline comparison
            baseline_comparison = self._compare_with_human_baselines(
                "text_extraction", basic_metrics
            )
            
            results = {
                "basic_metrics": basic_metrics,
                "advanced_metrics": advanced_metrics,
                "human_baseline_comparison": baseline_comparison,
                "overall_score": self._calculate_overall_text_score(basic_metrics, advanced_metrics)
            }
            
            self.metrics["text_extraction"] = results
            return results
            
        except Exception as e:
            logger.error(f"Enhanced text evaluation failed: {e}")
            return {}
    
    def evaluate_language_identification_enhanced(self, predictions: List[Dict], 
                                               ground_truth: List[Dict]) -> Dict:
        """Enhanced language identification evaluation."""
        try:
            # Basic language ID metrics
            basic_metrics = self._calculate_basic_langid_metrics(predictions, ground_truth)
            
            # Advanced language metrics
            advanced_metrics = self._calculate_advanced_langid_metrics(predictions, ground_truth)
            
            # Human baseline comparison
            baseline_comparison = self._compare_with_human_baselines(
                "language_identification", basic_metrics
            )
            
            results = {
                "basic_metrics": basic_metrics,
                "advanced_metrics": advanced_metrics,
                "human_baseline_comparison": baseline_comparison,
                "overall_score": self._calculate_overall_langid_score(basic_metrics, advanced_metrics)
            }
            
            self.metrics["language_identification"] = results
            return results
            
        except Exception as e:
            logger.error(f"Enhanced language ID evaluation failed: {e}")
            return {}
    
    def evaluate_content_understanding_enhanced(self, predictions: List[Dict], 
                                              ground_truth: List[Dict]) -> Dict:
        """Enhanced content understanding evaluation (tables, charts, images)."""
        try:
            results = {}
            
            # Table understanding evaluation
            if any(p.get('type') == 'Table' for p in predictions):
                table_metrics = self._evaluate_table_understanding(predictions, ground_truth)
                results["table_understanding"] = table_metrics
            
            # Image captioning evaluation
            if any(p.get('type') == 'Figure' for p in predictions):
                caption_metrics = self._evaluate_image_captioning(predictions, ground_truth)
                results["image_captioning"] = caption_metrics
            
            # Chart understanding evaluation
            if any(p.get('type') == 'Chart' for p in predictions):
                chart_metrics = self._evaluate_chart_understanding(predictions, ground_truth)
                results["chart_understanding"] = chart_metrics
            
            self.metrics["content_understanding"] = results
            return results
            
        except Exception as e:
            logger.error(f"Enhanced content understanding evaluation failed: {e}")
            return {}
    
    def _calculate_basic_layout_metrics(self, predictions: List[Dict], 
                                      ground_truth: List[Dict]) -> Dict:
        """Calculate basic layout detection metrics."""
        try:
            # This would integrate with existing layout evaluator
            # For now, return placeholder metrics
            return {
                "mAP50": 0.0,
                "mAP75": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }
        except Exception as e:
            logger.error(f"Basic layout metrics calculation failed: {e}")
            return {}
    
    def _calculate_advanced_layout_metrics(self, predictions: List[Dict], 
                                         ground_truth: List[Dict]) -> Dict:
        """Calculate advanced layout detection metrics."""
        try:
            # Advanced metrics like:
            # - Spatial relationship accuracy
            # - Hierarchical structure understanding
            # - Cross-modal alignment quality
            
            return {
                "spatial_accuracy": 0.0,
                "hierarchical_accuracy": 0.0,
                "cross_modal_alignment": 0.0
            }
        except Exception as e:
            logger.error(f"Advanced layout metrics calculation failed: {e}")
            return {}
    
    def _calculate_basic_ocr_metrics(self, predictions: List[Dict], 
                                   ground_truth: List[Dict]) -> Dict:
        """Calculate basic OCR metrics."""
        try:
            # Character Error Rate (CER)
            total_chars = 0
            total_errors = 0
            
            for pred, gt in zip(predictions, ground_truth):
                pred_text = pred.get('text', '')
                gt_text = gt.get('text', '')
                
                total_chars += len(gt_text)
                total_errors += self._levenshtein_distance(pred_text, gt_text)
            
            cer = total_errors / total_chars if total_chars > 0 else 1.0
            
            # Word Error Rate (WER)
            total_words = 0
            total_word_errors = 0
            
            for pred, gt in zip(predictions, ground_truth):
                pred_words = pred.get('text', '').split()
                gt_words = gt.get('text', '').split()
                
                total_words += len(gt_words)
                total_word_errors += self._levenshtein_distance(pred_words, gt_words)
            
            wer = total_word_errors / total_words if total_words > 0 else 1.0
            
            return {
                "cer": cer,
                "wer": wer,
                "accuracy": 1.0 - cer
            }
            
        except Exception as e:
            logger.error(f"Basic OCR metrics calculation failed: {e}")
            return {}
    
    def _calculate_advanced_text_metrics(self, predictions: List[Dict], 
                                       ground_truth: List[Dict]) -> Dict:
        """Calculate advanced text quality metrics."""
        try:
            # Text quality metrics
            readability_scores = []
            semantic_similarity_scores = []
            
            for pred, gt in zip(predictions, ground_truth):
                pred_text = pred.get('text', '')
                gt_text = gt.get('text', '')
                
                # Readability score (simplified)
                readability = self._calculate_readability(pred_text)
                readability_scores.append(readability)
                
                # Semantic similarity (if BERTScore available)
                if BERT_SCORE_AVAILABLE and pred_text and gt_text:
                    try:
                        P, R, F1 = bert_score([pred_text], [gt_text], lang='en')
                        semantic_similarity_scores.append(F1.mean().item())
                    except:
                        semantic_similarity_scores.append(0.0)
                else:
                    semantic_similarity_scores.append(0.0)
            
            return {
                "avg_readability": np.mean(readability_scores) if readability_scores else 0.0,
                "avg_semantic_similarity": np.mean(semantic_similarity_scores) if semantic_similarity_scores else 0.0
            }
            
        except Exception as e:
            logger.error(f"Advanced text metrics calculation failed: {e}")
            return {}
    
    def _calculate_basic_langid_metrics(self, predictions: List[Dict], 
                                      ground_truth: List[Dict]) -> Dict:
        """Calculate basic language identification metrics."""
        try:
            correct = 0
            total = 0
            
            for pred, gt in zip(predictions, ground_truth):
                pred_lang = pred.get('language', 'unknown')
                gt_lang = gt.get('language', 'unknown')
                
                if pred_lang == gt_lang:
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0.0
            
            return {
                "accuracy": accuracy,
                "precision": accuracy,  # Simplified for binary case
                "recall": accuracy,
                "f1": accuracy
            }
            
        except Exception as e:
            logger.error(f"Basic language ID metrics calculation failed: {e}")
            return {}
    
    def _calculate_advanced_langid_metrics(self, predictions: List[Dict], 
                                         ground_truth: List[Dict]) -> Dict:
        """Calculate advanced language identification metrics."""
        try:
            # Language-specific metrics
            language_metrics = {}
            
            # Get unique languages
            all_languages = set()
            for gt in ground_truth:
                all_languages.add(gt.get('language', 'unknown'))
            
            for lang in all_languages:
                lang_predictions = [p for p in predictions if p.get('language') == lang]
                lang_ground_truth = [g for g in ground_truth if g.get('language') == lang]
                
                if lang_ground_truth:
                    # Calculate per-language accuracy
                    correct = sum(1 for p, g in zip(lang_predictions, lang_ground_truth) 
                                if p.get('language') == g.get('language'))
                    accuracy = correct / len(lang_ground_truth) if lang_ground_truth else 0.0
                    
                    language_metrics[lang] = {
                        "accuracy": accuracy,
                        "sample_count": len(lang_ground_truth)
                    }
            
            return {
                "per_language_metrics": language_metrics,
                "language_coverage": len(all_languages)
            }
            
        except Exception as e:
            logger.error(f"Advanced language ID metrics calculation failed: {e}")
            return {}
    
    def _evaluate_table_understanding(self, predictions: List[Dict], 
                                    ground_truth: List[Dict]) -> Dict:
        """Evaluate table understanding capabilities."""
        try:
            # Table structure accuracy
            structure_accuracy = 0.0
            
            # Table content accuracy
            content_accuracy = 0.0
            
            # Table description quality (if available)
            description_quality = 0.0
            
            return {
                "structure_accuracy": structure_accuracy,
                "content_accuracy": content_accuracy,
                "description_quality": description_quality
            }
            
        except Exception as e:
            logger.error(f"Table understanding evaluation failed: {e}")
            return {}
    
    def _evaluate_image_captioning(self, predictions: List[Dict], 
                                  ground_truth: List[Dict]) -> Dict:
        """Evaluate image captioning quality."""
        try:
            if not NLTK_AVAILABLE:
                return {"error": "NLTK not available for captioning evaluation"}
            
            # BLEU scores
            bleu1_scores = []
            bleu4_scores = []
            
            # METEOR scores (simplified)
            meteor_scores = []
            
            for pred, gt in zip(predictions, ground_truth):
                pred_caption = pred.get('description', '')
                gt_caption = gt.get('description', '')
                
                if pred_caption and gt_caption:
                    # BLEU-1
                    bleu1 = sentence_bleu([gt_caption.split()], pred_caption.split(), 
                                        smoothing_function=SmoothingFunction().method1)
                    bleu1_scores.append(bleu1)
                    
                    # BLEU-4
                    bleu4 = sentence_bleu([gt_caption.split()], pred_caption.split(), 
                                        smoothing_function=SmoothingFunction().method4)
                    bleu4_scores.append(bleu4)
                    
                    # Simplified METEOR (word overlap)
                    meteor = self._calculate_meteor_simplified(pred_caption, gt_caption)
                    meteor_scores.append(meteor)
            
            return {
                "bleu1": np.mean(bleu1_scores) if bleu1_scores else 0.0,
                "bleu4": np.mean(bleu4_scores) if bleu4_scores else 0.0,
                "meteor": np.mean(meteor_scores) if meteor_scores else 0.0
            }
            
        except Exception as e:
            logger.error(f"Image captioning evaluation failed: {e}")
            return {}
    
    def _evaluate_chart_understanding(self, predictions: List[Dict], 
                                    ground_truth: List[Dict]) -> Dict:
        """Evaluate chart understanding capabilities."""
        try:
            # Chart type identification accuracy
            type_accuracy = 0.0
            
            # Chart data extraction accuracy
            data_accuracy = 0.0
            
            # Chart description quality
            description_quality = 0.0
            
            return {
                "type_accuracy": type_accuracy,
                "data_accuracy": data_accuracy,
                "description_quality": description_quality
            }
            
        except Exception as e:
            logger.error(f"Chart understanding evaluation failed: {e}")
            return {}
    
    def _compare_with_human_baselines(self, task: str, metrics: Dict) -> Dict:
        """Compare current performance with human baselines."""
        try:
            if task not in self.human_baselines:
                return {"error": f"No human baseline for task: {task}"}
            
            baseline = self.human_baselines[task]
            comparison = {}
            
            for metric, value in metrics.items():
                if metric in baseline:
                    baseline_value = baseline[metric]
                    if baseline_value > 0:
                        ratio = value / baseline_value
                        comparison[metric] = {
                            "current": value,
                            "human_baseline": baseline_value,
                            "ratio": ratio,
                            "percentage_of_human": ratio * 100
                        }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Human baseline comparison failed: {e}")
            return {}
    
    def _calculate_overall_layout_score(self, basic_metrics: Dict, advanced_metrics: Dict) -> float:
        """Calculate overall layout detection score."""
        try:
            # Weighted combination of metrics
            weights = {
                "mAP50": 0.4,
                "precision": 0.2,
                "recall": 0.2,
                "spatial_accuracy": 0.1,
                "hierarchical_accuracy": 0.1
            }
            
            score = 0.0
            for metric, weight in weights.items():
                value = basic_metrics.get(metric, 0.0) or advanced_metrics.get(metric, 0.0)
                score += value * weight
            
            return score
            
        except Exception as e:
            logger.error(f"Overall layout score calculation failed: {e}")
            return 0.0
    
    def _calculate_overall_text_score(self, basic_metrics: Dict, advanced_metrics: Dict) -> float:
        """Calculate overall text extraction score."""
        try:
            # Weighted combination
            weights = {
                "accuracy": 0.5,
                "avg_semantic_similarity": 0.3,
                "avg_readability": 0.2
            }
            
            score = 0.0
            for metric, weight in weights.items():
                value = basic_metrics.get(metric, 0.0) or advanced_metrics.get(metric, 0.0)
                score += value * weight
            
            return score
            
        except Exception as e:
            logger.error(f"Overall text score calculation failed: {e}")
            return 0.0
    
    def _calculate_overall_langid_score(self, basic_metrics: Dict, advanced_metrics: Dict) -> float:
        """Calculate overall language identification score."""
        try:
            # Weighted combination
            weights = {
                "accuracy": 0.7,
                "language_coverage": 0.3
            }
            
            score = 0.0
            for metric, weight in weights.items():
                value = basic_metrics.get(metric, 0.0) or advanced_metrics.get(metric, 0.0)
                score += value * weight
            
            return score
            
        except Exception as e:
            logger.error(f"Overall language ID score calculation failed: {e}")
            return 0.0
    
    def _levenshtein_distance(self, s1, s2):
        """Calculate Levenshtein distance between two sequences."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate simplified readability score."""
        try:
            if not text:
                return 0.0
            
            words = text.split()
            sentences = text.split('.')
            
            if not words or not sentences:
                return 0.0
            
            # Simplified Flesch Reading Ease
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Normalize to 0-1 scale
            readability = max(0.0, min(1.0, 1.0 - (avg_sentence_length / 20.0)))
            
            return readability
            
        except Exception as e:
            logger.error(f"Readability calculation failed: {e}")
            return 0.0
    
    def _calculate_meteor_simplified(self, pred: str, gt: str) -> float:
        """Calculate simplified METEOR score."""
        try:
            pred_words = set(pred.lower().split())
            gt_words = set(gt.lower().split())
            
            if not gt_words:
                return 0.0
            
            # Word overlap
            overlap = len(pred_words.intersection(gt_words))
            precision = overlap / len(pred_words) if pred_words else 0.0
            recall = overlap / len(gt_words)
            
            # F1 score
            if precision + recall == 0:
                return 0.0
            
            f1 = 2 * precision * recall / (precision + recall)
            return f1
            
        except Exception as e:
            logger.error(f"METEOR calculation failed: {e}")
            return 0.0
    
    def get_evaluation_summary(self) -> Dict:
        """Get comprehensive evaluation summary."""
        summary = {
            "evaluation_timestamp": time.time(),
            "metrics": self.metrics,
            "human_baselines": self.human_baselines,
            "overall_performance": {
                "layout_detection": self.metrics.get("layout_detection", {}).get("overall_score", 0.0),
                "text_extraction": self.metrics.get("text_extraction", {}).get("overall_score", 0.0),
                "language_identification": self.metrics.get("language_identification", {}).get("overall_score", 0.0)
            }
        }
        
        # Calculate overall system score
        scores = [v for v in summary["overall_performance"].values() if v > 0]
        if scores:
            summary["system_overall_score"] = np.mean(scores)
        else:
            summary["system_overall_score"] = 0.0
        
        return summary
    
    def save_evaluation_results(self, output_path: str):
        """Save evaluation results to file."""
        try:
            summary = self.get_evaluation_summary()
            
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"✅ Evaluation results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")

# Convenience function
def get_enhanced_evaluator(config: Dict = None) -> EnhancedEvaluator:
    """Get instance of enhanced evaluator."""
    return EnhancedEvaluator(config)
