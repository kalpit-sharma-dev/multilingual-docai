"""
OCR Evaluation Module

Evaluates OCR performance using:
- CER (Character Error Rate)
- WER (Word Error Rate)
- Language-specific metrics
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import json

try:
    import jiwer
except ImportError:
    jiwer = None
    logging.warning("jiwer not installed. Install with: pip install jiwer")

class OCREvaluator:
    """Evaluator for OCR performance."""
    
    def __init__(self):
        """Initialize the OCR evaluator."""
        self.metrics = ['CER', 'WER']
    
    def calculate_cer(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Calculate Character Error Rate.
        
        Args:
            predictions: List of predicted text strings
            ground_truth: List of ground truth text strings
            
        Returns:
            CER value between 0 and 1
        """
        if not predictions or not ground_truth:
            return 1.0
        
        try:
            if jiwer:
                # Use jiwer for CER calculation
                transformation = jiwer.Compose([
                    jiwer.ToLowerCase(),
                    jiwer.RemoveMultipleSpaces(),
                    jiwer.Strip()
                ])
                
                total_cer = 0.0
                count = 0
                
                for pred, gt in zip(predictions, ground_truth):
                    if gt.strip():  # Skip empty ground truth
                        cer = jiwer.character_error_rate(gt, pred)
                        total_cer += cer
                        count += 1
                
                return total_cer / count if count > 0 else 1.0
            else:
                # Simple character-level comparison
                return self._simple_cer(predictions, ground_truth)
                
        except Exception as e:
            logging.error(f"Error calculating CER: {e}")
            return 1.0
    
    def _simple_cer(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Simple character error rate calculation."""
        total_errors = 0
        total_chars = 0
        
        for pred, gt in zip(predictions, ground_truth):
            pred_chars = list(pred.lower().strip())
            gt_chars = list(gt.lower().strip())
            
            # Use Levenshtein distance
            distance = self._levenshtein_distance(pred_chars, gt_chars)
            total_errors += distance
            total_chars += len(gt_chars)
        
        return total_errors / total_chars if total_chars > 0 else 1.0
    
    def _levenshtein_distance(self, s1: List, s2: List) -> int:
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
    
    def calculate_wer(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Calculate Word Error Rate.
        
        Args:
            predictions: List of predicted text strings
            ground_truth: List of ground truth text strings
            
        Returns:
            WER value between 0 and 1
        """
        if not predictions or not ground_truth:
            return 1.0
        
        try:
            if jiwer:
                # Use jiwer for WER calculation
                transformation = jiwer.Compose([
                    jiwer.ToLowerCase(),
                    jiwer.RemoveMultipleSpaces(),
                    jiwer.Strip()
                ])
                
                total_wer = 0.0
                count = 0
                
                for pred, gt in zip(predictions, ground_truth):
                    if gt.strip():  # Skip empty ground truth
                        wer = jiwer.word_error_rate(gt, pred)
                        total_wer += wer
                        count += 1
                
                return total_wer / count if count > 0 else 1.0
            else:
                # Simple word-level comparison
                return self._simple_wer(predictions, ground_truth)
                
        except Exception as e:
            logging.error(f"Error calculating WER: {e}")
            return 1.0
    
    def _simple_wer(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Simple word error rate calculation."""
        total_errors = 0
        total_words = 0
        
        for pred, gt in zip(predictions, ground_truth):
            pred_words = pred.lower().strip().split()
            gt_words = gt.lower().strip().split()
            
            # Use Levenshtein distance on words
            distance = self._levenshtein_distance(pred_words, gt_words)
            total_errors += distance
            total_words += len(gt_words)
        
        return total_errors / total_words if total_words > 0 else 1.0
    
    def evaluate_ocr(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Evaluate OCR performance.
        
        Args:
            predictions: List of prediction dictionaries with 'text' field
            ground_truth: List of ground truth dictionaries with 'text' field
            
        Returns:
            Dictionary with OCR evaluation metrics
        """
        try:
            # Extract text from predictions and ground truth
            pred_texts = [p.get('text', '') for p in predictions]
            gt_texts = [g.get('text', '') for g in ground_truth]
            
            # Calculate metrics
            cer = self.calculate_cer(pred_texts, gt_texts)
            wer = self.calculate_wer(pred_texts, gt_texts)
            
            # Calculate per-language metrics if language info is available
            lang_metrics = self._calculate_language_metrics(predictions, ground_truth)
            
            return {
                'CER': cer,
                'WER': wer,
                'per_language': lang_metrics,
                'total_samples': len(predictions)
            }
            
        except Exception as e:
            logging.error(f"Error evaluating OCR: {e}")
            return {
                'CER': 1.0,
                'WER': 1.0,
                'per_language': {},
                'total_samples': 0,
                'error': str(e)
            }
    
    def _calculate_language_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Calculate metrics per language."""
        lang_metrics = {}
        
        try:
            # Group by language
            lang_groups = {}
            for pred, gt in zip(predictions, ground_truth):
                lang = pred.get('lang', 'unknown')
                if lang not in lang_groups:
                    lang_groups[lang] = {'preds': [], 'gts': []}
                lang_groups[lang]['preds'].append(pred.get('text', ''))
                lang_groups[lang]['gts'].append(gt.get('text', ''))
            
            # Calculate metrics for each language
            for lang, group in lang_groups.items():
                if len(group['preds']) > 0:
                    cer = self.calculate_cer(group['preds'], group['gts'])
                    wer = self.calculate_wer(group['preds'], group['gts'])
                    
                    lang_metrics[lang] = {
                        'CER': cer,
                        'WER': wer,
                        'samples': len(group['preds'])
                    }
        
        except Exception as e:
            logging.error(f"Error calculating language metrics: {e}")
        
        return lang_metrics
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logging.info(f"OCR evaluation results saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")

def evaluate_ocr(predictions: List[Dict], ground_truth: List[Dict], 
                output_path: str = None) -> Dict:
    """Convenience function for OCR evaluation.
    
    Args:
        predictions: List of prediction results
        ground_truth: List of ground truth annotations
        output_path: Optional path to save results
        
    Returns:
        Dictionary with OCR evaluation results
    """
    evaluator = OCREvaluator()
    results = evaluator.evaluate_ocr(predictions, ground_truth)
    
    if output_path:
        evaluator.save_results(results, output_path)
    
    return results

if __name__ == "__main__":
    # Test the OCR evaluator
    predictions = [
        {'text': 'Hello world', 'lang': 'en'},
        {'text': 'नमस्ते दुनिया', 'lang': 'hi'}
    ]
    
    ground_truth = [
        {'text': 'Hello world', 'lang': 'en'},
        {'text': 'नमस्ते दुनिया', 'lang': 'hi'}
    ]
    
    results = evaluate_ocr(predictions, ground_truth)
    print("OCR Evaluation Results:")
    print(json.dumps(results, indent=2)) 