"""
Language Identification Evaluation Module

Evaluates language identification performance using:
- Accuracy, Precision, Recall, F1-score
- Confusion matrix
- Per-language metrics
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import json
from collections import defaultdict

class LanguageEvaluator:
    """Evaluator for language identification performance."""
    
    def __init__(self, target_languages: List[str] = None):
        """Initialize the language evaluator.
        
        Args:
            target_languages: List of target language codes
        """
        self.target_languages = target_languages or ['en', 'hi', 'ur', 'ar', 'ne', 'fa']
    
    def calculate_accuracy(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Calculate overall accuracy.
        
        Args:
            predictions: List of predicted language codes
            ground_truth: List of ground truth language codes
            
        Returns:
            Accuracy value between 0 and 1
        """
        if not predictions or not ground_truth:
            return 0.0
        
        correct = sum(1 for pred, gt in zip(predictions, ground_truth) if pred == gt)
        return correct / len(predictions)
    
    def calculate_precision_recall_f1(self, predictions: List[str], ground_truth: List[str]) -> Dict:
        """Calculate precision, recall, and F1-score per language.
        
        Args:
            predictions: List of predicted language codes
            ground_truth: List of ground truth language codes
            
        Returns:
            Dictionary with precision, recall, F1 for each language
        """
        metrics = {}
        
        for lang in self.target_languages:
            # True positives: predicted as lang and actually lang
            tp = sum(1 for pred, gt in zip(predictions, ground_truth) 
                    if pred == lang and gt == lang)
            
            # False positives: predicted as lang but actually not lang
            fp = sum(1 for pred, gt in zip(predictions, ground_truth) 
                    if pred == lang and gt != lang)
            
            # False negatives: not predicted as lang but actually lang
            fn = sum(1 for pred, gt in zip(predictions, ground_truth) 
                    if pred != lang and gt == lang)
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[lang] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        
        return metrics
    
    def create_confusion_matrix(self, predictions: List[str], ground_truth: List[str]) -> Dict:
        """Create confusion matrix.
        
        Args:
            predictions: List of predicted language codes
            ground_truth: List of ground truth language codes
            
        Returns:
            Confusion matrix as dictionary
        """
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        
        for pred, gt in zip(predictions, ground_truth):
            confusion_matrix[gt][pred] += 1
        
        return dict(confusion_matrix)
    
    def evaluate_language_id(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Evaluate language identification performance.
        
        Args:
            predictions: List of prediction dictionaries with 'lang' field
            ground_truth: List of ground truth dictionaries with 'lang' field
            
        Returns:
            Dictionary with language identification evaluation metrics
        """
        try:
            # Extract language codes
            pred_langs = [p.get('lang', 'unknown') for p in predictions]
            gt_langs = [g.get('lang', 'unknown') for g in ground_truth]
            
            # Calculate overall accuracy
            accuracy = self.calculate_accuracy(pred_langs, gt_langs)
            
            # Calculate per-language metrics
            per_lang_metrics = self.calculate_precision_recall_f1(pred_langs, gt_langs)
            
            # Create confusion matrix
            confusion_matrix = self.create_confusion_matrix(pred_langs, gt_langs)
            
            # Calculate macro-averaged metrics
            macro_precision = np.mean([m['precision'] for m in per_lang_metrics.values()])
            macro_recall = np.mean([m['recall'] for m in per_lang_metrics.values()])
            macro_f1 = np.mean([m['f1'] for m in per_lang_metrics.values()])
            
            return {
                'accuracy': accuracy,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
                'per_language': per_lang_metrics,
                'confusion_matrix': confusion_matrix,
                'total_samples': len(predictions)
            }
            
        except Exception as e:
            logging.error(f"Error evaluating language identification: {e}")
            return {
                'accuracy': 0.0,
                'macro_precision': 0.0,
                'macro_recall': 0.0,
                'macro_f1': 0.0,
                'per_language': {},
                'confusion_matrix': {},
                'total_samples': 0,
                'error': str(e)
            }
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logging.info(f"Language identification evaluation results saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")

def evaluate_language_id(predictions: List[Dict], ground_truth: List[Dict], 
                        output_path: str = None) -> Dict:
    """Convenience function for language identification evaluation.
    
    Args:
        predictions: List of prediction results
        ground_truth: List of ground truth annotations
        output_path: Optional path to save results
        
    Returns:
        Dictionary with language identification evaluation results
    """
    evaluator = LanguageEvaluator()
    results = evaluator.evaluate_language_id(predictions, ground_truth)
    
    if output_path:
        evaluator.save_results(results, output_path)
    
    return results

if __name__ == "__main__":
    # Test the language identification evaluator
    predictions = [
        {'lang': 'en', 'confidence': 0.9},
        {'lang': 'hi', 'confidence': 0.8},
        {'lang': 'ar', 'confidence': 0.7}
    ]
    
    ground_truth = [
        {'lang': 'en'},
        {'lang': 'hi'},
        {'lang': 'ar'}
    ]
    
    results = evaluate_language_id(predictions, ground_truth)
    print("Language Identification Evaluation Results:")
    print(json.dumps(results, indent=2)) 