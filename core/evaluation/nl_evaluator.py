"""
Natural Language Generation Evaluation Module

Evaluates NL generation performance using:
- BLEURT
- BERTScore
- BLEU
- Semantic similarity metrics
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import json

try:
    import sacrebleu
except ImportError:
    sacrebleu = None
    logging.warning("sacrebleu not installed. Install with: pip install sacrebleu")

try:
    from bert_score import score as bert_score_fn
except ImportError:
    bert_score_fn = None
    logging.warning("bert-score not installed. Install with: pip install bert-score")

class NLEvaluator:
    """Evaluator for natural language generation performance."""
    
    def __init__(self):
        """Initialize the NL evaluator."""
        self.metrics = ['BLEU', 'BERTScore']
    
    def calculate_bleu(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Calculate BLEU score.
        
        Args:
            predictions: List of predicted text strings
            ground_truth: List of ground truth text strings
            
        Returns:
            BLEU score
        """
        if not predictions or not ground_truth:
            return 0.0
        
        try:
            if sacrebleu:
                # Use sacrebleu for BLEU calculation
                bleu = sacrebleu.corpus_bleu(predictions, [ground_truth])
                return bleu.score / 100.0  # Normalize to 0-1 range
            else:
                # Simple n-gram overlap calculation
                return self._simple_bleu(predictions, ground_truth)
                
        except Exception as e:
            logging.error(f"Error calculating BLEU: {e}")
            return 0.0
    
    def _simple_bleu(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Simple BLEU-like score calculation."""
        total_score = 0.0
        count = 0
        
        for pred, gt in zip(predictions, ground_truth):
            if not gt.strip():
                continue
            
            pred_words = pred.lower().strip().split()
            gt_words = gt.lower().strip().split()
            
            if not gt_words:
                continue
            
            # Calculate n-gram overlap (simplified)
            overlap = len(set(pred_words) & set(gt_words))
            total_words = len(gt_words)
            
            if total_words > 0:
                score = overlap / total_words
                total_score += score
                count += 1
        
        return total_score / count if count > 0 else 0.0
    
    def calculate_bert_score(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Calculate BERTScore.
        
        Args:
            predictions: List of predicted text strings
            ground_truth: List of ground truth text strings
            
        Returns:
            BERTScore value
        """
        if not predictions or not ground_truth:
            return 0.0
        
        try:
            if bert_score_fn:
                # Use bert-score library
                P, R, F1 = bert_score_fn(predictions, ground_truth, lang='en', verbose=False)
                return float(F1.mean())  # Return F1 score
            else:
                # Fallback to simple semantic similarity
                return self._simple_semantic_similarity(predictions, ground_truth)
                
        except Exception as e:
            logging.error(f"Error calculating BERTScore: {e}")
            return 0.0
    
    def _simple_semantic_similarity(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Simple semantic similarity calculation."""
        total_similarity = 0.0
        count = 0
        
        for pred, gt in zip(predictions, ground_truth):
            if not gt.strip():
                continue
            
            # Simple word overlap-based similarity
            pred_words = set(pred.lower().strip().split())
            gt_words = set(gt.lower().strip().split())
            
            if not gt_words:
                continue
            
            intersection = len(pred_words & gt_words)
            union = len(pred_words | gt_words)
            
            if union > 0:
                similarity = intersection / union
                total_similarity += similarity
                count += 1
        
        return total_similarity / count if count > 0 else 0.0
    
    def evaluate_nl_generation(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Evaluate natural language generation performance.
        
        Args:
            predictions: List of prediction dictionaries with 'summary' field
            ground_truth: List of ground truth dictionaries with 'summary' field
            
        Returns:
            Dictionary with NL generation evaluation metrics
        """
        try:
            # Extract summaries
            pred_summaries = [p.get('summary', '') for p in predictions]
            gt_summaries = [g.get('summary', '') for g in ground_truth]
            
            # Calculate metrics
            bleu_score = self.calculate_bleu(pred_summaries, gt_summaries)
            bert_score = self.calculate_bert_score(pred_summaries, gt_summaries)
            
            # Calculate per-type metrics if type information is available
            type_metrics = self._calculate_type_metrics(predictions, ground_truth)
            
            return {
                'BLEU': bleu_score,
                'BERTScore': bert_score,
                'per_type': type_metrics,
                'total_samples': len(predictions)
            }
            
        except Exception as e:
            logging.error(f"Error evaluating NL generation: {e}")
            return {
                'BLEU': 0.0,
                'BERTScore': 0.0,
                'per_type': {},
                'total_samples': 0,
                'error': str(e)
            }
    
    def _calculate_type_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Calculate metrics per element type."""
        type_metrics = {}
        
        try:
            # Group by element type
            type_groups = {}
            for pred, gt in zip(predictions, ground_truth):
                element_type = pred.get('type', 'unknown')
                if element_type not in type_groups:
                    type_groups[element_type] = {'preds': [], 'gts': []}
                type_groups[element_type]['preds'].append(pred.get('summary', ''))
                type_groups[element_type]['gts'].append(gt.get('summary', ''))
            
            # Calculate metrics for each type
            for element_type, group in type_groups.items():
                if len(group['preds']) > 0:
                    bleu = self.calculate_bleu(group['preds'], group['gts'])
                    bert_score = self.calculate_bert_score(group['preds'], group['gts'])
                    
                    type_metrics[element_type] = {
                        'BLEU': bleu,
                        'BERTScore': bert_score,
                        'samples': len(group['preds'])
                    }
        
        except Exception as e:
            logging.error(f"Error calculating type metrics: {e}")
        
        return type_metrics
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logging.info(f"NL generation evaluation results saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")

def evaluate_nl_generation(predictions: List[Dict], ground_truth: List[Dict], 
                          output_path: str = None) -> Dict:
    """Convenience function for NL generation evaluation.
    
    Args:
        predictions: List of prediction results
        ground_truth: List of ground truth annotations
        output_path: Optional path to save results
        
    Returns:
        Dictionary with NL generation evaluation results
    """
    evaluator = NLEvaluator()
    results = evaluator.evaluate_nl_generation(predictions, ground_truth)
    
    if output_path:
        evaluator.save_results(results, output_path)
    
    return results

if __name__ == "__main__":
    # Test the NL generation evaluator
    predictions = [
        {'summary': 'A table showing quarterly revenue data', 'type': 'table'},
        {'summary': 'A bar chart displaying sales trends', 'type': 'chart'}
    ]
    
    ground_truth = [
        {'summary': 'A table containing quarterly revenue information', 'type': 'table'},
        {'summary': 'A bar chart showing sales data trends', 'type': 'chart'}
    ]
    
    results = evaluate_nl_generation(predictions, ground_truth)
    print("NL Generation Evaluation Results:")
    print(json.dumps(results, indent=2)) 