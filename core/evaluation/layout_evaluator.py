"""
Layout Detection Evaluation Module

Evaluates layout detection performance using:
- mAP (mean Average Precision)
- IoU (Intersection over Union)
- Precision, Recall, F1-score
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    COCO = None
    logging.warning("pycocotools not installed. Install with: pip install pycocotools")

class LayoutEvaluator:
    """Evaluator for layout detection performance."""
    
    def __init__(self, iou_thresholds: List[float] = None):
        """Initialize the layout evaluator.
        
        Args:
            iou_thresholds: List of IoU thresholds for evaluation
        """
        self.iou_thresholds = iou_thresholds or [0.5, 0.75]
        self.classes = ['Background', 'Text', 'Title', 'List', 'Table', 'Figure']
        
    def calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union between two bounding boxes.
        
        Args:
            bbox1: [x1, y1, w1, h1] first bounding box
            bbox2: [x2, y2, w2, h2] second bounding box
            
        Returns:
            IoU value between 0 and 1
        """
        try:
            # Convert to [x1, y1, x2, y2] format
            x1_1, y1_1, w1, h1 = bbox1
            x2_1, y2_1 = x1_1 + w1, y1_1 + h1
            
            x1_2, y1_2, w2, h2 = bbox2
            x2_2, y2_2 = x1_2 + w2, y1_2 + h2
            
            # Calculate intersection
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            
            # Calculate union
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logging.error(f"Error calculating IoU: {e}")
            return 0.0
    
    def evaluate_single_image(self, predictions: List[Dict], ground_truth: List[Dict], 
                            iou_threshold: float = 0.5) -> Dict:
        """Evaluate layout detection for a single image.
        
        Args:
            predictions: List of predicted detections
            ground_truth: List of ground truth annotations
            iou_threshold: IoU threshold for matching
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Match predictions to ground truth
            matches = self._match_predictions_to_ground_truth(
                predictions, ground_truth, iou_threshold
            )
            
            # Calculate metrics
            tp = len(matches)  # True positives
            fp = len(predictions) - tp  # False positives
            fn = len(ground_truth) - tp  # False negatives
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'iou_threshold': iou_threshold
            }
            
        except Exception as e:
            logging.error(f"Error evaluating single image: {e}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'iou_threshold': iou_threshold
            }
    
    def _match_predictions_to_ground_truth(self, predictions: List[Dict], 
                                         ground_truth: List[Dict], 
                                         iou_threshold: float) -> List[Tuple]:
        """Match predictions to ground truth using IoU.
        
        Args:
            predictions: List of predicted detections
            ground_truth: List of ground truth annotations
            iou_threshold: IoU threshold for matching
            
        Returns:
            List of matched pairs (pred_idx, gt_idx)
        """
        matches = []
        used_gt = set()
        
        # Sort predictions by confidence score (descending)
        sorted_predictions = sorted(predictions, key=lambda x: x.get('score', 0), reverse=True)
        
        for pred_idx, pred in enumerate(sorted_predictions):
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in used_gt:
                    continue
                
                # Check if classes match
                if pred.get('cls') != gt.get('cls'):
                    continue
                
                # Calculate IoU
                iou = self.calculate_iou(pred.get('bbox', []), gt.get('bbox', []))
                
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                matches.append((pred_idx, best_gt_idx))
                used_gt.add(best_gt_idx)
        
        return matches
    
    def evaluate_dataset(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Evaluate layout detection on a dataset.
        
        Args:
            predictions: List of prediction results for all images
            ground_truth: List of ground truth annotations for all images
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        try:
            results = {}
            
            # Evaluate at different IoU thresholds
            for iou_threshold in self.iou_thresholds:
                iou_results = self._evaluate_at_iou_threshold(
                    predictions, ground_truth, iou_threshold
                )
                results[f'mAP@{int(iou_threshold*100)}'] = iou_results
            
            # Calculate overall mAP
            map_values = [results[f'mAP@{int(t*100)}']['mAP'] for t in self.iou_thresholds]
            results['mAP'] = np.mean(map_values)
            
            # Per-class metrics
            results['per_class'] = self._calculate_per_class_metrics(predictions, ground_truth)
            
            return results
            
        except Exception as e:
            logging.error(f"Error evaluating dataset: {e}")
            return {'error': str(e)}
    
    def _evaluate_at_iou_threshold(self, predictions: List[Dict], ground_truth: List[Dict], 
                                  iou_threshold: float) -> Dict:
        """Evaluate at a specific IoU threshold."""
        try:
            all_precisions = []
            all_recalls = []
            all_f1s = []
            
            # Group by image
            pred_by_image = self._group_by_image(predictions)
            gt_by_image = self._group_by_image(ground_truth)
            
            for image_id in set(list(pred_by_image.keys()) + list(gt_by_image.keys())):
                preds = pred_by_image.get(image_id, [])
                gts = gt_by_image.get(image_id, [])
                
                metrics = self.evaluate_single_image(preds, gts, iou_threshold)
                all_precisions.append(metrics['precision'])
                all_recalls.append(metrics['recall'])
                all_f1s.append(metrics['f1'])
            
            return {
                'mAP': np.mean(all_precisions),
                'mAR': np.mean(all_recalls),
                'mF1': np.mean(all_f1s),
                'std_precision': np.std(all_precisions),
                'std_recall': np.std(all_recalls),
                'std_f1': np.std(all_f1s)
            }
            
        except Exception as e:
            logging.error(f"Error evaluating at IoU threshold {iou_threshold}: {e}")
            return {'mAP': 0.0, 'mAR': 0.0, 'mF1': 0.0}
    
    def _group_by_image(self, annotations: List[Dict]) -> Dict:
        """Group annotations by image ID."""
        grouped = defaultdict(list)
        for ann in annotations:
            image_id = ann.get('image_id', 0)
            grouped[image_id].append(ann)
        return dict(grouped)
    
    def _calculate_per_class_metrics(self, predictions: List[Dict], 
                                   ground_truth: List[Dict]) -> Dict:
        """Calculate per-class metrics."""
        try:
            class_metrics = {}
            
            for class_name in self.classes:
                class_preds = [p for p in predictions if p.get('cls') == class_name]
                class_gts = [g for g in ground_truth if g.get('cls') == class_name]
                
                if not class_gts and not class_preds:
                    continue
                
                # Calculate metrics for this class
                metrics = self._calculate_class_metrics(class_preds, class_gts)
                class_metrics[class_name] = metrics
            
            return class_metrics
            
        except Exception as e:
            logging.error(f"Error calculating per-class metrics: {e}")
            return {}
    
    def _calculate_class_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Calculate metrics for a specific class."""
        try:
            # Use the highest IoU threshold for class-level evaluation
            iou_threshold = max(self.iou_thresholds)
            
            matches = self._match_predictions_to_ground_truth(
                predictions, ground_truth, iou_threshold
            )
            
            tp = len(matches)
            fp = len(predictions) - tp
            fn = len(ground_truth) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'total_predictions': len(predictions),
                'total_ground_truth': len(ground_truth)
            }
            
        except Exception as e:
            logging.error(f"Error calculating class metrics: {e}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'total_predictions': len(predictions),
                'total_ground_truth': len(ground_truth)
            }
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logging.info(f"Evaluation results saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")

def evaluate_layout(predictions: List[Dict], ground_truth: List[Dict], 
                   output_path: str = None) -> Dict:
    """Convenience function for layout evaluation.
    
    Args:
        predictions: List of prediction results
        ground_truth: List of ground truth annotations
        output_path: Optional path to save results
        
    Returns:
        Dictionary with evaluation results
    """
    evaluator = LayoutEvaluator()
    results = evaluator.evaluate_dataset(predictions, ground_truth)
    
    if output_path:
        evaluator.save_results(results, output_path)
    
    return results

if __name__ == "__main__":
    # Test the layout evaluator
    # Sample predictions and ground truth
    predictions = [
        {'image_id': 1, 'cls': 'Text', 'bbox': [10, 10, 100, 50], 'score': 0.9},
        {'image_id': 1, 'cls': 'Title', 'bbox': [10, 70, 100, 30], 'score': 0.8}
    ]
    
    ground_truth = [
        {'image_id': 1, 'cls': 'Text', 'bbox': [12, 12, 98, 48]},
        {'image_id': 1, 'cls': 'Title', 'bbox': [12, 72, 98, 28]}
    ]
    
    results = evaluate_layout(predictions, ground_truth)
    print("Layout Evaluation Results:")
    print(json.dumps(results, indent=2)) 