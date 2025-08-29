"""
Stage 1 Evaluator for PS-05 Document Understanding

Evaluates layout detection performance using mAP (Mean Average Precision)
at bbox threshold >= 0.5 as specified in the problem statement.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2

try:
    from pycocotools.cocoeval import COCOeval
    from pycocotools.coco import COCO
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("pycocotools not available. Install with: pip install pycocotools")

logger = logging.getLogger(__name__)

class Stage1Evaluator:
    """Evaluator for Stage 1 layout detection performance."""
    
    def __init__(self, iou_threshold: float = 0.5):
        """Initialize the Stage 1 evaluator.
        
        Args:
            iou_threshold: IoU threshold for evaluation (default: 0.5 as per problem statement)
        """
        self.iou_threshold = iou_threshold
        self.classes = ['Background', 'Text', 'Title', 'List', 'Table', 'Figure']
        self.class_ids = {name: i for i, name in enumerate(self.classes)}
        
    def calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union between two bounding boxes.
        
        Args:
            bbox1: First bbox in [x, y, w, h] format
            bbox2: Second bbox in [x, y, w, h] format
            
        Returns:
            IoU value between 0 and 1
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + w2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_predictions(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Evaluate layout detection predictions against ground truth.
        
        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not PYTORCH_AVAILABLE:
            return self._evaluate_simple(predictions, ground_truth)
        
        try:
            return self._evaluate_coco_format(predictions, ground_truth)
        except Exception as e:
            logger.warning(f"COCO evaluation failed: {e}. Falling back to simple evaluation.")
            return self._evaluate_simple(predictions, ground_truth)
    
    def _evaluate_simple(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Simple evaluation without pycocotools."""
        logger.info("Running simple evaluation...")
        
        # Group predictions and ground truth by image
        pred_by_image = self._group_by_image(predictions)
        gt_by_image = self._group_by_image(ground_truth)
        
        # Calculate metrics for each class
        class_metrics = {}
        for class_name in self.classes:
            class_metrics[class_name] = self._calculate_class_metrics(
                pred_by_image, gt_by_image, class_name
            )
        
        # Calculate overall mAP
        aps = [metrics['AP'] for metrics in class_metrics.values() if metrics['AP'] > 0]
        mAP = np.mean(aps) if aps else 0.0
        
        # Calculate overall precision and recall
        total_tp = sum(metrics['TP'] for metrics in class_metrics.values())
        total_fp = sum(metrics['FP'] for metrics in class_metrics.values())
        total_fn = sum(metrics['FN'] for metrics in class_metrics.values())
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        
        return {
            'mAP': mAP,
            'mAP50': mAP,  # Same as mAP for single IoU threshold
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'class_metrics': class_metrics,
            'evaluation_method': 'simple'
        }
    
    def _evaluate_coco_format(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Evaluate using COCO format for more accurate metrics."""
        logger.info("Running COCO format evaluation...")
        
        # Convert to COCO format
        coco_pred = self._convert_to_coco_format(predictions, is_gt=False)
        coco_gt = self._convert_to_coco_format(ground_truth, is_gt=True)
        
        # Create COCO objects
        coco_gt_obj = COCO()
        coco_gt_obj.dataset = coco_gt
        coco_gt_obj.createIndex()
        
        coco_pred_obj = COCO()
        coco_pred_obj.dataset = coco_pred
        coco_pred_obj.createIndex()
        
        # Run evaluation
        coco_eval = COCOeval(coco_gt_obj, coco_pred_obj, 'bbox')
        coco_eval.params.iouThrs = [self.iou_threshold]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        mAP = coco_eval.stats[0]  # AP at IoU threshold
        mAP50 = coco_eval.stats[1]  # AP at IoU=0.5
        
        return {
            'mAP': float(mAP),
            'mAP50': float(mAP50),
            'evaluation_method': 'coco',
            'coco_eval': coco_eval
        }
    
    def _group_by_image(self, annotations: List[Dict]) -> Dict[str, List[Dict]]:
        """Group annotations by image file."""
        grouped = {}
        for ann in annotations:
            image_name = ann.get('image_name', ann.get('file_name', 'unknown'))
            if image_name not in grouped:
                grouped[image_name] = []
            grouped[image_name].append(ann)
        return grouped
    
    def _calculate_class_metrics(self, pred_by_image: Dict, gt_by_image: Dict, 
                                class_name: str) -> Dict:
        """Calculate metrics for a specific class."""
        class_id = self.class_ids[class_name]
        
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives
        
        for image_name in pred_by_image.keys():
            preds = [p for p in pred_by_image.get(image_name, []) 
                    if p.get('class_id') == class_id or p.get('cls') == class_name]
            gts = [g for g in gt_by_image.get(image_name, []) 
                   if g.get('category_id') == class_id or g.get('class_id') == class_id]
            
            # Match predictions to ground truth
            matched_gt = set()
            for pred in preds:
                pred_bbox = pred.get('bbox', [0, 0, 1, 1])
                best_iou = 0
                best_gt_idx = -1
                
                for i, gt in enumerate(gts):
                    if i in matched_gt:
                        continue
                    
                    gt_bbox = gt.get('bbox', [0, 0, 1, 1])
                    iou = self.calculate_iou(pred_bbox, gt_bbox)
                    
                    if iou > best_iou and iou >= self.iou_threshold:
                        best_iou = iou
                        best_gt_idx = i
                
                if best_gt_idx >= 0:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
            
            # Count unmatched ground truth as false negatives
            fn += len(gts) - len(matched_gt)
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Calculate AP (Average Precision) - simplified version
        ap = precision if recall > 0 else 0.0
        
        return {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'AP': ap
        }
    
    def _convert_to_coco_format(self, annotations: List[Dict], is_gt: bool) -> Dict:
        """Convert annotations to COCO format."""
        coco_format = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add categories
        for i, class_name in enumerate(self.classes):
            coco_format['categories'].append({
                'id': i,
                'name': class_name,
                'supercategory': 'document'
            })
        
        # Group by image
        grouped = self._group_by_image(annotations)
        
        image_id = 0
        ann_id = 0
        
        for image_name, anns in grouped.items():
            # Add image info
            coco_format['images'].append({
                'id': image_id,
                'file_name': image_name,
                'width': 800,  # Default values - should be extracted from actual images
                'height': 600
            })
            
            # Add annotations
            for ann in anns:
                bbox = ann.get('bbox', [0, 0, 1, 1])
                
                if is_gt:
                    category_id = ann.get('category_id', 1)
                else:
                    category_id = ann.get('class_id', 1)
                
                coco_format['annotations'].append({
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': bbox,
                    'area': bbox[2] * bbox[3],
                    'iscrowd': 0
                })
                ann_id += 1
            
            image_id += 1
        
        return coco_format
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to file.
        
        Args:
            results: Evaluation results dictionary
            output_path: Path to save results
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Evaluation results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def print_summary(self, results: Dict):
        """Print evaluation summary."""
        print("\n" + "="*50)
        print("STAGE 1 EVALUATION RESULTS")
        print("="*50)
        print(f"mAP (IoU ≥ {self.iou_threshold}): {results.get('mAP', 0):.4f}")
        print(f"mAP50: {results.get('mAP50', 0):.4f}")
        
        if 'class_metrics' in results:
            print(f"\nPer-class metrics:")
            for class_name, metrics in results['class_metrics'].items():
                print(f"  {class_name}:")
                print(f"    Precision: {metrics['Precision']:.4f}")
                print(f"    Recall: {metrics['Recall']:.4f}")
                print(f"    AP: {metrics['AP']:.4f}")
        
        print(f"\nEvaluation method: {results.get('evaluation_method', 'unknown')}")
        print("="*50)

def evaluate_stage1(predictions_file: str, ground_truth_file: str, 
                   output_file: str = None) -> Dict:
    """Convenience function to evaluate Stage 1 performance.
    
    Args:
        predictions_file: Path to predictions JSON file
        ground_truth_file: Path to ground truth JSON file
        output_file: Optional path to save results
        
    Returns:
        Evaluation results dictionary
    """
    try:
        # Load predictions and ground truth
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
        
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
        
        # Initialize evaluator
        evaluator = Stage1Evaluator()
        
        # Run evaluation
        results = evaluator.evaluate_predictions(predictions, ground_truth)
        
        # Print summary
        evaluator.print_summary(results)
        
        # Save results if output file specified
        if output_file:
            evaluator.save_results(results, output_file)
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 layout detection")
    parser.add_argument('--predictions', required=True, help='Predictions JSON file')
    parser.add_argument('--ground-truth', required=True, help='Ground truth JSON file')
    parser.add_argument('--output', help='Output results file')
    
    args = parser.parse_args()
    
    results = evaluate_stage1(args.predictions, args.ground_truth, args.output)
