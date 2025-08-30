#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for PS-05

Integrates all advanced models and evaluation metrics:
- LayoutLMv3, Donut, BLIP models
- Advanced evaluation metrics (BLEURT, BERTScore)
- Human-level performance baselines
- Cross-modal evaluation
"""

import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Add the parent directory to the Python path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
try:
    from core.models.advanced_models import get_advanced_models
    from core.evaluation.enhanced_evaluator import get_enhanced_evaluator
    from core.evaluation.layout_evaluator import LayoutEvaluator
    from core.evaluation.ocr_evaluator import OCREvaluator
    from core.evaluation.langid_evaluator import LangIDEvaluator
except ImportError as e:
    logging.error(f"Failed to import evaluation modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    """Comprehensive evaluator that integrates all evaluation components."""
    
    def __init__(self, config_path: str = "configs/ps05_config.yaml"):
        self.config = self._load_config(config_path)
        self.results = {}
        
        # Initialize advanced models
        logger.info("🚀 Initializing advanced models...")
        self.advanced_models = get_advanced_models()
        
        # Initialize evaluators
        logger.info("📊 Initializing evaluators...")
        self.enhanced_evaluator = get_enhanced_evaluator(self.config)
        self.layout_evaluator = LayoutEvaluator()
        self.ocr_evaluator = OCREvaluator()
        self.langid_evaluator = LangIDEvaluator()
        
        logger.info("✅ Comprehensive evaluator initialized!")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file."""
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def evaluate_system_comprehensive(self, test_data: List[Dict], 
                                    ground_truth: List[Dict]) -> Dict:
        """Comprehensive system evaluation."""
        logger.info("🎯 Starting comprehensive system evaluation...")
        
        start_time = time.time()
        
        try:
            # 1. Layout Detection Evaluation
            logger.info("📐 Evaluating layout detection...")
            layout_results = self._evaluate_layout_detection(test_data, ground_truth)
            self.results["layout_detection"] = layout_results
            
            # 2. Text Extraction Evaluation
            logger.info("📝 Evaluating text extraction...")
            text_results = self._evaluate_text_extraction(test_data, ground_truth)
            self.results["text_extraction"] = text_results
            
            # 3. Language Identification Evaluation
            logger.info("🌍 Evaluating language identification...")
            langid_results = self._evaluate_language_identification(test_data, ground_truth)
            self.results["language_identification"] = langid_results
            
            # 4. Content Understanding Evaluation
            logger.info("🧠 Evaluating content understanding...")
            content_results = self._evaluate_content_understanding(test_data, ground_truth)
            self.results["content_understanding"] = content_results
            
            # 5. Advanced Model Performance
            logger.info("🤖 Evaluating advanced models...")
            advanced_results = self._evaluate_advanced_models(test_data, ground_truth)
            self.results["advanced_models"] = advanced_results
            
            # 6. Overall System Performance
            logger.info("📊 Calculating overall system performance...")
            overall_results = self._calculate_overall_performance()
            self.results["overall_performance"] = overall_results
            
            # 7. Human Baseline Comparison
            logger.info("👥 Comparing with human baselines...")
            baseline_comparison = self._compare_with_human_baselines()
            self.results["human_baseline_comparison"] = baseline_comparison
            
            evaluation_time = time.time() - start_time
            logger.info(f"✅ Comprehensive evaluation completed in {evaluation_time:.2f} seconds")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed: {e}")
            return {}
    
    def _evaluate_layout_detection(self, test_data: List[Dict], 
                                 ground_truth: List[Dict]) -> Dict:
        """Evaluate layout detection performance."""
        try:
            # Basic layout evaluation
            basic_results = self.layout_evaluator.evaluate(test_data, ground_truth)
            
            # Enhanced layout evaluation
            enhanced_results = self.enhanced_evaluator.evaluate_layout_detection_enhanced(
                test_data, ground_truth
            )
            
            # Advanced model evaluation (if available)
            advanced_results = {}
            if self.advanced_models.layoutlmv3_model:
                advanced_results = self._evaluate_layoutlmv3_performance(test_data, ground_truth)
            
            return {
                "basic_metrics": basic_results,
                "enhanced_metrics": enhanced_results,
                "advanced_model_metrics": advanced_results,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Layout detection evaluation failed: {e}")
            return {"error": str(e)}
    
    def _evaluate_text_extraction(self, test_data: List[Dict], 
                                ground_truth: List[Dict]) -> Dict:
        """Evaluate text extraction performance."""
        try:
            # Basic OCR evaluation
            basic_results = self.ocr_evaluator.evaluate(test_data, ground_truth)
            
            # Enhanced text evaluation
            enhanced_results = self.enhanced_evaluator.evaluate_text_extraction_enhanced(
                test_data, ground_truth
            )
            
            # Advanced model evaluation (Donut)
            advanced_results = {}
            if self.advanced_models.donut_model:
                advanced_results = self._evaluate_donut_performance(test_data, ground_truth)
            
            return {
                "basic_metrics": basic_results,
                "enhanced_metrics": enhanced_results,
                "advanced_model_metrics": advanced_results,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Text extraction evaluation failed: {e}")
            return {"error": str(e)}
    
    def _evaluate_language_identification(self, test_data: List[Dict], 
                                       ground_truth: List[Dict]) -> Dict:
        """Evaluate language identification performance."""
        try:
            # Basic language ID evaluation
            basic_results = self.langid_evaluator.evaluate(test_data, ground_truth)
            
            # Enhanced language ID evaluation
            enhanced_results = self.enhanced_evaluator.evaluate_language_identification_enhanced(
                test_data, ground_truth
            )
            
            # FastText evaluation (if available)
            fasttext_results = {}
            if self.advanced_models.fasttext_langid:
                fasttext_results = self._evaluate_fasttext_performance(test_data, ground_truth)
            
            return {
                "basic_metrics": basic_results,
                "enhanced_metrics": enhanced_results,
                "fasttext_metrics": fasttext_results,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Language identification evaluation failed: {e}")
            return {"error": str(e)}
    
    def _evaluate_content_understanding(self, test_data: List[Dict], 
                                     ground_truth: List[Dict]) -> Dict:
        """Evaluate content understanding performance."""
        try:
            # Enhanced content understanding evaluation
            enhanced_results = self.enhanced_evaluator.evaluate_content_understanding_enhanced(
                test_data, ground_truth
            )
            
            # BLIP model evaluation (if available)
            blip_results = {}
            if self.advanced_models.blip_model:
                blip_results = self._evaluate_blip_performance(test_data, ground_truth)
            
            return {
                "enhanced_metrics": enhanced_results,
                "blip_metrics": blip_results,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Content understanding evaluation failed: {e}")
            return {"error": str(e)}
    
    def _evaluate_advanced_models(self, test_data: List[Dict], 
                                ground_truth: List[Dict]) -> Dict:
        """Evaluate advanced model performance."""
        try:
            results = {}
            
            # Model availability info
            model_info = self.advanced_models.get_model_info()
            results["model_availability"] = model_info
            
            # Performance metrics for each available model
            if model_info["models_loaded"]["LayoutLMv3"]:
                results["layoutlmv3"] = self._evaluate_layoutlmv3_performance(test_data, ground_truth)
            
            if model_info["models_loaded"]["Donut"]:
                results["donut"] = self._evaluate_donut_performance(test_data, ground_truth)
            
            if model_info["models_loaded"]["BLIP"]:
                results["blip"] = self._evaluate_blip_performance(test_data, ground_truth)
            
            if model_info["models_loaded"]["FastText"]:
                results["fasttext"] = self._evaluate_fasttext_performance(test_data, ground_truth)
            
            return results
            
        except Exception as e:
            logger.error(f"Advanced models evaluation failed: {e}")
            return {"error": str(e)}
    
    def _evaluate_layoutlmv3_performance(self, test_data: List[Dict], 
                                       ground_truth: List[Dict]) -> Dict:
        """Evaluate LayoutLMv3 performance."""
        try:
            # This would run LayoutLMv3 on test data and compare with ground truth
            # For now, return placeholder metrics
            return {
                "mAP50": 0.0,
                "mAP75": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "inference_time_per_sample": 0.0
            }
        except Exception as e:
            logger.error(f"LayoutLMv3 evaluation failed: {e}")
            return {"error": str(e)}
    
    def _evaluate_donut_performance(self, test_data: List[Dict], 
                                  ground_truth: List[Dict]) -> Dict:
        """Evaluate Donut performance."""
        try:
            # This would run Donut on test data and compare with ground truth
            # For now, return placeholder metrics
            return {
                "text_extraction_accuracy": 0.0,
                "inference_time_per_sample": 0.0
            }
        except Exception as e:
            logger.error(f"Donut evaluation failed: {e}")
            return {"error": str(e)}
    
    def _evaluate_blip_performance(self, test_data: List[Dict], 
                                 ground_truth: List[Dict]) -> Dict:
        """Evaluate BLIP performance."""
        try:
            # This would run BLIP on test data and compare with ground truth
            # For now, return placeholder metrics
            return {
                "caption_quality_score": 0.0,
                "inference_time_per_sample": 0.0
            }
        except Exception as e:
            logger.error(f"BLIP evaluation failed: {e}")
            return {"error": str(e)}
    
    def _evaluate_fasttext_performance(self, test_data: List[Dict], 
                                     ground_truth: List[Dict]) -> Dict:
        """Evaluate FastText performance."""
        try:
            # This would run FastText on test data and compare with ground truth
            # For now, return placeholder metrics
            return {
                "language_identification_accuracy": 0.0,
                "inference_time_per_sample": 0.0
            }
        except Exception as e:
            logger.error(f"FastText evaluation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_overall_performance(self) -> Dict:
        """Calculate overall system performance."""
        try:
            # Extract scores from different components
            layout_score = self.results.get("layout_detection", {}).get("enhanced_metrics", {}).get("overall_score", 0.0)
            text_score = self.results.get("text_extraction", {}).get("enhanced_metrics", {}).get("overall_score", 0.0)
            langid_score = self.results.get("language_identification", {}).get("enhanced_metrics", {}).get("overall_score", 0.0)
            
            # Calculate weighted overall score
            weights = {
                "layout_detection": 0.4,
                "text_extraction": 0.35,
                "language_identification": 0.25
            }
            
            overall_score = (
                layout_score * weights["layout_detection"] +
                text_score * weights["text_extraction"] +
                langid_score * weights["language_identification"]
            )
            
            return {
                "overall_score": overall_score,
                "component_scores": {
                    "layout_detection": layout_score,
                    "text_extraction": text_score,
                    "language_identification": langid_score
                },
                "weights": weights,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Overall performance calculation failed: {e}")
            return {"error": str(e)}
    
    def _compare_with_human_baselines(self) -> Dict:
        """Compare system performance with human baselines."""
        try:
            # Get human baselines from enhanced evaluator
            human_baselines = self.enhanced_evaluator.human_baselines
            
            # Get current performance
            current_performance = self.results.get("overall_performance", {})
            overall_score = current_performance.get("overall_score", 0.0)
            
            # Calculate percentage of human performance
            # Assuming human baseline is 1.0 (100%)
            human_percentage = overall_score * 100
            
            return {
                "human_baselines": human_baselines,
                "current_performance": current_performance,
                "human_percentage": human_percentage,
                "gap_to_human": 1.0 - overall_score,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Human baseline comparison failed: {e}")
            return {"error": str(e)}
    
    def generate_evaluation_report(self) -> str:
        """Generate comprehensive evaluation report."""
        try:
            report = []
            report.append("=" * 80)
            report.append("🎯 PS-05 COMPREHENSIVE EVALUATION REPORT")
            report.append("=" * 80)
            report.append("")
            
            # Overall Performance
            overall = self.results.get("overall_performance", {})
            if overall:
                report.append("📊 OVERALL SYSTEM PERFORMANCE")
                report.append("-" * 40)
                report.append(f"Overall Score: {overall.get('overall_score', 0.0):.4f}")
                report.append(f"Layout Detection: {overall.get('component_scores', {}).get('layout_detection', 0.0):.4f}")
                report.append(f"Text Extraction: {overall.get('component_scores', {}).get('text_extraction', 0.0):.4f}")
                report.append(f"Language ID: {overall.get('component_scores', {}).get('language_identification', 0.0):.4f}")
                report.append("")
            
            # Human Baseline Comparison
            baseline = self.results.get("human_baseline_comparison", {})
            if baseline:
                report.append("👥 HUMAN BASELINE COMPARISON")
                report.append("-" * 40)
                report.append(f"Human Performance: {baseline.get('human_percentage', 0.0):.1f}%")
                report.append(f"Gap to Human: {baseline.get('gap_to_human', 0.0):.4f}")
                report.append("")
            
            # Model Availability
            advanced = self.results.get("advanced_models", {})
            if advanced:
                model_info = advanced.get("model_availability", {})
                report.append("🤖 ADVANCED MODEL STATUS")
                report.append("-" * 40)
                for model, available in model_info.get("models_loaded", {}).items():
                    status = "✅ Available" if available else "❌ Not Available"
                    report.append(f"{model}: {status}")
                report.append("")
            
            # Detailed Metrics
            for component, results in self.results.items():
                if component in ["overall_performance", "human_baseline_comparison"]:
                    continue
                
                report.append(f"📋 {component.upper().replace('_', ' ')}")
                report.append("-" * 40)
                
                if "error" in results:
                    report.append(f"❌ Error: {results['error']}")
                else:
                    # Add key metrics
                    for metric_type, metrics in results.items():
                        if isinstance(metrics, dict) and "error" not in metrics:
                            report.append(f"  {metric_type}:")
                            for key, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    report.append(f"    {key}: {value:.4f}")
                                else:
                                    report.append(f"    {key}: {value}")
                
                report.append("")
            
            report.append("=" * 80)
            report.append(f"Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("=" * 80)
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return f"Error generating report: {e}"
    
    def save_results(self, output_path: str):
        """Save evaluation results to file."""
        try:
            # Save detailed results
            results_path = Path(output_path)
            results_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"✅ Detailed results saved to {results_path}")
            
            # Save evaluation report
            report_path = results_path.parent / f"{results_path.stem}_report.txt"
            report = self.generate_evaluation_report()
            
            with open(report_path, 'w') as f:
                f.write(report)
            
            logger.info(f"✅ Evaluation report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive PS-05 evaluation")
    parser.add_argument("--test-data", required=True, help="Path to test data JSON")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth JSON")
    parser.add_argument("--output", default="results/comprehensive_evaluation.json", 
                       help="Output path for results")
    parser.add_argument("--config", default="configs/ps05_config.yaml", 
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    try:
        # Load test data and ground truth
        logger.info("📂 Loading test data and ground truth...")
        
        with open(args.test_data, 'r') as f:
            test_data = json.load(f)
        
        with open(args.ground_truth, 'r') as f:
            ground_truth = json.load(f)
        
        logger.info(f"✅ Loaded {len(test_data)} test samples and {len(ground_truth)} ground truth samples")
        
        # Initialize evaluator
        evaluator = ComprehensiveEvaluator(args.config)
        
        # Run comprehensive evaluation
        results = evaluator.evaluate_system_comprehensive(test_data, ground_truth)
        
        # Save results
        evaluator.save_results(args.output)
        
        # Print summary
        print("\n" + "="*80)
        print("🎉 COMPREHENSIVE EVALUATION COMPLETED!")
        print("="*80)
        
        overall = results.get("overall_performance", {})
        if overall:
            print(f"📊 Overall System Score: {overall.get('overall_score', 0.0):.4f}")
        
        baseline = results.get("human_baseline_comparison", {})
        if baseline:
            print(f"👥 Human Performance: {baseline.get('human_percentage', 0.0):.1f}%")
        
        print(f"📁 Results saved to: {args.output}")
        print("📄 Evaluation report also generated")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Comprehensive evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
