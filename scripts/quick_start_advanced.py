#!/usr/bin/env python3
"""
Quick Start Script for Advanced PS-05 Features

Demonstrates the new advanced capabilities:
- DocLayNet dataset ingestion
- Advanced model integration
- Enhanced evaluation
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are available."""
    logger.info("🔍 Checking dependencies...")
    
    missing_deps = []
    
    # Check for advanced models
    try:
        import transformers
        logger.info("✅ Transformers available")
    except ImportError:
        missing_deps.append("transformers")
        logger.warning("❌ Transformers not available")
    
    try:
        import datasets
        logger.info("✅ Datasets available")
    except ImportError:
        missing_deps.append("datasets")
        logger.warning("❌ Datasets not available")
    
    try:
        import fasttext
        logger.info("✅ FastText available")
    except ImportError:
        missing_deps.append("fasttext")
        logger.warning("❌ FastText not available")
    
    try:
        import bert_score
        logger.info("✅ BERTScore available")
    except ImportError:
        missing_deps.append("bert-score")
        logger.warning("❌ BERTScore not available")
    
    if missing_deps:
        logger.warning(f"⚠️  Missing dependencies: {', '.join(missing_deps)}")
        logger.info("💡 Install with: pip install " + " ".join(missing_deps))
        return False
    
    logger.info("✅ All dependencies available!")
    return True

def download_doclaynet():
    """Download DocLayNet dataset."""
    logger.info("📥 Downloading DocLayNet dataset...")
    
    try:
        from scripts.ingest_datasets import DatasetIngester
        
        ingester = DatasetIngester()
        success = ingester.download_doclaynet(use_hf=True)
        
        if success:
            logger.info("✅ DocLayNet downloaded successfully!")
            return True
        else:
            logger.error("❌ DocLayNet download failed!")
            return False
            
    except Exception as e:
        logger.error(f"Failed to download DocLayNet: {e}")
        return False

def test_advanced_models():
    """Test advanced model integration."""
    logger.info("🤖 Testing advanced models...")
    
    try:
        from core.models.advanced_models import get_advanced_models
        
        models = get_advanced_models()
        model_info = models.get_model_info()
        
        logger.info("📊 Model Status:")
        for model, available in model_info["models_loaded"].items():
            status = "✅ Available" if available else "❌ Not Available"
            logger.info(f"  {model}: {status}")
        
        return model_info["models_loaded"]
        
    except Exception as e:
        logger.error(f"Failed to test advanced models: {e}")
        return {}

def test_enhanced_evaluation():
    """Test enhanced evaluation capabilities."""
    logger.info("📊 Testing enhanced evaluation...")
    
    try:
        from core.evaluation.enhanced_evaluator import get_enhanced_evaluator
        
        evaluator = get_enhanced_evaluator()
        
        # Test human baselines
        baselines = evaluator.human_baselines
        logger.info("👥 Human Baselines:")
        for task, metrics in baselines.items():
            logger.info(f"  {task}:")
            for metric, value in metrics.items():
                logger.info(f"    {metric}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to test enhanced evaluation: {e}")
        return False

def create_sample_data():
    """Create sample data for testing."""
    logger.info("📝 Creating sample test data...")
    
    sample_data = [
        {
            "id": "sample_1",
            "type": "Text",
            "bbox": [100, 200, 300, 50],
            "text": "This is a sample text in English",
            "language": "en",
            "confidence": 0.95
        },
        {
            "id": "sample_2",
            "type": "Table",
            "bbox": [400, 300, 200, 150],
            "description": "Sample table with data",
            "confidence": 0.88
        },
        {
            "id": "sample_3",
            "type": "Figure",
            "bbox": [600, 400, 100, 100],
            "description": "Sample image or chart",
            "confidence": 0.92
        }
    ]
    
    # Save sample data
    output_dir = Path("data/sample")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(output_dir / "test_data.json", 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    logger.info(f"✅ Sample data saved to {output_dir / 'test_data.json'}")
    return str(output_dir / "test_data.json")

def run_comprehensive_evaluation(test_data_path: str):
    """Run comprehensive evaluation on sample data."""
    logger.info("🎯 Running comprehensive evaluation...")
    
    try:
        from scripts.comprehensive_evaluation import ComprehensiveEvaluator
        
        # Create ground truth (same as test data for demo)
        import json
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        ground_truth = test_data  # For demo purposes
        
        # Initialize evaluator
        evaluator = ComprehensiveEvaluator()
        
        # Run evaluation
        results = evaluator.evaluate_system_comprehensive(test_data, ground_truth)
        
        # Save results
        output_path = "results/quick_start_evaluation.json"
        evaluator.save_results(output_path)
        
        logger.info(f"✅ Evaluation completed! Results saved to {output_path}")
        
        # Print summary
        overall = results.get("overall_performance", {})
        if overall:
            logger.info(f"📊 Overall Score: {overall.get('overall_score', 0.0):.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Comprehensive evaluation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Quick start for advanced PS-05 features")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation")
    
    args = parser.parse_args()
    
    logger.info("🚀 PS-05 Advanced Features Quick Start")
    logger.info("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        logger.warning("⚠️  Some dependencies are missing. Advanced features may not work.")
    
    # Download DocLayNet (if not skipped)
    if not args.skip_download:
        if not download_doclaynet():
            logger.error("❌ Dataset download failed. Exiting.")
            return
    
    # Test advanced models
    model_status = test_advanced_models()
    
    # Test enhanced evaluation
    if not test_enhanced_evaluation():
        logger.error("❌ Enhanced evaluation test failed.")
    
    # Create sample data
    test_data_path = create_sample_data()
    
    # Run comprehensive evaluation (if not skipped)
    if not args.skip_evaluation:
        if not run_comprehensive_evaluation(test_data_path):
            logger.error("❌ Comprehensive evaluation failed.")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("🎉 QUICK START COMPLETED!")
    logger.info("=" * 50)
    
    if model_status:
        available_models = sum(model_status.values())
        total_models = len(model_status)
        logger.info(f"🤖 Models: {available_models}/{total_models} available")
    
    logger.info("📁 Sample data created")
    logger.info("📊 Evaluation framework ready")
    logger.info("🚀 Ready for advanced document understanding!")
    
    # Next steps
    logger.info("\n📋 Next Steps:")
    logger.info("1. Download more datasets: python scripts/ingest_datasets.py --dataset all")
    logger.info("2. Train advanced models with DocLayNet")
    logger.info("3. Run comprehensive evaluation on real data")
    logger.info("4. Integrate with your existing pipeline")

if __name__ == "__main__":
    main()
