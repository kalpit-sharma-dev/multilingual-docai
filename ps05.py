#!/usr/bin/env python3
"""
PS-05 Intelligent Multilingual Document Understanding System

Main command-line interface for the PS-05 document understanding system.
Supports training, inference, evaluation, and deployment.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import subprocess

# Import our modules
from src.pipeline.infer_page import PS05Pipeline, process_batch
from src.evaluation.layout_evaluator import evaluate_layout
from scripts.train_layout import main as train_layout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_inference(args):
    """Run inference on document images."""
    try:
        pipeline = PS05Pipeline(args.config)
        
        if args.batch:
            # Batch processing
            image_paths = list(Path(args.input).glob("*.png")) + list(Path(args.input).glob("*.jpg"))
            if not image_paths:
                logger.error(f"No images found in {args.input}")
                return
            
            logger.info(f"Processing {len(image_paths)} images in batch mode")
            results = process_batch(
                [str(p) for p in image_paths],
                args.config,
                args.stage,
                args.output
            )
            
            logger.info(f"Batch processing completed. Results saved to {args.output}")
            
        else:
            # Single image processing
            if not Path(args.input).exists():
                logger.error(f"Input file not found: {args.input}")
                return
            
            logger.info(f"Processing single image: {args.input}")
            result = pipeline.process_image(args.input, args.stage)
            
            # Save result
            Path(args.output).mkdir(parents=True, exist_ok=True)
            output_path = Path(args.output) / "result.json"
            
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Processing completed. Result saved to {output_path}")
            logger.info(f"Processing time: {result.get('processing_time', 0):.2f}s")
            
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)

def run_training(args):
    """Run model training."""
    try:
        logger.info("Starting model training...")
        
        # Set up training arguments
        sys.argv = [
            'train_layout.py',
            '--config', args.config,
            '--data', args.data,
            '--output', args.output,
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size)
        ]
        
        if args.validate:
            sys.argv.append('--validate')
        
        # Run training
        train_layout()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

def run_evaluation(args):
    """Run model evaluation."""
    try:
        logger.info("Starting model evaluation...")
        
        # Load predictions and ground truth
        import json
        
        with open(args.predictions, 'r') as f:
            predictions = json.load(f)
        
        with open(args.ground_truth, 'r') as f:
            ground_truth = json.load(f)
        
        # Run evaluation
        results = evaluate_layout(predictions, ground_truth, args.output)
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to {args.output}")
        
        # Print summary
        if 'mAP' in results:
            logger.info(f"Overall mAP: {results['mAP']:.3f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

def run_server(args):
    """Run the FastAPI server."""
    try:
        import uvicorn
        from backend.app.main import app
        
        logger.info(f"Starting PS-05 API server on {args.host}:{args.port}")
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers
        )
        
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PS-05 Intelligent Multilingual Document Understanding System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference on a single image
  python ps05.py infer --input image.png --output results/ --stage 2
  
  # Process a batch of images
  python ps05.py infer --input data/images/ --output results/ --batch --stage 3
  
  # Train the layout detection model
  python ps05.py train --data data/train/ --output models/ --epochs 100
  
  # Evaluate model performance
  python ps05.py eval --predictions preds.json --ground-truth gt.json --output eval_results.json
  
  # Start the API server
  python ps05.py server --host 0.0.0.0 --port 8000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference on documents')
    infer_parser.add_argument('--input', required=True, help='Input image or directory')
    infer_parser.add_argument('--output', default='outputs', help='Output directory')
    infer_parser.add_argument('--config', default='configs/ps05_config.yaml', help='Configuration file')
    infer_parser.add_argument('--stage', type=int, default=1, choices=[1, 2, 3], 
                             help='Processing stage (1: Layout, 2: +OCR, 3: +NL)')
    infer_parser.add_argument('--batch', action='store_true', help='Batch processing mode')
    infer_parser.set_defaults(func=run_inference)
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--data', required=True, help='Training data directory')
    train_parser.add_argument('--output', default='models', help='Output directory')
    train_parser.add_argument('--config', default='configs/ps05_config.yaml', help='Configuration file')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    train_parser.add_argument('--validate', action='store_true', help='Run validation after training')
    train_parser.set_defaults(func=run_training)
    
    # Evaluation command
    eval_parser = subparsers.add_parser('eval', help='Evaluate model performance')
    eval_parser.add_argument('--predictions', required=True, help='Predictions JSON file')
    eval_parser.add_argument('--ground-truth', required=True, help='Ground truth JSON file')
    eval_parser.add_argument('--output', required=True, help='Output results file')
    eval_parser.set_defaults(func=run_evaluation)
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument('--host', default='0.0.0.0', help='Server host')
    server_parser.add_argument('--port', type=int, default=8000, help='Server port')
    server_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    server_parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    server_parser.set_defaults(func=run_server)
    
    # Submission packager
    pack_parser = subparsers.add_parser("pack", help="Create submission package (zip)")
    pack_parser.add_argument("--input", required=True, help="Image file or directory")
    pack_parser.add_argument("--output", required=True, help="Output directory")
    pack_parser.add_argument("--stage", type=int, default=3, choices=[1,2,3])
    pack_parser.add_argument("--zip-name", default="submission.zip")
    pack_parser.add_argument("--config", default=None)

    # Overlay viewer
    overlay_parser = subparsers.add_parser("overlay", help="Create overlay image for QA")
    overlay_parser.add_argument("--image", required=True)
    overlay_parser.add_argument("--json", required=True)
    overlay_parser.add_argument("--stage", type=int, default=3, choices=[1,2,3])
    overlay_parser.add_argument("--out", required=True)

    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Run the appropriate function
    if args.command == "pack":
        cmd = [
            sys.executable, "scripts/pack_submission.py",
            "--input", args.input,
            "--output", args.output,
            "--stage", str(args.stage),
            "--zip-name", args.zip_name
        ]
        if args.config:
            cmd.extend(["--config", args.config])
        subprocess.run(cmd, check=True)
    elif args.command == "overlay":
        cmd = [
            sys.executable, "scripts/overlay_viewer.py",
            "--image", args.image,
            "--json", args.json,
            "--stage", str(args.stage),
            "--out", args.out
        ]
        subprocess.run(cmd, check=True)
    else:
        args.func(args)

if __name__ == "__main__":
    main() 