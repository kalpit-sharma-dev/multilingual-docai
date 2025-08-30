#!/usr/bin/env python3
"""
PS-05 Main Entry Point

This is the main entry point for the PS-05 Document Understanding System.
It provides easy access to all functionality through a simple command-line interface.
"""

import argparse
import sys
import os
from pathlib import Path

def main():
    """Main entry point for PS-05 system."""
    parser = argparse.ArgumentParser(
        description="PS-05 Document Understanding System - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Run the complete 3-stage pipeline
  python ps05.py pipeline --stage all

  # Run individual stages
  python ps05.py pipeline --stage 1  # Layout detection
  python ps05.py pipeline --stage 2  # Text extraction  
  python ps05.py pipeline --stage 3  # Content understanding

  # Data cleaning and EDA
  python ps05.py clean --input data/train --output results/cleaned --mode cleaning_with_eda
  python ps05.py clean --input data/train --output results/eda --mode eda_only

  # Training pipeline
  python ps05.py train --prepare-data --input data/train --output data/training_prepared
  python ps05.py train --train-model

  # Backend API
  python ps05.py backend --start
  python ps05.py backend --status

  # Utilities
  python ps05.py cleanup
  python ps05.py pack-submission
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run the 3-stage pipeline')
    pipeline_parser.add_argument('--stage', choices=['1', '2', '3', 'all'], 
                                default='all', help='Pipeline stage to run')
    pipeline_parser.add_argument('--input', default='data/test', 
                                help='Input directory (default: data/test)')
    pipeline_parser.add_argument('--output', default='results', 
                                help='Output directory (default: results)')
    
    # Cleaning command
    clean_parser = subparsers.add_parser('clean', help='Data cleaning and EDA')
    clean_parser.add_argument('--input', required=True, help='Input data directory')
    clean_parser.add_argument('--output', required=True, help='Output directory for results')
    clean_parser.add_argument('--mode', choices=['eda_only', 'cleaning_with_eda'], 
                             default='cleaning_with_eda', help='Cleaning mode')
    clean_parser.add_argument('--dataset-type', choices=['auto', 'images', 'documents', 'mixed'],
                             default='auto', help='Dataset type for cleaning')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Training pipeline')
    train_parser.add_argument('--prepare-data', action='store_true', 
                             help='Prepare training data')
    train_parser.add_argument('--train-model', action='store_true', 
                             help='Train YOLO model')
    train_parser.add_argument('--input', help='Input data directory')
    train_parser.add_argument('--output', help='Output directory')
    
    # Backend command
    backend_parser = subparsers.add_parser('backend', help='Backend API management')
    backend_parser.add_argument('--start', action='store_true', 
                               help='Start the backend server')
    backend_parser.add_argument('--status', action='store_true', 
                               help='Check backend status')
    backend_parser.add_argument('--port', default=8000, type=int, 
                               help='Port for backend server')
    
    # Utilities command
    utils_parser = subparsers.add_parser('utils', help='Utility functions')
    utils_parser.add_argument('--cleanup', action='store_true', 
                             help='Clean up repository')
    utils_parser.add_argument('--pack-submission', action='store_true', 
                             help='Pack submission files')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute commands
    try:
        if args.command == 'pipeline':
            run_pipeline(args)
        elif args.command == 'clean':
            run_cleaning(args)
        elif args.command == 'train':
            run_training(args)
        elif args.command == 'backend':
            run_backend(args)
        elif args.command == 'utils':
            run_utilities(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def run_pipeline(args):
    """Run the 3-stage pipeline."""
    print(f"🚀 Running PS-05 Pipeline - Stage: {args.stage}")
    
    # Import and run the pipeline
    try:
        sys.path.append('scripts/core')
        from run_stages import main as run_stages_main
        
        # Set up arguments for run_stages
        sys.argv = [
            'run_stages.py',
            '--stage', args.stage,
            '--input', args.input,
            '--output', args.output
        ]
        
        run_stages_main()
        
    except ImportError as e:
        print(f"❌ Failed to import pipeline: {e}")
        print("💡 Make sure you're in the project root directory")
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")

def run_cleaning(args):
    """Run data cleaning and EDA."""
    print(f"🧹 Running Data Cleaning - Mode: {args.mode}")
    
    try:
        sys.path.append('scripts/cleaning')
        from eda_with_cleaning import main as cleaning_main
        
        # Set up arguments for cleaning script
        sys.argv = [
            'eda_with_cleaning.py',
            '--input', args.input,
            '--output', args.output,
            '--mode', args.mode,
            '--dataset-type', args.dataset_type
        ]
        
        cleaning_main()
        
    except ImportError as e:
        print(f"❌ Failed to import cleaning script: {e}")
        print("💡 Make sure you're in the project root directory")
    except Exception as e:
        print(f"❌ Cleaning execution failed: {e}")

def run_training(args):
    """Run training pipeline."""
    print("🎯 Running Training Pipeline")
    
    try:
        if args.prepare_data:
            print("📊 Preparing training data...")
            sys.path.append('scripts/training')
            from prepare_training_data import main as prep_main
            
            sys.argv = [
                'prepare_training_data.py',
                '--input', args.input or 'data/train',
                '--output', args.output or 'data/training_prepared'
            ]
            prep_main()
        
        if args.train_model:
            print("🏋️ Training YOLO model...")
            sys.path.append('scripts/training')
            from train_stage1 import main as train_main
            
            sys.argv = ['train_stage1.py']
            train_main()
            
    except ImportError as e:
        print(f"❌ Failed to import training script: {e}")
        print("💡 Make sure you're in the project root directory")
    except Exception as e:
        print(f"❌ Training execution failed: {e}")

def run_backend(args):
    """Manage backend API."""
    if args.start:
        print("🚀 Starting PS-05 Backend Server...")
        print(f"📡 Server will be available at: http://localhost:{args.port}")
        print("💡 Use Ctrl+C to stop the server")
        
        try:
            os.chdir('backend')
            os.system(f'python -m uvicorn app.main:app --host 0.0.0.0 --port {args.port}')
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
    
    elif args.status:
        print("📊 Backend Status:")
        print("💡 To check status, start the backend first with: python ps05.py backend --start")

def run_utilities(args):
    """Run utility functions."""
    if args.cleanup:
        print("🧹 Running repository cleanup...")
        try:
            sys.path.append('scripts/utilities')
            from cleanup import main as cleanup_main
            cleanup_main()
        except ImportError as e:
            print(f"❌ Failed to import cleanup script: {e}")
    
    elif args.pack_submission:
        print("📦 Packing submission files...")
        try:
            sys.path.append('scripts/utilities')
            from pack_submission import main as pack_main
            pack_main()
        except ImportError as e:
            print(f"❌ Failed to import pack script: {e}")

if __name__ == "__main__":
    main() 