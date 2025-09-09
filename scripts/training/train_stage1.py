#!/usr/bin/env python3
"""
Stage 1 Training (Enhanced Wrapper)

Preserves previous functionality (prep → train → optional validate → optional submission)
while delegating core training to the canonical YOLO trainer and dataset preparer.
"""

import argparse
import logging
import json
import shutil
from pathlib import Path
from typing import Dict
import os
import sys

import torch

# Ensure local training scripts are importable when running directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    # Prefer local sibling modules
    from prepare_dataset import prepare_dataset
    from train_yolo import train_yolo_model, load_config
except Exception:
    # Fallback if repo is installed as a package with a different layout
    from scripts.training.prepare_dataset import prepare_dataset  # type: ignore
    from scripts.training.train_yolo import train_yolo_model, load_config  # type: ignore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _log_gpu_info():
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {name} | Mem: {mem:.2f} GB | CUDA: {torch.version.cuda}")
        else:
            logger.warning("CUDA not available, using CPU")
    except Exception:
        pass


def _find_trained_weights(output_dir: Path) -> Path:
    # Ultralytics default path: <output>/layout_detection/weights/best.pt
    candidate = output_dir / 'layout_detection' / 'weights' / 'best.pt'
    return candidate if candidate.exists() else Path("")


def _write_submission(model_path: Path, output_dir: Path, classes: list) -> str:
    sub_dir = output_dir / 'stage1_submission'
    sub_dir.mkdir(parents=True, exist_ok=True)

    # Copy model
    dest = sub_dir / model_path.name
    shutil.copy2(model_path, dest)

    # Info JSON
    info = {
        "stage": 1,
        "description": "PS-05 Stage 1 Layout Detection Model",
        "classes": classes,
        "model_file": model_path.name,
        "evaluation_metric": "mAP@0.5",
        "output_format": "JSON per image with bbox [x,y,h,w], class and class_id"
    }
    with open(sub_dir / 'submission_info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)

    # README (minimal)
    readme = sub_dir / 'README.md'
    readme.write_text(
        (
            "# PS-05 Stage 1 Submission\n\n"
            "This package contains the trained Stage-1 model (YOLO) for layout detection.\n\n"
            "- Classes: {classes}\n"
            "- Metric: mAP@0.5\n\n"
            "Outputs for each image are JSON files with bbox [x,y,h,w], class and class_id.\n"
        ).format(classes=", ".join(classes)),
        encoding='utf-8'
    )
    return str(sub_dir)


def main():
    parser = argparse.ArgumentParser(description="Train Stage 1 layout detection model")
    parser.add_argument('--data', required=True, help='Input data directory (images + per-image JSON)')
    parser.add_argument('--output', required=True, help='Output directory root')
    parser.add_argument('--config', default='configs/ps05_config.yaml', help='Configuration file')

    # Training hyperparams (forwarded)
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=0, help='Override batch size (0 = use config/default)')
    parser.add_argument('--lr0', type=float, default=0.0, help='Override initial learning rate (0 = use config/default)')
    parser.add_argument('--imgsz', type=int, default=640, help='Training image size')
    parser.add_argument('--weights', type=str, default='', help='Starting weights for fine-tuning')
    parser.add_argument('--workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--device', type=str, default='0', help="CUDA device string/index (e.g., '0' or 'cpu')")

    # Pipeline toggles
    parser.add_argument('--skip-dataset-prep', action='store_true', help='Skip dataset preparation step')
    parser.add_argument('--skip-validation', action='store_true', help='Skip post-training validation log')
    parser.add_argument('--create-submission', action='store_true', help='Create submission package with trained weights')

    args = parser.parse_args()

    # Load config & classes
    config: Dict = load_config(args.config) or {}
    classes = (
        config.get('models', {})
              .get('layout', {})
              .get('classes', ['Background', 'Text', 'Title', 'List', 'Table', 'Figure'])
    )

    # Output dirs
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log_gpu_info()

    # Step 1: Prepare dataset
    dataset_yaml = None
    if not args.skip_dataset_prep:
        logger.info("Preparing dataset via prepare_dataset.py ...")
        dataset_yaml = prepare_dataset(args.data, str(out_dir / 'dataset'))
        if not dataset_yaml:
            candidate = out_dir / 'dataset' / 'dataset.yaml'
            if candidate.exists():
                dataset_yaml = str(candidate)
            else:
                logger.error("Dataset preparation failed and dataset.yaml not found")
                return
    else:
        candidate = out_dir / 'dataset' / 'dataset.yaml'
        if not candidate.exists():
            logger.error(f"--skip-dataset-prep set but missing {candidate}")
            return
        dataset_yaml = str(candidate)

    # Step 2: Train
    logger.info("Starting training (delegated to train_yolo.py) ...")
    ok = train_yolo_model(
        config=config,
        dataset_yaml=dataset_yaml,
        output_dir=str(out_dir),
        epochs=args.epochs,
        weights=args.weights,
        imgsz=args.imgsz,
        batch_override=args.batch,
        lr0_override=args.lr0,
        workers=args.workers,
        device_str=args.device,
    )

    if not ok:
        logger.error("Training failed")
        return

    # Step 3: Validation (lightweight logging)
    if not args.skip_validation:
        weights_path = _find_trained_weights(out_dir)
        if weights_path:
            logger.info(f"Best weights: {weights_path}")
        else:
            logger.warning("Could not locate best.pt; check training logs for metrics")

    # Step 4: Optional submission package
    if args.create_submission:
        weights_path = _find_trained_weights(out_dir)
        if not weights_path:
            logger.error("Cannot create submission: best.pt not found")
            return
        sub_path = _write_submission(weights_path, out_dir, classes)
        logger.info(f"Submission package created at: {sub_path}")


if __name__ == "__main__":
    main()


