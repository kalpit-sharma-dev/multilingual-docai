#!/usr/bin/env python3
"""
Deprecated: Use scripts/training/prepare_dataset.py instead.

This script is retained as a thin wrapper for backward compatibility.
It forwards to prepare_dataset.prepare_dataset with equivalent defaults.
"""

import json
import os
from pathlib import Path
import shutil
from typing import Dict, List
import logging

from scripts.training.prepare_dataset import prepare_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPreparer:
    """Deprecated wrapper that forwards to prepare_dataset.prepare_dataset."""

    def __init__(self, input_dir: str = "data/train", output_dir: str = "data/yolo_dataset"):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def prepare_dataset(self) -> bool:
        # Forward to canonical implementation
        result = prepare_dataset(self.input_dir, self.output_dir)
        return bool(result)

def main():
    """Main function."""
    preparer = DatasetPreparer()
    success = preparer.prepare_dataset()
    if success:
        logger.info("🎉 Dataset is ready for training! (wrapper)")
        logger.info("📁 See output directory for dataset.yaml")
    else:
        logger.error("❌ Dataset preparation failed! (wrapper)")

if __name__ == "__main__":
    main()
