#!/usr/bin/env python3
"""
Deprecated wrapper: use scripts/training/prepare_dataset.py instead.

This file is kept to avoid breaking existing workflows/imports.
It forwards to the canonical data preparer.
"""

import argparse
import logging
from pathlib import Path

from scripts.training.prepare_dataset import prepare_dataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="(Deprecated) Prepare dataset wrapper")
    parser.add_argument('--data', required=True, help='Input data directory')
    parser.add_argument('--output', required=True, help='Output directory')
    args = parser.parse_args()

    logger.warning("prepare_current_dataset.py is deprecated; forwarding to prepare_dataset.py")
    result = prepare_dataset(args.data, args.output)
    if result:
        logger.info("Dataset prepared successfully (wrapper)")
    else:
        logger.error("Dataset preparation failed (wrapper)")


if __name__ == "__main__":
    main()


