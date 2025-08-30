#!/usr/bin/env python3
"""
Cleanup Script for PS-05 Repository

Removes temporary files and organizes the repository.
"""

import os
import shutil
from pathlib import Path

def cleanup_repository():
    """Clean up the repository."""
    print("🧹 Cleaning up PS-05 repository...")
    
    # Remove temporary files and directories
    temp_dirs = [
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        ".mypy_cache"
    ]
    
    # Remove cache directories
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name in ["__pycache__", ".pytest_cache", ".mypy_cache"]:
                cache_path = Path(root) / dir_name
                print(f"🗑️  Removing: {cache_path}")
                shutil.rmtree(cache_path, ignore_errors=True)
    
    # Remove Python cache files
    for root, dirs, files in os.walk("."):
        for file_name in files:
            if file_name.endswith((".pyc", ".pyo", ".pyd")):
                file_path = Path(root) / file_name
                print(f"🗑️  Removing: {file_path}")
                file_path.unlink(missing_ok=True)
    
    # Create clean directory structure
    clean_dirs = [
        "data/test",
        "results",
        "models"
    ]
    
    for dir_path in clean_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created: {dir_path}")
    
    print("✅ Cleanup completed!")
    print("\n📋 Repository is now clean and organized!")
    print("🎯 Use 'python ps05.py --help' to get started")

def main():
    """Main function for the cleanup script."""
    cleanup_repository()

if __name__ == "__main__":
    main()
