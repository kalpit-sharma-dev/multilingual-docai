"""
Unified Cleaning Service for PS-05

This service integrates both image and document cleaning to provide
a comprehensive data preparation pipeline for the PS-05 project.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
from datetime import datetime

from .image_cleaner import ImageCleaningService
try:
    from .document_cleaner import DocumentCleaningService  # type: ignore
except Exception as e:  # pragma: no cover
    DocumentCleaningService = None  # type: ignore
    logger = logging.getLogger(__name__)
    logger.warning(f"DocumentCleaningService unavailable, document cleaning will be disabled: {e}")
from .eda_service import DatasetEDA

logger = logging.getLogger(__name__)

class UnifiedCleaningService:
    """Unified service for cleaning both images and documents."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.image_cleaner = ImageCleaningService(self.config.get("image_cleaning", {}))
        self.document_cleaner = None
        try:
            if DocumentCleaningService is not None:
                self.document_cleaner = DocumentCleaningService(self.config.get("document_cleaning", {}))
        except Exception as e:
            logger.warning(f"Document cleaner initialization failed; disabling document cleaning: {e}")
        self.cleaning_log = []
        self.eda_service: Optional[DatasetEDA] = None
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for unified cleaning."""
        return {
            "image_cleaning": {
                "target_size": (800, 1000),
                "min_resolution": (100, 100),
                "max_file_size_mb": 50,
                "noise_reduction": True,
                "exif_normalization": True,
                "augmentation": True,
                "supported_formats": ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
            },
            "document_cleaning": {
                "supported_formats": ['.pdf', '.docx', '.pptx', '.txt'],
                "encoding": 'utf-8',
                "min_text_length": 50,
                "remove_boilerplate": True,
                "normalize_text": True,
                "language_detection": True,
                "structure_recovery": True,
                "deduplication": True,
                "output_format": "json"
            },
            "workflow": {
                "clean_images_first": True,
                "clean_documents_first": False,
                "parallel_processing": False,
                "save_intermediate_results": True,
                "generate_combined_report": True
            }
        }
    
    def clean_dataset(
        self, 
        input_dir: Path, 
        output_dir: Path,
        dataset_type: str = "auto",
        run_eda: bool = True
    ) -> Dict:
        """
        Clean entire dataset (images and/or documents) based on content.
        
        Args:
            input_dir: Directory containing raw data
            output_dir: Directory for cleaned data
            dataset_type: "auto", "images", "documents", or "mixed"
            
        Returns:
            Dictionary with comprehensive cleaning results
        """
        try:
            logger.info(f"Starting unified dataset cleaning for {input_dir}")
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Detect dataset type if auto
            if dataset_type == "auto":
                dataset_type = self._detect_dataset_type(input_dir)
            
            logger.info(f"Detected dataset type: {dataset_type}")
            
            # Initialize results
            results = {
                "dataset_type": dataset_type,
                "input_directory": str(input_dir),
                "output_directory": str(output_dir),
                "cleaning_timestamp": datetime.now().isoformat(),
                "eda_before_cleaning": {},
                "image_cleaning": {},
                "document_cleaning": {},
                "combined_statistics": {},
                "cleaning_log": []
            }
            
            # Run EDA before cleaning if enabled
            if run_eda:
                results["eda_before_cleaning"] = self._run_eda_analysis(input_dir, "before_cleaning")
            
            # Clean based on dataset type
            if dataset_type in ["images", "mixed"]:
                results["image_cleaning"] = self._clean_images(input_dir, output_dir)
            
            if dataset_type in ["documents", "mixed"]:
                results["document_cleaning"] = self._clean_documents(input_dir, output_dir)
            
            # Run EDA after cleaning if enabled
            if run_eda:
                results["eda_after_cleaning"] = self._run_eda_analysis(output_dir, "after_cleaning")
            
            # Generate combined statistics
            results["combined_statistics"] = self._generate_combined_statistics(results)
            
            # Save comprehensive report
            self._save_comprehensive_report(output_dir, results)
            
            logger.info("Unified dataset cleaning completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Unified dataset cleaning failed: {e}")
            raise
    
    def _detect_dataset_type(self, input_dir: Path) -> str:
        """Automatically detect if directory contains images, documents, or both."""
        image_count = 0
        document_count = 0
        
        # Count image files
        image_extensions = self.config["image_cleaning"]["supported_formats"]
        for ext in image_extensions:
            image_count += len(list(input_dir.glob(f"*{ext}")))
            image_count += len(list(input_dir.glob(f"*{ext.upper()}")))
        
        # Count document files
        document_extensions = self.config["document_cleaning"]["supported_formats"]
        for ext in document_extensions:
            document_count += len(list(input_dir.glob(f"*{ext}")))
            document_count += len(list(input_dir.glob(f"*{ext.upper()}")))
        
        logger.info(f"Detected {image_count} images and {document_count} documents")
        
        if image_count > 0 and document_count > 0:
            return "mixed"
        elif image_count > 0:
            return "images"
        elif document_count > 0:
            return "documents"
        else:
            return "unknown"
    
    def _clean_images(self, input_dir: Path, output_dir: Path) -> Dict:
        """Clean image dataset."""
        logger.info("Starting image cleaning...")
        
        # Create image output directory
        image_output_dir = output_dir / "cleaned_images"
        
        try:
            # Check for annotations
            annotations_file = input_dir / "annotations.json"
            if not annotations_file.exists():
                annotations_file = None
            
            # Clean images
            image_results = self.image_cleaner.clean_dataset(
                input_dir=input_dir,
                output_dir=image_output_dir,
                annotations_file=annotations_file
            )
            
            logger.info("Image cleaning completed")
            return image_results
            
        except Exception as e:
            logger.error(f"Image cleaning failed: {e}")
            return {"error": str(e)}
    
    def _clean_documents(self, input_dir: Path, output_dir: Path) -> Dict:
        """Clean document dataset."""
        logger.info("Starting document cleaning...")
        if self.document_cleaner is None:
            logger.warning("Document cleaning is unavailable (langdetect or dependencies missing)")
            return {"error": "Document cleaning unavailable: missing optional dependency"}
        
        # Create document output directory
        document_output_dir = output_dir / "cleaned_documents"
        
        try:
            # Clean documents
            document_results = self.document_cleaner.clean_dataset(
                input_dir=input_dir,
                output_dir=document_output_dir
            )
            
            logger.info("Document cleaning completed")
            return document_results
            
        except Exception as e:
            logger.error(f"Document cleaning failed: {e}")
            return {"error": str(e)}
    
    def _generate_combined_statistics(self, results: Dict) -> Dict:
        """Generate combined statistics from both cleaning processes."""
        combined_stats = {
            "total_input_files": 0,
            "total_cleaned_files": 0,
            "total_removed_files": 0,
            "cleaning_efficiency": 0.0,
            "file_type_breakdown": {},
            "language_distribution": {},
            "quality_metrics": {}
        }
        
        # Aggregate image statistics
        if "image_cleaning" in results and results["image_cleaning"]:
            img_stats = results["image_cleaning"].get("cleaning_summary", {})
            combined_stats["total_input_files"] += img_stats.get("total_images", 0)
            combined_stats["total_cleaned_files"] += img_stats.get("cleaned_images", 0)
            combined_stats["total_removed_files"] += (
                img_stats.get("removed_corrupt", 0) +
                img_stats.get("removed_duplicates", 0) +
                img_stats.get("removed_low_res", 0) +
                img_stats.get("removed_outliers", 0)
            )
            
            # Add image quality metrics
            combined_stats["quality_metrics"]["image_quality"] = {
                "corruption_rate": img_stats.get("removed_corrupt", 0) / max(img_stats.get("total_images", 1), 1),
                "duplication_rate": img_stats.get("removed_duplicates", 0) / max(img_stats.get("total_images", 1), 1),
                "augmentation_count": img_stats.get("augmented_images", 0)
            }
        
        # Aggregate document statistics
        if "document_cleaning" in results and results["document_cleaning"]:
            doc_stats = results["document_cleaning"].get("cleaning_summary", {})
            combined_stats["total_input_files"] += doc_stats.get("total_documents", 0)
            combined_stats["total_cleaned_files"] += doc_stats.get("cleaned_documents", 0)
            combined_stats["total_removed_files"] += (
                doc_stats.get("removed_corrupt", 0) +
                doc_stats.get("removed_duplicates", 0)
            )
            
            # Add document quality metrics
            combined_stats["quality_metrics"]["document_quality"] = {
                "corruption_rate": doc_stats.get("removed_corrupt", 0) / max(doc_stats.get("total_documents", 1), 1),
                "duplication_rate": doc_stats.get("removed_duplicates", 0) / max(doc_stats.get("total_documents", 1), 1),
                "language_diversity": len(doc_stats.get("language_detected", {}))
            }
            
            # Add language distribution
            combined_stats["language_distribution"] = doc_stats.get("language_detected", {})
        
        # Calculate overall efficiency
        if combined_stats["total_input_files"] > 0:
            combined_stats["cleaning_efficiency"] = (
                combined_stats["total_cleaned_files"] / combined_stats["total_input_files"]
            ) * 100
        
        return combined_stats
    
    def _save_comprehensive_report(self, output_dir: Path, results: Dict):
        """Save comprehensive cleaning report."""
        report_path = output_dir / "comprehensive_cleaning_report.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comprehensive report saved: {report_path}")
    
    def get_cleaning_status(self, dataset_id: str) -> Dict:
        """Get cleaning status for a specific dataset."""
        # This would typically check against a database or file system
        # For now, return basic status
        return {
            "dataset_id": dataset_id,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "services_available": {
                "image_cleaning": True,
                "document_cleaning": True,
                "unified_cleaning": True
            }
        }
    
    def get_cleaning_capabilities(self) -> Dict:
        """Get information about cleaning capabilities."""
        return {
            "image_cleaning": {
                "supported_formats": self.config["image_cleaning"]["supported_formats"],
                "cleaning_tasks": [
                    "Corrupt file removal",
                    "Deduplication",
                    "Resizing & rescaling",
                    "Color space conversion",
                    "Low-resolution handling",
                    "Noise reduction",
                    "Data augmentation",
                    "Annotation cleaning",
                    "EXIF normalization",
                    "Outlier detection"
                ],
                "augmentation_enabled": self.config["image_cleaning"]["augmentation"]
            },
            "document_cleaning": {
                "supported_formats": self.config["document_cleaning"]["supported_formats"],
                "cleaning_tasks": [
                    "Text extraction & encoding",
                    "Boilerplate removal",
                    "Hyphenation handling",
                    "Non-text element removal",
                    "Text normalization",
                    "Special character removal",
                    "Tokenization",
                    "Metadata extraction",
                    "Language detection",
                    "Structure recovery",
                    "Deduplication"
                ],
                "language_detection": self.config["document_cleaning"]["language_detection"]
            },
            "workflow_options": {
                "parallel_processing": self.config["workflow"]["parallel_processing"],
                "intermediate_results": self.config["workflow"]["save_intermediate_results"],
                "combined_reporting": self.config["workflow"]["generate_combined_report"]
            }
        }
    
    def _run_eda_analysis(self, data_dir: Path, analysis_type: str) -> Dict:
        """Run EDA analysis on the dataset."""
        try:
            logger.info(f"Running {analysis_type} EDA analysis...")
            
            # Initialize EDA service if not already done
            if self.eda_service is None:
                self.eda_service = DatasetEDA(str(data_dir))
            
            # Create temporary output directory for EDA results
            eda_output_dir = data_dir / f"eda_{analysis_type}"
            eda_output_dir.mkdir(exist_ok=True)
            
            # Run EDA analysis
            self.eda_service.run_complete_analysis(str(eda_output_dir))
            
            # Load EDA results
            eda_results_path = eda_output_dir / "eda_results.json"
            if eda_results_path.exists():
                with open(eda_results_path, 'r', encoding='utf-8') as f:
                    eda_results = json.load(f)
                
                # Add analysis metadata
                eda_results["analysis_type"] = analysis_type
                eda_results["analysis_timestamp"] = datetime.now().isoformat()
                eda_results["output_directory"] = str(eda_output_dir)
                
                logger.info(f"{analysis_type} EDA analysis completed")
                return eda_results
            else:
                logger.warning(f"EDA results not found at {eda_results_path}")
                return {"error": "EDA results not generated"}
                
        except Exception as e:
            logger.error(f"EDA analysis failed for {analysis_type}: {e}")
            return {"error": str(e)}
    
    def get_eda_comparison(self, dataset_id: str) -> Dict:
        """Get comparison between before and after cleaning EDA results."""
        try:
            # This would typically load from stored results
            # For now, return a template structure
            return {
                "dataset_id": dataset_id,
                "comparison_available": True,
                "before_cleaning": "eda_before_cleaning results",
                "after_cleaning": "eda_after_cleaning results",
                "improvements": {
                    "file_quality": "Improved file integrity",
                    "format_consistency": "Standardized formats",
                    "annotation_quality": "Validated annotations",
                    "data_distribution": "Balanced class distribution"
                }
            }
        except Exception as e:
            logger.error(f"Failed to get EDA comparison: {e}")
            return {"error": str(e)}
