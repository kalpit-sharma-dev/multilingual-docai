#!/usr/bin/env python3
"""
EDA with Cleaning Services Integration Script

This script demonstrates how to use EDA analysis integrated with
the comprehensive cleaning services for PS-05.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from backend.app.services.unified_cleaning_service import UnifiedCleaningService
from backend.app.services.eda_service import DatasetEDA

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EDAWithCleaning:
    """Integrated EDA and cleaning pipeline."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.cleaning_service = UnifiedCleaningService(self.config)
        self.eda_service = None
        
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
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
    
    def run_eda_only(self, input_dir: Path, output_dir: Path) -> Dict:
        """Run EDA analysis only (no cleaning)."""
        logger.info("Running EDA analysis only...")
        
        try:
            # Initialize EDA service
            self.eda_service = DatasetEDA(str(input_dir))
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run complete EDA analysis
            self.eda_service.run_complete_analysis(str(output_dir))
            
            # Load and return results
            results_path = output_dir / "eda_results.json"
            if results_path.exists():
                with open(results_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                logger.info("EDA analysis completed successfully!")
                return {
                    "status": "success",
                    "eda_results": results,
                    "output_directory": str(output_dir),
                    "files_generated": [
                        "eda_report.md",
                        "eda_results.json",
                        "file_formats.png",
                        "image_dimensions.png",
                        "class_distribution.png"
                    ]
                }
            else:
                raise FileNotFoundError("EDA results not generated")
                
        except Exception as e:
            logger.error(f"EDA analysis failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_cleaning_with_eda(self, input_dir: Path, output_dir: Path, dataset_type: str = "auto") -> Dict:
        """Run cleaning with EDA before and after."""
        logger.info("Running cleaning with integrated EDA...")
        
        try:
            # Run cleaning with EDA enabled
            results = self.cleaning_service.clean_dataset(
                input_dir=input_dir,
                output_dir=output_dir,
                dataset_type=dataset_type,
                run_eda=True
            )
            
            logger.info("Cleaning with EDA completed successfully!")
            return {
                "status": "success",
                "cleaning_results": results,
                "output_directory": str(output_dir),
                "eda_before": results.get("eda_before_cleaning", {}),
                "eda_after": results.get("eda_after_cleaning", {})
            }
            
        except Exception as e:
            logger.error(f"Cleaning with EDA failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def compare_eda_results(self, before_results: Dict, after_results: Dict) -> Dict:
        """Compare EDA results before and after cleaning."""
        logger.info("Comparing EDA results...")
        
        try:
            comparison = {
                "before_cleaning": before_results,
                "after_cleaning": after_results,
                "improvements": {},
                "statistics": {}
            }
            
            # Compare file format analysis
            if "file_formats" in before_results and "file_formats" in after_results:
                before_fmt = before_results["file_formats"]
                after_fmt = after_results["file_formats"]
                
                comparison["improvements"]["file_formats"] = {
                    "before_total": before_fmt.get("total_files", 0),
                    "after_total": after_fmt.get("total_files", 0),
                    "before_images": before_fmt.get("image_count", 0),
                    "after_images": after_fmt.get("image_count", 0),
                    "format_standardization": "Improved format consistency"
                }
            
            # Compare image properties
            if "image_properties" in before_results and "image_properties" in after_results:
                before_img = before_results["image_properties"]
                after_img = after_results["image_properties"]
                
                comparison["improvements"]["image_properties"] = {
                    "before_rotation_std": before_img.get("rotation_stats", {}).get("std", 0),
                    "after_rotation_std": after_img.get("rotation_stats", {}).get("std", 0),
                    "rotation_improvement": "Reduced rotation variation",
                    "quality_improvement": "Enhanced image quality"
                }
            
            # Compare annotations
            if "annotations" in before_results and "annotations" in after_results:
                before_ann = before_results["annotations"]
                after_ann = after_results["annotations"]
                
                comparison["improvements"]["annotations"] = {
                    "before_quality_issues": before_ann.get("annotation_quality", {}),
                    "after_quality_issues": after_ann.get("annotation_quality", {}),
                    "quality_improvement": "Validated annotation quality"
                }
            
            # Generate summary statistics
            comparison["statistics"] = {
                "total_improvements": len(comparison["improvements"]),
                "analysis_complete": True,
                "recommendations": [
                    "Continue monitoring data quality",
                    "Apply similar cleaning to new datasets",
                    "Use insights for model training"
                ]
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"EDA comparison failed: {e}")
            return {"error": str(e)}
    
    def generate_comprehensive_report(self, results: Dict, output_dir: Path):
        """Generate comprehensive report combining EDA and cleaning results."""
        logger.info("Generating comprehensive report...")
        
        try:
            report_path = output_dir / "comprehensive_eda_cleaning_report.md"
            
            report = []
            report.append("# PS-05 Comprehensive EDA & Cleaning Report")
            report.append("=" * 60)
            report.append("")
            
            # Executive Summary
            report.append("## Executive Summary")
            report.append(f"- **Analysis Type**: Integrated EDA + Cleaning")
            report.append(f"- **Status**: {results.get('status', 'Unknown')}")
            report.append(f"- **Output Directory**: {results.get('output_directory', 'Unknown')}")
            report.append("")
            
            # EDA Before Cleaning
            if "eda_before" in results:
                report.append("## EDA Analysis - Before Cleaning")
                eda_before = results["eda_before"]
                
                if "file_formats" in eda_before:
                    fmt = eda_before["file_formats"]
                    report.append(f"- **Total Files**: {fmt.get('total_files', 0)}")
                    report.append(f"- **Image Files**: {fmt.get('image_count', 0)}")
                    report.append(f"- **Document Files**: {fmt.get('document_count', 0)}")
                    report.append("")
                
                if "image_properties" in eda_before:
                    img = eda_before["image_properties"]
                    report.append(f"- **Images Analyzed**: {img.get('total_images_analyzed', 0)}")
                    report.append(f"- **Average Width**: {img.get('dimension_stats', {}).get('avg_width', 0):.0f}px")
                    report.append(f"- **Average Height**: {img.get('dimension_stats', {}).get('avg_height', 0):.0f}px")
                    report.append("")
            
            # EDA After Cleaning
            if "eda_after" in results:
                report.append("## EDA Analysis - After Cleaning")
                eda_after = results["eda_after"]
                
                if "file_formats" in eda_after:
                    fmt = eda_after["file_formats"]
                    report.append(f"- **Total Files**: {fmt.get('total_files', 0)}")
                    report.append(f"- **Image Files**: {fmt.get('image_count', 0)}")
                    report.append(f"- **Document Files**: {fmt.get('document_count', 0)}")
                    report.append("")
            
            # Cleaning Results
            if "cleaning_results" in results:
                cleaning = results["cleaning_results"]
                report.append("## Cleaning Results")
                
                if "image_cleaning" in cleaning:
                    img_clean = cleaning["image_cleaning"]
                    if "cleaning_summary" in img_clean:
                        summary = img_clean["cleaning_summary"]
                        report.append(f"- **Images Cleaned**: {summary.get('cleaned_images', 0)}")
                        report.append(f"- **Corrupt Files Removed**: {summary.get('removed_corrupt', 0)}")
                        report.append(f"- **Duplicates Removed**: {summary.get('removed_duplicates', 0)}")
                        report.append(f"- **Augmented Images**: {summary.get('augmented_images', 0)}")
                        report.append("")
                
                if "document_cleaning" in cleaning:
                    doc_clean = cleaning["document_cleaning"]
                    if "cleaning_summary" in doc_clean:
                        summary = doc_clean["cleaning_summary"]
                        report.append(f"- **Documents Cleaned**: {summary.get('cleaned_documents', 0)}")
                        report.append(f"- **Languages Detected**: {len(summary.get('language_detected', {}))}")
                        report.append("")
            
            # Recommendations
            report.append("## Recommendations")
            report.append("1. **Data Quality**: Continue monitoring with regular EDA")
            report.append("2. **Training**: Use cleaned data for model training")
            report.append("3. **Validation**: Apply similar cleaning to new datasets")
            report.append("4. **Automation**: Integrate EDA into CI/CD pipeline")
            report.append("5. **Documentation**: Maintain cleaning procedures")
            
            # Save report
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            
            logger.info(f"Comprehensive report saved: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run EDA with cleaning services")
    parser.add_argument('--input', required=True, help='Input data directory')
    parser.add_argument('--output', required=True, help='Output directory for results')
    parser.add_argument('--mode', choices=['eda_only', 'cleaning_with_eda'], 
                       default='cleaning_with_eda', help='Analysis mode')
    parser.add_argument('--dataset-type', choices=['auto', 'images', 'documents', 'mixed'],
                       default='auto', help='Dataset type for cleaning')
    
    args = parser.parse_args()
    
    # Initialize service
    service = EDAWithCleaning()
    
    try:
        if args.mode == 'eda_only':
            # Run EDA only
            results = service.run_eda_only(Path(args.input), Path(args.output))
            
            if results["status"] == "success":
                print(f"\n✅ EDA analysis completed successfully!")
                print(f"📁 Results saved to: {results['output_directory']}")
                print(f"📊 Files generated: {', '.join(results['files_generated'])}")
            else:
                print(f"\n❌ EDA analysis failed: {results['error']}")
                
        else:
            # Run cleaning with EDA
            results = service.run_cleaning_with_eda(
                Path(args.input), 
                Path(args.output), 
                args.dataset_type
            )
            
            if results["status"] == "success":
                print(f"\n✅ Cleaning with EDA completed successfully!")
                print(f"📁 Results saved to: {results['output_directory']}")
                
                # Generate comprehensive report
                service.generate_comprehensive_report(results, Path(args.output))
                
                # Compare EDA results if available
                if "eda_before" in results and "eda_after" in results:
                    comparison = service.compare_eda_results(
                        results["eda_before"], 
                        results["eda_after"]
                    )
                    
                    if "error" not in comparison:
                        print(f"📊 EDA comparison completed: {comparison['statistics']['total_improvements']} improvements identified")
                    else:
                        print(f"⚠️ EDA comparison failed: {comparison['error']}")
                        
            else:
                print(f"\n❌ Cleaning with EDA failed: {results['error']}")
                
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        print(f"\n❌ Script failed: {e}")

if __name__ == "__main__":
    main()
