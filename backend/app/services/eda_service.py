"""
EDA Service for PS-05

This module provides EDA functionality that can be imported
by other services in the backend.
"""

import json
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
from PIL import Image
import seaborn as sns
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

class DatasetEDA:
    """Dataset Exploratory Data Analysis."""
    
    def __init__(self, data_dir: str):
        """Initialize EDA with data directory.
        
        Args:
            data_dir: Path to dataset directory
        """
        self.data_dir = Path(data_dir)
        self.results = {}
        
    def analyze_file_formats(self) -> Dict:
        """Analyze file formats in the dataset."""
        logger.info("Analyzing file formats...")
        
        file_formats = defaultdict(list)
        total_files = 0
        
        # Find all files
        for file_path in self.data_dir.rglob("*"):
            if file_path.is_file():
                total_files += 1
                suffix = file_path.suffix.lower()
                file_formats[suffix].append(str(file_path))
        
        # Count formats
        format_counts = {fmt: len(files) for fmt, files in file_formats.items()}
        
        # Identify image files
        image_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        image_files = []
        for fmt in image_formats:
            if fmt in file_formats:
                image_files.extend(file_formats[fmt])
        
        # Identify annotation files
        annotation_formats = ['.json', '.xml', '.txt']
        annotation_files = []
        for fmt in annotation_formats:
            if fmt in file_formats:
                annotation_files.extend(file_formats[fmt])
        
        # Identify document files
        document_formats = ['.pdf', '.doc', '.docx', '.ppt', '.pptx']
        document_files = []
        for fmt in document_formats:
            if fmt in file_formats:
                document_files.extend(file_formats[fmt])
        
        analysis = {
            'total_files': total_files,
            'format_counts': format_counts,
            'image_files': image_files,
            'annotation_files': annotation_files,
            'document_files': document_files,
            'image_count': len(image_files),
            'annotation_count': len(annotation_files),
            'document_count': len(document_files)
        }
        
        self.results['file_formats'] = analysis
        return analysis
    
    def analyze_image_properties(self) -> Dict:
        """Analyze image properties including dimensions, rotation, and quality."""
        logger.info("Analyzing image properties...")
        
        image_files = self.results.get('file_formats', {}).get('image_files', [])
        if not image_files:
            logger.warning("No image files found for analysis")
            return {}
        
        properties = {
            'dimensions': [],
            'aspect_ratios': [],
            'file_sizes': [],
            'rotation_angles': [],
            'quality_metrics': []
        }
        
        # Sample images for analysis (limit to prevent memory issues)
        sample_size = min(100, len(image_files))
        sample_images = image_files[:sample_size]
        
        for img_path in sample_images:
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Basic properties
                height, width = img.shape[:2]
                properties['dimensions'].append((width, height))
                properties['aspect_ratios'].append(width / height)
                
                # File size
                file_size = Path(img_path).stat().st_size / 1024  # KB
                properties['file_sizes'].append(file_size)
                
                # Detect rotation using Hough Line Transform
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
                
                if lines is not None:
                    angles = []
                    for line in lines:
                        rho, theta = line[0]
                        angle = np.degrees(theta)
                        if angle < 90:
                            angles.append(angle)
                        else:
                            angles.append(angle - 90)
                    
                    if angles:
                        avg_angle = np.mean(angles)
                        properties['rotation_angles'].append(avg_angle)
                
                # Quality metrics
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                properties['quality_metrics'].append(laplacian_var)
                
            except Exception as e:
                logger.warning(f"Error analyzing {img_path}: {e}")
                continue
        
        # Calculate statistics
        analysis = {
            'total_images_analyzed': len(properties['dimensions']),
            'sample_size': sample_size,
            'dimension_stats': {
                'widths': [d[0] for d in properties['dimensions']],
                'heights': [d[1] for d in properties['dimensions']],
                'min_width': min([d[0] for d in properties['dimensions']]) if properties['dimensions'] else 0,
                'max_width': max([d[0] for d in properties['dimensions']]) if properties['dimensions'] else 0,
                'min_height': min([d[1] for d in properties['dimensions']]) if properties['dimensions'] else 0,
                'max_height': max([d[1] for d in properties['dimensions']]) if properties['dimensions'] else 0,
                'avg_width': np.mean([d[0] for d in properties['dimensions']]) if properties['dimensions'] else 0,
                'avg_height': np.mean([d[1] for d in properties['dimensions']]) if properties['dimensions'] else 0
            },
            'aspect_ratio_stats': {
                'min': min(properties['aspect_ratios']) if properties['aspect_ratios'] else 0,
                'max': max(properties['aspect_ratios']) if properties['aspect_ratios'] else 0,
                'mean': np.mean(properties['aspect_ratios']) if properties['aspect_ratios'] else 0,
                'std': np.std(properties['aspect_ratios']) if properties['aspect_ratios'] else 0
            },
            'file_size_stats': {
                'min': min(properties['file_sizes']) if properties['file_sizes'] else 0,
                'max': max(properties['file_sizes']) if properties['file_sizes'] else 0,
                'mean': np.mean(properties['file_sizes']) if properties['file_sizes'] else 0,
                'std': np.std(properties['file_sizes']) if properties['file_sizes'] else 0
            },
            'rotation_stats': {
                'min': min(properties['rotation_angles']) if properties['rotation_angles'] else 0,
                'max': max(properties['rotation_angles']) if properties['rotation_angles'] else 0,
                'mean': np.mean(properties['rotation_angles']) if properties['rotation_angles'] else 0,
                'std': np.std(properties['rotation_angles']) if properties['rotation_angles'] else 0
            },
            'quality_stats': {
                'min': min(properties['quality_metrics']) if properties['quality_metrics'] else 0,
                'max': max(properties['quality_metrics']) if properties['quality_metrics'] else 0,
                'mean': np.mean(properties['quality_metrics']) if properties['quality_metrics'] else 0,
                'std': np.std(properties['quality_metrics']) if properties['quality_metrics'] else 0
            }
        }
        
        self.results['image_properties'] = analysis
        return analysis
    
    def analyze_annotations(self) -> Dict:
        """Analyze annotation patterns and quality."""
        logger.info("Analyzing annotations...")
        
        annotation_files = self.results.get('file_formats', {}).get('annotation_files', [])
        if not annotation_files:
            logger.warning("No annotation files found for analysis")
            return {}
        
        annotation_stats = {
            'total_annotations': 0,
            'class_distribution': Counter(),
            'bbox_statistics': [],
            'annotation_quality': {
                'missing_bbox': 0,
                'invalid_bbox': 0,
                'missing_class': 0
            }
        }
        
        # Sample annotations for analysis
        sample_size = min(100, len(annotation_files))
        sample_annotations = annotation_files[:sample_size]
        
        for ann_path in sample_annotations:
            try:
                with open(ann_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Count annotations
                annotations = data.get('annotations', [])
                annotation_stats['total_annotations'] += len(annotations)
                
                # Analyze each annotation
                for ann in annotations:
                    # Class distribution
                    category_id = ann.get('category_id', 'unknown')
                    annotation_stats['class_distribution'][category_id] += 1
                    
                    # Bbox analysis
                    bbox = ann.get('bbox', [])
                    if len(bbox) == 4:
                        x, y, w, h = bbox
                        annotation_stats['bbox_statistics'].append({
                            'x': x, 'y': y, 'width': w, 'height': h,
                            'area': w * h,
                            'aspect_ratio': w / h if h > 0 else 0
                        })
                    else:
                        annotation_stats['annotation_quality']['invalid_bbox'] += 1
                    
                    # Check for missing class
                    if 'category_id' not in ann:
                        annotation_stats['annotation_quality']['missing_class'] += 1
                
            except Exception as e:
                logger.warning(f"Error analyzing annotation {ann_path}: {e}")
                continue
        
        # Calculate bbox statistics
        if annotation_stats['bbox_statistics']:
            bbox_data = pd.DataFrame(annotation_stats['bbox_statistics'])
            bbox_stats = {
                'width_stats': bbox_data['width'].describe().to_dict(),
                'height_stats': bbox_data['height'].describe().to_dict(),
                'area_stats': bbox_data['area'].describe().to_dict(),
                'aspect_ratio_stats': bbox_data['aspect_ratio'].describe().to_dict()
            }
            annotation_stats['bbox_statistics'] = bbox_stats
        
        annotation_stats['sample_size'] = sample_size
        self.results['annotations'] = annotation_stats
        return annotation_stats
    
    def generate_visualizations(self, output_dir: str):
        """Generate EDA visualizations."""
        logger.info("Generating visualizations...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. File format distribution
            if 'file_formats' in self.results:
                format_counts = self.results['file_formats']['format_counts']
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.pie(format_counts.values(), labels=format_counts.keys(), autopct='%1.1f%%')
                plt.title('File Format Distribution')
                
                plt.subplot(1, 2, 2)
                plt.bar(format_counts.keys(), format_counts.values())
                plt.title('File Format Counts')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_path / 'file_formats.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 2. Image dimensions scatter plot
            if 'image_properties' in self.results:
                props = self.results['image_properties']
                if 'dimension_stats' in props:
                    plt.figure(figsize=(10, 8))
                    plt.scatter(props['dimension_stats']['widths'], props['dimension_stats']['heights'], alpha=0.6)
                    plt.xlabel('Width (pixels)')
                    plt.ylabel('Height (pixels)')
                    plt.title('Image Dimensions Distribution')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(output_path / 'image_dimensions.png', dpi=300, bbox_inches='tight')
                    plt.close()
            
            # 3. Class distribution
            if 'annotations' in self.results:
                ann_stats = self.results['annotations']
                if 'class_distribution' in ann_stats:
                    plt.figure(figsize=(10, 6))
                    classes = list(ann_stats['class_distribution'].keys())
                    counts = list(ann_stats['class_distribution'].values())
                    plt.bar(classes, counts)
                    plt.title('Annotation Class Distribution')
                    plt.xlabel('Class ID')
                    plt.ylabel('Count')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(output_path / 'class_distribution.png', dpi=300, bbox_inches='tight')
                    plt.close()
            
            logger.info("Visualizations generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
    
    def generate_report(self, output_dir: str):
        """Generate comprehensive EDA report."""
        logger.info("Generating EDA report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report = []
        report.append("# PS-05 Dataset EDA Report")
        report.append("=" * 50)
        report.append("")
        
        # File format summary
        if 'file_formats' in self.results:
            fmt_data = self.results['file_formats']
            report.append("## File Format Analysis")
            report.append(f"- Total files: {fmt_data['total_files']}")
            report.append(f"- Image files: {fmt_data['image_count']}")
            report.append(f"- Annotation files: {fmt_data['annotation_count']}")
            report.append(f"- Document files: {fmt_data['document_count']}")
            report.append("")
            
            report.append("### Format Distribution:")
            for fmt, count in fmt_data['format_counts'].items():
                report.append(f"- {fmt}: {count} files")
            report.append("")
        
        # Image properties summary
        if 'image_properties' in self.results:
            img_data = self.results['image_properties']
            report.append("## Image Properties Analysis")
            report.append(f"- Images analyzed: {img_data['total_images_analyzed']} (sample size: {img_data['sample_size']})")
            report.append("")
            
            if 'dimension_stats' in img_data:
                dims = img_data['dimension_stats']
                report.append("### Dimensions:")
                report.append(f"- Width range: {dims['min_width']:.0f} - {dims['max_width']:.0f} pixels")
                report.append(f"- Height range: {dims['min_height']:.0f} - {dims['max_height']:.0f} pixels")
                report.append(f"- Average width: {dims['avg_width']:.0f} pixels")
                report.append(f"- Average height: {dims['avg_height']:.0f} pixels")
                report.append("")
            
            if 'rotation_stats' in img_data:
                rot = img_data['rotation_stats']
                report.append("### Rotation Analysis:")
                report.append(f"- Rotation range: {rot['min']:.2f}° - {rot['max']:.2f}°")
                report.append(f"- Average rotation: {rot['mean']:.2f}°")
                report.append(f"- Rotation std: {rot['std']:.2f}°")
                report.append("")
        
        # Annotation summary
        if 'annotations' in self.results:
            ann_data = self.results['annotations']
            report.append("## Annotation Analysis")
            report.append(f"- Total annotations: {ann_data['total_annotations']}")
            report.append(f"- Sample size: {ann_data['sample_size']}")
            report.append("")
            
            if 'class_distribution' in ann_data:
                report.append("### Class Distribution:")
                for class_id, count in ann_data['class_distribution'].most_common():
                    report.append(f"- Class {class_id}: {count} annotations")
                report.append("")
            
            if 'annotation_quality' in ann_data:
                qual = ann_data['annotation_quality']
                report.append("### Quality Issues:")
                report.append(f"- Missing bbox: {qual['missing_bbox']}")
                report.append(f"- Invalid bbox: {qual['invalid_bbox']}")
                report.append(f"- Missing class: {qual['missing_class']}")
                report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("1. **Deskewing**: Implement robust deskewing for rotated images")
        report.append("2. **Multi-format Support**: Add support for PDF, DOC, PPT formats")
        report.append("3. **Data Augmentation**: Use rotation, scaling, noise for training")
        report.append("4. **Quality Control**: Validate annotation quality and consistency")
        report.append("5. **Format Conversion**: Convert all inputs to image format for processing")
        
        # Save report
        report_path = output_path / 'eda_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logger.info(f"EDA report saved to {report_path}")
        
        # Save results as JSON
        results_path = output_path / 'eda_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"EDA results saved to {results_path}")
    
    def run_complete_analysis(self, output_dir: str):
        """Run complete EDA analysis."""
        logger.info("Starting complete dataset analysis...")
        
        # Run all analyses
        self.analyze_file_formats()
        self.analyze_image_properties()
        self.analyze_annotations()
        
        # Generate outputs
        self.generate_visualizations(output_dir)
        self.generate_report(output_dir)
        
        logger.info("Complete EDA analysis finished!")
