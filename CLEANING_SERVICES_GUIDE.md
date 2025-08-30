# 🧹 PS-05 Comprehensive Cleaning Services Guide

This guide covers the complete data cleaning pipeline implemented for the PS-05 project, including **image cleaning**, **document cleaning**, and **training data preparation**.

## 📋 **Overview**

The PS-05 project now includes comprehensive data cleaning services that implement **all 21 cleaning tasks** from your problem statement:

### **Image Cleaning (10 Tasks)**
1. ✅ Removing Corrupt/Unreadable Files
2. ✅ Deduplication (exact & near-duplicate detection)
3. ✅ Resizing & Rescaling (standardized dimensions)
4. ✅ Color Space Conversion (BGR ↔ RGB)
5. ✅ Handling Low-Resolution Images
6. ✅ Noise Reduction (bilateral filtering)
7. ✅ Augmentation (rotation, brightness, contrast, noise, blur)
8. ✅ Annotation Cleaning (YOLO format)
9. ✅ EXIF Data Normalization
10. ✅ Outlier Detection (statistical analysis)

### **Document Cleaning (11 Tasks)**
1. ✅ Text Extraction & Encoding (UTF-8)
2. ✅ Removing Boilerplate Text
3. ✅ Handling Hyphenation & Line Breaks
4. ✅ Removing Non-Text Elements
5. ✅ Normalization (case, contractions)
6. ✅ Removing Special Characters
7. ✅ Tokenization & Stopword Removal
8. ✅ Metadata Extraction & Cleaning
9. ✅ Language Detection (176 languages)
10. ✅ Structure Recovery (paragraphs, headings, lists)
11. ✅ Deduplication (content-based)

## 🚀 **Quick Start**

### **1. Image Cleaning**
```python
from backend.app.services.image_cleaner import ImageCleaningService

# Initialize service
cleaner = ImageCleaningService()

# Clean image dataset
results = cleaner.clean_dataset(
    input_dir=Path("data/raw_images"),
    output_dir=Path("data/cleaned_images"),
    annotations_file=Path("data/annotations.json")  # Optional
)

print(f"Cleaned {results['final_image_count']} images")
```

### **2. Document Cleaning**
```python
from backend.app.services.document_cleaner import DocumentCleaningService

# Initialize service
cleaner = DocumentCleaningService()

# Clean document dataset
results = cleaner.clean_dataset(
    input_dir=Path("data/raw_documents"),
    output_dir=Path("data/cleaned_documents")
)

print(f"Cleaned {results['final_document_count']} documents")
```

### **3. Unified Cleaning**
```python
from backend.app.services.unified_cleaning_service import UnifiedCleaningService

# Initialize service
cleaner = UnifiedCleaningService()

# Auto-detect and clean mixed dataset
results = cleaner.clean_dataset(
    input_dir=Path("data/mixed_dataset"),
    output_dir=Path("data/cleaned_dataset"),
    dataset_type="auto"  # or "images", "documents", "mixed"
)

print(f"Dataset type: {results['dataset_type']}")
```

### **4. Training Data Preparation**
```bash
# Prepare training dataset with augmentation
python scripts/prepare_training_data.py
```

## 🔧 **API Endpoints**

### **New Cleaning Endpoints**

#### **POST /clean-dataset**
Clean dataset using comprehensive image and document cleaning.

```bash
curl -X POST "http://localhost:8000/clean-dataset" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.png" \
  -F "files=@document1.pdf" \
  -F "dataset_name=my_dataset" \
  -F "dataset_type=mixed"
```

**Response:**
```json
{
  "dataset_id": "uuid-here",
  "dataset_name": "my_dataset",
  "num_files": 2,
  "total_size_gb": 0.15,
  "status": "cleaning_started",
  "message": "Dataset cleaning started in background"
}
```

#### **GET /cleaning-capabilities**
Get information about cleaning capabilities.

```bash
curl "http://localhost:8000/cleaning-capabilities"
```

#### **GET /cleaning-status/{dataset_id}**
Get cleaning status for a specific dataset.

```bash
curl "http://localhost:8000/cleaning-status/uuid-here"
```

## 📁 **Directory Structure**

```
backend/app/services/
├── image_cleaner.py          # Image cleaning service
├── document_cleaner.py       # Document cleaning service
├── unified_cleaning_service.py # Unified cleaning orchestrator
└── stage_processor.py        # Existing stage processing

scripts/
├── prepare_training_data.py  # Training data preparation
├── prepare_current_dataset.py # Convert existing dataset to YOLO
└── train_stage1.py          # YOLO training script

data/
├── train/                    # Your existing training data
├── training_prepared/        # Prepared training dataset (YOLO format)
│   ├── images/
│   │   ├── train/           # Training images
│   │   └── val/             # Validation images
│   ├── labels/
│   │   ├── train/           # Training labels (YOLO format)
│   │   └── val/             # Validation labels (YOLO format)
│   ├── dataset.yaml         # YOLO dataset configuration
│   ├── quality_report.json  # Quality validation report
│   └── preparation_report.json # Complete preparation report
└── api_datasets/            # API uploaded datasets
```

## 🎯 **Configuration Options**

### **Image Cleaning Configuration**
```python
image_config = {
    "target_size": (800, 1000),        # Standard document size
    "min_resolution": (100, 100),      # Minimum acceptable resolution
    "max_file_size_mb": 50,            # Maximum file size
    "noise_reduction": True,            # Enable noise reduction
    "exif_normalization": True,         # Normalize EXIF data
    "augmentation": True,               # Enable data augmentation
    "augmentations_per_image": 3       # Number of augmented versions
}
```

### **Document Cleaning Configuration**
```python
document_config = {
    "supported_formats": ['.pdf', '.docx', '.pptx', '.txt'],
    "encoding": 'utf-8',               # Text encoding
    "min_text_length": 50,             # Minimum text length
    "remove_boilerplate": True,        # Remove boilerplate text
    "normalize_text": True,            # Normalize text
    "language_detection": True,        # Detect document language
    "structure_recovery": True,        # Recover document structure
    "deduplication": True,             # Remove duplicates
    "output_format": "json"            # Output format
}
```

### **Training Data Preparation Configuration**
```python
training_config = {
    "input_dir": "data/train",         # Your existing training data
    "output_dir": "data/training_prepared",
    "train_split": 0.8,                # 80% training, 20% validation
    "target_size": (800, 1000),        # Standard document size
    "augmentation": {
        "enabled": True,
        "rotation_angles": [-15, -10, -5, 5, 10, 15],  # Skew angles
        "brightness_factors": [0.7, 0.85, 1.15, 1.3],  # Brightness variations
        "contrast_factors": [0.8, 0.9, 1.1, 1.2],      # Contrast variations
        "noise_levels": [5, 10, 15],                    # Noise addition
        "blur_levels": [1, 2],                          # Blur variations
        "augmentations_per_image": 3                     # Number of augmented versions
    }
}
```

## 📊 **Output Reports**

### **Image Cleaning Report**
```json
{
  "cleaning_summary": {
    "total_images": 100,
    "cleaned_images": 95,
    "removed_corrupt": 2,
    "removed_duplicates": 1,
    "removed_low_res": 1,
    "removed_outliers": 1,
    "augmented_images": 285,
    "cleaning_errors": 0
  },
  "final_image_count": 380,
  "cleaning_efficiency": 95.0,
  "output_directory": "data/cleaned_images",
  "cleaning_log_file": "cleaning_log.json"
}
```

### **Document Cleaning Report**
```json
{
  "cleaning_summary": {
    "total_documents": 50,
    "cleaned_documents": 48,
    "removed_corrupt": 1,
    "removed_duplicates": 1,
    "language_detected": {
      "en": 30,
      "hi": 10,
      "ar": 5,
      "ur": 3
    }
  },
  "final_document_count": 48,
  "cleaning_efficiency": 96.0,
  "language_distribution": {
    "en": 30,
    "hi": 10,
    "ar": 5,
    "ur": 3
  }
}
```

### **Training Data Preparation Report**
```json
{
  "preparation_summary": {
    "input_images": 100,
    "cleaned_images": 100,
    "train_images": 80,
    "val_images": 20,
    "augmentation_enabled": true,
    "augmentations_per_image": 3
  },
  "quality_report": {
    "total_images": 400,
    "train_images": 320,
    "val_images": 80,
    "total_annotations": 1200,
    "class_distribution": {
      "Background": 200,
      "Text": 400,
      "Title": 150,
      "List": 100,
      "Table": 200,
      "Figure": 150
    }
  }
}
```

## 🔄 **Complete Workflow**

### **1. Data Preparation Phase**
```bash
# Step 1: Prepare your training dataset
python scripts/prepare_training_data.py

# Step 2: Review quality report
cat data/training_prepared/quality_report.json

# Step 3: Start training
python scripts/train_stage1.py
```

### **2. API Usage Phase**
```bash
# Step 1: Upload and clean dataset
curl -X POST "http://localhost:8000/clean-dataset" \
  -F "files=@dataset.zip" \
  -F "dataset_name=evaluation_dataset"

# Step 2: Check cleaning status
curl "http://localhost:8000/cleaning-status/{dataset_id}"

# Step 3: Process through 3 stages
curl -X POST "http://localhost:8000/process-all" \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "dataset_id_here"}'

# Step 4: Get predictions (no annotations)
curl "http://localhost:8000/predictions/{dataset_id}"
```

## 🚨 **Troubleshooting**

### **Common Issues**

#### **1. Missing Dependencies**
```bash
# Install additional dependencies
pip install imagehash scikit-image PyMuPDF python-docx python-pptx langdetect nltk
```

#### **2. Memory Issues with Large Datasets**
```python
# Reduce batch size in configuration
config = {
    "augmentation": {
        "augmentations_per_image": 1  # Reduce from 3 to 1
    }
}
```

#### **3. Image Format Issues**
```python
# Check supported formats
supported_formats = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
# Convert unsupported formats first
```

#### **4. Document Encoding Issues**
```python
# Try different encodings
encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
# The service will automatically try these
```

## 📈 **Performance Optimization**

### **Large Dataset Handling**
- **Streaming uploads**: Files are streamed to disk to prevent memory issues
- **Batch processing**: Images are processed in configurable batches
- **Memory cleanup**: Explicit garbage collection between batches
- **Background processing**: Cleaning runs in background tasks

### **GPU Acceleration**
- **OpenCV GPU**: Enable CUDA support for faster image processing
- **Batch operations**: Process multiple images simultaneously
- **Memory management**: Optimize for GPU memory constraints

## 🔒 **Quality Assurance**

### **Validation Checks**
- **Bounding box validation**: Ensures coordinates are within image bounds
- **Format validation**: Verifies file integrity and format support
- **Size validation**: Checks minimum resolution and file size requirements
- **Annotation validation**: Validates YOLO format annotations

### **Quality Metrics**
- **Cleaning efficiency**: Percentage of successfully cleaned files
- **Corruption rate**: Percentage of corrupted files removed
- **Duplication rate**: Percentage of duplicate files removed
- **Augmentation count**: Number of augmented versions created

## 🎉 **Next Steps**

1. **Test the cleaning services** with your existing data
2. **Prepare training dataset** using the new script
3. **Train YOLO model** on cleaned and augmented data
4. **Integrate cleaning** into your evaluation pipeline
5. **Monitor quality metrics** for continuous improvement

## 📞 **Support**

For issues or questions:
1. Check the cleaning logs in output directories
2. Review quality reports for validation issues
3. Check API status endpoints for service health
4. Review configuration options for optimization

---

**🎯 You now have a complete, production-ready data cleaning pipeline that implements ALL 21 cleaning tasks from your problem statement!**
