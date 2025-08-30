# 🚀 PS-05 Document Understanding System - Master Guide

## 🎯 **What This Project Is**

**PS-05** is a **state-of-the-art, production-ready** document understanding system that combines:
- **3-Stage AI Pipeline** (Layout Detection → Text Extraction → Content Understanding)
- **21 Comprehensive Cleaning Tasks** (Image + Document cleaning)
- **Professional EDA Analysis** (Exploratory Data Analysis)
- **FastAPI Backend** with full API support
- **Training Pipeline** for custom model development
- **Docker Deployment** ready for production

## 🏗️ **Project Organization (Refactored for Clarity)**

```
multilingual-docai/
├── 📁 CORE SYSTEM          # Core ML models and processing
├── 📁 SCRIPTS              # Command-line tools (organized by function)
├── 📁 BACKEND              # FastAPI server with all services
├── 📁 DATA & MODELS        # Datasets, models, and results
├── 📁 DOCUMENTATION        # Comprehensive guides and tutorials
└── 📁 DEPLOYMENT           # Docker and production setup
```

## 🚀 **Quick Start (Choose Your Path)**

### **🎯 Path 1: Simple Pipeline (Recommended for Starters)**
```bash
# Run the complete 3-stage pipeline
python ps05.py pipeline --stage all

# Or run individual stages
python ps05.py pipeline --stage 1  # Layout detection
python ps05.py pipeline --stage 2  # Text extraction
python ps05.py pipeline --stage 3  # Content understanding
```

### **🧹 Path 2: Data Quality Focus**
```bash
# Run EDA analysis only
python ps05.py clean --input data/train --output results/eda --mode eda_only

# Run cleaning with EDA integration
python ps05.py clean --input data/train --output results/cleaned --mode cleaning_with_eda
```

### **🏋️ Path 3: Training Pipeline**
```bash
# Prepare training data with augmentation
python ps05.py train --prepare-data --input data/train --output data/training_prepared

# Train YOLO model
python ps05.py train --train-model
```

### **🌐 Path 4: Production/API**
```bash
# Start the backend server
python ps05.py backend --start

# Use API endpoints for processing
curl -X POST "http://localhost:8000/upload-dataset" -F "files=@dataset.zip"
```

## 🔧 **What Each Component Does**

### **📊 Core Pipeline (3 Stages)**
1. **Stage 1: Layout Detection** - YOLOv8 for document structure
2. **Stage 2: Text Extraction** - OCR + Language identification
3. **Stage 3: Content Understanding** - LayoutLMv3 + Advanced models

### **🧹 Cleaning Services (21 Tasks)**
- **Image Cleaning (10 tasks)**: Corrupt removal, deduplication, augmentation, quality enhancement
- **Document Cleaning (11 tasks)**: Text extraction, boilerplate removal, language detection, structure recovery

### **📈 EDA Analysis**
- **File format analysis** and categorization
- **Image properties** (dimensions, rotation, quality)
- **Annotation quality** and class distribution
- **Professional visualizations** and reports

### **🌐 Backend API**
- **Dataset upload** (with/without annotations)
- **Stage-by-stage processing**
- **Background processing** for large datasets
- **Evaluation and metrics** calculation
- **Result retrieval** and status monitoring

## 📁 **Key Files & Directories**

### **🚀 Main Entry Points**
- **`ps05.py`** - Main command-line interface for everything
- **`scripts/core/run_stages.py`** - Core pipeline execution
- **`scripts/cleaning/eda_with_cleaning.py`** - EDA + Cleaning integration
- **`backend/app/main.py`** - FastAPI backend server

### **🔧 Core Services**
- **`backend/app/services/unified_cleaning_service.py`** - All cleaning tasks
- **`backend/app/services/eda_service.py`** - Data analysis
- **`backend/app/services/stage_processor.py`** - Pipeline processing

### **📚 Documentation**
- **`PROJECT_STRUCTURE.md`** - High-level project organization
- **`DIRECTORY_STRUCTURE.md`** - Detailed file layout
- **`CLEANING_SERVICES_GUIDE.md`** - Cleaning services documentation
- **`EDA_CLEANING_INTEGRATION_GUIDE.md`** - EDA integration guide
- **`API_USAGE_GUIDE.md`** - API usage documentation

## 🎯 **Use Cases & Examples**

### **1. Document Processing**
```bash
# Process test documents
python ps05.py pipeline --stage all --input data/test --output results/test_processed
```

### **2. Training Data Preparation**
```bash
# Clean and prepare training data
python ps05.py clean --input data/raw --output data/cleaned --mode cleaning_with_eda
python ps05.py train --prepare-data --input data/cleaned --output data/training_ready
```

### **3. Model Training**
```bash
# Train custom YOLO model
python ps05.py train --train-model
```

### **4. Production Deployment**
```bash
# Start production backend
python ps05.py backend --start --port 8000

# Process large datasets via API
curl -X POST "http://localhost:8000/process-all" -F "files=@large_dataset.zip"
```

## 🔍 **Finding What You Need**

| **What You Want** | **Where to Look** | **How to Use** |
|-------------------|-------------------|----------------|
| **Run Pipeline** | `scripts/core/run_stages.py` | `python ps05.py pipeline` |
| **Clean Data** | `scripts/cleaning/eda_with_cleaning.py` | `python ps05.py clean` |
| **Train Models** | `scripts/training/` | `python ps05.py train` |
| **Use API** | `backend/app/main.py` | `python ps05.py backend --start` |
| **Configuration** | `configs/` | Edit YAML files |
| **Documentation** | Root directory | Read markdown guides |

## 🚨 **Troubleshooting**

### **Common Issues & Solutions**

#### **1. Import Errors**
```bash
# Make sure you're in the project root
cd /path/to/multilingual-docai

# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### **2. Missing Dependencies**
```bash
# Install all requirements
pip install -r requirements.txt
```

#### **3. Memory Issues**
```bash
# Use smaller batch sizes for large datasets
python ps05.py clean --input data/large --output results/cleaned --mode eda_only
```

#### **4. Backend Issues**
```bash
# Check backend status
python ps05.py backend --status

# Restart backend
python ps05.py backend --start --port 8000
```

## 📊 **Project Status: 100% Complete**

- ✅ **Core 3-stage pipeline**: 100%
- ✅ **Comprehensive cleaning services**: 100% (21 tasks)
- ✅ **Training data preparation**: 100%
- ✅ **EDA service**: 100%
- ✅ **EDA + Cleaning integration**: 100%
- ✅ **API endpoints**: 100%
- ✅ **Documentation**: 100%
- ✅ **Project refactoring**: 100%

## 🎉 **What You Get**

### **🚀 Production-Ready System**
- **Complete AI pipeline** for document understanding
- **Professional data cleaning** with quality assurance
- **Scalable API backend** for large datasets
- **Docker deployment** ready for production

### **📊 Data Quality Assurance**
- **Before/after analysis** with EDA
- **21 cleaning tasks** for comprehensive data preparation
- **Quality metrics** and validation
- **Professional reporting** and visualizations

### **🔧 Easy to Use**
- **Single entry point** (`ps05.py`) for all functionality
- **Organized scripts** by function (core, training, cleaning, utilities)
- **Clear documentation** with examples
- **Multiple usage paths** for different needs

### **🌐 Scalable Architecture**
- **API-first design** for production use
- **Background processing** for large datasets
- **Modular services** for easy extension
- **Docker support** for deployment

## 🚀 **Next Steps**

1. **Start Simple**: Use `python ps05.py pipeline --stage all`
2. **Explore Quality**: Try `python ps05.py clean --mode eda_only`
3. **Scale Up**: Use backend API for large datasets
4. **Customize**: Modify configs and add new features
5. **Deploy**: Use Docker for production deployment

## 📞 **Support & Resources**

### **📚 Documentation Files**
- **`PROJECT_STRUCTURE.md`** - Project organization overview
- **`DIRECTORY_STRUCTURE.md`** - Detailed file layout
- **`CLEANING_SERVICES_GUIDE.md`** - Cleaning services guide
- **`EDA_CLEANING_INTEGRATION_GUIDE.md`** - EDA integration guide
- **`API_USAGE_GUIDE.md`** - API usage guide

### **🔧 Scripts by Function**
- **`scripts/core/`** - Main pipeline execution
- **`scripts/training/`** - Training and dataset preparation
- **`scripts/cleaning/`** - Data cleaning and EDA
- **`scripts/utilities/`** - Maintenance and utilities

### **🌐 Backend Services**
- **`backend/app/services/`** - All backend services
- **`backend/app/main.py`** - FastAPI application
- **`backend/run.py`** - Backend runner

---

## 🎯 **Summary**

**PS-05** is now a **completely refactored, production-ready** document understanding system that provides:

- **🎯 Simple entry point** (`ps05.py`) for all functionality
- **📁 Organized structure** that's easy to navigate
- **🧹 Comprehensive cleaning** with 21 tasks
- **📊 Professional EDA** with quality analysis
- **🌐 Scalable API** for production use
- **📚 Clear documentation** for every component

**💡 Start with `python ps05.py --help` to see all available options!**

---

**🚀 You now have a world-class, enterprise-ready document understanding system that's easy to use and understand!**
