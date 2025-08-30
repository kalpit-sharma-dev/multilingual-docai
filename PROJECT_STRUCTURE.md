# 🏗️ PS-05 Project Structure Guide

## 📂 **Project Organization**

```
multilingual-docai/
├── 📁 CORE SYSTEM (Main Implementation)
│   ├── core/                    # Core ML models and processing
│   ├── backend/                 # FastAPI backend with all services
│   └── configs/                 # Configuration files
│
├── 📁 SCRIPTS (Command Line Tools)
│   ├── core/                    # Core pipeline scripts
│   ├── training/                # Training and dataset preparation
│   ├── cleaning/                # Data cleaning and EDA
│   └── utilities/               # Utility and maintenance scripts
│
├── 📁 DATA & MODELS
│   ├── data/                    # Dataset storage
│   ├── models/                  # Trained models
│   └── results/                 # Processing results
│
├── 📁 DOCUMENTATION
│   ├── docs/                    # Technical documentation
│   └── guides/                  # User guides and tutorials
│
└── 📁 DEPLOYMENT
    ├── docker/                  # Docker configuration
    └── frontend/                # Web interface (if needed)
```

## 🎯 **What Each Directory Contains**

### **📁 CORE SYSTEM**
- **`core/`**: YOLOv8, LayoutLMv3, OCR models, evaluation
- **`backend/`**: FastAPI server with all services (cleaning, EDA, processing)
- **`configs/`**: YAML configuration files for different components

### **📁 SCRIPTS**
- **`scripts/core/`**: Main pipeline execution (`run_stages.py`)
- **`scripts/training/`**: Dataset preparation, training scripts
- **`scripts/cleaning/`**: EDA, data cleaning, quality analysis
- **`scripts/utilities/`**: Cleanup, maintenance, submission tools

### **📁 DATA & MODELS**
- **`data/`**: Your datasets (train, test, validation)
- **`models/`**: Pre-trained and fine-tuned models
- **`results/`**: Output from processing, cleaning, and EDA

## 🚀 **How to Use This Project**

### **Option 1: Simple Pipeline (Recommended for Starters)**
```bash
# Run the complete 3-stage pipeline
python scripts/core/run_stages.py --stage all

# Run individual stages
python scripts/core/run_stages.py --stage 1  # Layout detection
python scripts/core/run_stages.py --stage 2  # Text extraction
python scripts/core/run_stages.py --stage 3  # Content understanding
```

### **Option 2: Data Cleaning & EDA**
```bash
# Run EDA analysis only
python scripts/cleaning/eda_with_cleaning.py --input data/train --output results/eda --mode eda_only

# Run cleaning with EDA
python scripts/cleaning/eda_with_cleaning.py --input data/train --output results/cleaned --mode cleaning_with_eda
```

### **Option 3: Training Pipeline**
```bash
# Prepare training data
python scripts/training/prepare_training_data.py --input data/train --output data/training_prepared

# Train YOLO model
python scripts/training/train_stage1.py
```

### **Option 4: Backend API**
```bash
# Start the backend server
cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Use API endpoints for processing
curl -X POST "http://localhost:8000/upload-dataset" -F "files=@dataset.zip"
```

## 🔧 **Key Files Explained**

### **Main Entry Points:**
- **`scripts/core/run_stages.py`** - Main pipeline execution
- **`scripts/cleaning/eda_with_cleaning.py`** - EDA + Cleaning integration
- **`backend/app/main.py`** - FastAPI backend server

### **Core Services:**
- **`backend/app/services/unified_cleaning_service.py`** - All cleaning tasks
- **`backend/app/services/eda_service.py`** - Data analysis
- **`backend/app/services/stage_processor.py`** - Pipeline processing

### **Configuration:**
- **`configs/ps05_config.yaml`** - Main configuration
- **`configs/competition_config.yaml`** - Competition settings

## 📚 **Documentation Files**

- **`README.md`** - Main project overview
- **`README_QUICK_START.md`** - Quick start guide
- **`CLEANING_SERVICES_GUIDE.md`** - Cleaning services documentation
- **`EDA_CLEANING_INTEGRATION_GUIDE.md`** - EDA integration guide
- **`API_USAGE_GUIDE.md`** - API usage documentation

## 🎯 **Quick Start Path**

1. **Start Simple**: Use `scripts/core/run_stages.py`
2. **Add Quality**: Use `scripts/cleaning/eda_with_cleaning.py`
3. **Scale Up**: Use backend API for large datasets
4. **Customize**: Modify configs and add new features

## 🔍 **Finding What You Need**

- **Want to run the pipeline?** → `scripts/core/run_stages.py`
- **Want to clean data?** → `scripts/cleaning/eda_with_cleaning.py`
- **Want to train models?** → `scripts/training/`
- **Want to use API?** → `backend/app/main.py`
- **Need help?** → Check the guides in root directory

---

**💡 This structure keeps all functionality while making it easy to find what you need!**
