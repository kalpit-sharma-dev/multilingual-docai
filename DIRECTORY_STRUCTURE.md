# 📁 PS-05 Project Directory Structure

## 🏗️ **Complete Project Layout**

```
multilingual-docai/
├── 📁 CORE SYSTEM (Main Implementation)
│   ├── core/                           # Core ML models and processing
│   │   ├── models/                     # YOLOv8, LayoutLMv3, OCR models
│   │   ├── evaluation/                 # Evaluation metrics and tools
│   │   └── preprocessing/              # Image and document preprocessing
│   │
│   ├── backend/                        # FastAPI backend with all services
│   │   ├── app/                        # Main application
│   │   │   ├── main.py                 # FastAPI app with all endpoints
│   │   │   ├── services/               # All backend services
│   │   │   │   ├── unified_cleaning_service.py    # Main cleaning service
│   │   │   │   ├── eda_service.py                  # EDA analysis service
│   │   │   │   ├── image_cleaner.py                # Image cleaning (10 tasks)
│   │   │   │   ├── document_cleaner.py             # Document cleaning (11 tasks)
│   │   │   │   ├── stage_processor.py              # Pipeline processing
│   │   │   │   ├── evaluation_service.py           # Evaluation and metrics
│   │   │   │   ├── document_processor.py           # Document processing
│   │   │   │   └── file_manager.py                 # File management
│   │   │   ├── models/                # API data models
│   │   │   ├── controllers/            # API controllers
│   │   │   └── database/              # Database models
│   │   ├── config/                     # Backend configuration
│   │   └── run.py                      # Backend runner
│   │
│   └── configs/                        # Configuration files
│       ├── ps05_config.yaml            # Main configuration
│       ├── competition_config.yaml     # Competition settings
│       └── dataset_registry.yaml       # Dataset registry
│
├── 📁 SCRIPTS (Command Line Tools)
│   ├── core/                           # Core pipeline scripts
│   │   ├── run_stages.py               # Main 3-stage pipeline execution
│   │   └── ingest_datasets.py          # Dataset ingestion and preparation
│   │
│   ├── training/                       # Training and dataset preparation
│   │   ├── prepare_training_data.py    # Training data preparation with augmentation
│   │   ├── prepare_current_dataset.py  # Convert existing dataset to YOLO format
│   │   ├── prepare_dataset.py          # General dataset preparation
│   │   ├── train_stage1.py             # YOLO training script
│   │   └── train_yolo.py               # Alternative YOLO training
│   │
│   ├── cleaning/                       # Data cleaning and EDA
│   │   ├── eda_with_cleaning.py       # EDA + Cleaning integration script
│   │   ├── dataset_eda.py             # Standalone EDA analysis
│   │   └── enhanced_preprocessing.py   # Advanced preprocessing
│   │
│   └── utilities/                      # Utility and maintenance scripts
│       ├── cleanup.py                  # Repository cleanup
│       ├── pack_submission.py          # Create submission package
│       ├── quick_start.sh              # Quick start script (Linux/Mac)
│       └── quick_start.bat             # Quick start script (Windows)
│
├── 📁 DATA & MODELS
│   ├── data/                           # Dataset storage
│   │   ├── train/                      # Training dataset
│   │   ├── test/                       # Test dataset
│   │   └── validation/                 # Validation dataset
│   │
│   ├── models/                         # Trained models
│   ├── yolov8x.pt                      # Pre-trained YOLOv8 model
│   └── results/                        # Processing results
│       ├── stage1/                     # Stage 1 results (layout detection)
│       ├── stage2/                     # Stage 2 results (text extraction)
│       ├── stage3/                     # Stage 3 results (content understanding)
│       └── cleaning/                   # Cleaning and EDA results
│
├── 📁 DOCUMENTATION
│   ├── docs/                           # Technical documentation
│   └── guides/                         # User guides and tutorials
│
├── 📁 DEPLOYMENT
│   ├── docker/                         # Docker configuration
│   │   ├── Dockerfile                  # Main Dockerfile
│   │   └── docker-compose.yml          # Docker Compose configuration
│   │
│   └── frontend/                       # Web interface (if needed)
│
├── 📄 MAIN FILES
│   ├── ps05.py                         # Main entry point script
│   ├── README.md                        # Main project overview
│   ├── README_QUICK_START.md           # Quick start guide
│   ├── PROJECT_STRUCTURE.md            # This file - project organization
│   ├── DIRECTORY_STRUCTURE.md          # Detailed directory layout
│   ├── CLEANING_SERVICES_GUIDE.md      # Cleaning services documentation
│   ├── EDA_CLEANING_INTEGRATION_GUIDE.md # EDA integration guide
│   ├── API_USAGE_GUIDE.md              # API usage documentation
│   ├── API_USAGE_GUIDE_NO_ANNOTATIONS.md # No-annotation API guide
│   ├── CLEAN_REPOSITORY_GUIDE.md       # Repository cleanup guide
│   ├── requirements.txt                 # Python dependencies
│   ├── .gitignore                      # Git ignore file
│   └── LICENSE                         # Project license
│
└── 📁 EXTERNAL
    └── .git/                           # Git repository
```

## 🎯 **Key Entry Points**

### **1. Main Entry Point: `ps05.py`**
```bash
# Run the complete 3-stage pipeline
python ps05.py pipeline --stage all

# Data cleaning and EDA
python ps05.py clean --input data/train --output results/cleaned

# Training pipeline
python ps05.py train --prepare-data --train-model

# Backend API
python ps05.py backend --start
```

### **2. Direct Script Access**
```bash
# Core pipeline
python scripts/core/run_stages.py --stage all

# Data cleaning
python scripts/cleaning/eda_with_cleaning.py --mode cleaning_with_eda

# Training
python scripts/training/prepare_training_data.py
```

### **3. Backend API**
```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 🔧 **Service Architecture**

### **Backend Services (`backend/app/services/`)**
- **`unified_cleaning_service.py`** - Orchestrates all cleaning tasks
- **`eda_service.py`** - Exploratory Data Analysis
- **`image_cleaner.py`** - 10 image cleaning tasks
- **`document_cleaner.py`** - 11 document cleaning tasks
- **`stage_processor.py`** - 3-stage pipeline processing
- **`evaluation_service.py`** - Performance evaluation
- **`document_processor.py`** - Document processing core
- **`file_manager.py`** - File operations and management

### **Core Models (`core/models/`)**
- **YOLOv8** - Layout detection
- **LayoutLMv3** - Document understanding
- **OCR models** - Text extraction
- **Advanced models** - State-of-the-art implementations

## 📊 **Data Flow**

```
Input Data → EDA Analysis → Cleaning Services → Pipeline Processing → Results
    ↓              ↓              ↓              ↓              ↓
  Raw Files    Quality      Cleaned Data   3-Stage      JSON Output +
                Report                      Pipeline    Visualizations
```

## 🚀 **Getting Started Paths**

### **Path 1: Simple Pipeline (Recommended)**
1. Use `python ps05.py pipeline --stage all`
2. Check results in `results/` directory
3. Customize with configuration files

### **Path 2: Data Quality Focus**
1. Run EDA: `python ps05.py clean --mode eda_only`
2. Clean data: `python ps05.py clean --mode cleaning_with_eda`
3. Run pipeline on cleaned data

### **Path 3: Training Focus**
1. Prepare data: `python ps05.py train --prepare-data`
2. Train model: `python ps05.py train --train-model`
3. Use trained model in pipeline

### **Path 4: Production/API**
1. Start backend: `python ps05.py backend --start`
2. Use API endpoints for processing
3. Scale with Docker deployment

## 🔍 **Finding What You Need**

| **What You Want** | **Where to Look** | **How to Use** |
|-------------------|-------------------|----------------|
| **Run Pipeline** | `scripts/core/run_stages.py` | `python ps05.py pipeline` |
| **Clean Data** | `scripts/cleaning/eda_with_cleaning.py` | `python ps05.py clean` |
| **Train Models** | `scripts/training/` | `python ps05.py train` |
| **Use API** | `backend/app/main.py` | `python ps05.py backend --start` |
| **Configuration** | `configs/` | Edit YAML files |
| **Documentation** | Root directory | Read markdown guides |

## 💡 **Tips for Navigation**

1. **Start with `ps05.py`** - It's the main entry point for everything
2. **Use `PROJECT_STRUCTURE.md`** - For high-level understanding
3. **Check `DIRECTORY_STRUCTURE.md`** - For detailed file locations
4. **Read the guides** - They contain specific usage examples
5. **Explore `scripts/`** - Organized by functionality

---

**🎯 This structure keeps all functionality while making it easy to find what you need!**
