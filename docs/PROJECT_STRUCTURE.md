# 🏗️ PS-05 Project Structure

## 📁 Directory Overview

```
clean_repo/
├── 📚 docs/                          # Documentation
│   ├── COMPLETE_PROJECT_GUIDE.md    # Main project guide
│   ├── PROJECT_STRUCTURE.md         # This file
│   ├── REFACTORING_SUMMARY.md       # Refactoring details
│   └── MODEL_ARCHITECTURE.md        # Model architecture details
├── 🔧 backend/                       # FastAPI backend server
│   ├── app/                         # API endpoints and business logic
│   │   ├── api/                     # API routes
│   │   ├── core/                    # Core functionality
│   │   ├── models/                  # Data models
│   │   └── services/                # Business logic services
│   ├── config/                      # Configuration files
│   ├── requirements.txt             # Python dependencies
│   └── run.py                      # Server startup script
├── 📱 frontend/                     # React Native mobile app
│   ├── components/                  # Reusable UI components
│   ├── screens/                     # App screens
│   ├── navigation/                  # Navigation configuration
│   ├── utils/                       # Utility functions
│   └── package.json                # Node.js dependencies
├── 🧠 core/                         # Core ML components
│   ├── stage1/                     # Layout detection models
│   │   ├── models/                 # Trained models
│   │   ├── training/               # Training outputs
│   │   └── weights/                # Model weights
│   ├── stage2/                     # OCR and text processing
│   ├── stage3/                     # NLG and summarization
│   ├── preprocessing/               # Data preprocessing modules
│   │   ├── preprocess.py           # Basic preprocessing functions
│   │   ├── deskew.py               # Advanced deskewing algorithms
│   │   ├── document_processor.py   # Multi-format document handling
│   │   └── data/                   # Additional preprocessing modules
│   └── augment/                     # Data augmentation scripts
│       ├── doc_augs.py             # Document augmentation
│       └── synth_lines.py          # Synthetic line generation
├── 📜 scripts/                      # Utility scripts
│   ├── dataset_eda.py              # Dataset Exploratory Data Analysis
│   ├── enhanced_preprocessing.py    # Comprehensive data preprocessing
│   ├── prepare_dataset.py          # Dataset preparation for YOLO
│   ├── train_stage1.py             # Stage 1 model training
│   ├── pack_submission.py          # Submission package creation
│   ├── quick_start.sh              # Linux/macOS setup
│   └── quick_start.bat             # Windows setup
├── ⚙️ configs/                      # Configuration files
│   ├── ps05_config.yaml            # Main configuration
│   ├── backend_config.yaml         # Backend settings
│   └── model_config.yaml           # Model parameters
├── 📊 data/                         # Data storage
│   ├── train/                      # Training dataset
│   ├── processed/                  # Processed data
│   └── yolo_dataset/               # YOLO format dataset
├── 📈 results/                      # Processing results
│   ├── stage1/                     # Stage 1 results
│   ├── stage2/                     # Stage 2 results
│   └── stage3/                     # Stage 3 results
├── 🚀 ps05.py                      # Main entry point
├── 📋 requirements.txt              # Python dependencies
└── 📖 README.md                    # Project overview
```

## 🔍 Detailed Component Breakdown

### 📚 Documentation (`docs/`)
- **COMPLETE_PROJECT_GUIDE.md**: Comprehensive guide covering all stages
- **PROJECT_STRUCTURE.md**: This detailed structure document
- **REFACTORING_SUMMARY.md**: What was refactored and why
- **MODEL_ARCHITECTURE.md**: Technical model architecture details

### 🔧 Backend (`backend/`)
The FastAPI-based backend server providing REST API endpoints.

#### Key Components:
- **`app/api/`**: API route definitions
- **`app/core/`**: Core business logic
- **`app/models/`**: Data models and schemas
- **`app/services/`**: Service layer for ML operations
- **`config/`**: Configuration management
- **`run.py`**: Server startup and configuration

#### API Endpoints:
- `POST /api/v1/process`: Process documents through all stages
- `GET /api/v1/status`: Check processing status
- `GET /api/v1/results/{id}`: Retrieve processing results
- `POST /api/v1/upload`: Upload documents for processing

### 📱 Frontend (`frontend/`)
React Native mobile application for document processing and results visualization.

#### Key Components:
- **`components/`**: Reusable UI components
- **`screens/`**: Main application screens
- **`navigation/`**: Navigation configuration
- **`utils/`**: Utility functions and helpers

#### Features:
- Document upload interface
- Real-time processing status
- Results visualization
- Multi-language support

### 🧠 Core ML (`core/`)
Core machine learning components organized by processing stages.

#### Stage 1: Layout Detection (`core/stage1/`)
- **Purpose**: Detect and classify document layout elements
- **Classes**: Background, Text, Title, List, Table, Figure
- **Model**: YOLOv8-based custom training
- **Output**: Bounding boxes with class labels and confidence scores

#### Stage 2: Text Extraction & OCR (`core/stage2/`)
- **Purpose**: Extract text from detected layout elements
- **Features**: Multilingual OCR, language identification
- **Languages**: English, Hindi, Urdu, Arabic, Nepali, Persian
- **Output**: Text regions with language and confidence information

#### Stage 3: Natural Language Generation (`core/stage3/`)
- **Purpose**: Generate natural language descriptions
- **Features**: Table summarization, chart description, map analysis
- **Output**: Natural language summaries and descriptions

#### Preprocessing (`core/preprocessing/`)
- **Purpose**: Data preprocessing and augmentation
- **Features**: Deskewing, noise removal, format conversion
- **Support**: Multiple document formats (PDF, DOC, PPT, images)
- **Key Modules**:
  - **`preprocess.py`**: Basic preprocessing functions (deskew, resize, normalize)
  - **`deskew.py`**: Advanced deskewing with multiple algorithms
  - **`document_processor.py`**: Multi-format document processing

#### Data Augmentation (`core/augment/`)
- **Purpose**: Generate synthetic data and augment existing datasets
- **Features**:
  - **`doc_augs.py`**: Document augmentation (perspective, rotation, noise)
  - **`synth_lines.py`**: Synthetic multi-language text line generation

### 📜 Scripts (`scripts/`)
Utility scripts for various operations.

#### Key Scripts:
- **`dataset_eda.py`**: Comprehensive dataset analysis and visualization
- **`enhanced_preprocessing.py`**: Multi-format document preprocessing pipeline
- **`prepare_dataset.py`**: Convert annotations to YOLO format
- **`train_stage1.py`**: Stage 1 model training
- **`pack_submission.py`**: Create submission packages
- **`quick_start.sh/.bat`**: Automated setup scripts

#### EDA & Preprocessing Scripts:
- **`dataset_eda.py`**: 
  - File format analysis and distribution
  - Image properties (dimensions, rotation, quality)
  - Annotation distribution and patterns
  - Data quality assessment
  - Visualization plots (PNG format)

- **`enhanced_preprocessing.py`**:
  - Multi-format document support
  - Image deskewing and rotation correction
  - Noise removal and denoising
  - Image enhancement and normalization
  - Format conversion to images
  - Processing manifest generation

### ⚙️ Configuration (`configs/`)
Configuration files for different components.

#### Key Configs:
- **`ps05_config.yaml`**: Main system configuration
- **`backend_config.yaml`**: Backend server settings
- **`model_config.yaml`**: Model parameters and hyperparameters

### 📊 Data (`data/`)
Data storage and organization.

#### Structure:
- **`train/`**: Raw training dataset
- **`processed/`**: Preprocessed data
- **`yolo_dataset/`**: YOLO format dataset for training

### 📈 Results (`results/`)
Processing results and outputs.

#### Organization:
- **`stage1/`**: Layout detection results
- **`stage2/`**: OCR and text extraction results
- **`stage3/`**: Natural language generation results

## 🔄 Data Flow

```
Input Document → EDA Analysis → Preprocessing → Stage 1 → Stage 2 → Stage 3 → Results
      ↓              ↓              ↓           ↓        ↓        ↓        ↓
   PDF/Image    Quality      Enhanced    Layout     Text      NLG     JSON
   Files        Report       Images      Detection  Regions   Output   Results
```

## 🎯 Key Files

### Main Entry Points:
- **`ps05.py`**: Command-line interface for all operations
- **`backend/run.py`**: Backend server startup
- **`frontend/App.tsx`**: Frontend application entry

### Configuration:
- **`configs/ps05_config.yaml`**: Central configuration
- **`requirements.txt`**: Python dependencies
- **`frontend/package.json`**: Node.js dependencies

### Training & Models:
- **`scripts/train_stage1.py`**: Model training script
- **`core/stage1/weights/best.pt`**: Trained model weights
- **`scripts/enhanced_preprocessing.py`**: Data preprocessing

### EDA & Analysis:
- **`scripts/dataset_eda.py`**: Dataset analysis and visualization
- **`core/preprocessing/deskew.py`**: Advanced deskewing algorithms
- **`core/preprocessing/document_processor.py`**: Multi-format processing

## 🚀 Getting Started

1. **Clone Repository**: `git clone <url> && cd clean_repo`
2. **Run Quick Start**: `./scripts/quick_start.sh` (Linux/macOS) or `scripts\quick_start.bat` (Windows)
3. **Analyze Dataset**: `python scripts/dataset_eda.py --data data/train --output eda_results`
4. **Preprocess Data**: `python scripts/enhanced_preprocessing.py --input data/train --output data/processed`
5. **Test Stage 1**: `python ps05.py infer --input test.png --output results/ --stage 1`
6. **Start Backend**: `cd backend && python run.py`
7. **Start Frontend**: `cd frontend && npm start`

## 📝 File Naming Conventions

- **Directories**: lowercase with underscores (`stage1`, `data_processing`)
- **Python Files**: lowercase with underscores (`train_stage1.py`)
- **Configuration Files**: lowercase with underscores (`ps05_config.yaml`)
- **Model Files**: descriptive names (`best.pt`, `layout_detector.pt`)
- **Documentation**: UPPERCASE with underscores (`COMPLETE_PROJECT_GUIDE.md`)

## 🔧 Development Workflow

1. **Feature Development**: Create feature branch from main
2. **Testing**: Run tests and validate functionality
3. **Documentation**: Update relevant documentation
4. **Code Review**: Submit pull request for review
5. **Integration**: Merge to main branch after approval

## 📊 Performance Considerations

- **GPU Memory**: Optimize batch sizes for available GPU memory
- **Data Loading**: Use efficient data loading and preprocessing
- **Model Optimization**: Enable mixed precision training and inference
- **Caching**: Implement result caching for repeated operations

## 🔍 EDA & Preprocessing Capabilities

### Dataset Analysis:
- **File Format Analysis**: Support for PNG, JPG, PDF, DOC, PPT
- **Image Properties**: Dimensions, rotation, quality metrics
- **Annotation Analysis**: Distribution, patterns, quality assessment
- **Visualization**: Automatic plot generation and reports

### Preprocessing Pipeline:
- **Multi-format Support**: Handle various document types
- **Image Enhancement**: Deskewing, noise removal, normalization
- **Quality Control**: Processing manifest and metadata
- **Batch Processing**: Efficient handling of large datasets

### Data Augmentation:
- **Synthetic Generation**: Multi-language text lines
- **Document Variations**: Rotation, noise, compression effects
- **Quality Variations**: Blur, brightness, contrast adjustments

---

**Last Updated**: December 2024  
**Version**: 1.0.0
