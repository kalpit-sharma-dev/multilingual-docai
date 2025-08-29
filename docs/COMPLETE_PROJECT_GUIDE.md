# 🚀 PS-05 Document Understanding System - Complete Project Guide

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Installation & Setup](#installation--setup)
5. [Data Analysis & Preprocessing](#data-analysis--preprocessing)
6. [Stage 1: Layout Detection](#stage-1-layout-detection)
7. [Stage 2: Text Extraction & OCR](#stage-2-text-extraction--ocr)
8. [Stage 3: Natural Language Generation](#stage-3-natural-language-generation)
9. [Backend API](#backend-api)
10. [Frontend Application](#frontend-application)
11. [Deployment](#deployment)
12. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

PS-05 is a comprehensive Document Understanding System that processes multilingual documents through three stages:

- **Stage 1**: Layout detection and element classification
- **Stage 2**: Text extraction and OCR processing
- **Stage 3**: Natural language generation and summarization

### Supported Document Types
- Images: PNG, JPG, JPEG, BMP, TIFF
- Documents: PDF, DOC, DOCX, PPT, PPTX
- Languages: English, Hindi, Urdu, Arabic, Nepali, Persian

---

## 🏗️ Repository Structure

```
clean_repo/
├── backend/                 # FastAPI backend server
│   ├── app/                # API endpoints and business logic
│   ├── config/             # Configuration files
│   ├── requirements.txt    # Python dependencies
│   └── run.py             # Server startup script
├── frontend/               # React Native mobile app
│   ├── components/         # Reusable UI components
│   ├── screens/            # App screens
│   ├── navigation/         # Navigation configuration
│   └── package.json       # Node.js dependencies
├── core/                   # Core ML components
│   ├── stage1/            # Layout detection models
│   ├── stage2/            # OCR and text processing
│   ├── stage3/            # NLG and summarization
│   ├── preprocessing/      # Data preprocessing modules
│   └── augment/            # Data augmentation scripts
├── scripts/                # Utility scripts
├── configs/                # Configuration files
├── data/                   # Data storage
├── results/                # Processing results
└── docs/                   # Documentation
```

---

## ⚙️ Prerequisites

### System Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **GPU**: NVIDIA GPU with CUDA support (RTX 2070 or better)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB free space

### Software Requirements
- **Python**: 3.8+ with pip
- **Node.js**: 16+ with npm
- **CUDA**: 11.8+ (for GPU acceleration)
- **Git**: Latest version

---

## 🚀 Installation & Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd clean_repo
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

### 4. Core Dependencies
```bash
cd ..
pip install -r requirements.txt
```

---

## 📊 Data Analysis & Preprocessing

### Exploratory Data Analysis (EDA)
```bash
# Run comprehensive dataset analysis
python scripts/dataset_eda.py --data data/train --output eda_results

# This generates:
# - eda_results/eda_report.md: Comprehensive analysis report
# - eda_results/eda_results.json: Raw analysis data
# - eda_results/*.png: Visualization plots
```

**EDA Features:**
- File format analysis and distribution
- Image properties (dimensions, rotation, quality)
- Annotation distribution and patterns
- Data quality assessment
- Visualizations and statistics

### Enhanced Data Preprocessing
```bash
# Comprehensive preprocessing pipeline
python scripts/enhanced_preprocessing.py --input data/train --output data/processed

# Features:
# - Multi-format support (PNG, JPG, PDF, DOC, PPT)
# - Image deskewing and rotation correction
# - Noise removal and denoising
# - Image enhancement and normalization
# - Format conversion to images
```

**Preprocessing Capabilities:**
- **Deskewing**: Automatic rotation detection and correction
- **Noise Removal**: Bilateral, Gaussian, and median filtering
- **Image Enhancement**: CLAHE, histogram equalization
- **Format Conversion**: PDF/DOC/PPT to image conversion
- **Quality Assessment**: Processing manifest and metadata

### Data Augmentation
```bash
# Synthetic data generation
python core/augment/synth_lines.py

# Document augmentation
python core/augment/doc_augs.py
```

**Augmentation Features:**
- **Synthetic Text Lines**: Multi-language text generation
- **Document Augmentation**: Perspective, rotation, noise, compression
- **Quality Variations**: Blur, brightness, contrast adjustments

### Preprocessing Modules
- **`core/preprocessing/preprocess.py`**: Basic preprocessing functions
- **`core/preprocessing/deskew.py`**: Advanced deskewing algorithms
- **`core/preprocessing/document_processor.py`**: Multi-format document handling

---

## 🎯 Stage 1: Layout Detection

### Overview
Stage 1 detects and classifies document layout elements into 6 categories:
- Background, Text, Title, List, Table, Figure

### Quick Start
```bash
# 1. Run EDA to understand your dataset
python scripts/dataset_eda.py --data data/train --output eda_results

# 2. Preprocess dataset
python scripts/enhanced_preprocessing.py --input data/train --output data/processed

# 3. Prepare YOLO dataset
python scripts/prepare_dataset.py --data data/processed --output data/yolo_dataset

# 4. Train model (GPU recommended)
python scripts/train_stage1.py --data data/processed --output outputs/stage1 --epochs 100 --batch-size 8

# 5. Test inference
python ps05.py infer --input test_image.png --output results/ --stage 1
```

### Complete Stage 1 Pipeline
```bash
# Full automation script
#!/bin/bash
echo "🚀 Starting Stage 1 Pipeline..."

# Environment setup
python -m venv ps05_env
source ps05_env/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python PyMuPDF python-docx python-pptx tqdm pyyaml albumentations
pip install easyocr pytesseract langdetect transformers pycocotools jiwer sacrebleu bert-score

# GPU verification
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"

# Data analysis and preprocessing
python scripts/dataset_eda.py --data data/train --output eda_results
python scripts/enhanced_preprocessing.py --input data/train --output data/enhanced_processed_complete

# Dataset preparation
python scripts/prepare_dataset.py --data data/enhanced_processed_complete --output data/yolo_enhanced_dataset

# Model training
python scripts/train_stage1.py --data data/enhanced_processed_complete --output outputs/stage1_enhanced --epochs 5 --batch-size 4 --create-submission

# Testing
python ps05.py infer --input data/enhanced_processed_complete/images/doc_00000_processed.png --output results/stage1_test --stage 1

# Submission package
python scripts/pack_submission.py --input data/enhanced_processed_complete/images/ --output results/stage1_submission --stage 1 --zip-name stage1_submission.zip

echo "✅ Stage 1 Complete!"
```

### Expected Outputs
- **EDA Results**: `eda_results/eda_report.md` and visualizations
- **Preprocessed Data**: `data/enhanced_processed_complete/`
- **Trained Model**: `outputs/stage1_enhanced/training/layout_detector3/weights/best.pt`
- **Results**: `results/stage1_test/result.json`
- **Submission**: `results/stage1_submission/stage1_submission.zip`

---

## 📝 Stage 2: Text Extraction & OCR

### Overview
Stage 2 extracts text from detected layout elements and performs language identification.

### Implementation
```bash
# Stage 2 is integrated into the main pipeline
python ps05.py infer --input document.png --output results/ --stage 2
```

### Features
- **Multilingual OCR**: English, Hindi, Urdu, Arabic, Nepali, Persian
- **Language Detection**: Automatic script and language identification
- **Text Region Extraction**: Precise bounding box text extraction
- **Quality Assessment**: Confidence scoring and error detection

---

## 🗣️ Stage 3: Natural Language Generation

### Overview
Stage 3 generates natural language descriptions and summaries of visual elements.

### Implementation
```bash
# Stage 3 is integrated into the main pipeline
python ps05.py infer --input document.png --output results/ --stage 3
```

### Features
- **Table Summarization**: Structured data to natural language
- **Chart Description**: Visual chart analysis and description
- **Map Analysis**: Geographic feature identification and description
- **Cross-Reference Detection**: Element relationship analysis

---

## 🔌 Backend API

### Start Server
```bash
cd backend
python run.py
```

### API Endpoints
- `POST /api/v1/process`: Process document through all stages
- `GET /api/v1/status`: Check processing status
- `GET /api/v1/results/{id}`: Retrieve processing results
- `POST /api/v1/upload`: Upload document for processing

### Configuration
Edit `backend/config/config.yaml` for:
- Model paths
- Processing parameters
- API settings

---

## 📱 Frontend Application

### Development Mode
```bash
cd frontend
npm start
```

### Build for Production
```bash
cd frontend
npm run build
```

### Features
- Document upload and processing
- Real-time processing status
- Results visualization
- Multi-language support

---

## 🚀 Deployment

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build
```

### Manual Deployment
```bash
# Backend
cd backend
python run.py --host 0.0.0.0 --port 8000

# Frontend
cd frontend
npm run build
serve -s build -l 3000
```

---

## 🔧 Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Model Training Fails
```bash
# Check GPU memory
nvidia-smi

# Reduce batch size
python scripts/train_stage1.py --batch-size 2

# Check dataset integrity
python scripts/prepare_dataset.py --data data/processed --output data/yolo_dataset
```

#### OCR Processing Errors
```bash
# Install language packs
pip install easyocr[all]

# Check image format
python -c "import cv2; img = cv2.imread('test.png'); print(img.shape)"
```

#### Preprocessing Issues
```bash
# Check document format support
python scripts/enhanced_preprocessing.py --help

# Verify preprocessing output
ls -la data/processed/images/
ls -la data/processed/annotations/
```

### Performance Optimization

#### GPU Memory Management
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Optimize batch size for your GPU
# RTX 2070 (8GB): batch_size = 4
# RTX 3080 (10GB): batch_size = 8
# RTX 4090 (24GB): batch_size = 16
```

#### Data Preprocessing
```bash
# Use parallel processing
python scripts/enhanced_preprocessing.py --input data/train --output data/processed --workers 4

# Optimize image resolution
python scripts/enhanced_preprocessing.py --max-width 1920 --max-height 1080
```

---

## 📊 Evaluation & Metrics

### Stage 1 Metrics
- **mAP**: Mean Average Precision at IoU ≥ 0.5
- **Precision**: True positive rate
- **Recall**: Detection completeness
- **F1-Score**: Harmonic mean of precision and recall

### Stage 2 Metrics
- **CER**: Character Error Rate
- **WER**: Word Error Rate
- **Language Accuracy**: Correct language identification rate

### Stage 3 Metrics
- **BLEU**: Text generation quality
- **BERTScore**: Semantic similarity
- **User Satisfaction**: Human evaluation scores

---

## 🔄 Complete Pipeline Execution

### Single Command Execution
```bash
# Process document through all stages
python ps05.py infer --input document.pdf --output results/ --stage 3
```

### Batch Processing
```bash
# Process multiple documents
python ps05.py infer --input data/documents/ --output results/ --batch --stage 3
```

### Custom Configuration
```bash
# Use custom config
python ps05.py infer --input document.png --output results/ --stage 3 --config configs/custom_config.yaml
```

---

## 📚 Additional Resources

### Documentation
- [API Reference](backend/README.md)
- [Frontend Guide](frontend/README.md)
- [Model Architecture](docs/MODEL_ARCHITECTURE.md)

### Training Data
- [Dataset Preparation](docs/DATASET_PREPARATION.md)
- [Data Augmentation](docs/DATA_AUGMENTATION.md)
- [Validation Strategies](docs/VALIDATION.md)

### Deployment
- [Docker Guide](docs/DOCKER_DEPLOYMENT.md)
- [Cloud Deployment](docs/CLOUD_DEPLOYMENT.md)
- [Performance Tuning](docs/PERFORMANCE_TUNING.md)

---

## 🆘 Support & Contact

### Issues
- Create GitHub issue for bugs
- Check troubleshooting section
- Review error logs in `logs/` directory

### Contributing
- Follow coding standards
- Add tests for new features
- Update documentation

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Last Updated**: December 2024  
**Version**: 1.0.0  
**Status**: Production Ready ✅
