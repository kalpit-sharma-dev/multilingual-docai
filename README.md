# 🚀 PS-05 Document Understanding System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive Document Understanding System that processes multilingual documents through three stages: Layout Detection, Text Extraction & OCR, and Natural Language Generation.

## 🎯 Features

- **Stage 1**: Layout detection and element classification (6 classes)
- **Stage 2**: Multilingual OCR with language identification
- **Stage 3**: Natural language generation and summarization
- **Multi-format Support**: PNG, JPG, PDF, DOC, DOCX, PPT, PPTX
- **Multi-language Support**: English, Hindi, Urdu, Arabic, Nepali, Persian
- **GPU Acceleration**: Optimized for NVIDIA GPUs (RTX 2070+)
- **Full-stack Application**: Backend API + Mobile Frontend

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Stage 1:      │    │   Stage 2:      │    │   Stage 3:      │
│   Layout        │───▶│   Text          │───▶│   Natural       │
│   Detection     │    │   Extraction    │    │   Language      │
│                 │    │   & OCR         │    │   Generation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA 11.8+
- Node.js 16+ (for frontend)

### 1. Clone Repository
```bash
git clone <repository-url>
cd clean_repo
```

### 2. Run Quick Start Script
```bash
# Linux/macOS
chmod +x scripts/quick_start.sh
./scripts/quick_start.sh

# Windows
scripts\quick_start.bat
```

### 3. Test Stage 1
```bash
# Test inference with trained model
python ps05.py infer --input test_image.png --output results/ --stage 1
```

## 📚 Documentation

- **[Complete Project Guide](docs/COMPLETE_PROJECT_GUIDE.md)** - Comprehensive guide for all stages
- **[API Reference](backend/README.md)** - Backend API documentation
- **[Frontend Guide](frontend/README.md)** - Mobile app documentation

## 🎯 Stage-by-Stage Usage

### Stage 1: Layout Detection
```bash
# Train model
python scripts/train_stage1.py --data data/train --output outputs/stage1 --epochs 100

# Run inference
python ps05.py infer --input document.png --output results/ --stage 1
```

### Stage 2: Text Extraction & OCR
```bash
# Process with OCR
python ps05.py infer --input document.png --output results/ --stage 2
```

### Stage 3: Natural Language Generation
```bash
# Complete pipeline
python ps05.py infer --input document.png --output results/ --stage 3
```

## 🔌 Backend API

```bash
cd backend
python run.py
```

**API Endpoints:**
- `POST /api/v1/process` - Process document through all stages
- `GET /api/v1/status` - Check processing status
- `GET /api/v1/results/{id}` - Retrieve results

## 📱 Frontend Application

```bash
cd frontend
npm install
npm start
```

**Features:**
- Document upload and processing
- Real-time status updates
- Results visualization
- Multi-language support

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up --build

# Or use individual containers
docker build -t ps05-backend backend/
docker build -t ps05-frontend frontend/
```

## 📊 Performance

### GPU Requirements
- **RTX 2070 (8GB)**: Batch size 4, ~42ms inference
- **RTX 3080 (10GB)**: Batch size 8, ~35ms inference
- **RTX 4090 (24GB)**: Batch size 16, ~25ms inference

### Model Performance
- **Stage 1**: mAP ≥ 0.85 at IoU threshold 0.5
- **Stage 2**: CER < 0.05, WER < 0.10
- **Stage 3**: BLEU ≥ 0.75, BERTScore ≥ 0.80

## 🔧 Configuration

Edit `configs/ps05_config.yaml` for:
- Model parameters
- Processing settings
- GPU optimization
- Language support

## 🆘 Troubleshooting

### Common Issues
1. **GPU Not Detected**: Check CUDA installation and PyTorch version
2. **Training Fails**: Reduce batch size for GPU memory constraints
3. **OCR Errors**: Install language packs with `pip install easyocr[all]`

### Performance Tips
- Use appropriate batch size for your GPU
- Enable mixed precision training for faster convergence
- Use data augmentation for better model generalization

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLOv8 for layout detection
- EasyOCR for multilingual text recognition
- Transformers for natural language generation
- FastAPI for backend framework
- React Native for mobile frontend

---

**Status**: Production Ready ✅  
**Last Updated**: December 2024  
**Version**: 1.0.0
