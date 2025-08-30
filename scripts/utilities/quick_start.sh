#!/bin/bash
# 🚀 PS-05 Quick Start Script for Linux/macOS

echo "🚀 Starting PS-05 Document Understanding System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "ps05.py" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    print_error "Python 3.8+ required, found $python_version"
    exit 1
fi

print_success "Python version check passed: $python_version"

# Create virtual environment
print_status "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip

# Install PyTorch with CUDA support
print_status "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
print_status "Installing other dependencies..."
pip install -r requirements.txt

# Install additional packages
print_status "Installing additional packages..."
pip install ultralytics opencv-python PyMuPDF python-docx python-pptx tqdm pyyaml albumentations
pip install easyocr pytesseract langdetect transformers pycocotools jiwer sacrebleu bert-score

# Check GPU availability
print_status "Checking GPU availability..."
if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    gpu_info=$(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')" 2>/dev/null)
    print_success "GPU detected: $gpu_info"
else
    print_warning "GPU check failed or CUDA not available"
fi

# Check if training data exists
if [ -d "data/train" ]; then
    print_status "Training data found, you can now run Stage 1 training"
    echo ""
    echo "To run Stage 1 training:"
    echo "  python scripts/train_stage1.py --data data/train --output outputs/stage1 --epochs 5 --batch-size 4"
    echo ""
    echo "To test inference:"
    echo "  python ps05.py infer --input test_image.png --output results/ --stage 1"
else
    print_warning "No training data found in data/train/"
    echo "Please add your training dataset to data/train/ directory"
fi

# Check if trained model exists
if [ -f "outputs/stage1_enhanced/training/layout_detector3/weights/best.pt" ]; then
    print_success "Trained model found!"
    echo "You can now run inference with the trained model"
else
    print_warning "No trained model found"
    echo "You need to train the model first or download a pre-trained one"
fi

print_success "Setup complete! 🎉"
echo ""
echo "Next steps:"
echo "1. Add your training data to data/train/ (if not already done)"
echo "2. Run Stage 1 training: python scripts/train_stage1.py --help"
echo "3. Test inference: python ps05.py infer --help"
echo "4. Start backend: cd backend && python run.py"
echo "5. Start frontend: cd frontend && npm start"
echo ""
echo "For detailed instructions, see docs/COMPLETE_PROJECT_GUIDE.md"
