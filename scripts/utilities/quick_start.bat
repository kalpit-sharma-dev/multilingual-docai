@echo off
REM 🚀 PS-05 Quick Start Script for Windows

echo 🚀 Starting PS-05 Document Understanding System...

REM Check if we're in the right directory
if not exist "ps05.py" (
    echo [ERROR] Please run this script from the project root directory
    pause
    exit /b 1
)

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+ and add it to PATH
    pause
    exit /b 1
)

echo [SUCCESS] Python found

REM Create virtual environment
echo [INFO] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo [SUCCESS] Virtual environment created
) else (
    echo [WARNING] Virtual environment already exists
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo [INFO] Installing Python dependencies...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA support
echo [INFO] Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Install other dependencies
echo [INFO] Installing other dependencies...
pip install -r requirements.txt

REM Install additional packages
echo [INFO] Installing additional packages...
pip install ultralytics opencv-python PyMuPDF python-docx python-pptx tqdm pyyaml albumentations
pip install easyocr pytesseract langdetect transformers pycocotools jiwer sacrebleu bert-score

REM Check GPU availability
echo [INFO] Checking GPU availability...
python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo [WARNING] GPU check failed or CUDA not available
) else (
    echo [SUCCESS] GPU check completed
)

REM Check if training data exists
if exist "data\train" (
    echo [INFO] Training data found, you can now run Stage 1 training
    echo.
    echo To run Stage 1 training:
    echo   python scripts\train_stage1.py --data data\train --output outputs\stage1 --epochs 5 --batch-size 4
    echo.
    echo To test inference:
    echo   python ps05.py infer --input test_image.png --output results\ --stage 1
) else (
    echo [WARNING] No training data found in data\train\
    echo Please add your training dataset to data\train\ directory
)

REM Check if trained model exists
if exist "outputs\stage1_enhanced\training\layout_detector3\weights\best.pt" (
    echo [SUCCESS] Trained model found!
    echo You can now run inference with the trained model
) else (
    echo [WARNING] No trained model found
    echo You need to train the model first or download a pre-trained one
)

echo [SUCCESS] Setup complete! 🎉
echo.
echo Next steps:
echo 1. Add your training data to data\train\ (if not already done)
echo 2. Run Stage 1 training: python scripts\train_stage1.py --help
echo 3. Test inference: python ps05.py infer --help
echo 4. Start backend: cd backend ^&^& python run.py
echo 5. Start frontend: cd frontend ^&^& npm start
echo.
echo For detailed instructions, see docs\COMPLETE_PROJECT_GUIDE.md

pause
