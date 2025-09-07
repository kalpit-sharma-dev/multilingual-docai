@echo off
setlocal enabledelayedexpansion

rem Stage 1 Fine-tuning helper for Windows (RTX 2070, 8GB)
rem Run this in Command Prompt/PowerShell from the repo root or double-click.

rem -------- Resolve repo root (absolute path) --------
for %%I in ("%~dp0\..\..") do set "REPO_ROOT=%%~fI\"

rem -------- Config (edit if needed) --------
set "INPUT_DIR=%REPO_ROOT%data\train"
set "YOLO_DATASET=%REPO_ROOT%yolo_dataset"
set "OUTPUT_DIR=%REPO_ROOT%models\layout_detection"
set "DEFAULT_WEIGHTS=%REPO_ROOT%models\layout_yolo_6class.pt"
set "PYEXE=python"
set "EPOCHS=100"
set "IMGSZ=640"
set "BATCH=8"
set "LR0=0.001"
set "WORKERS=0"
set "DEVICE=0"
rem ----------------------------------------

echo [train_stage1] Repo root: %REPO_ROOT%
echo [train_stage1] Input dir: %INPUT_DIR%
echo [train_stage1] YOLO dataset: %YOLO_DATASET%
echo [train_stage1] Output dir: %OUTPUT_DIR%

rem 0) Optional: print CUDA availability
%PYEXE% -c "import torch, sys; print('cuda?', torch.cuda.is_available(), 'cuda', getattr(torch.version,'cuda', None))" || echo [train_stage1] Warning: torch not available

rem 1) Prepare YOLO dataset from INPUT_DIR
echo [train_stage1] Preparing YOLO dataset...
%PYEXE% "%REPO_ROOT%scripts\training\prepare_dataset.py" --data "%INPUT_DIR%" --output "%YOLO_DATASET%" --train-ratio 0.8 --val-ratio 0.2 --test-ratio 0.0
if errorlevel 1 goto :error

rem 2) Choose starting weights
set "START_WEIGHTS=%DEFAULT_WEIGHTS%"
if not exist "%DEFAULT_WEIGHTS%" (
  echo [train_stage1] Default 6-class weights not found. Falling back to yolov8x.pt
  set "START_WEIGHTS=yolov8x.pt"
)
echo [train_stage1] Using weights: %START_WEIGHTS%

rem 3) Launch training (workers=0 on Windows to avoid DataLoader issues)
echo [train_stage1] Starting training...
%PYEXE% "%REPO_ROOT%scripts\training\train_yolo.py" ^
  --data "%YOLO_DATASET%\dataset.yaml" ^
  --output "%OUTPUT_DIR%" ^
  --epochs %EPOCHS% ^
  --weights "%START_WEIGHTS%" ^
  --imgsz %IMGSZ% ^
  --batch %BATCH% ^
  --lr0 %LR0% ^
  --workers %WORKERS% ^
  --device %DEVICE%
if errorlevel 1 goto :error

echo [train_stage1] Training completed successfully.
echo [train_stage1] Best weights (expected): %OUTPUT_DIR%\layout_detection\weights\best.pt
exit /b 0

:error
echo [train_stage1] ERROR: Training failed. Review logs above.
exit /b 1

