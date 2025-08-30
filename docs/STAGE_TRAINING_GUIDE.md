### Stage-wise Training Guide (DocLayNet and similar datasets)

## 0) Prerequisites
- Python env or Docker (GPU recommended)
- Data prepared (images + annotations or image-only for prediction)

## 1) EDA and Cleaning (optional but recommended)
- EDA only:
```bash
python scripts/cleaning/eda_with_cleaning.py \
  --input /path/to/raw_data \
  --output /path/to/eda_output \
  --mode eda_only
```
- Cleaning + EDA:
```bash
python scripts/cleaning/eda_with_cleaning.py \
  --input /path/to/raw_data \
  --output /path/to/cleaned_data \
  --mode cleaning_with_eda \
  --dataset-type images
```

## 2) Prepare dataset for Stage 1 (YOLO format)
```bash
python scripts/training/prepare_dataset.py \
  --data /path/to/cleaned_data \
  --output /path/to/yolo_dataset \
  --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1
```
- Output will include `dataset.yaml` under the output directory.

## 3) Train Stage 1 (YOLOv8 Layout Detection)
```bash
python scripts/training/train_yolo.py \
  --data /path/to/yolo_dataset/dataset.yaml \
  --output models/layout_detection \
  --epochs 50
```
- Check `models/layout_detection/layout_detection/weights/` for weights.

## 4) Evaluate/Inspect Stage 1
- Use YOLO’s built-in metrics and the generated plots in the output directory.
- Optionally run your validation pipeline.

## 5) Stage 2 (OCR + Language ID) fine-tuning
- OCR (EasyOCR/Tesseract) is not typically “trained” here; ensure cleaned, well-formed images.
- If you fine-tune a language classifier, place the model under `models/` and wire it in code (e.g., `core/models/langid_classifier.py`).

## 6) Stage 3 (Content Understanding) fine-tuning
- BLIP-2 and similar LMMs are large; for competition timelines, keep default checkpoints.
- If you fine-tune, cache models in `models/` and set envs:
```bash
export TRANSFORMERS_CACHE=/app/models
export HF_HOME=/app/models
```

## 7) Inference via API (using your new models)
- Mount your new models into the container and run processing:
```bash
docker run -d --rm --name ps05-backend-gpu -p 8000:8000 --gpus all \
  -e TRANSFORMERS_CACHE=/app/models -e HF_HOME=/app/models -e MPLCONFIGDIR=/tmp \
  -v /data/ps05_models:/app/models \
  -v /data/ps05_eval/EVAL_2025/images:/app/data/api_datasets/EVAL_2025/images:ro \
  -v /data/ps05_results:/app/data/api_results \
  ps05-backend:gpu

curl -X POST http://localhost:8000/process-all \
  -F "dataset_id=EVAL_2025" \
  -F "parallel_processing=true" \
  -F "max_workers=8" \
  -F "gpu_acceleration=true" \
  -F "batch_size=50" \
  -F "optimization_level=speed"
```

## 8) Tips for DocLayNet
- Ensure correct class mapping for Stage 1 (6 classes: Background, Text, Title, List, Table, Figure)
- Verify annotation conversion in `prepare_dataset.py` fits your source annotations; adjust mapping if needed.

## 9) Where to plug in custom models
- Stage 1: YOLO weights path (training output) → feed into your pipeline if you extend inference
- Stage 2: Replace or wrap OCR/langid components in `core` or `backend/app/services`
- Stage 3: Swap BLIP-2 checkpoint or add a wrapper in `OptimizedProcessingService`

## 10) Validate end-to-end
- Run a small subset through all three stages and inspect outputs before full-scale training or evaluation.

