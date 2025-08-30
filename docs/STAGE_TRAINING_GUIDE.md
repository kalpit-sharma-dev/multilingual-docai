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
  -e USE_PADDLEOCR=1 \  # optional: PaddleOCR primary
  # Optional specialized models
  -e LAYOUTLMV3_CHECKPOINT=/app/models/layoutlmv3-6class \
  -e CHART_CAPTION_CHECKPOINT=/app/models/pix2struct-chart \
  -e TABLE_T2T_CHECKPOINT=/app/models/table-t2t \
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
- If you fine-tune LayoutLMv3 for 6-class refinement, set `LAYOUTLMV3_CHECKPOINT` and mount the weights.
- For higher table/chart scores, enable `TABLE_T2T_CHECKPOINT` and `CHART_CAPTION_CHECKPOINT` and mount the models.

## 9) Where to plug in custom models
- Stage 1: YOLO weights path (training output) → feed into your pipeline if you extend inference
- Stage 2: Replace or wrap OCR/langid components in `core` or `backend/app/services`
- Stage 3: Swap BLIP-2 checkpoint or add a wrapper in `OptimizedProcessingService`

## 10) Validate end-to-end
- Run a small subset through all three stages and inspect outputs before full-scale training or evaluation.

---

## 11) Fine-tuning optional models (online)

- YOLOv8 (layout detection)
```bash
# Prepare data
python scripts/training/prepare_dataset.py --data /data/my_docs --output /data/yolo_dataset
# Train (internet ok for pip/model pulls)
python scripts/training/train_yolo.py \
  --data /data/yolo_dataset/dataset.yaml \
  --output models/layout_detection \
  --epochs 100
```

- fastText (language ID)
```bash
# Train supervised model; each line: __label__en your text here ...
fasttext supervised -input lang_train.txt -output models/lid_custom -lr 0.5 -epoch 25 -wordNgrams 2
# Use by placing models/lid_custom.bin at /app/lid.176.bin (mount or copy)
```

- BLIP-2 (image captioning; PEFT/LoRA recommended)
```bash
# Example (Transformers + PEFT); assumes dataset of (image, text)
# 1) Install deps (internet)
pip install transformers peft accelerate datasets
# 2) Run your LoRA script (pseudo)
python train_blip2_lora.py \
  --base Salesforce/blip2-opt-2.7b \
  --data /data/blip2_pairs \
  --output models/blip2-finetuned
# Use by pointing loader to models/blip2-finetuned
```

- LayoutLMv3 (6-class refinement)
```bash
# Option A: Use API training endpoint
curl -X POST http://localhost:8000/train-layout-model \
  -F "train_data_dir=/app/datasets/train" \
  -F "val_data_dir=/app/datasets/val" \
  -F "output_dir=/app/models/layoutlmv3-6class" \
  -F "epochs=10" -F "batch_size=16" -F "learning_rate=1e-4" -F "mixed_precision=true"
# Option B: Transformers training script (custom); save to models/layoutlmv3-6class
# Enable with: -e LAYOUTLMV3_CHECKPOINT=/app/models/layoutlmv3-6class
```

- Pix2Struct (charts)
```bash
# Fine-tune Pix2StructForConditionalGeneration on chart (image, caption) pairs
# Save to models/pix2struct-chart; enable with: -e CHART_CAPTION_CHECKPOINT
```

- Table T2T (table → text)
```bash
# Fine-tune a seq2seq LM (e.g., t5-base) on pairs: "summarize table: <linearized>" → "<description>"
# Save to models/table-t2t; enable with: -e TABLE_T2T_CHECKPOINT
```

