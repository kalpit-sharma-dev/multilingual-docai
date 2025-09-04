## PS-05 Document AI - Knowledge Transfer Guide

This document provides a concise, end-to-end overview for onboarding, maintenance, and handover of the PS-05 multilingual document understanding project.

### 1) System Overview
- **Purpose**: Three-stage pipeline for document layout detection, OCR + language ID, and content understanding, producing per-image JSON.
- **Stack**: FastAPI backend (Python), GPU-accelerated ML (PyTorch, Ultralytics, Hugging Face), Docker/Docker Compose.
- **Primary entrypoint**: `backend/app/main.py` (FastAPI app with endpoints for upload, processing, evaluation, and monitoring).

### 2) Repository Layout (key paths)
- `backend/app/main.py`: FastAPI application and endpoints for dataset ops and processing.
- `backend/app/services/optimized_processing_service.py`: GPU-optimized 3-stage processing (parallelized). Loads/uses all core models.
- `backend/app/services/stage_processor.py`: Alternate stage-by-stage orchestrator with saving and memory cleanup.
- `backend/app/controllers/document_controller.py`: Document-level endpoints (upload, async processing, metrics).
- `backend/app/models/schemas.py`: Pydantic schemas for API IO (includes `BoundingBox {x,y,width,height}`).
- `core/`: Classic pipelines, evaluators, and models used by stage processor.
- `configs/`: Challenge and model configs (e.g., `competition_config.yaml`, `ps05_config.yaml`).
- `docs/`: Runbook and training guides; this KT doc.
- `docker-compose.gpu.yml`, `Dockerfile.gpu`: GPU deployment.
- `models/`: Cached/embedded model weights and artifacts.

### 3) Processing Pipeline
- Stage 1: Layout Detection
  - Default model: YOLOv8x (Ultralytics) reading weights via `YOLO_WEIGHTS` (e.g., `models/yolov8x.pt` or `models/layout_yolo_6class.pt`).
  - Optional refinement: LayoutLMv3 SequenceClassification head (6-class) via `LAYOUTLMV3_CHECKPOINT`.
  - Output: Elements with canonical labels [Background, Text, Title, List, Table, Figure] and bbox `[x,y,w,h]`.
- Stage 2: OCR + Language ID
  - OCR: EasyOCR by default; PaddleOCR (primary) if `USE_PADDLEOCR=1`.
  - Language ID: fastText `lid.176.bin` (176 langs) if present; fallback to `'en'`.
  - Output: Text lines with bbox `[x,y,w,h]`, confidence, language.
- Stage 3: Content Understanding
  - BLIP‑2 for image captioning (optional) via `BLIP2_CHECKPOINT`.
  - Pix2Struct for charts (optional) via `CHART_CAPTION_CHECKPOINT`.
  - Table-to-text via seq2seq LM (optional) `TABLE_T2T_CHECKPOINT` (e.g., FLAN‑T5).
  - Output: Whole-image caption and per-element descriptions (Tables/Figures/Charts) as available.

Primary implementation: `OptimizedProcessingService` methods
- `_detect_layout_gpu`, `_extract_text_and_language`, `_understand_content_gpu`
- `_refine_layout_with_layoutlm` (optional refinement)
- `_caption_from_pil`, `_caption_chart_from_pil`, `_table_to_text`

### 4) Models and Configuration
- YOLOv8x (Ultralytics) — layout detection; `YOLO_WEIGHTS` env.
- LayoutLMv3 (SequenceClassification + Processor) — optional; `LAYOUTLMV3_CHECKPOINT`.
- EasyOCR — default OCR; `EASYOCR_MODEL_PATH`.
- PaddleOCR — optional when `USE_PADDLEOCR=1`.
- fastText language ID — `lid.176.bin` at repo root or `/app/lid.176.bin`.
- BLIP‑2 — `BLIP2_CHECKPOINT` (e.g., `Salesforce/blip2-opt-2.7b`).
- Pix2Struct — `CHART_CAPTION_CHECKPOINT` (e.g., `google/pix2struct-textcaps-base`).
- Table T2T — `TABLE_T2T_CHECKPOINT` (e.g., `google/flan-t5-small`).
- Prefetch script: `scripts/utilities/fetch_models.py` downloads defaults into `models/`.

### 5) API Endpoints (selected)
- `POST /upload-dataset`: Upload images (and optional annotations) → stores under `data/api_datasets/<dataset_id>`.
- `POST /process-stage`: Run 1 stage with GPU optimization (batch size, speed vs memory).
- `POST /process-all`: Run all 3 stages in parallel (optimized for A100) and persist results.
- `POST /evaluate`: Compute mAP and related metrics (requires annotations).
- `GET /predictions/{dataset_id}`: Consolidated predictions for datasets without annotations.
- `GET /results/{dataset_id}`: Retrieve saved results.
- `GET /processing-stats`, `GET /training-stats`, `GET /status`, `GET /health`.

### 6) Deployment
- GPU single-container: `Dockerfile.gpu` + `docker run --gpus all` with volumes for datasets, models, results.
- Compose: `docker-compose.gpu.yml` with envs for optional models and GPU flags.
- Offline mode: put all model folders/files under `./models` prior to build; image bundles models for air‑gapped usage.

### 7) Monitoring and Ops
- Health and status endpoints provide system info (device, models loaded, counts, memory usage if CUDA available).
- Logs: `docker-compose -f docker-compose.gpu.yml logs -f ps05-gpu` or container shell.
- GPU: `nvidia-smi`, and FastAPI `/processing-stats`.

### 8) Data and Results
- Uploaded datasets: `data/api_datasets/<DATASET_ID>/{images,labels,annotations.json}`.
- Results: `data/api_results/<DATASET_ID>/...` with per-stage JSON files and GPU-optimized output directories.

### 9) Recent Changes You Should Know
- Bounding box standardization across pipeline to `[x, y, width, height]`:
  - Updated `OptimizedProcessingService` to emit YOLO results as `[x,y,w,h]`.
  - Replaced OCR quad converter with `_convert_quad_to_hbb_xywh(...)` and integrated it wherever OCR bboxes are produced.
  - Fixed region checks, crops, and per-region text collection to use `[x,y,w,h]` consistently.
  - Updated per-element captioning paths to crop with `[x,y,w,h]`.
  - Implication: All downstream consumers should expect `[x,y,w,h]`. This matches `BoundingBox` in `schemas.py`.

Locations touched for bbox consistency:
- `backend/app/services/optimized_processing_service.py`
  - Standardized YOLO result assembly to `[x,y,w,h]`.
  - New helper `_convert_quad_to_hbb_xywh` used for EasyOCR quadrilaterals.
  - `_collect_text_in_region`, `_refine_layout_with_layoutlm`, and captioning methods now operate on `[x,y,w,h]`.

Validation tip:
- Use `docs/EVALUATION_DAY_RUNBOOK.md` and `scripts/utilities/schema_check.py` to verify output schema.

### 10) How to Extend or Replace Models
- Stage 1: Swap YOLO weights via env or update loader in `optimized_processing_service.py`.
- Layout refinement: point `LAYOUTLMV3_CHECKPOINT` to a fine‑tuned 6-class head.
- Stage 2: Switch OCR engine preference by `USE_PADDLEOCR` or wrap your own in `core/models/ocr_engine.py`.
- Language ID: replace `lid.176.bin` with custom fastText model if desired.
- Stage 3: Provide fine‑tuned BLIP‑2/Pix2Struct/T2T checkpoints via env and mount paths.

### 11) Performance Settings (A100 defaults)
- Batch sizes: Stage 1/3 ≈ 50; Stage 2 slightly smaller if memory constrained.
- Mixed precision (FP16) and TF32 enabled for CUDA; memory fraction set around 0.9.
- Parallel tasks per batch in `process_dataset_parallel` for throughput.

### 12) Common Issues
- Missing heavy deps or models: endpoints may return 503 or warnings; ensure GPU image build and models present.
- OOM: reduce batch size via API params, clear caches, or disable optional models.
- Incorrect outputs: verify bbox orientation `[x,y,w,h]`, ensure env vars for optional checkpoints are set.

### 13) Quick Commands
- Build GPU image (with models cached):
  - `docker build --build-arg INSTALL_GPU_DEPS=1 -t ps05-backend:gpu .`
- Run processing:
  - `curl -X POST http://localhost:8000/process-all -F "dataset_id=<ID>" -F "batch_size=50" -F "optimization_level=speed"`
- Check stats:
  - `curl http://localhost:8000/processing-stats`

---

For deeper details, see `README.md`, `docs/EVALUATION_DAY_RUNBOOK.md`, and `docs/STAGE_TRAINING_GUIDE.md`.


