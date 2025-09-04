### Evaluation Day Runbook (A100 GPU, Docker)

## 1) Prerequisites (host machine)
- NVIDIA drivers installed and NVIDIA Container Toolkit configured
- Docker installed and able to run `--gpus all`
- No outbound internet required if models are preloaded
- A large, fast disk for dataset and results (20GB+ dataset)

## 2) Prepare dataset on host
- Choose a dataset identifier: `<DATASET_ID>` (example: `EVAL_2025`)
- Create the expected layout:
  - `/data/ps05_eval/<DATASET_ID>/images` containing images: .jpg .jpeg .png .tif .tiff
- If your inputs are PDFs/DOCs/PPTs, pre-convert to images on the host for speed and consistency.
  - Example (PDF → PNG):
    ```bash
    mkdir -p /data/ps05_eval/<DATASET_ID>/images
    find /data/ps05_eval/pdfs -name "*.pdf" -print0 | while IFS= read -r -d '' f; do
      bn="$(basename "${f%.*}")"
      pdftoppm -png "$f" "/data/ps05_eval/<DATASET_ID>/images/${bn}"
    done
    ```

## 3) Build GPU image (offline-ready)
```bash
cd /path/to/multilingual-docai
docker build --build-arg INSTALL_GPU_DEPS=1 -t ps05-backend:gpu .
```

Optional: embed models into the image (offline)
- Place all required models in `./models` before build (YOLOv8 weights, LayoutLMv3, BLIP‑2, Pix2Struct, T2T‑Gen, fastText `lid.176.bin`).
- The Dockerfile copies `models/` into `/app/models` so the container does not need the internet.

## 4) Run container with volumes (offline)
```bash
docker run -d --rm --name ps05-backend-gpu -p 8000:8000 --gpus all \
  -e TRANSFORMERS_CACHE=/app/models -e HF_HOME=/app/models -e MPLCONFIGDIR=/tmp \
  -e USE_PADDLEOCR=1 \  # optional: PaddleOCR primary with EasyOCR fallback
  # Optional specialized models (mount and set if available)
  -e LAYOUTLMV3_CHECKPOINT=/app/models/layoutlmv3-6class \
  -e CHART_CAPTION_CHECKPOINT=/app/models/pix2struct-chart \
  -e TABLE_T2T_CHECKPOINT=/app/models/table-t2t \
  -v /data/ps05_eval/<DATASET_ID>/images:/app/data/api_datasets/<DATASET_ID>/images:ro \
  -v /data/ps05_results:/app/data/api_results \
  -v /data/ps05_models:/app/models \
  ps05-backend:gpu
```

Notes:
- `:ro` for dataset mount protects source data
- `models` volume persists Transformers/YOLO caches
- `results` volume persists outputs across restarts

## 5) Health check
```bash
curl http://localhost:8000/health
```

## 6) Process the dataset
- All stages (GPU, parallel):
```bash
curl -X POST http://localhost:8000/process-all \
  -F "dataset_id=<DATASET_ID>" \
  -F "parallel_processing=true" \
  -F "max_workers=8" \
  -F "gpu_acceleration=true" \
  -F "batch_size=50" \
  -F "optimization_level=speed"
```

Notes on outputs:
- Bounding boxes are standardized to `[x, y, w, h]` (HBB).
- Per-element captions are generated for Table/Figure; charts/maps under Figure use a chart model if provided, else BLIP-2.

## 7) Offline image save/load
```bash
# On your build machine (with models included)
docker save -o ps05-backend-gpu-offline.tar ps05-backend:gpu

# At the venue (no internet)
docker load -i ps05-backend-gpu-offline.tar
```

## 8) Offline models folder structure (example)
```
models/
  yolov8x.pt                         # YOLOv8 weights
  layoutlmv3-6class/                 # optional fine-tuned checkpoint
    config.json
    pytorch_model.bin
    preprocessor_config.json
    tokenizer.json (if applicable)
  blip2-opt-2.7b/                    # BLIP-2 files (if embedded)
    config.json ...
  pix2struct-chart/                  # optional chart captioner
    config.json ...
  table-t2t/                         # optional table-to-text model
    config.json ...
  lid.176.bin                        # fastText language ID
```

- Single stage (1, 2, or 3):
```bash
curl -X POST http://localhost:8000/process-stage \
  -F "dataset_id=<DATASET_ID>" \
  -F "stage=1" \
  -F "optimization_level=speed" \
  -F "batch_size=50" \
  -F "gpu_acceleration=true"
```

## 7) Retrieve results
- API:
```bash
curl http://localhost:8000/results/<DATASET_ID>
```
- On host (mounted):
```
/data/ps05_results/<DATASET_ID>/...
```

## 8) Alternative: docker compose (GPU)
- The repo has `docker-compose.gpu.yml` with standard mounts. Example to add a specific dataset mapping under `services.ps05-gpu.volumes`:
```yaml
    # Example: mount a specific evaluation dataset (read-only)
    # - /data/ps05_eval/<DATASET_ID>/images:/app/data/api_datasets/<DATASET_ID>/images:ro
```
- Bring up:
```bash
docker compose -f docker-compose.gpu.yml up -d --build
```

## 9) Troubleshooting quick checks
- 503 on processing endpoints: missing heavy dependency; verify GPU build and internet or pre-cached models
- Slow first run: large model downloads; persist `/app/models` across runs
- Path not found: ensure dataset is mounted at `/app/data/api_datasets/<DATASET_ID>/images`
- GPU not used: confirm `--gpus all` and host `nvidia-smi`

## 10) Cleanup
```bash
docker stop ps05-backend-gpu
```

## 11) Timed rehearsal and schema check
- Rehearsal (assumes dataset mounted inside container):
```bash
bash scripts/utilities/rehearsal.sh <DATASET_ID> http://localhost:8000
```
- Schema check:
```bash
python scripts/utilities/schema_check.py results/<DATASET_ID>
```


