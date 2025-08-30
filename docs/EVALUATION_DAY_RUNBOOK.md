### Evaluation Day Runbook (A100 GPU, Docker)

## 1) Prerequisites (host machine)
- NVIDIA drivers installed and NVIDIA Container Toolkit configured
- Docker installed and able to run `--gpus all`
- Outbound internet (or pre-populated `models/` volume) for first-run model downloads
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

## 3) Build GPU image
```bash
cd /path/to/multilingual-docai
docker build --build-arg INSTALL_GPU_DEPS=1 -t ps05-backend:gpu .
```

## 4) Run container with volumes (recommended)
```bash
docker run -d --rm --name ps05-backend-gpu -p 8000:8000 --gpus all \
  -e TRANSFORMERS_CACHE=/app/models -e HF_HOME=/app/models -e MPLCONFIGDIR=/tmp \
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


