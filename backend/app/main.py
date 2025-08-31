#!/usr/bin/env python3
"""
PS-05 Backend API - Complete Solution

Handles all 3 stages:
- Stage 1: Layout Detection (mAP evaluation)
- Stage 2: Text Extraction + Language ID
- Stage 3: Content Understanding + Natural Language

Features:
- Dataset upload via API (with or without annotations)
- Stage-by-stage processing
- mAP calculation (when annotations available)
- JSON output generation for evaluation
- Docker-ready
- Handles large datasets (20GB+)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import shutil
import uuid
from datetime import datetime

# NOTE: Heavy services are imported lazily to allow the API to start even if
# optional ML dependencies aren't installed in the current image.
from .models.schemas import (
    ProcessingRequest, ProcessingResponse, StageResult, 
    DatasetUploadResponse, EvaluationResult, NoAnnotationResponse
)

# Type-only imports (avoid importing heavy deps at runtime)
try:  # typing guard for editors; does nothing at runtime if missing
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from .services.document_processor import DocumentProcessor
        from .services.stage_processor import StageProcessor
        from .services.evaluation_service import EvaluationService
        from .services.unified_cleaning_service import UnifiedCleaningService
        from .services.gpu_training_service import GPUTrainingService, TrainingConfig
        from .services.optimized_processing_service import OptimizedProcessingService, ProcessingConfig
except Exception:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PS-05 Document Understanding API",
    description="Complete 3-stage document understanding pipeline with mAP evaluation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singletons for heavy services
document_processor = None
stage_processor = None
evaluation_service = None
cleaning_service = None
gpu_training_service = None
optimized_service = None


def get_stage_processor():
    global stage_processor
    if stage_processor is None:
        try:
            from .services.stage_processor import StageProcessor
            stage_processor = StageProcessor()
        except Exception as e:
            logging.getLogger(__name__).warning(f"StageProcessor unavailable: {e}")
            stage_processor = None
    return stage_processor


def get_evaluation_service():
    global evaluation_service
    if evaluation_service is None:
        try:
            from .services.evaluation_service import EvaluationService
            evaluation_service = EvaluationService()
        except Exception as e:
            logging.getLogger(__name__).warning(f"EvaluationService unavailable: {e}")
            evaluation_service = None
    return evaluation_service


def get_cleaning_service():
    global cleaning_service
    if cleaning_service is None:
        try:
            from .services.unified_cleaning_service import UnifiedCleaningService
            cleaning_service = UnifiedCleaningService()
        except Exception as e:
            logging.getLogger(__name__).warning(f"UnifiedCleaningService unavailable: {e}")
            cleaning_service = None
    return cleaning_service


def get_gpu_training_service():
    global gpu_training_service
    if gpu_training_service is None:
        try:
            from .services.gpu_training_service import GPUTrainingService
            gpu_training_service = GPUTrainingService()
        except Exception as e:
            logging.getLogger(__name__).warning(f"GPUTrainingService unavailable: {e}")
            gpu_training_service = None
    return gpu_training_service


def create_optimized_service(config=None):
    try:
        from .services.optimized_processing_service import OptimizedProcessingService
        return OptimizedProcessingService(config)
    except Exception as e:
        logging.getLogger(__name__).error(f"OptimizedProcessingService unavailable: {e}")
        raise HTTPException(status_code=503, detail="Optimized processing service unavailable. Ensure ML dependencies are installed.")

# Global storage for datasets and results
DATASET_STORAGE = Path("data/api_datasets")
RESULTS_STORAGE = Path("data/api_results")
DATASET_STORAGE.mkdir(parents=True, exist_ok=True)
RESULTS_STORAGE.mkdir(parents=True, exist_ok=True)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "PS-05 Document Understanding API",
        "version": "1.0.0",
        "stages": ["Layout Detection", "Text Extraction + Language ID", "Content Understanding"],
        "evaluation_modes": ["With Annotations (mAP)", "Without Annotations (Prediction Only)"],
        "endpoints": {
            "upload_dataset": "/upload-dataset",
            "process_stage": "/process-stage (GPU Optimized)",
            "process_all": "/process-all (GPU Optimized)",
            "evaluate": "/evaluate",
            "get_results": "/results/{dataset_id}",
            "get_predictions": "/predictions/{dataset_id}",
            "status": "/status",
            "clean_dataset": "/clean-dataset",
            "cleaning_capabilities": "/cleaning-capabilities",
            "run_eda": "/run-eda",
            "eda_results": "/eda-results/{dataset_id}",
            "training": "/train-layout-model, /train-yolo-model",
            "gpu_monitoring": "/training-stats, /processing-stats"
        },
        "gpu_optimization": {
            "enabled": True,
            "target_gpu": "NVIDIA A100",
            "batch_size": "50 images (A100 optimized)",
            "parallel_processing": True,
            "mixed_precision": "FP16 enabled"
        }
    }

@app.get("/status")
async def get_status():
    """Get system status and available models."""
    try:
        status = {
            "api_status": "running",
            "timestamp": datetime.now().isoformat(),
            "available_models": (get_stage_processor().get_available_models() if get_stage_processor() else []),
            "storage": {
                "datasets": len(list(DATASET_STORAGE.iterdir())),
                "results": len(list(RESULTS_STORAGE.iterdir())),
                "total_size_gb": sum(f.stat().st_size for f in DATASET_STORAGE.rglob('*') if f.is_file()) / (1024**3)
            },
            "max_dataset_size_gb": 50,  # Support up to 50GB datasets
            "supported_formats": ["PNG", "JPEG", "JPG", "TIFF"],
            "processing_capabilities": {
                "batch_processing": True,
                "background_jobs": True,
                "memory_optimization": True
            }
        }
        return status
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-dataset", response_model=DatasetUploadResponse)
async def upload_dataset(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    annotations: Optional[UploadFile] = File(None),
    dataset_name: Optional[str] = None
):
    """
    Upload dataset for processing.
    
    - files: List of document images (PNG/JPEG) - supports large datasets (20GB+)
    - annotations: Optional ground truth annotations for evaluation
    - dataset_name: Optional name for the dataset
    """
    try:
        # Generate unique dataset ID
        dataset_id = str(uuid.uuid4())
        dataset_dir = DATASET_STORAGE / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        uploaded_files = []
        total_size_bytes = 0
        
        # Save uploaded images with progress tracking
        for file in files:
            if file.content_type.startswith("image/"):
                file_path = images_dir / file.filename
                
                # Stream large files to avoid memory issues
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # Get file size
                file_size = file_path.stat().st_size
                total_size_bytes += file_size
                uploaded_files.append({
                    "filename": file.filename,
                    "size_bytes": file_size,
                    "size_mb": file_size / (1024**2)
                })
        
        # Save annotations if provided
        annotation_data = None
        if annotations:
            annotation_path = dataset_dir / "annotations.json"
            with open(annotation_path, "wb") as buffer:
                shutil.copyfileobj(annotations.file, buffer)
            
            # Load and validate annotations
            with open(annotation_path, 'r') as f:
                annotation_data = json.load(f)
        
        # Create dataset info
        dataset_info = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name or f"Dataset_{dataset_id[:8]}",
            "upload_time": datetime.now().isoformat(),
            "num_images": len(uploaded_files),
            "total_size_gb": total_size_bytes / (1024**3),
            "has_annotations": annotation_data is not None,
            "images": uploaded_files,
            "evaluation_mode": "with_annotations" if annotation_data else "prediction_only"
        }
        
        # Save dataset info
        with open(dataset_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info(f"Dataset uploaded: {dataset_id} with {len(uploaded_files)} images ({total_size_bytes/(1024**3):.2f} GB)")
        
        return DatasetUploadResponse(
            dataset_id=dataset_id,
            message="Dataset uploaded successfully",
            num_images=len(uploaded_files),
            has_annotations=annotation_data is not None,
            total_size_gb=total_size_bytes / (1024**3),
            evaluation_mode="with_annotations" if annotation_data else "prediction_only"
        )
        
    except Exception as e:
        logger.error(f"Dataset upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-stage", tags=["Processing"])
async def process_stage(
    dataset_id: str = Form(...),
    stage: int = Form(...),
    optimization_level: str = Form("speed"),
    batch_size: int = Form(50),
    gpu_acceleration: bool = Form(True)
):
    """
    Process a specific stage for a dataset with GPU optimization.
    
    **Available Stages:**
    - **Stage 1**: Layout Detection (YOLOv8, LayoutLMv3, Mask R-CNN)
    - **Stage 2**: Text Extraction + Language Identification (EasyOCR, Tesseract, fastText)
    - **Stage 3**: Content Understanding + Natural Language Generation (Table Transformer, BLIP, OFA)
    
    **Optimization Options:**
    - **optimization_level**: "speed" (default) or "memory"
    - **batch_size**: Number of images per batch (default: 50 for A100)
    - **gpu_acceleration**: Enable GPU acceleration (default: True)
    """
    try:
        # Get dataset path (images live under data/api_datasets/{id}/images)
        dataset_path = str(DATASET_STORAGE / dataset_id / "images")
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Configure optimization
        from .services.optimized_processing_service import ProcessingConfig  # type: ignore
        config = ProcessingConfig(
            batch_size=batch_size,
            gpu_acceleration=gpu_acceleration,
            memory_optimization=(optimization_level == "memory")
        )
        
        # Create optimized service instance
        service = create_optimized_service(config)
        
        # Process specific stage
        if stage == 1:
            # Layout detection
            results = await service._detect_layout_gpu(dataset_path)
        elif stage == 2:
            # Text extraction + Language ID
            results = await service._extract_text_and_language(dataset_path)
        elif stage == 3:
            # Content understanding
            results = await service._understand_content_gpu(dataset_path)
        else:
            raise HTTPException(status_code=400, detail="Invalid stage number")
        
        return {
            "status": "completed",
            "dataset_id": dataset_id,
            "stage": stage,
            "results": results,
            "gpu_stats": service.get_processing_stats(),
            "optimization": {
                "level": optimization_level,
                "batch_size": batch_size,
                "gpu_acceleration": gpu_acceleration
            }
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in stage processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-all", tags=["Processing"])
async def process_all_stages(
    dataset_id: str = Form(...),
    parallel_processing: bool = Form(True),
    max_workers: int = Form(8),
    gpu_acceleration: bool = Form(True),
    batch_size: int = Form(50),
    optimization_level: str = Form("speed")
):
    """
    Process all 3 stages for a dataset with GPU optimization.
    This runs in the background and returns a job ID.
    Optimized for large datasets (20GB+) and A100 GPU.
    
    **Processing Flow:**
    1. Layout Detection → 2. Text Extraction + Language ID → 3. Content Understanding
    
    **Optimization Options:**
    - **parallel_processing**: Run all stages simultaneously (default: True)
    - **max_workers**: Number of parallel workers (default: 8)
    - **gpu_acceleration**: Enable GPU acceleration (default: True)
    - **batch_size**: Images per batch (default: 50 for A100)
    - **optimization_level**: "speed" (default) or "memory"
    """
    try:
        # Get dataset path (images live under data/api_datasets/{id}/images)
        dataset_path = str(DATASET_STORAGE / dataset_id / "images")
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Configure for maximum speed
        from .services.optimized_processing_service import ProcessingConfig  # type: ignore
        config = ProcessingConfig(
            batch_size=batch_size,
            max_workers=max_workers,
            gpu_acceleration=gpu_acceleration,
            memory_optimization=(optimization_level == "memory"),
            save_intermediate=True
        )
        
        # Create optimized service
        service = create_optimized_service(config)
        
        # Start parallel processing
        output_dir = str(RESULTS_STORAGE / dataset_id / "gpu_optimized")
        results = await service.process_dataset_parallel(dataset_path, output_dir)
        
        return {
            "status": "completed",
            "dataset_id": dataset_id,
            "total_images": results["total_images"],
            "processing_time": results["processing_time"],
            "speed_images_per_second": results["speed_images_per_second"],
            "output_directory": results["output_directory"],
            "gpu_stats": service.get_processing_stats(),
            "optimization": {
                "parallel_processing": parallel_processing,
                "max_workers": max_workers,
                "gpu_acceleration": gpu_acceleration,
                "batch_size": config.batch_size,
                "level": optimization_level
            }
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate", response_model=EvaluationResult)
async def evaluate_dataset(dataset_id: str):
    """
    Evaluate dataset performance and calculate mAP scores.
    Requires ground truth annotations.
    """
    try:
        dataset_dir = DATASET_STORAGE / dataset_id
        if not dataset_dir.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Check if annotations exist
        annotation_path = dataset_dir / "annotations.json"
        if not annotation_path.exists():
            raise HTTPException(
                status_code=400, 
                detail="No ground truth annotations found for evaluation. Use /predictions endpoint for prediction-only datasets."
            )
        
        # Run evaluation
        service = get_evaluation_service()
        if service is None:
            raise HTTPException(status_code=503, detail="Evaluation service unavailable")
        evaluation_result = await service.evaluate_dataset(
            dataset_id=dataset_id
        )
        
        return evaluation_result
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/{dataset_id}", response_model=NoAnnotationResponse)
async def get_predictions(dataset_id: str):
    """
    Get predictions for datasets without annotations.
    This is the main endpoint for evaluator testing.
    """
    try:
        dataset_dir = DATASET_STORAGE / dataset_id
        results_dir = RESULTS_STORAGE / dataset_id
        
        if not dataset_dir.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if not results_dir.exists():
            raise HTTPException(
                status_code=400, 
                detail="No results found. Please process the dataset first using /process-all endpoint."
            )
        
        # Load all stage results
        predictions = {}
        for result_file in results_dir.glob("stage_*_results.json"):
            stage = result_file.stem.split("_")[1]  # Extract stage number
            with open(result_file, 'r') as f:
                predictions[f"stage_{stage}"] = json.load(f)
        
        # Get dataset info
        dataset_info_path = dataset_dir / "dataset_info.json"
        dataset_info = {}
        if dataset_info_path.exists():
            with open(dataset_info_path, 'r') as f:
                dataset_info = json.load(f)
        
        return NoAnnotationResponse(
            dataset_id=dataset_id,
            dataset_name=dataset_info.get("dataset_name", "Unknown"),
            num_images=dataset_info.get("num_images", 0),
            total_size_gb=dataset_info.get("total_size_gb", 0),
            predictions=predictions,
            timestamp=datetime.now().isoformat(),
            message="Predictions generated successfully. No ground truth available for mAP calculation."
        )
        
    except Exception as e:
        logger.error(f"Failed to get predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{dataset_id}")
async def get_results(dataset_id: str):
    """Get processing results for a dataset."""
    try:
        results_dir = RESULTS_STORAGE / dataset_id
        if not results_dir.exists():
            raise HTTPException(status_code=404, detail="Results not found")
        
        # Load results
        results = {}
        for result_file in results_dir.glob("*.json"):
            with open(result_file, 'r') as f:
                results[result_file.stem] = json.load(f)
        
        return {
            "dataset_id": dataset_id,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/datasets")
async def list_datasets():
    """List all uploaded datasets."""
    try:
        datasets = []
        for dataset_dir in DATASET_STORAGE.iterdir():
            if dataset_dir.is_dir():
                info_path = dataset_dir / "dataset_info.json"
                if info_path.exists():
                    with open(info_path, 'r') as f:
                        datasets.append(json.load(f))
        
        return {"datasets": datasets}
        
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset and its results."""
    try:
        dataset_dir = DATASET_STORAGE / dataset_id
        results_dir = RESULTS_STORAGE / dataset_id
        
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        
        if results_dir.exists():
            shutil.rmtree(results_dir)
        
        return {"message": f"Dataset {dataset_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Failed to delete dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clean-dataset")
async def clean_dataset(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    dataset_name: Optional[str] = None,
    dataset_type: str = "auto"
):
    """
    Clean dataset using comprehensive image and document cleaning.
    
    - files: List of files to clean (images and/or documents)
    - dataset_name: Optional name for the dataset
    - dataset_type: "auto", "images", "documents", or "mixed"
    """
    try:
        # Generate dataset ID
        dataset_id = str(uuid.uuid4())
        dataset_dir = DATASET_STORAGE / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded files
        total_size_bytes = 0
        for file in files:
            file_path = dataset_dir / file.filename
            with open(file_path, "wb") as buffer:
                content = file.file.read()
                buffer.write(content)
                total_size_bytes += len(content)
        
        # Save dataset info
        dataset_info = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name or f"dataset_{dataset_id[:8]}",
            "num_files": len(files),
            "total_size_bytes": total_size_bytes,
            "total_size_gb": total_size_bytes / (1024**3),
            "upload_timestamp": datetime.now().isoformat(),
            "dataset_type": dataset_type
        }
        
        with open(dataset_dir / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f)
        
        # Start cleaning in background
        # Ensure cleaning service is initialized lazily
        _ = get_cleaning_service()
        background_tasks.add_task(
            _clean_dataset_background,
            dataset_id,
            dataset_dir,
            dataset_type
        )
        
        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset_info["dataset_name"],
            "num_files": len(files),
            "total_size_gb": dataset_info["total_size_gb"],
            "status": "cleaning_started",
            "message": "Dataset cleaning started in background"
        }
        
    except Exception as e:
        logger.error(f"Dataset cleaning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _clean_dataset_background(dataset_id: str, dataset_dir: Path, dataset_type: str):
    """Background task for dataset cleaning."""
    try:
        # Create output directory for cleaned data
        output_dir = RESULTS_STORAGE / dataset_id / "cleaned_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run cleaning
        cleaning_results = cleaning_service.clean_dataset(
            input_dir=dataset_dir,
            output_dir=output_dir,
            dataset_type=dataset_type
        )
        
        # Save cleaning results
        with open(output_dir / "cleaning_results.json", "w") as f:
            json.dump(cleaning_results, f)
        
        logger.info(f"Dataset {dataset_id} cleaning completed")
        
    except Exception as e:
        logger.error(f"Background cleaning failed for {dataset_id}: {e}")

@app.get("/cleaning-capabilities")
async def get_cleaning_capabilities():
    """Get information about cleaning capabilities."""
    try:
        capabilities = cleaning_service.get_cleaning_capabilities()
        return {
            "cleaning_capabilities": capabilities,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get cleaning capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cleaning-status/{dataset_id}")
async def get_cleaning_status(dataset_id: str):
    """Get cleaning status for a specific dataset."""
    try:
        status = cleaning_service.get_cleaning_status(dataset_id)
        return status
    except Exception as e:
        logger.error(f"Failed to get cleaning status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run-eda")
async def run_eda_analysis(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    dataset_name: Optional[str] = None
):
    """
    Run EDA analysis on uploaded dataset.
    
    - files: List of files to analyze
    - dataset_name: Optional name for the dataset
    """
    try:
        # Generate dataset ID
        dataset_id = str(uuid.uuid4())
        dataset_dir = DATASET_STORAGE / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded files
        total_size_bytes = 0
        for file in files:
            file_path = dataset_dir / file.filename
            with open(file_path, "wb") as buffer:
                content = file.file.read()
                buffer.write(content)
                total_size_bytes += len(content)
        
        # Save dataset info
        dataset_info = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name or f"eda_dataset_{dataset_id[:8]}",
            "num_files": len(files),
            "total_size_bytes": total_size_bytes,
            "total_size_gb": total_size_bytes / (1024**3),
            "upload_timestamp": datetime.now().isoformat(),
            "analysis_type": "eda_only"
        }
        
        with open(dataset_dir / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f)
        
        # Start EDA analysis in background
        background_tasks.add_task(
            _run_eda_background,
            dataset_id,
            dataset_dir
        )
        
        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset_info["dataset_name"],
            "num_files": len(files),
            "total_size_gb": dataset_info["total_size_gb"],
            "status": "eda_started",
            "message": "EDA analysis started in background"
        }
        
    except Exception as e:
        logger.error(f"EDA analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _run_eda_background(dataset_id: str, dataset_dir: Path):
    """Background task for EDA analysis."""
    try:
        # Create output directory for EDA results
        output_dir = RESULTS_STORAGE / dataset_id / "eda_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run EDA analysis
        from .services.unified_cleaning_service import UnifiedCleaningService
        eda_service = UnifiedCleaningService()
        
        # Run EDA only (no cleaning)
        eda_results = eda_service._run_eda_analysis(dataset_dir, "standalone")
        
        # Save EDA results
        with open(output_dir / "eda_results.json", "w") as f:
            json.dump(eda_results, f)
        
        logger.info(f"Dataset {dataset_id} EDA analysis completed")
        
    except Exception as e:
        logger.error(f"Background EDA failed for {dataset_id}: {e}")

@app.get("/eda-results/{dataset_id}")
async def get_eda_results(dataset_id: str):
    """Get EDA results for a specific dataset."""
    try:
        results_dir = RESULTS_STORAGE / dataset_id / "eda_results"
        if not results_dir.exists():
            raise HTTPException(status_code=404, detail="EDA results not found")
        
        # Load EDA results
        eda_results_path = results_dir / "eda_results.json"
        if eda_results_path.exists():
            with open(eda_results_path, 'r') as f:
                eda_results = json.load(f)
            
            return {
                "dataset_id": dataset_id,
                "eda_results": eda_results,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="EDA results file not found")
        
    except Exception as e:
        logger.error(f"Failed to get EDA results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train-layout-model", tags=["Training"])
async def train_layout_model(
    train_data_dir: str = Form(...),
    val_data_dir: str = Form(...),
    output_dir: str = Form(...),
    epochs: int = Form(50),
    batch_size: int = Form(16),
    learning_rate: float = Form(1e-4),
    mixed_precision: bool = Form(True)
):
    """
    Train LayoutLMv3 model for document layout classification.
    Optimized for A100 GPU with PyTorch.
    """
    try:
        # Configure training
        from .services.gpu_training_service import TrainingConfig, GPUTrainingService  # type: ignore
        config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            mixed_precision=mixed_precision,
            gpu_acceleration=True
        )
        
        # Create training service
        service = GPUTrainingService(config)
        
        # Setup training
        service.setup_layout_training()
        
        # Start training
        results = service.train_layout_model(train_data_dir, val_data_dir, output_dir)
        
        # Cleanup
        service.cleanup()
        
        return {
            "status": "success",
            "message": "Layout model training completed",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in layout model training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train-yolo-model", tags=["Training"])
async def train_yolo_model(
    data_yaml_path: str = Form(...),
    output_dir: str = Form(...),
    epochs: int = Form(50),
    batch_size: int = Form(16),
    learning_rate: float = Form(1e-4)
):
    """
    Train YOLOv8 model for document object detection.
    Optimized for A100 GPU.
    """
    try:
        # Configure training
        from .services.gpu_training_service import TrainingConfig, GPUTrainingService  # type: ignore
        config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gpu_acceleration=True
        )
        
        # Create training service
        service = GPUTrainingService(config)
        
        # Setup training
        service.setup_yolo_training()
        
        # Start training
        results = service.train_yolo_model(data_yaml_path, output_dir)
        
        # Cleanup
        service.cleanup()
        
        return {
            "status": "success",
            "message": "YOLO model training completed",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in YOLO model training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training-stats", tags=["Training"])
async def get_training_stats():
    """
    Get current training statistics and GPU usage.
    """
    try:
        service = get_gpu_training_service()
        if service is None:
            raise HTTPException(status_code=503, detail="GPU training service unavailable")
        stats = service.get_training_stats()
        return {
            "status": "success",
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting training stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/processing-stats", tags=["System"])
async def get_processing_stats():
    """
    Get current processing statistics and GPU usage.
    """
    try:
        # Prefer full stats when heavy deps are available
        try:
            service = create_optimized_service()
            stats = service.get_processing_stats()
            return {
                "status": "success",
                "stats": stats,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as svc_err:
            # Lightweight fallback: report device and (if available) GPU basics without loading models
            logger.warning(f"OptimizedProcessingService unavailable, returning lightweight stats: {svc_err}")
            fallback_stats = {}
            try:
                import torch  # local import to avoid hard dependency at startup
                device = "cuda" if torch.cuda.is_available() else "cpu"
                fallback_stats["device"] = device
                if device == "cuda":
                    fallback_stats.update({
                        "gpu_name": torch.cuda.get_device_name(),
                        "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                        "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                        "gpu_memory_cached_gb": torch.cuda.memory_reserved() / 1e9
                    })
            except Exception as torch_err:
                fallback_stats["device"] = "unknown"
                logger.warning(f"Torch not available for stats: {torch_err}")
            return {
                "status": "degraded",
                "stats": fallback_stats,
                "message": "Optimized processing service unavailable. Returning lightweight stats.",
                "timestamp": datetime.now().isoformat()
            }
    except HTTPException as he:
        # Even in HTTP errors, try to return a degraded response instead of failing hard
        logger.warning(f"HTTP error in processing-stats: {he.detail}")
        return {
            "status": "degraded",
            "stats": {},
            "message": he.detail,
            "timestamp": datetime.now().isoformat()
        }

@app.get("/health")
async def health_check():
    """Simple health check for load balancers."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)