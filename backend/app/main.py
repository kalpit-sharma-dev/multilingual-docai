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

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
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

# Import our core modules
from .services.document_processor import DocumentProcessor
from .services.stage_processor import StageProcessor
from .services.evaluation_service import EvaluationService
from .services.unified_cleaning_service import UnifiedCleaningService
from .models.schemas import (
    ProcessingRequest, ProcessingResponse, StageResult, 
    DatasetUploadResponse, EvaluationResult, NoAnnotationResponse
)

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

# Initialize services
document_processor = DocumentProcessor()
stage_processor = StageProcessor()
evaluation_service = EvaluationService()
cleaning_service = UnifiedCleaningService()

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
            "process_stage": "/process-stage",
            "process_all": "/process-all",
            "evaluate": "/evaluate",
            "get_results": "/results/{dataset_id}",
            "get_predictions": "/predictions/{dataset_id}",
            "status": "/status",
            "clean_dataset": "/clean-dataset",
            "cleaning_capabilities": "/cleaning-capabilities"
        }
    }

@app.get("/status")
async def get_status():
    """Get system status and available models."""
    try:
        status = {
            "api_status": "running",
            "timestamp": datetime.now().isoformat(),
            "available_models": stage_processor.get_available_models(),
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

@app.post("/process-stage", response_model=StageResult)
async def process_stage(request: ProcessingRequest):
    """
    Process a specific stage for a dataset.
    
    Stages:
    - 1: Layout Detection
    - 2: Text Extraction + Language ID
    - 3: Content Understanding
    """
    try:
        dataset_dir = DATASET_STORAGE / request.dataset_id
        if not dataset_dir.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Process the requested stage
        result = await stage_processor.process_stage(
            dataset_id=request.dataset_id,
            stage=request.stage,
            config=request.config
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Stage processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-all", response_model=ProcessingResponse)
async def process_all_stages(
    background_tasks: BackgroundTasks,
    dataset_id: str,
    config: Optional[Dict] = None
):
    """
    Process all 3 stages for a dataset.
    This runs in the background and returns a job ID.
    Optimized for large datasets (20GB+).
    """
    try:
        dataset_dir = DATASET_STORAGE / dataset_id
        if not dataset_dir.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Start background processing
        job_id = str(uuid.uuid4())
        background_tasks.add_task(
            stage_processor.process_all_stages,
            dataset_id=dataset_id,
            job_id=job_id,
            config=config
        )
        
        return ProcessingResponse(
            job_id=job_id,
            message="Processing started in background",
            status="processing"
        )
        
    except Exception as e:
        logger.error(f"All-stage processing failed: {e}")
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
        evaluation_result = await evaluation_service.evaluate_dataset(
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