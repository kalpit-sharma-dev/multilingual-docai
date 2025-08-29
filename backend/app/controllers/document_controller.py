from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import logging

from app.database.database import get_db
from app.services.document_service import DocumentService
from app.services.file_manager import FileManager
from app.models.schemas import (
    DocumentUploadRequest, DocumentResponse, BatchProcessingRequest, 
    BatchProcessingResponse, ProcessingStage, ProcessingStatus,
    HealthResponse, InfoResponse, SystemMetricsResponse
)
from app.config.settings import settings
from app.models.document_model import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize services
document_service = DocumentService()
file_manager = FileManager()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Get system metrics
        import psutil
        from datetime import datetime
        
        # Calculate uptime (simplified)
        # Placeholder uptime; replace with real uptime tracking if needed
        uptime = 0.0
        
        # Get system stats
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        
        # Get queue stats (simplified)
        active_jobs = 0
        queue_size = 0
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            version=settings.VERSION,
            uptime=uptime,
            active_jobs=active_jobs,
            queue_size=queue_size
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version=settings.VERSION,
            uptime=0,
            active_jobs=0,
            queue_size=0
        )

@router.get("/info", response_model=InfoResponse)
async def get_system_info():
    """Get system information."""
    try:
        return InfoResponse(
            name=settings.PROJECT_NAME,
            version=settings.VERSION,
            description=settings.DESCRIPTION,
            supported_languages=settings.SUPPORTED_LANGUAGES,
            supported_stages=settings.SUPPORTED_STAGES,
            max_file_size=settings.MAX_FILE_SIZE,
            allowed_extensions=settings.ALLOWED_EXTENSIONS
        )
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system information")

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    stage: int = Form(3, description="Processing stage (1, 2, or 3)"),
    high_quality: bool = Form(False, description="Use high quality processing"),
    language_hint: Optional[str] = Form(None, description="Language hint for OCR"),
    background: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Upload and process a document.
    
    - **file**: Document file to upload
    - **stage**: Processing stage (1=Layout, 2=OCR+Language, 3=Full Analysis)
    - **high_quality**: Use high quality processing (slower but more accurate)
    - **language_hint**: Language hint for OCR (en, hi, ur, ar, ne, fa)
    """
    try:
        # Validate stage
        if stage not in [1, 2, 3]:
            raise HTTPException(status_code=400, detail="Stage must be 1, 2, or 3")
        
        # Save uploaded file
        file_path, filename = await file_manager.save_uploaded_file(file)
        
        # Process document
        response = await document_service.process_document(
            db=db,
            file_path=file_path,
            original_filename=file.filename or "unknown",
            stage=ProcessingStage(stage),
            high_quality=high_quality,
            language_hint=language_hint
        )
        
        logger.info(f"Document uploaded and processed successfully: {response.document_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")

@router.post("/upload/async", response_model=dict)
async def upload_document_async(
    file: UploadFile = File(...),
    stage: int = Form(3, description="Processing stage (1, 2, or 3)"),
    high_quality: bool = Form(False, description="Use high quality processing"),
    language_hint: Optional[str] = Form(None, description="Language hint for OCR"),
    db: Session = Depends(get_db)
):
    """
    Upload document for asynchronous processing.
    
    Returns document ID for status tracking.
    """
    try:
        # Validate stage
        if stage not in [1, 2, 3]:
            raise HTTPException(status_code=400, detail="Stage must be 1, 2, or 3")
        
        # Save uploaded file
        file_path, filename = await file_manager.save_uploaded_file(file)
        
        # Start async processing
        document_id = await document_service.process_document_async(
            db=db,
            file_path=file_path,
            original_filename=file.filename or "unknown",
            stage=ProcessingStage(stage),
            high_quality=high_quality,
            language_hint=language_hint
        )
        
        logger.info(f"Async document processing started: {document_id}")
        return {
            "document_id": document_id,
            "status": "processing",
            "message": "Document processing started. Use GET /status/{document_id} to check progress."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Async document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Async document upload failed: {str(e)}")

@router.post("/batch", response_model=BatchProcessingResponse)
async def process_batch(
    batch_request: BatchProcessingRequest,
    db: Session = Depends(get_db)
):
    """
    Process multiple documents in batch.
    
    - **documents**: List of document file paths
    - **stage**: Processing stage for all documents
    - **priority**: Processing priority (1-10)
    """
    try:
        response = await document_service.process_batch(db, batch_request)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@router.get("/status/{document_id}", response_model=DocumentResponse)
async def get_document_status(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Get document processing status and results.
    
    - **document_id**: Document identifier
    """
    try:
        response = document_service.get_document_status(db, document_id)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document status: {str(e)}")

@router.get("/documents", response_model=List[DocumentResponse])
async def get_documents(
    status: Optional[ProcessingStatus] = Query(None, description="Filter by processing status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of documents to return"),
    offset: int = Query(0, ge=0, description="Number of documents to skip"),
    db: Session = Depends(get_db)
):
    """
    Get list of documents with optional filtering.
    
    - **status**: Filter by processing status (pending, processing, completed, failed)
    - **limit**: Maximum number of documents to return (1-1000)
    - **offset**: Number of documents to skip for pagination
    """
    try:
        documents = document_service.get_documents_by_status(db, status, limit, offset)
        return documents
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")

@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a document and its associated files.
    
    - **document_id**: Document identifier
    """
    try:
        success = document_service.delete_document(db, document_id)
        if success:
            return {"message": "Document deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete document")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@router.get("/documents/{document_id}/download")
async def download_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Download the original document file.
    
    - **document_id**: Document identifier
    """
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if not file_manager.validate_image_file(document.file_path):
            raise HTTPException(status_code=404, detail="Document file not found or invalid")
        
        return FileResponse(
            path=document.file_path,
            filename=document.original_filename,
            media_type=document.mime_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download document: {str(e)}")

@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics():
    """Get system performance metrics."""
    try:
        import psutil
        from datetime import datetime
        
        # Get system metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        
        # Get GPU usage if available (simplified)
        gpu_usage = None
        try:
            # In production, use proper GPU monitoring libraries
            pass
        except:
            pass
        
        # Get processing metrics (simplified)
        active_jobs = 0
        queue_size = 0
        average_processing_time = 0.0
        throughput = 0.0
        
        return SystemMetricsResponse(
            timestamp=datetime.utcnow(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            active_jobs=active_jobs,
            queue_size=queue_size,
            average_processing_time=average_processing_time,
            throughput=throughput
        )
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {str(e)}")

@router.get("/storage/stats")
async def get_storage_stats():
    """Get storage statistics."""
    try:
        stats = file_manager.get_storage_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get storage stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get storage stats: {str(e)}")

@router.post("/storage/backup")
async def create_backup(backup_name: Optional[str] = None):
    """Create a backup of the upload directory."""
    try:
        backup_path = file_manager.create_backup(backup_name)
        return {
            "message": "Backup created successfully",
            "backup_path": backup_path
        }
        
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create backup: {str(e)}")

@router.post("/storage/restore")
async def restore_backup(backup_path: str):
    """Restore from a backup file."""
    try:
        success = file_manager.restore_backup(backup_path)
        if success:
            return {"message": "Backup restored successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to restore backup")
        
    except Exception as e:
        logger.error(f"Failed to restore backup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to restore backup: {str(e)}")

@router.get("/languages")
async def get_supported_languages():
    """Get list of supported languages."""
    try:
        return {
            "supported_languages": settings.SUPPORTED_LANGUAGES,
            "language_names": {
                "en": "English",
                "hi": "Hindi",
                "ur": "Urdu",
                "ar": "Arabic",
                "ne": "Nepali",
                "fa": "Persian"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get supported languages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get supported languages: {str(e)}")

@router.get("/stages")
async def get_processing_stages():
    """Get information about processing stages."""
    try:
        return {
            "stages": [
                {
                    "stage": 1,
                    "name": "Layout Detection",
                    "description": "Detect text, titles, lists, tables, and figures",
                    "features": ["Element detection", "Bounding boxes", "Confidence scores"]
                },
                {
                    "stage": 2,
                    "name": "OCR + Language ID",
                    "description": "Extract text with language identification",
                    "features": ["Text extraction", "Language detection", "Multilingual support"]
                },
                {
                    "stage": 3,
                    "name": "Full Analysis",
                    "description": "Complete analysis with natural language descriptions",
                    "features": ["Table analysis", "Chart detection", "Figure description", "Map analysis"]
                }
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get processing stages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get processing stages: {str(e)}")
