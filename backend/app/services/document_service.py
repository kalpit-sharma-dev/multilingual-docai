import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from fastapi import HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
import uuid
import os
from pathlib import Path

from app.models.document_model import Document, ProcessingJob, ProcessingResult as DBProcessingResult
from app.models.schemas import (
    ProcessingResult, DocumentResponse, ProcessingStage, ProcessingStatus,
    BatchProcessingRequest, BatchProcessingResponse, LayoutElement, PreprocessingInfo, TextLine, TableResult, ChartResult, FigureResult, MapResult
)
from app.services.document_processor import DocumentProcessor
from app.services.file_manager import FileManager
from app.config.settings import settings

logger = logging.getLogger(__name__)

class DocumentService:
    """Main service for document processing operations."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.file_manager = FileManager()
    
    async def process_document(
        self,
        db: Session,
        file_path: str,
        original_filename: str,
        stage: ProcessingStage = ProcessingStage.FULL_ANALYSIS,
        high_quality: bool = False,
        language_hint: Optional[str] = None
    ) -> DocumentResponse:
        """
        Process a single document through the specified stages.
        
        Args:
            db: Database session
            file_path: Path to the uploaded file
            original_filename: Original filename
            stage: Processing stage (1, 2, or 3)
            high_quality: Use high quality processing
            language_hint: Language hint for OCR
            
        Returns:
            DocumentResponse with processing results
        """
        try:
            # Create document record
            document = self._create_document_record(
                db, file_path, original_filename, stage
            )
            
            # Convert file to image if needed
            image_path = self.file_manager.convert_to_image(file_path)
            
            # Optimize image for processing
            optimized_path = self.file_manager.optimize_image(image_path, high_quality)
            
            # Process document
            processing_result = self.document_processor.process_document(
                optimized_path,
                stage=stage,
                high_quality=high_quality,
                language_hint=language_hint
            )
            
            # Update document record with results
            self._update_document_with_results(db, document, processing_result)
            
            # Clean up temporary files
            temp_files = [image_path, optimized_path]
            if image_path != file_path:
                temp_files.append(file_path)
            self.file_manager.cleanup_temp_files(temp_files)
            
            # Create response
            response = self._create_document_response(document, processing_result)
            
            logger.info(f"Document processed successfully: {document.id}")
            return response
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            
            # Update document status to failed
            if 'document' in locals():
                self._mark_document_failed(db, document, str(e))
            
            raise HTTPException(
                status_code=500,
                detail=f"Document processing failed: {str(e)}"
            )
    
    async def process_document_async(
        self,
        db: Session,
        file_path: str,
        original_filename: str,
        stage: ProcessingStage = ProcessingStage.FULL_ANALYSIS,
        high_quality: bool = False,
        language_hint: Optional[str] = None
    ) -> str:
        """
        Start asynchronous document processing.
        
        Returns:
            Document ID for tracking
        """
        try:
            # Create document record
            document = self._create_document_record(
                db, file_path, original_filename, stage
            )
            
            # Start background processing
            asyncio.create_task(
                self._process_document_background(
                    document.id, file_path, stage, high_quality, language_hint
                )
            )
            
            logger.info(f"Async document processing started: {document.id}")
            return str(document.id)
            
        except Exception as e:
            logger.error(f"Failed to start async processing: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start processing: {str(e)}"
            )
    
    async def _process_document_background(
        self,
        document_id: int,
        file_path: str,
        stage: ProcessingStage,
        high_quality: bool,
        language_hint: Optional[str]
    ):
        """Background document processing task."""
        try:
            # Get database session
            from app.database.database import get_db_context
            
            with get_db_context() as db:
                # Get document
                document = db.query(Document).filter(Document.id == document_id).first()
                if not document:
                    logger.error(f"Document not found: {document_id}")
                    return
                
                # Update status to processing
                document.processing_status = ProcessingStatus.PROCESSING
                document.processing_start_time = datetime.utcnow()
                db.commit()
                
                # Convert file to image if needed
                image_path = self.file_manager.convert_to_image(file_path)
                
                # Optimize image for processing
                optimized_path = self.file_manager.optimize_image(image_path, high_quality)
                
                # Process document
                processing_result = self.document_processor.process_document(
                    optimized_path,
                    stage=stage,
                    high_quality=high_quality,
                    language_hint=language_hint
                )
                
                # Update document record with results
                self._update_document_with_results(db, document, processing_result)
                
                # Clean up temporary files
                temp_files = [image_path, optimized_path]
                if image_path != file_path:
                    temp_files.append(file_path)
                self.file_manager.cleanup_temp_files(temp_files)
                
                logger.info(f"Background document processing completed: {document_id}")
                
        except Exception as e:
            logger.error(f"Background document processing failed: {e}")
            
            # Update document status to failed
            try:
                with get_db_context() as db:
                    document = db.query(Document).filter(Document.id == document_id).first()
                    if document:
                        self._mark_document_failed(db, document, str(e))
            except Exception as update_error:
                logger.error(f"Failed to update document status: {update_error}")
    
    async def process_batch(
        self,
        db: Session,
        batch_request: BatchProcessingRequest
    ) -> BatchProcessingResponse:
        """
        Process multiple documents in batch.
        
        Args:
            db: Database session
            batch_request: Batch processing request
            
        Returns:
            BatchProcessingResponse with batch information
        """
        try:
            batch_id = str(uuid.uuid4())
            results = []
            
            # Process each document
            for doc_path in batch_request.documents:
                try:
                    # For batch processing, we'll use async processing
                    document_id = await self.process_document_async(
                        db, doc_path, Path(doc_path).name, batch_request.stage
                    )
                    
                    # Create placeholder response
                    doc_response = DocumentResponse(
                        document_id=document_id,
                        filename=Path(doc_path).name,
                        status=ProcessingStatus.PROCESSING,
                        stage=batch_request.stage,
                        upload_timestamp=datetime.utcnow(),
                        processing_time=None,
                        result=None,
                        error_message=None
                    )
                    
                    results.append(doc_response)
                    
                except Exception as e:
                    logger.error(f"Failed to process document {doc_path}: {e}")
                    
                    # Create error response
                    doc_response = DocumentResponse(
                        document_id="",
                        filename=Path(doc_path).name,
                        status=ProcessingStatus.FAILED,
                        stage=batch_request.stage,
                        upload_timestamp=datetime.utcnow(),
                        processing_time=None,
                        result=None,
                        error_message=str(e)
                    )
                    
                    results.append(doc_response)
            
            # Create batch response
            completed = len([r for r in results if r.status == ProcessingStatus.COMPLETED])
            failed = len([r for r in results if r.status == ProcessingStatus.FAILED])
            processing = len([r for r in results if r.status == ProcessingStatus.PROCESSING])
            
            batch_response = BatchProcessingResponse(
                batch_id=batch_id,
                total_documents=len(batch_request.documents),
                completed_documents=completed,
                failed_documents=failed,
                processing_documents=processing,
                results=results,
                estimated_completion_time=None
            )
            
            logger.info(f"Batch processing started: {batch_id}")
            return batch_response
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Batch processing failed: {str(e)}"
            )
    
    def get_document_status(self, db: Session, document_id: str) -> DocumentResponse:
        """
        Get document processing status.
        
        Args:
            db: Database session
            document_id: Document ID
            
        Returns:
            DocumentResponse with current status
        """
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            
            # Get processing result if available
            processing_result = None
            if document.processing_status == ProcessingStatus.COMPLETED:
                processing_result = self._get_processing_result_from_db(document)
            
            response = self._create_document_response(document, processing_result)
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get document status: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get document status: {str(e)}"
            )
    
    def get_documents_by_status(
        self,
        db: Session,
        status: Optional[ProcessingStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[DocumentResponse]:
        """
        Get documents filtered by status.
        
        Args:
            db: Database session
            status: Optional status filter
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            List of DocumentResponse objects
        """
        try:
            query = db.query(Document)
            
            if status:
                query = query.filter(Document.processing_status == status)
            
            documents = query.offset(offset).limit(limit).all()
            
            responses = []
            for document in documents:
                processing_result = None
                if document.processing_status == ProcessingStatus.COMPLETED:
                    processing_result = self._get_processing_result_from_db(document)
                
                response = self._create_document_response(document, processing_result)
                responses.append(response)
            
            return responses
            
        except Exception as e:
            logger.error(f"Failed to get documents by status: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get documents: {str(e)}"
            )
    
    def delete_document(self, db: Session, document_id: str) -> bool:
        """
        Delete a document and its associated files.
        
        Args:
            db: Database session
            document_id: Document ID
            
        Returns:
            True if deletion successful
        """
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            
            # Delete associated files
            try:
                if os.path.exists(document.file_path):
                    os.remove(document.file_path)
            except Exception as e:
                logger.warning(f"Failed to delete file {document.file_path}: {e}")
            
            # Delete from database
            db.delete(document)
            db.commit()
            
            logger.info(f"Document deleted successfully: {document_id}")
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete document: {str(e)}"
            )
    
    def _create_document_record(
        self,
        db: Session,
        file_path: str,
        original_filename: str,
        stage: ProcessingStage
    ) -> Document:
        """Create a new document record in the database."""
        try:
            # Get file info
            file_info = self.file_manager.get_file_info(file_path)
            
            # Create document record
            document = Document(
                filename=file_info["filename"],
                original_filename=original_filename,
                file_path=file_path,
                file_size=file_info["size"],
                mime_type=file_info.get("format", "image/jpeg"),
                processing_stage=stage,
                processing_status=ProcessingStatus.PENDING,
                upload_timestamp=datetime.utcnow()
            )
            
            db.add(document)
            db.commit()
            db.refresh(document)
            
            return document
            
        except Exception as e:
            logger.error(f"Failed to create document record: {e}")
            raise
    
    def _update_document_with_results(
        self,
        db: Session,
        document: Document,
        processing_result: ProcessingResult
    ):
        """Update document record with processing results."""
        try:
            document.processing_status = ProcessingStatus.COMPLETED
            document.processing_end_time = datetime.utcnow()
            document.processing_time = processing_result.processing_time
            
            # Store results in JSON fields
            document.layout_elements = [elem.dict() for elem in processing_result.elements]
            
            if processing_result.text_lines:
                document.text_lines = [line.dict() for line in processing_result.text_lines]
            
            if processing_result.tables or processing_result.charts or processing_result.figures or processing_result.maps:
                analysis_results = {}
                if processing_result.tables:
                    analysis_results["tables"] = [table.dict() for table in processing_result.tables]
                if processing_result.charts:
                    analysis_results["charts"] = [chart.dict() for chart in processing_result.charts]
                if processing_result.figures:
                    analysis_results["figures"] = [figure.dict() for figure in processing_result.figures]
                if processing_result.maps:
                    analysis_results["maps"] = [map_obj.dict() for map_obj in processing_result.maps]
                
                document.analysis_results = analysis_results
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to update document with results: {e}")
            raise
    
    def _mark_document_failed(
        self,
        db: Session,
        document: Document,
        error_message: str
    ):
        """Mark document as failed with error message."""
        try:
            document.processing_status = ProcessingStatus.FAILED
            document.processing_end_time = datetime.utcnow()
            document.error_message = error_message
            document.retry_count += 1
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to mark document as failed: {e}")
    
    def _get_processing_result_from_db(self, document: Document) -> Optional[ProcessingResult]:
        """Reconstruct ProcessingResult from database fields."""
        try:
            if not document.layout_elements:
                return None
            
            # Reconstruct layout elements
            elements = []
            for elem_data in document.layout_elements:
                element = LayoutElement(**elem_data)
                elements.append(element)
            
            # Create preprocessing info
            preprocess = PreprocessingInfo(
                deskew_angle=0.0,  # Default value
                resolution=None,
                quality_score=None
            )
            
            # Create processing result
            result = ProcessingResult(
                page=1,
                size={'w': 0, 'h': 0},  # Default values
                elements=elements,
                preprocess=preprocess,
                processing_time=document.processing_time or 0.0
            )
            
            # Add text lines if available
            if document.text_lines:
                text_lines = []
                for line_data in document.text_lines:
                    text_line = TextLine(**line_data)
                    text_lines.append(text_line)
                result.text_lines = text_lines
            
            # Add analysis results if available
            if document.analysis_results:
                analysis = document.analysis_results
                
                if "tables" in analysis:
                    tables = [TableResult(**table_data) for table_data in analysis["tables"]]
                    result.tables = tables
                
                if "charts" in analysis:
                    charts = [ChartResult(**chart_data) for chart_data in analysis["charts"]]
                    result.charts = charts
                
                if "figures" in analysis:
                    figures = [FigureResult(**figure_data) for figure_data in analysis["figures"]]
                    result.figures = figures
                
                if "maps" in analysis:
                    maps = [MapResult(**map_data) for map_data in analysis["maps"]]
                    result.maps = maps
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to reconstruct processing result: {e}")
            return None
    
    def _create_document_response(
        self,
        document: Document,
        processing_result: Optional[ProcessingResult]
    ) -> DocumentResponse:
        """Create DocumentResponse from document and processing result."""
        return DocumentResponse(
            document_id=str(document.id),
            filename=document.original_filename,
            status=document.processing_status,
            stage=document.processing_stage,
            upload_timestamp=document.upload_timestamp,
            processing_time=document.processing_time,
            result=processing_result,
            error_message=document.error_message
        )
