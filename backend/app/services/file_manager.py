import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, Tuple, List
import logging
from datetime import datetime
from fastapi import UploadFile, HTTPException
from PIL import Image
import cv2
import numpy as np
from app.config.settings import settings

logger = logging.getLogger(__name__)

class FileManager:
    """Service for managing file uploads, storage, and validation."""
    
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.max_file_size = settings.MAX_FILE_SIZE
        self.allowed_extensions = settings.ALLOWED_EXTENSIONS
        
        # Ensure upload directory exists
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.upload_dir / "images").mkdir(exist_ok=True)
        (self.upload_dir / "processed").mkdir(exist_ok=True)
        (self.upload_dir / "temp").mkdir(exist_ok=True)
    
    async def save_uploaded_file(self, file: UploadFile) -> Tuple[str, str]:
        """
        Save uploaded file and return file path and filename.
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            Tuple of (file_path, filename)
        """
        try:
            # Validate file
            self._validate_file(file)
            
            # Generate unique filename
            file_extension = Path(file.filename).suffix.lower()
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            
            # Determine save path
            if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                save_dir = self.upload_dir / "images"
            else:
                save_dir = self.upload_dir / "temp"
            
            file_path = save_dir / unique_filename
            
            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            logger.info(f"File saved successfully: {file_path}")
            return str(file_path), unique_filename
            
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    def _validate_file(self, file: UploadFile) -> None:
        """Validate uploaded file."""
        # Check file size
        if file.size and file.size > self.max_file_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File size {file.size} bytes exceeds maximum allowed size {self.max_file_size} bytes"
            )
        
        # Check file extension
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in self.allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File extension {file_extension} not allowed. Allowed extensions: {', '.join(self.allowed_extensions)}"
            )
        
        # Check MIME type
        if file.content_type and not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type: {file.content_type}. Only image files are allowed."
            )
    
    def convert_to_image(self, file_path: str) -> str:
        """
        Convert uploaded file to image format if needed.
        
        Args:
            file_path: Path to the uploaded file
            
        Returns:
            Path to the converted image file
        """
        try:
            file_path = Path(file_path)
            
            # If already an image, return the path
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                return str(file_path)
            
            # Convert PDF or other formats to image
            if file_path.suffix.lower() == '.pdf':
                return self._convert_pdf_to_image(file_path)
            
            # For other formats, try to open with PIL
            try:
                with Image.open(file_path) as img:
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save as JPEG
                    output_path = file_path.parent / f"{file_path.stem}.jpg"
                    img.save(output_path, 'JPEG', quality=95)
                    
                    logger.info(f"File converted to image: {output_path}")
                    return str(output_path)
                    
            except Exception as e:
                logger.error(f"Failed to convert file to image: {e}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Failed to convert file to image format: {str(e)}"
                )
                
        except Exception as e:
            logger.error(f"File conversion failed: {e}")
            raise HTTPException(status_code=500, detail=f"File conversion failed: {str(e)}")
    
    def _convert_pdf_to_image(self, pdf_path: Path) -> str:
        """Convert PDF to image using pdf2image or similar library."""
        try:
            # This is a placeholder - in production, use pdf2image or similar
            # For now, we'll raise an error
            raise NotImplementedError("PDF to image conversion not implemented yet")
            
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise HTTPException(
                status_code=400, 
                detail="PDF files are not supported yet. Please upload an image file."
            )
    
    def optimize_image(self, image_path: str, high_quality: bool = False) -> str:
        """
        Optimize image for processing.
        
        Args:
            image_path: Path to the image file
            high_quality: Whether to use high quality processing
            
        Returns:
            Path to the optimized image
        """
        try:
            image_path = Path(image_path)
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("Failed to load image")
            
            # Resize if too large (for performance)
            max_dimension = 4000 if high_quality else 2000
            height, width = image.shape[:2]
            
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                logger.info(f"Image resized from {width}x{height} to {new_width}x{new_height}")
            
            # Save optimized image
            output_path = image_path.parent / f"{image_path.stem}_optimized.jpg"
            cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            logger.info(f"Image optimized and saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Image optimization failed: {e}")
            # Return original path if optimization fails
            return image_path
    
    def cleanup_temp_files(self, file_paths: List[str]) -> None:
        """
        Clean up temporary files.
        
        Args:
            file_paths: List of file paths to clean up
        """
        try:
            for file_path in file_paths:
                try:
                    path = Path(file_path)
                    if path.exists() and path.is_file():
                        path.unlink()
                        logger.info(f"Temporary file cleaned up: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up file {file_path}: {e}")
                    
        except Exception as e:
            logger.error(f"File cleanup failed: {e}")
    
    def get_file_info(self, file_path: str) -> dict:
        """
        Get information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Get basic file info
            stat = path.stat()
            file_info = {
                "filename": path.name,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "extension": path.suffix.lower(),
                "path": str(path)
            }
            
            # Get image-specific info if it's an image
            if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                try:
                    with Image.open(path) as img:
                        file_info.update({
                            "width": img.width,
                            "height": img.height,
                            "mode": img.mode,
                            "format": img.format
                        })
                except Exception as e:
                    logger.warning(f"Failed to get image info: {e}")
            
            return file_info
            
        except Exception as e:
            logger.error(f"Failed to get file info: {e}")
            raise
    
    def validate_image_file(self, file_path: str) -> bool:
        """
        Validate that a file is a valid image.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if valid image, False otherwise
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return False
            
            # Try to open with PIL
            with Image.open(path) as img:
                img.verify()
            
            return True
            
        except Exception as e:
            logger.warning(f"Image validation failed for {file_path}: {e}")
            return False
    
    def get_storage_stats(self) -> dict:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            total_size = 0
            file_count = 0
            
            for file_path in self.upload_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            return {
                "total_files": file_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "upload_directory": str(self.upload_dir),
                "max_file_size_bytes": self.max_file_size,
                "max_file_size_mb": round(self.max_file_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    def create_backup(self, backup_name: str = None) -> str:
        """
        Create a backup of the upload directory.
        
        Args:
            backup_name: Optional custom backup name
            
        Returns:
            Path to the backup file
        """
        try:
            if backup_name is None:
                backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            backup_path = self.upload_dir.parent / f"{backup_name}.zip"
            
            # Create zip backup
            shutil.make_archive(
                str(backup_path).replace('.zip', ''),
                'zip',
                self.upload_dir
            )
            
            logger.info(f"Backup created successfully: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            raise
    
    def restore_backup(self, backup_path: str) -> bool:
        """
        Restore from a backup file.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            True if restore successful, False otherwise
        """
        try:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
            # Create temporary directory for extraction
            temp_dir = self.upload_dir.parent / "temp_restore"
            temp_dir.mkdir(exist_ok=True)
            
            # Extract backup
            shutil.unpack_archive(backup_path, temp_dir, 'zip')
            
            # Find extracted directory
            extracted_dir = next(temp_dir.iterdir())
            
            # Backup current upload directory
            current_backup = self.upload_dir.parent / f"current_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if self.upload_dir.exists():
                shutil.move(str(self.upload_dir), str(current_backup))
            
            # Move extracted directory to upload directory
            shutil.move(str(extracted_dir), str(self.upload_dir))
            
            # Clean up
            shutil.rmtree(temp_dir)
            
            logger.info(f"Backup restored successfully from: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Backup restore failed: {e}")
            return False
