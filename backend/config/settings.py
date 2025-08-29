from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "PS-05 Document AI"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Intelligent Multilingual Document Understanding System"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8081"]
    
    # Database Configuration
    DATABASE_URL: str = "sqlite:///./ps05_document_ai.db"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_DB: str = "ps05_document_ai"
    
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379"
    
    # File Storage Configuration
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".pdf", ".tiff", ".bmp"]
    
    # Model Configuration
    MODEL_CACHE_DIR: str = "./models"
    EASYOCR_MODEL_PATH: str = "./models/easyocr"
    LAYOUT_MODEL_PATH: str = "./models/layout"
    OCR_MODEL_PATH: str = "./models/ocr"
    
    # Processing Configuration
    MAX_WORKERS: int = 4
    PROCESSING_TIMEOUT: int = 300  # 5 minutes
    BATCH_SIZE: int = 10
    
    # Supported Languages
    SUPPORTED_LANGUAGES: List[str] = [
        "en", "hi", "ur", "ar", "ne", "fa"
    ]
    
    # Supported Processing Stages
    SUPPORTED_STAGES: List[int] = [1, 2, 3]
    
    # Security Configuration
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Monitoring Configuration
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 8001
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
