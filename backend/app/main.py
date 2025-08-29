from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import time
import traceback
from contextlib import asynccontextmanager

from app.controllers.document_controller import router as document_router
from app.config.settings import settings
from app.database.database import init_db

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting PS-05 Document AI Backend...")
    
    try:
        # Initialize database
        init_db()
        logger.info("Database initialized successfully")
        
        # Initialize other services here if needed
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down PS-05 Document AI Backend...")
    # Cleanup code here if needed

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Global exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "error_code": "VALIDATION_ERROR",
            "details": exc.errors(),
            "timestamp": time.time()
        }
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "error_code": f"HTTP_{exc.status_code}",
            "details": exc.detail,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "error_code": "INTERNAL_ERROR",
            "details": "An unexpected error occurred",
            "timestamp": time.time()
        }
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "description": settings.DESCRIPTION,
        "docs": "/docs",
        "health": "/api/v1/health",
        "info": "/api/v1/info",
        "endpoints": {
            "document_processing": "/api/v1/upload",
            "batch_processing": "/api/v1/batch",
            "status_check": "/api/v1/status/{document_id}",
            "documents": "/api/v1/documents",
            "metrics": "/api/v1/metrics",
            "storage": "/api/v1/storage/stats",
            "languages": "/api/v1/languages",
            "stages": "/api/v1/stages"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.VERSION
    }

# Include API routers
app.include_router(
    document_router,
    prefix=settings.API_V1_STR,
    tags=["document-analysis"]
)

# Add startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info(f"PS-05 Document AI Backend v{settings.VERSION} is starting up...")
    logger.info(f"Environment: {'DEBUG' if settings.DEBUG else 'PRODUCTION'}")
    logger.info(f"Database URL: {settings.DATABASE_URL}")
    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")
    logger.info(f"Supported languages: {', '.join(settings.SUPPORTED_LANGUAGES)}")
    logger.info(f"Supported stages: {settings.SUPPORTED_STAGES}")

# Add shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("PS-05 Document AI Backend is shutting down...")

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )