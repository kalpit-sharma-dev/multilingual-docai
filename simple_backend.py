#!/usr/bin/env python3
"""
Simple Test Backend for PS-05 Challenge
This will work without complex dependencies
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime

# Create a simple FastAPI app
app = FastAPI(
    title="PS-05 Document Understanding API (Simple)",
    description="Simple working backend for Docker testing",
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

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "PS-05 Document Understanding API (Simple)",
        "version": "1.0.0",
        "status": "working",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/test")
async def test_endpoint():
    """Test endpoint."""
    return {
        "message": "Backend is working correctly!",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("🚀 Starting Simple PS-05 Backend...")
    print("📡 Server will be available at: http://0.0.0.0:8000")
    print("📚 API Documentation: http://0.0.0.0:8000/docs")
    print("💡 Use Ctrl+C to stop the server")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
