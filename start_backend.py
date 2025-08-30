#!/usr/bin/env python3
"""
Simple Backend Startup Script for PS-05 Challenge
Clean, reliable startup for Docker containers
"""

import uvicorn
import sys
import os
from pathlib import Path

def main():
    """Start the PS-05 backend server."""
    try:
        # Add current directory to Python path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        # Set environment variables
        os.environ.setdefault('PYTHONPATH', str(current_dir))
        os.environ.setdefault('ENVIRONMENT', 'production')
        
        # Import the FastAPI app
        from backend.app.main import app
        
        print("🚀 Starting PS-05 Backend Server...")
        print("📡 Server will be available at: http://0.0.0.0:8000")
        print("📚 API Documentation: http://0.0.0.0:8000/docs")
        print("💡 Use Ctrl+C to stop the server")
        
        # Start the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
        
    except ImportError as e:
        print(f"❌ Failed to import backend: {e}")
        print("💡 Make sure all dependencies are installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
