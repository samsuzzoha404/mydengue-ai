#!/usr/bin/env python3
"""
Simple FastAPI server startup script that properly sets up the Python path
"""
import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Now import and run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    from app.main import app
    
    print("🚀 Starting Advanced Dengue Guard AI Backend...")
    print(f"📁 Backend Directory: {backend_dir}")
    print(f"🐍 Python Path: {sys.path[:3]}")  # Show first 3 paths
    print("🌐 Server will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("=" * 60)
    
    # Start the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(backend_dir)],
        log_level="info"
    )