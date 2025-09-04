"""
AI Coach Application Entry Point

Main FastAPI application for pose analysis and coaching feedback.
Provides video upload, pose detection, and real-time chat functionality.
"""

import uvicorn
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables first
load_dotenv()

def main():
    """
    Run the AI Coach application.
    
    Starts the FastAPI server with proper configuration for development
    and production environments.
    """
    # Ensure uploads directory exists
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    # Get configuration from environment
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    
    # Import app here to avoid MediaPipe initialization during import
    from ai_coach.api import app
    
    # Run the application
    uvicorn.run(
        "src.main:get_app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )

def get_app():
    """Get the FastAPI app instance."""
    from ai_coach.api import app
    return app

if __name__ == "__main__":
    main()