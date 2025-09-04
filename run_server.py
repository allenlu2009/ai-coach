#!/usr/bin/env python3
"""
Simple server runner that avoids MediaPipe initialization issues.
"""

import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Try GPU first, fallback to CPU if needed
# Comment out the GPU disable to allow RTX 3060 acceleration
# os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"  # Use CPU only to avoid GPU issues
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"  # Disable hardware transforms for stability

def main():
    """Run the AI Coach server with stability fixes."""
    try:
        import uvicorn
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        # Ensure uploads directory exists
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        
        print("🏃‍♀️ Starting AI Coach Server...")
        print("📍 Server will be available at: http://localhost:8000")  
        print("🚀 GPU acceleration enabled for RTX 3060")
        
        # Run with string import to delay MediaPipe initialization
        uvicorn.run(
            "ai_coach.api:app",
            host="127.0.0.1",
            port=8000,
            reload=False,  # Disable reload to avoid MediaPipe re-init issues
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        print("💡 Try running: pip install uvicorn[standard]")
        sys.exit(1)

if __name__ == "__main__":
    main()