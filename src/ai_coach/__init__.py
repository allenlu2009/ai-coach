"""
AI Coach - Athlete Pose Analysis System

A comprehensive system for analyzing athlete poses from uploaded videos and providing 
conversational feedback through an AI-powered chat interface.

This package provides:
- MediaPipe-based pose detection and 3D landmark extraction
- Real-time video processing optimized for RTX 3060 GPU  
- AI-powered coaching feedback and conversation
- FastAPI web interface with video upload and chat capabilities
"""

__version__ = "0.1.0"
__author__ = "AI Coach Team"
__email__ = "team@aicoach.local"

# Core modules
from .models import (
    PoseLandmark,
    FrameAnalysis, 
    VideoAnalysis,
    CoachingFeedback,
)

# Main components - conditional imports for testing
try:
    from .pose_analyzer import PoseAnalyzer
    from .video_processor import VideoProcessor  
    from .coach_agent import CoachAgent
    from .api import create_app
    _FULL_IMPORTS = True
except ImportError:
    # Gracefully handle missing dependencies for basic testing
    PoseAnalyzer = None
    VideoProcessor = None
    CoachAgent = None
    create_app = None
    _FULL_IMPORTS = False

__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__email__",
    
    # Data models
    "PoseLandmark",
    "FrameAnalysis",
    "VideoAnalysis", 
    "CoachingFeedback",
    
    # Core components
    "PoseAnalyzer",
    "VideoProcessor",
    "CoachAgent",
    
    # API
    "create_app",
]