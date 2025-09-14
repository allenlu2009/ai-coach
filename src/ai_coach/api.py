"""
FastAPI backend for AI coach pose analysis system.

This module provides REST API endpoints and WebSocket connections for video upload,
pose analysis, coaching feedback, and real-time chat functionality.
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .models import (
    VideoAnalysis,
    CoachingFeedback, 
    ChatMessage,
    ProcessingStatus,
)
from .video_processor import VideoProcessor
from .coach_agent import CoachAgent
from .rtm_pose_analyzer import RTMPoseAnalyzer

logger = logging.getLogger(__name__)


class UploadResponse(BaseModel):
    """Response model for video upload."""
    video_id: str
    status: str
    message: str
    estimated_processing_time: Optional[float] = None


class AnalysisRequest(BaseModel):
    """Request model for manual analysis trigger."""
    movement_type: str = "general"


class ChatResponse(BaseModel):
    """Response model for chat messages."""
    message: str
    timestamp: datetime
    session_id: str


class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.user_sessions: Dict[str, str] = {}  # websocket -> session_id
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept WebSocket connection and add to session."""
        await websocket.accept()
        
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        
        self.active_connections[session_id].append(websocket)
        self.user_sessions[id(websocket)] = session_id
        
        logger.info(f"WebSocket connected for session {session_id}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        ws_id = id(websocket)
        if ws_id in self.user_sessions:
            session_id = self.user_sessions[ws_id]
            
            if session_id in self.active_connections:
                self.active_connections[session_id].remove(websocket)
                if not self.active_connections[session_id]:
                    del self.active_connections[session_id]
            
            del self.user_sessions[ws_id]
            logger.info(f"WebSocket disconnected for session {session_id}")
    
    async def send_personal_message(self, message: str, session_id: str):
        """Send message to all connections in a session."""
        if session_id in self.active_connections:
            dead_connections = []
            
            for websocket in self.active_connections[session_id]:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending WebSocket message: {e}")
                    dead_connections.append(websocket)
            
            # Clean up dead connections
            for dead_ws in dead_connections:
                self.disconnect(dead_ws)
    
    async def broadcast_progress(self, video_id: str, progress: Dict[str, Any]):
        """Broadcast processing progress to all sessions."""
        message = json.dumps({
            "type": "progress_update",
            "video_id": video_id,
            "data": progress
        })
        
        # Send to all active connections
        for session_connections in self.active_connections.values():
            for websocket in session_connections:
                try:
                    await websocket.send_text(message)
                except Exception:
                    pass  # Will be cleaned up on next send attempt


def create_app(uploads_dir: str = "uploads", use_gpu_encoding: bool = False, create_video_overlay: bool = False, frame_skip: int = 3) -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Args:
        uploads_dir: Directory for video uploads and processing
        use_gpu_encoding: Whether to use GPU acceleration for FFmpeg video encoding
        create_video_overlay: Whether to create video overlays (default: False for JSON-only)
        frame_skip: Analyze every Nth frame for performance (default: 3)
        
    Returns:
        Configured FastAPI app instance
    """
    app = FastAPI(
        title="AI Coach - Pose Analysis System",
        description="AI-powered athlete pose analysis and coaching system",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware for frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Initialize core components - use RTMPose if available, fallback to MediaPipe
    use_rtmpose = os.getenv('USE_RTMPOSE', 'true').lower() == 'true'
    use_3d = os.getenv('USE_3D_POSE', 'false').lower() == 'true'
    
    if use_rtmpose:
        try:
            pose_analyzer = RTMPoseAnalyzer(
                use_gpu_encoding=use_gpu_encoding, 
                frame_skip=frame_skip,
                use_3d=use_3d
            )
            if use_3d:
                logger.info("🚀 Using RTMPose analyzer with 3D pose estimation enabled")
            else:
                logger.info("🚀 Using RTMPose analyzer for ultra-fast 2D pose detection")
        except Exception as e:
            logger.error(f"RTMPose initialization failed: {e}")
            raise RuntimeError(f"RTMPose required but failed to initialize: {e}")
    else:
        # Default to RTMPose with 3D support
        pose_analyzer = RTMPoseAnalyzer(use_gpu_encoding=use_gpu_encoding, frame_skip=frame_skip, use_3d=use_3d)
    video_processor = VideoProcessor(
        uploads_dir=uploads_dir, 
        pose_analyzer=pose_analyzer, 
        use_gpu_encoding=use_gpu_encoding,
        create_video_overlay=create_video_overlay,
        frame_skip=frame_skip
    )
    coach_agent = CoachAgent()
    websocket_manager = WebSocketManager()
    
    # Mount static files for frontend
    frontend_path = Path(__file__).parent.parent / "frontend" / "static"
    if frontend_path.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
    
    @app.get("/")
    async def root():
        """Root endpoint - serve frontend HTML."""
        frontend_file = Path(__file__).parent.parent / "frontend" / "static" / "index.html"
        if frontend_file.exists():
            return FileResponse(str(frontend_file))
        else:
            # Fallback to API info if frontend not found
            return {
                "message": "AI Coach - Pose Analysis System",
                "version": "0.1.0",
                "docs": "/docs",
                "status": "running",
                "note": "Frontend files not found"
            }
            
    @app.get("/api")
    async def api_info():
        """API information endpoint."""
        return {
            "message": "AI Coach - Pose Analysis System API",
            "version": "0.1.0",
            "docs": "/docs",
            "status": "running"
        }
    
    @app.post("/upload", response_model=UploadResponse)
    async def upload_video(
        background_tasks: BackgroundTasks,
        video_file: UploadFile = File(...),
        movement_type: str = "general"
    ):
        """
        Upload video file for pose analysis.
        
        Args:
            video_file: Uploaded video file
            movement_type: Type of movement for specialized analysis
            
        Returns:
            Upload response with video ID and status
        """
        try:
            # Process video upload
            temp_path = await video_processor.process_upload(video_file)
            
            # Generate unique video ID
            video_id = str(uuid.uuid4())
            
            # Estimate processing time based on file size
            file_size_mb = Path(temp_path).stat().st_size / (1024 * 1024)
            estimated_time = max(30, min(300, file_size_mb * 2))  # 2 seconds per MB, min 30s, max 5min
            
            # Start background analysis
            background_tasks.add_task(
                analyze_video_background,
                video_processor,
                coach_agent,
                websocket_manager,
                temp_path,
                video_id,
                movement_type
            )
            
            logger.info(f"Video uploaded for analysis: {video_id}")
            
            return UploadResponse(
                video_id=video_id,
                status="processing",
                message="Video uploaded successfully. Analysis in progress.",
                estimated_processing_time=estimated_time
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Upload error: {e}")
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    @app.get("/analyze/{video_id}")
    async def get_analysis_results(video_id: str):
        """
        Get pose analysis results for a video.
        
        Args:
            video_id: Unique video identifier
            
        Returns:
            VideoAnalysis results or processing status
        """
        try:
            # Check if analysis is complete
            analysis = await video_processor.get_analysis_results(video_id)
            
            if analysis:
                return analysis
            
            # Check if processing is in progress
            job_status = video_processor.get_job_status(video_id)
            
            if job_status:
                return {
                    "video_id": video_id,
                    "status": job_status["status"],
                    "progress": job_status.get("progress", 0),
                    "message": "Analysis in progress"
                }
            
            # Video not found
            raise HTTPException(status_code=404, detail="Video analysis not found")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving analysis for {video_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve analysis")
    
    @app.get("/feedback/{video_id}")
    async def get_coaching_feedback(video_id: str, movement_type: str = "general"):
        """
        Get AI coaching feedback for analyzed video.
        
        Args:
            video_id: Unique video identifier
            movement_type: Type of movement for specialized coaching
            
        Returns:
            CoachingFeedback with analysis and suggestions
        """
        try:
            # Get analysis results
            analysis = await video_processor.get_analysis_results(video_id)
            
            if not analysis:
                raise HTTPException(status_code=404, detail="Video analysis not found")
            
            if analysis.status != ProcessingStatus.COMPLETED:
                raise HTTPException(status_code=400, detail="Analysis not yet complete")
            
            # Generate coaching feedback
            feedback = await coach_agent.generate_feedback(analysis, movement_type)
            
            logger.info(f"Generated coaching feedback for {video_id}")
            return feedback
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating feedback for {video_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate coaching feedback")
    
    @app.api_route("/videos/{video_id}/preview", methods=["GET", "HEAD"])
    async def get_video_preview(video_id: str):
        """
        Get processed video with pose overlay visualization.
        
        Args:
            video_id: Unique video identifier
            
        Returns:
            Video file response with pose landmarks overlay
        """
        try:
            logger.info(f"🎥 Video preview request for {video_id}")
            
            # Get processed video path
            processed_path = await video_processor.get_processed_video_path(video_id)
            logger.info(f"📁 Resolved path: {processed_path}")
            
            if not processed_path:
                logger.error(f"❌ No path resolved for {video_id}")
                raise HTTPException(status_code=404, detail="Processed video path not found")
            
            if not processed_path.exists():
                logger.error(f"❌ File does not exist: {processed_path}")
                raise HTTPException(status_code=404, detail="Processed video file not found")
            
            logger.info(f"✅ Video file found: {processed_path} ({processed_path.stat().st_size} bytes)")
            
            # Return streaming response with proper headers for inline video playback
            def generate_video_stream():
                with open(processed_path, "rb") as video_file:
                    while chunk := video_file.read(8192):  # 8KB chunks
                        yield chunk
            
            # Use StreamingResponse to avoid Content-Disposition: attachment header
            from fastapi.responses import StreamingResponse
            response = StreamingResponse(
                generate_video_stream(),
                media_type="video/mp4"
            )
            
            # Set proper headers for video streaming
            file_size = processed_path.stat().st_size
            response.headers["Content-Length"] = str(file_size)
            response.headers["Accept-Ranges"] = "bytes"
            response.headers["Cache-Control"] = "no-cache"
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"💥 Error serving video preview for {video_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to serve video preview")
    
    @app.get("/videos/{video_id}/download")
    async def download_video(video_id: str):
        """
        Download processed video with pose overlay visualization.
        
        Args:
            video_id: Unique video identifier
            
        Returns:
            Video file download response
        """
        try:
            # Get processed video path
            processed_path = await video_processor.get_processed_video_path(video_id)
            
            if not processed_path or not processed_path.exists():
                raise HTTPException(status_code=404, detail="Processed video not found")
            
            # Return video file for download with proper filename
            return FileResponse(
                path=str(processed_path),
                media_type="video/mp4",
                filename=f"{video_id}_pose_analysis.mp4",
                headers={
                    "Content-Disposition": f'attachment; filename="{video_id}_pose_analysis.mp4"',
                    "Accept-Ranges": "bytes"
                }
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error downloading video for {video_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to download video")
    
    @app.websocket("/chat/{session_id}")
    async def websocket_chat_endpoint(websocket: WebSocket, session_id: str):
        """
        WebSocket endpoint for real-time chat with AI coach.
        
        Args:
            websocket: WebSocket connection
            session_id: Chat session identifier
        """
        await websocket_manager.connect(websocket, session_id)
        
        try:
            # Send welcome message
            welcome_message = {
                "type": "message",
                "role": "assistant", 
                "content": "Hello! I'm your AI coach. Upload a video and I'll help analyze your movement and provide coaching feedback.",
                "timestamp": datetime.utcnow().isoformat()
            }
            await websocket.send_text(json.dumps(welcome_message))
            
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                user_message = message_data.get("content", "")
                video_id = message_data.get("video_id")
                
                # Generate coach response
                response = await coach_agent.handle_chat_message(
                    session_id, user_message, video_id
                )
                
                # Send response back to client
                response_message = {
                    "type": "message",
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.utcnow().isoformat(),
                    "session_id": session_id
                }
                
                await websocket.send_text(json.dumps(response_message))
                
        except WebSocketDisconnect:
            websocket_manager.disconnect(websocket)
            logger.info(f"WebSocket disconnected for session {session_id}")
        except Exception as e:
            logger.error(f"WebSocket error for session {session_id}: {e}")
            websocket_manager.disconnect(websocket)
    
    @app.get("/status")
    async def get_system_status():
        """Get system status and statistics."""
        try:
            # Get processing statistics
            performance_stats = pose_analyzer.get_performance_stats()
            storage_stats = video_processor.get_storage_stats()
            active_jobs = video_processor.get_all_active_jobs()
            
            return {
                "system_status": "healthy",
                "active_jobs": len(active_jobs),
                "performance": performance_stats,
                "storage": storage_stats,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "system_status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow()
            }
    
    @app.post("/admin/cleanup")
    async def cleanup_old_files(max_age_hours: int = 24):
        """Administrative endpoint to clean up old files."""
        try:
            # Clean up video files
            file_stats = await video_processor.cleanup_old_files(max_age_hours)
            
            # Clean up chat sessions
            session_stats = coach_agent.cleanup_old_sessions(max_age_hours)
            
            return {
                "files_cleaned": file_stats,
                "sessions_cleaned": session_stats,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.utcnow()}
    
    return app


async def analyze_video_background(
    video_processor: VideoProcessor,
    coach_agent: CoachAgent, 
    websocket_manager: WebSocketManager,
    temp_path: str,
    video_id: str,
    movement_type: str
):
    """
    Background task for video analysis with progress updates.
    
    Args:
        video_processor: VideoProcessor instance
        coach_agent: CoachAgent instance
        websocket_manager: WebSocket manager for progress updates
        temp_path: Path to temporary video file
        video_id: Unique video identifier
        movement_type: Type of movement for analysis
    """
    try:
        # Notify clients that processing started
        await websocket_manager.broadcast_progress(video_id, {
            "status": "processing",
            "progress": 0,
            "message": "Starting pose analysis..."
        })
        
        # Run pose analysis
        analysis = await video_processor.analyze_video_async(
            temp_path, video_id
        )
        
        if analysis.status == ProcessingStatus.COMPLETED:
            # Notify progress
            await websocket_manager.broadcast_progress(video_id, {
                "status": "generating_feedback", 
                "progress": 80,
                "message": "Generating coaching feedback..."
            })
            
            # Generate coaching feedback
            feedback = await coach_agent.generate_feedback(analysis, movement_type)
            
            # Notify completion
            await websocket_manager.broadcast_progress(video_id, {
                "status": "completed",
                "progress": 100,
                "message": "Analysis complete!",
                "analysis_summary": feedback.analysis_summary
            })
        else:
            # Notify failure
            await websocket_manager.broadcast_progress(video_id, {
                "status": "failed",
                "progress": 0,
                "message": f"Analysis failed: {analysis.error_message}"
            })
        
        # Clean up temporary file
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to clean up temp file {temp_path}: {e}")
        
        logger.info(f"Background analysis completed for {video_id}")
        
    except Exception as e:
        logger.error(f"Background analysis error for {video_id}: {e}")
        
        # Notify clients of error
        await websocket_manager.broadcast_progress(video_id, {
            "status": "failed",
            "progress": 0,
            "message": f"Analysis failed: {str(e)}"
        })


# Create default app instance
app = create_app()