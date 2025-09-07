"""
Video processing pipeline for AI coach system.

This module handles video file uploads, validation, temporary file management,
and orchestrates the pose analysis workflow with progress tracking.
"""

import asyncio
import tempfile
import os
import uuid
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta

import cv2
import aiofiles
from fastapi import UploadFile, HTTPException

from .models import (
    VideoAnalysis,
    VideoMetadata, 
    ProcessingStatus,
)
from .pose_analyzer import PoseAnalyzer

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Handles video upload, validation, and pose analysis coordination.
    
    This processor manages the complete workflow from file upload to analysis results,
    with proper temporary file management and progress tracking.
    """
    
    # Supported video formats
    SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    
    # File size limits (in MB)
    MAX_FILE_SIZE_MB = 100
    MAX_DURATION_SECONDS = 300  # 5 minutes
    
    def __init__(self, 
                 uploads_dir: str = "uploads",
                 pose_analyzer: Optional[PoseAnalyzer] = None,
                 use_gpu_encoding: bool = False,
                 create_video_overlay: bool = False,
                 frame_skip: int = 3):
        """
        Initialize the video processor.
        
        Args:
            uploads_dir: Directory for temporary video storage
            pose_analyzer: PoseAnalyzer instance (created if None)
            use_gpu_encoding: Whether to use GPU acceleration for FFmpeg video encoding
            create_video_overlay: Whether to create video overlay (default: False for JSON-only)
            frame_skip: Analyze every Nth frame for performance (default: 3)
        """
        self.uploads_dir = Path(uploads_dir)
        self.temp_dir = self.uploads_dir / "temp"
        self.processed_dir = self.uploads_dir / "processed"
        self.results_dir = self.uploads_dir / "results"
        
        # Ensure directories exist
        for directory in [self.temp_dir, self.processed_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Store configuration
        self.create_video_overlay = create_video_overlay
        
        # Initialize pose analyzer with frame skipping
        self.pose_analyzer = pose_analyzer or PoseAnalyzer(
            use_gpu_encoding=use_gpu_encoding, 
            frame_skip=frame_skip
        )
        
        # Active processing jobs (video_id -> progress info)
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        
        overlay_mode = "enabled" if create_video_overlay else "disabled (JSON-only)"
        logger.info(f"VideoProcessor initialized - uploads dir: {self.uploads_dir}, overlay: {overlay_mode}")
    
    async def validate_video_file(self, file: UploadFile) -> Dict[str, Any]:
        """
        Validate uploaded video file before processing.
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            Dictionary with validation results
            
        Raises:
            HTTPException: If validation fails
        """
        try:
            # Check file extension
            if not file.filename:
                raise HTTPException(status_code=400, detail="No filename provided")
            
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in self.SUPPORTED_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported format {file_ext}. Supported: {', '.join(self.SUPPORTED_FORMATS)}"
                )
            
            # Check file size (if available)
            file_size = 0
            if hasattr(file, 'size') and file.size:
                file_size = file.size
                if file_size > self.MAX_FILE_SIZE_MB * 1024 * 1024:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {self.MAX_FILE_SIZE_MB}MB"
                    )
            
            return {
                "valid": True,
                "filename": file.filename,
                "format": file_ext,
                "size_bytes": file_size,
                "size_mb": file_size / (1024 * 1024) if file_size > 0 else 0
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error validating file: {e}")
            raise HTTPException(status_code=400, detail=f"File validation error: {str(e)}")
    
    async def process_upload(self, file: UploadFile) -> str:
        """
        Process uploaded video file and create temporary file.
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            Path to temporary file
            
        Raises:
            HTTPException: If upload processing fails
        """
        try:
            # Validate file first
            validation = await self.validate_video_file(file)
            
            # Generate unique video ID and temporary filename
            video_id = str(uuid.uuid4())
            file_ext = Path(file.filename).suffix.lower()
            temp_filename = f"{video_id}{file_ext}"
            temp_path = self.temp_dir / temp_filename
            
            # GOTCHA: FastAPI UploadFile to OpenCV pipeline requires temporary file
            async with aiofiles.open(temp_path, 'wb') as temp_file:
                # Read file in chunks to handle large files
                chunk_size = 1024 * 1024  # 1MB chunks
                while chunk := await file.read(chunk_size):
                    await temp_file.write(chunk)
            
            # Additional validation with OpenCV
            cap = cv2.VideoCapture(str(temp_path))
            if not cap.isOpened():
                # Clean up failed upload
                temp_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail="Invalid video format or corrupted file")
            
            # Check video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            # Check duration limits
            if duration > self.MAX_DURATION_SECONDS:
                temp_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Video too long. Maximum duration: {self.MAX_DURATION_SECONDS}s"
                )
            
            logger.info(f"Video uploaded successfully: {video_id} ({duration:.1f}s, {frame_count} frames)")
            return str(temp_path)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing upload: {e}")
            raise HTTPException(status_code=500, detail=f"Upload processing error: {str(e)}")
    
    async def analyze_video_async(self, 
                                 video_path: str, 
                                 video_id: Optional[str] = None,
                                 progress_callback: Optional[Callable] = None) -> VideoAnalysis:
        """
        Analyze video asynchronously with progress tracking.
        
        Args:
            video_path: Path to video file
            video_id: Unique video identifier (generated if None)
            progress_callback: Optional callback for progress updates
            
        Returns:
            VideoAnalysis results
        """
        if not video_id:
            video_id = str(uuid.uuid4())
        
        # Initialize job tracking
        self.active_jobs[video_id] = {
            "status": ProcessingStatus.PROCESSING,
            "progress": 0.0,
            "start_time": datetime.utcnow(),
            "video_path": video_path
        }
        
        try:
            # Run pose analysis (this is CPU/GPU intensive)
            analysis = await self.pose_analyzer.analyze_video(video_path, video_id)
            
            # Save analysis results
            await self._save_analysis_results(analysis)
            
            # Conditionally create processed video with pose overlay
            if analysis.status == ProcessingStatus.COMPLETED and self.create_video_overlay:
                logger.info(f"🎬 Analysis completed, creating processed video for {video_id}")
                await self._create_processed_video(video_path, analysis)
            elif analysis.status == ProcessingStatus.COMPLETED:
                logger.info(f"✅ Analysis completed, skipping video overlay creation (JSON-only mode)")
            else:
                logger.warning(f"⚠️ Analysis not completed, skipping video creation. Status: {analysis.status}")
            
            # Update job status AFTER video creation is complete
            self.active_jobs[video_id]["status"] = ProcessingStatus.COMPLETED
            self.active_jobs[video_id]["progress"] = 100.0
            
            logger.info(f"Video analysis completed: {video_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing video {video_id}: {e}")
            
            # Update job status to failed
            if video_id in self.active_jobs:
                self.active_jobs[video_id]["status"] = ProcessingStatus.FAILED
                self.active_jobs[video_id]["error"] = str(e)
            
            # Create failed analysis result
            analysis = VideoAnalysis(
                video_id=video_id,
                status=ProcessingStatus.FAILED,
                metadata=VideoMetadata(
                    video_id=video_id,
                    filename=Path(video_path).name,
                    file_size_mb=0.0,
                    duration_seconds=0.0,
                    fps=0.0,
                    total_frames=0,
                    resolution_width=0,
                    resolution_height=0,
                    format="unknown"
                ),
                frame_analyses=[],
                processing_time_seconds=0.0,
                gpu_memory_used_mb=0.0,
                poses_detected_count=0,
                error_message=str(e)
            )
            
            await self._save_analysis_results(analysis)
            return analysis
    
    async def _save_analysis_results(self, analysis: VideoAnalysis) -> None:
        """Save analysis results to JSON file."""
        try:
            results_file = self.results_dir / f"{analysis.video_id}.json"
            
            # Convert to dict for JSON serialization
            analysis_dict = analysis.dict()
            
            async with aiofiles.open(results_file, 'w') as f:
                await f.write(json.dumps(analysis_dict, indent=2, default=str))
            
            logger.info(f"Analysis results saved: {results_file}")
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
    
    async def _create_processed_video(self, video_path: str, analysis: VideoAnalysis) -> None:
        """Create processed video with pose overlay."""
        try:
            logger.info(f"🎥 Starting processed video creation for {analysis.video_id}")
            processed_path = self.processed_dir / f"{analysis.video_id}_processed.mp4"
            logger.info(f"📁 Target processed path: {processed_path}")
            
            # Create pose overlay video (this is CPU intensive, run in thread pool)
            logger.info(f"🔄 Running pose overlay creation in executor...")
            success = await asyncio.get_event_loop().run_in_executor(
                None,
                self.pose_analyzer.create_pose_overlay_video,
                video_path,
                str(processed_path),
                analysis
            )
            
            logger.info(f"🔄 Pose overlay creation result: {success}")
            
            if success:
                logger.info(f"Processed video created: {processed_path}")
            else:
                logger.warning(f"Failed to create processed video for {analysis.video_id}")
            
        except Exception as e:
            logger.error(f"Error creating processed video: {e}")
    
    async def get_analysis_results(self, video_id: str) -> Optional[VideoAnalysis]:
        """
        Retrieve analysis results for a video.
        
        Args:
            video_id: Unique video identifier
            
        Returns:
            VideoAnalysis object if found, None otherwise
        """
        try:
            results_file = self.results_dir / f"{video_id}.json"
            
            if not results_file.exists():
                return None
            
            async with aiofiles.open(results_file, 'r') as f:
                data = json.loads(await f.read())
            
            # Reconstruct VideoAnalysis from dict
            return VideoAnalysis(**data)
            
        except Exception as e:
            logger.error(f"Error loading analysis results for {video_id}: {e}")
            return None
    
    def get_job_status(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get current job status and progress."""
        return self.active_jobs.get(video_id)
    
    def get_all_active_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Get all active job statuses."""
        return self.active_jobs.copy()
    
    async def cleanup_old_files(self, max_age_hours: int = 24) -> Dict[str, int]:
        """
        Clean up old temporary and processed files.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
            
        Returns:
            Dictionary with cleanup statistics
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        stats = {"temp_files": 0, "processed_files": 0, "result_files": 0}
        
        try:
            # Clean temp files
            for temp_file in self.temp_dir.iterdir():
                if temp_file.is_file():
                    file_time = datetime.fromtimestamp(temp_file.stat().st_mtime)
                    if file_time < cutoff_time:
                        temp_file.unlink()
                        stats["temp_files"] += 1
            
            # Clean processed files  
            for processed_file in self.processed_dir.iterdir():
                if processed_file.is_file():
                    file_time = datetime.fromtimestamp(processed_file.stat().st_mtime)
                    if file_time < cutoff_time:
                        processed_file.unlink()
                        stats["processed_files"] += 1
            
            # Clean result files (keep longer - 7 days)
            result_cutoff = datetime.utcnow() - timedelta(days=7)
            for result_file in self.results_dir.iterdir():
                if result_file.is_file() and result_file.suffix == '.json':
                    file_time = datetime.fromtimestamp(result_file.stat().st_mtime)
                    if file_time < result_cutoff:
                        result_file.unlink()
                        stats["result_files"] += 1
            
            # Clean up completed jobs from memory
            completed_jobs = []
            for job_id, job_info in self.active_jobs.items():
                if job_info.get("start_time"):
                    if job_info["start_time"] < cutoff_time:
                        completed_jobs.append(job_id)
            
            for job_id in completed_jobs:
                del self.active_jobs[job_id]
            
            logger.info(f"Cleanup completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return stats
    
    async def get_processed_video_path(self, video_id: str) -> Optional[Path]:
        """Get path to processed video with pose overlay."""
        processed_path = self.processed_dir / f"{video_id}_processed.mp4"
        return processed_path if processed_path.exists() else None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage usage statistics."""
        try:
            stats = {
                "temp_files": len(list(self.temp_dir.iterdir())),
                "processed_files": len(list(self.processed_dir.iterdir())),
                "result_files": len(list(self.results_dir.glob("*.json"))),
                "active_jobs": len(self.active_jobs)
            }
            
            # Calculate total storage usage
            total_size = 0
            for directory in [self.temp_dir, self.processed_dir, self.results_dir]:
                for file_path in directory.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
            
            stats["total_size_mb"] = total_size / (1024 * 1024)
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating storage stats: {e}")
            return {"error": str(e)}