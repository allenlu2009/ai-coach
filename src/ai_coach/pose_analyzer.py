"""
MediaPipe pose analysis engine for AI coach system.

This module provides the core pose detection and 3D landmark extraction functionality
using MediaPipe, optimized for RTX 3060 GPU and following patterns from examples/pose.py.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import logging
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .models import (
    PoseLandmark,
    FrameAnalysis,
    VideoAnalysis,
    VideoMetadata,
    ProcessingStatus,
)

logger = logging.getLogger(__name__)


class PoseAnalyzer:
    """
    MediaPipe-based pose analyzer optimized for coaching applications.
    
    This analyzer extracts 33 3D pose landmarks from video frames and provides
    comprehensive analysis results optimized for RTX 3060 GPU usage.
    """
    
    def __init__(self, use_gpu: bool = True, model_complexity: int = 2):
        """
        Initialize the pose analyzer.
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
            model_complexity: MediaPipe model complexity (0, 1, or 2)
                             2 = highest accuracy, best for coaching
        """
        self.use_gpu = use_gpu and self._check_gpu_availability()
        self.model_complexity = model_complexity
        
        # Initialize MediaPipe components
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # CRITICAL: MediaPipe configuration optimized for WSL2
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            enable_segmentation=True,  # Required for world landmarks
            min_detection_confidence=0.5,  # Lower threshold for WSL2 compatibility
            min_tracking_confidence=0.3,   # Lower threshold for better detection
            smooth_landmarks=True
        )
        
        # Performance tracking
        self.processing_stats = {
            "frames_processed": 0,
            "poses_detected": 0,
            "total_processing_time": 0.0,
            "gpu_memory_peak": 0.0
        }
        
        logger.info(f"PoseAnalyzer initialized - GPU: {self.use_gpu}, Complexity: {model_complexity}")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for acceleration."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using CPU only")
            return False
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU only")
            return False
        
        # Check RTX 3060 memory constraints
        device_props = torch.cuda.get_device_properties(0)
        total_memory_gb = device_props.total_memory / (1024**3)
        
        logger.info(f"GPU detected: {device_props.name}, {total_memory_gb:.1f}GB VRAM")
        
        if total_memory_gb < 4.0:
            logger.warning("Low GPU memory, consider using CPU mode")
        
        return True
    
    def _extract_landmarks(self, results) -> List[PoseLandmark]:
        """
        Extract 3D landmarks from MediaPipe results.
        
        Args:
            results: MediaPipe pose detection results
            
        Returns:
            List of 33 PoseLandmark objects (empty if no pose detected)
        """
        if not results.pose_world_landmarks:
            return []
        
        landmarks = []
        for landmark in results.pose_world_landmarks.landmark:
            # CRITICAL: Use world coordinates for 3D analysis
            pose_landmark = PoseLandmark(
                x=landmark.x,  # Meters relative to hip center
                y=landmark.y,  # Meters relative to hip center  
                z=landmark.z,  # Meters relative to hip center
                visibility=landmark.visibility
            )
            landmarks.append(pose_landmark)
        
        return landmarks
    
    def _calculate_confidence(self, landmarks: List[PoseLandmark]) -> float:
        """
        Calculate overall confidence score for pose detection.
        
        Args:
            landmarks: List of pose landmarks
            
        Returns:
            Confidence score between 0 and 1
        """
        if not landmarks:
            return 0.0
        
        # Calculate average visibility, weighted by important landmarks
        important_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # Core body landmarks
        
        if len(landmarks) != 33:
            return 0.0
        
        important_visibilities = [landmarks[i].visibility for i in important_indices if i < len(landmarks)]
        all_visibilities = [lm.visibility for lm in landmarks]
        
        # Weighted average: 70% important landmarks, 30% all landmarks
        if important_visibilities:
            important_avg = sum(important_visibilities) / len(important_visibilities)
            all_avg = sum(all_visibilities) / len(all_visibilities)
            confidence = 0.7 * important_avg + 0.3 * all_avg
        else:
            confidence = sum(all_visibilities) / len(all_visibilities)
        
        return confidence
    
    def analyze_frame(self, frame: np.ndarray, frame_number: int, timestamp_ms: float) -> FrameAnalysis:
        """
        Analyze a single video frame for pose detection.
        
        Args:
            frame: Video frame as numpy array (BGR format)
            frame_number: Frame number in sequence
            timestamp_ms: Timestamp in milliseconds
            
        Returns:
            FrameAnalysis object with pose detection results
        """
        try:
            # CRITICAL: Color conversion for MediaPipe  
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Debug: Log frame info every 30 frames
            if frame_number % 30 == 0:
                logger.info(f"Processing frame {frame_number}: {rgb_frame.shape}")
            
            # Process the frame
            results = self.pose.process(rgb_frame)
            
            # Debug: Check if pose was detected
            pose_detected = results.pose_world_landmarks is not None
            if frame_number % 30 == 0:
                logger.info(f"Frame {frame_number}: Pose detected = {pose_detected}")
            
            # Extract landmarks
            landmarks = self._extract_landmarks(results)
            
            # Calculate confidence
            confidence = self._calculate_confidence(landmarks)
            
            # Update stats
            self.processing_stats["frames_processed"] += 1
            if landmarks:
                self.processing_stats["poses_detected"] += 1
            
            return FrameAnalysis(
                frame_number=frame_number,
                timestamp_ms=timestamp_ms,
                landmarks=landmarks,
                confidence_score=confidence,
                pose_detected=len(landmarks) == 33
            )
            
        except Exception as e:
            logger.error(f"Error analyzing frame {frame_number}: {e}")
            return FrameAnalysis(
                frame_number=frame_number,
                timestamp_ms=timestamp_ms,
                landmarks=[],
                confidence_score=0.0,
                pose_detected=False
            )
    
    async def analyze_video(self, video_path: str, video_id: str) -> VideoAnalysis:
        """
        Analyze complete video for pose detection and 3D landmarks.
        
        Args:
            video_path: Path to video file
            video_id: Unique video identifier
            
        Returns:
            VideoAnalysis object with complete results
        """
        start_time = time.time()
        initial_gpu_memory = 0.0
        peak_gpu_memory = 0.0
        
        # Track GPU memory if available
        if self.use_gpu and TORCH_AVAILABLE:
            initial_gpu_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
        
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            # Extract video metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Get file info
            file_path = Path(video_path)
            file_size_mb = file_path.stat().st_size / (1024**2)
            
            metadata = VideoMetadata(
                video_id=video_id,
                filename=file_path.name,
                file_size_mb=file_size_mb,
                duration_seconds=duration,
                fps=fps,
                total_frames=total_frames,
                resolution_width=width,
                resolution_height=height,
                format=file_path.suffix.lstrip('.')
            )
            
            # Process frames
            frame_analyses = []
            frame_number = 0
            
            logger.info(f"Processing video: {total_frames} frames at {fps:.1f} FPS")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate timestamp
                timestamp_ms = (frame_number / fps) * 1000 if fps > 0 else 0
                
                # Analyze frame
                analysis = self.analyze_frame(frame, frame_number, timestamp_ms)
                frame_analyses.append(analysis)
                
                frame_number += 1
                
                # CRITICAL: GPU memory management for RTX 3060
                if frame_number % 100 == 0:
                    if self.use_gpu and TORCH_AVAILABLE:
                        current_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
                        peak_gpu_memory = max(peak_gpu_memory, current_memory)
                        torch.cuda.empty_cache()  # Clear cache periodically
                        
                    logger.info(f"Processed {frame_number}/{total_frames} frames "
                              f"({100*frame_number/total_frames:.1f}%)")
            
            cap.release()
            
            # Final GPU memory check
            if self.use_gpu and TORCH_AVAILABLE:
                final_memory = torch.cuda.memory_allocated() / (1024**2)  # MB  
                peak_gpu_memory = max(peak_gpu_memory, final_memory)
            
            # Calculate final statistics
            processing_time = time.time() - start_time
            poses_detected = sum(1 for analysis in frame_analyses if analysis.pose_detected)
            
            # Create analysis result
            video_analysis = VideoAnalysis(
                video_id=video_id,
                status=ProcessingStatus.COMPLETED,
                metadata=metadata,
                frame_analyses=frame_analyses,
                processing_time_seconds=processing_time,
                gpu_memory_used_mb=peak_gpu_memory,
                poses_detected_count=poses_detected
            )
            
            logger.info(f"Video analysis completed: {poses_detected}/{total_frames} poses detected "
                       f"({video_analysis.pose_detection_rate:.1%} success rate) "
                       f"in {processing_time:.1f}s")
            
            return video_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing video {video_id}: {e}")
            processing_time = time.time() - start_time
            
            # Return failed analysis
            return VideoAnalysis(
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
                processing_time_seconds=processing_time,
                gpu_memory_used_mb=peak_gpu_memory,
                poses_detected_count=0,
                error_message=str(e)
            )
    
    def create_pose_overlay_video(self, 
                                 video_path: str, 
                                 output_path: str, 
                                 analysis: VideoAnalysis) -> bool:
        """
        Create a video with pose landmarks overlay.
        
        Args:
            video_path: Path to original video
            output_path: Path for output video with pose overlay
            analysis: VideoAnalysis results for overlay
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_number = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get corresponding analysis if available
                if frame_number < len(analysis.frame_analyses):
                    frame_analysis = analysis.frame_analyses[frame_number]
                    
                    if frame_analysis.pose_detected and len(frame_analysis.landmarks) == 33:
                        # Convert landmarks back to image coordinates for drawing
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = self.pose.process(rgb_frame)
                        
                        if results.pose_landmarks:
                            # Draw pose landmarks
                            self.mp_drawing.draw_landmarks(
                                frame,
                                results.pose_landmarks,
                                self.mp_pose.POSE_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                            )
                            
                        # Add confidence score text
                        cv2.putText(
                            frame,
                            f"Confidence: {frame_analysis.confidence_score:.2f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2
                        )
                
                out.write(frame)
                frame_number += 1
            
            cap.release()
            out.release()
            
            logger.info(f"Pose overlay video created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating pose overlay video: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.processing_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.processing_stats = {
            "frames_processed": 0,
            "poses_detected": 0,
            "total_processing_time": 0.0,
            "gpu_memory_peak": 0.0
        }
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()