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
            enable_segmentation=False,  # Disable segmentation to fix dimension errors
            min_detection_confidence=0.7,  # Increase for better quality detection
            min_tracking_confidence=0.5,   # Increase for better tracking
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
        # Fallback to regular landmarks if world landmarks unavailable
        if results.pose_world_landmarks:
            landmark_source = results.pose_world_landmarks.landmark
        elif results.pose_landmarks:
            landmark_source = results.pose_landmarks.landmark
        else:
            return []
        
        landmarks = []
        for landmark in landmark_source:
            # Use available coordinate system (world if available, otherwise normalized)
            pose_landmark = PoseLandmark(
                x=landmark.x,
                y=landmark.y,  
                z=getattr(landmark, 'z', 0.0),  # Z may not exist in regular landmarks
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
            logger.info(f"🎬 Starting pose overlay video creation: {output_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"❌ Cannot open input video: {video_path}")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"📊 Video properties: {width}x{height} @ {fps}fps")
            
            # Create video writer - skip OpenCV VideoWriter and use FFmpeg directly
            temp_output_path = str(output_path).replace('.mp4', '_temp.avi')
            logger.info(f"📁 Creating temp AVI: {temp_output_path}")
            
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use MJPG for reliable OpenCV output
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.error(f"❌ Cannot create output video writer")
                return False
            
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
            
            logger.info(f"✅ AVI creation complete: {temp_output_path}")
            logger.info(f"📊 AVI file size: {Path(temp_output_path).stat().st_size} bytes")
            
            # Convert MJPG AVI to browser-compatible H.264 MP4 using FFmpeg
            import subprocess
            import os
            try:
                logger.info(f"🎬 Converting {temp_output_path} to {output_path}")
                
                # Use FFmpeg to create browser-compatible video with ultra-conservative settings
                result = subprocess.run([
                    'ffmpeg', '-i', temp_output_path,
                    '-c:v', 'libx264',
                    '-profile:v', 'baseline',     # Baseline profile for maximum compatibility
                    '-level', '3.0',             # Lower level for broader compatibility
                    '-pix_fmt', 'yuv420p',
                    '-crf', '28',                # Higher CRF (lower quality) for stability
                    '-preset', 'ultrafast',      # Fastest encoding for simpler output
                    '-movflags', '+faststart',   # Optimize for web streaming
                    '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
                    '-y', str(output_path)
                ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
                
                # Check FFmpeg result explicitly
                if result.returncode != 0:
                    logger.error(f"❌ FFmpeg failed with return code {result.returncode}")
                    logger.error(f"FFmpeg stdout: {result.stdout}")
                    logger.error(f"FFmpeg stderr: {result.stderr}")
                    return False
                
                logger.info(f"✅ FFmpeg conversion successful")
                
                # Verify output file was created
                if not Path(output_path).exists():
                    logger.error(f"❌ Output file not created: {output_path}")
                    return False
                    
                output_size = Path(output_path).stat().st_size
                logger.info(f"📊 Output file size: {output_size} bytes")
                
                # Clean up temporary AVI file
                os.remove(temp_output_path)
                logger.info(f"Pose overlay video converted to browser-compatible H.264: {output_path}")
                
                # Create a copy in pose_analysis_videos folder for easy access
                try:
                    pose_videos_dir = Path("pose_analysis_videos")
                    pose_videos_dir.mkdir(exist_ok=True)
                    
                    # Create a descriptive filename
                    video_filename = Path(output_path).stem.replace("_processed", "_pose_analysis_h264")
                    copy_path = pose_videos_dir / f"{video_filename}.mp4"
                    
                    # Copy the H.264 video to pose_analysis_videos folder
                    import shutil
                    shutil.copy2(str(output_path), str(copy_path))
                    logger.info(f"Created copy in pose_analysis_videos: {copy_path}")
                    
                except Exception as copy_error:
                    logger.warning(f"Failed to copy to pose_analysis_videos: {copy_error}")
                    # Don't fail the whole process if copy fails
                
            except subprocess.CalledProcessError as e:
                logger.error(f"💥 H.264 conversion failed: {e}")
                logger.error(f"FFmpeg stdout: {e.stdout}")
                logger.error(f"FFmpeg stderr: {e.stderr}")
                # Clean up temp files if they exist
                try:
                    os.remove(temp_output_path)
                except:
                    pass
                return False
            except Exception as e:
                logger.error(f"H.264 conversion error: {e}")
                return False
            
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