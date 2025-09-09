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
import subprocess
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
    
    def __init__(self, use_gpu: bool = True, model_complexity: int = 2, use_gpu_encoding: bool = False, frame_skip: int = 3):
        """
        Initialize the pose analyzer.
        
        Args:
            use_gpu: Whether to use GPU acceleration for MediaPipe pose detection
            model_complexity: MediaPipe model complexity (0, 1, or 2)
                             2 = highest accuracy, best for coaching
            use_gpu_encoding: Whether to use GPU acceleration for FFmpeg video encoding
            frame_skip: Analyze every Nth frame for 3x speedup (default: 3 for 30fps→10fps analysis)
        """
        self.use_gpu = use_gpu and self._check_gpu_availability()
        self.use_gpu_encoding = use_gpu_encoding and self._check_nvenc_availability()
        self.model_complexity = model_complexity
        self.frame_skip = max(1, frame_skip)  # Ensure at least 1 (no skipping)
        
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
        
        logger.info(f"PoseAnalyzer initialized - GPU: {self.use_gpu}, Complexity: {model_complexity}, Frame Skip: {self.frame_skip}")
    
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
    
    def _check_nvenc_availability(self) -> bool:
        """Check if NVIDIA NVENC GPU encoding is available."""
        try:
            # Check if FFmpeg has NVENC support
            result = subprocess.run(
                ['ffmpeg', '-encoders'], 
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                logger.warning("FFmpeg not available, cannot use GPU encoding")
                return False
            
            # Check for NVENC encoders
            has_nvenc = 'h264_nvenc' in result.stdout
            
            if not has_nvenc:
                logger.warning("NVENC encoder not found in FFmpeg, using CPU encoding")
                return False
            
            # Test NVENC encoder by encoding a small test
            try:
                test_result = subprocess.run([
                    'ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=0.1:size=320x240:rate=1',
                    '-c:v', 'h264_nvenc', '-f', 'null', '-'
                ], capture_output=True, text=True, timeout=15)
                
                if test_result.returncode == 0:
                    logger.info("✅ NVIDIA NVENC GPU encoding available and functional")
                    return True
                else:
                    logger.warning("NVENC test failed, falling back to CPU encoding")
                    return False
                    
            except subprocess.TimeoutExpired:
                logger.warning("NVENC test timed out, falling back to CPU encoding")
                return False
                
        except Exception as e:
            logger.error(f"Error checking NVENC availability: {e}")
            return False
    
    def _extract_landmarks(self, results) -> List[PoseLandmark]:
        """
        Extract normalized 2D landmarks from MediaPipe results for video overlay.
        
        Args:
            results: MediaPipe pose detection results
            
        Returns:
            List of 33 PoseLandmark objects with normalized coordinates (0-1)
        """
        # Always use normalized landmarks for video overlay (not world landmarks)
        if not results.pose_landmarks:
            return []
        
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            # Use normalized 2D coordinates (0-1) for proper video overlay
            pose_landmark = PoseLandmark(
                x=landmark.x,  # Normalized x coordinate (0-1)
                y=landmark.y,  # Normalized y coordinate (0-1) 
                z=getattr(landmark, 'z', 0.0),  # Normalized z depth
                visibility=landmark.visibility
            )
            landmarks.append(pose_landmark)
        
        return landmarks
    
    def _interpolate_pose(self, prev_landmarks: List[PoseLandmark], next_landmarks: List[PoseLandmark], ratio: float) -> List[PoseLandmark]:
        """
        Interpolate pose landmarks between two frames.
        
        Args:
            prev_landmarks: Landmarks from previous frame with pose
            next_landmarks: Landmarks from next frame with pose  
            ratio: Interpolation ratio (0.0 = prev, 1.0 = next)
            
        Returns:
            Interpolated landmarks
        """
        if not prev_landmarks or not next_landmarks or len(prev_landmarks) != len(next_landmarks):
            return []
        
        interpolated = []
        for i in range(len(prev_landmarks)):
            prev_lm = prev_landmarks[i]
            next_lm = next_landmarks[i]
            
            # Interpolate coordinates
            x = prev_lm.x + ratio * (next_lm.x - prev_lm.x)
            y = prev_lm.y + ratio * (next_lm.y - prev_lm.y)
            z = prev_lm.z + ratio * (next_lm.z - prev_lm.z)
            visibility = min(prev_lm.visibility, next_lm.visibility)  # Use minimum visibility
            
            interpolated.append(PoseLandmark(x=x, y=y, z=z, visibility=visibility))
        
        return interpolated

    def _fill_skipped_frames(self, frame_analyses: List[FrameAnalysis]) -> List[FrameAnalysis]:
        """
        Fill in skipped frames with interpolated pose data.
        
        Args:
            frame_analyses: List of frame analyses with placeholders
            
        Returns:
            List with interpolated poses for skipped frames
        """
        if self.frame_skip <= 1:
            return frame_analyses  # No skipping, return as-is
        
        # Find frames with actual pose data
        pose_frames = []
        for i, analysis in enumerate(frame_analyses):
            if analysis.pose_detected and len(analysis.landmarks) == 33:
                pose_frames.append(i)
        
        if len(pose_frames) < 2:
            return frame_analyses  # Not enough pose data to interpolate
        
        # Interpolate missing frames between pose frames
        updated_analyses = frame_analyses.copy()
        
        for i in range(len(pose_frames) - 1):
            start_idx = pose_frames[i]
            end_idx = pose_frames[i + 1]
            
            if end_idx - start_idx <= 1:
                continue  # No frames to interpolate
            
            start_landmarks = frame_analyses[start_idx].landmarks
            end_landmarks = frame_analyses[end_idx].landmarks
            
            # Interpolate frames between start_idx and end_idx
            for frame_idx in range(start_idx + 1, end_idx):
                ratio = (frame_idx - start_idx) / (end_idx - start_idx)
                interpolated_landmarks = self._interpolate_pose(start_landmarks, end_landmarks, ratio)
                
                if interpolated_landmarks:
                    # Update the placeholder with interpolated data
                    updated_analyses[frame_idx] = FrameAnalysis(
                        frame_number=frame_analyses[frame_idx].frame_number,
                        timestamp_ms=frame_analyses[frame_idx].timestamp_ms,
                        pose_detected=True,
                        landmarks=interpolated_landmarks,
                        confidence_score=0.8  # Lower confidence for interpolated poses
                    )
        
        logger.info(f"🔄 Interpolated poses for {sum(1 for a in updated_analyses if a.confidence_score == 0.8)} skipped frames")
        return updated_analyses

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
            overall_avg = sum(all_visibilities) / len(all_visibilities)
            return 0.7 * important_avg + 0.3 * overall_avg
        
        # Fallback to overall average
        return sum(all_visibilities) / len(all_visibilities)
    
    def analyze_frame(self, frame: np.ndarray, frame_number: int, timestamp_ms: float) -> FrameAnalysis:
        """
        Analyze a single frame for pose detection.
        
        Args:
            frame: Input frame as numpy array
            frame_number: Frame sequence number
            timestamp_ms: Timestamp in milliseconds
            
        Returns:
            FrameAnalysis object with pose detection results
        """
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run pose detection
            results = self.pose.process(rgb_frame)
            
            # Extract landmarks
            landmarks = self._extract_landmarks(results)
            pose_detected = len(landmarks) == 33  # MediaPipe returns 33 landmarks
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(landmarks) if pose_detected else 0.0
            
            # Update performance statistics
            self.processing_stats["frames_processed"] += 1
            if pose_detected:
                self.processing_stats["poses_detected"] += 1
            
            return FrameAnalysis(
                frame_number=frame_number,
                timestamp_ms=timestamp_ms,
                pose_detected=pose_detected,
                landmarks=landmarks,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error analyzing frame {frame_number}: {e}")
            return FrameAnalysis(
                frame_number=frame_number,
                timestamp_ms=timestamp_ms,
                pose_detected=False,
                landmarks=[],
                confidence_score=0.0
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
            
            # Process frames with optional frame skipping for performance
            frame_analyses = []
            frame_number = 0
            analyzed_frames = 0
            
            effective_fps = fps / self.frame_skip if self.frame_skip > 1 else fps
            logger.info(f"Processing video: {total_frames} frames at {fps:.1f} FPS")
            if self.frame_skip > 1:
                logger.info(f"⚡ Frame skipping enabled: analyzing every {self.frame_skip} frames (effective: {effective_fps:.1f} FPS)")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames for performance optimization
                if frame_number % self.frame_skip == 0:
                    # Calculate timestamp
                    timestamp_ms = (frame_number / fps) * 1000 if fps > 0 else 0
                    
                    # Analyze frame
                    analysis = self.analyze_frame(frame, frame_number, timestamp_ms)
                    frame_analyses.append(analysis)
                    analyzed_frames += 1
                else:
                    # Create a placeholder analysis for skipped frames (no pose detection)
                    timestamp_ms = (frame_number / fps) * 1000 if fps > 0 else 0
                    placeholder_analysis = FrameAnalysis(
                        frame_number=frame_number,
                        timestamp_ms=timestamp_ms,
                        pose_detected=False,
                        landmarks=[],
                        confidence_score=0.0
                    )
                    frame_analyses.append(placeholder_analysis)
                
                frame_number += 1
                
                # CRITICAL: GPU memory management for RTX 3060
                if frame_number % 100 == 0:
                    if self.use_gpu and TORCH_AVAILABLE:
                        current_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
                        peak_gpu_memory = max(peak_gpu_memory, current_memory)
                        torch.cuda.empty_cache()  # Clear cache periodically
                        
                    if self.frame_skip > 1:
                        logger.info(f"Processed {frame_number}/{total_frames} frames "
                                  f"({100*frame_number/total_frames:.1f}%) - Analyzed: {analyzed_frames}")
                    else:
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
            
            logger.info(f"✅ Analysis completed: {analyzed_frames}/{frame_number} frames analyzed in {processing_time:.2f}s")
            if self.frame_skip > 1:
                speedup = self.frame_skip
                logger.info(f"⚡ Frame skipping speedup: ~{speedup:.1f}x faster ({analyzed_frames} vs {frame_number} frames)")
                
                # Fill in skipped frames with interpolated poses for smooth video overlay
                frame_analyses = self._fill_skipped_frames(frame_analyses)
            
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
        Create video with pose overlay using direct FFmpeg piping (no AVI intermediate).
        
        Args:
            video_path: Path to original video
            output_path: Path for output video with pose overlay
            analysis: VideoAnalysis results for overlay
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"🎬 Starting direct pose overlay video creation: {output_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"❌ Cannot open input video: {video_path}")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"📊 Video properties: {width}x{height} @ {fps}fps")
            logger.info("⚡ Using direct FFmpeg piping (no intermediate AVI)")
            
            # Build FFmpeg command for direct piping
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-f', 'rawvideo',  # Input format
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',  # Size
                '-pix_fmt', 'bgr24',  # OpenCV uses BGR
                '-r', str(fps),  # Frame rate
                '-i', '-',  # Read from stdin
            ]
            
            # Add encoding parameters
            if self.use_gpu_encoding:
                logger.info("🚀 Using NVIDIA NVENC GPU encoding")
                ffmpeg_cmd.extend([
                    '-c:v', 'h264_nvenc',
                    '-preset', 'fast',
                    '-profile:v', 'baseline',
                    '-level', '3.0',
                    '-pix_fmt', 'yuv420p',
                    '-cq', '28',
                    '-b:v', '2M',
                    '-maxrate', '4M',
                    '-bufsize', '8M',
                    '-rc:v', 'vbr',
                    '-movflags', '+faststart',
                    '-avoid_negative_ts', 'make_zero'
                ])
            else:
                logger.info("🖥️ Using CPU libx264 encoding")
                ffmpeg_cmd.extend([
                    '-c:v', 'libx264',
                    '-profile:v', 'baseline',
                    '-level', '3.0',
                    '-pix_fmt', 'yuv420p',
                    '-crf', '28',
                    '-preset', 'ultrafast',
                    '-movflags', '+faststart',
                    '-avoid_negative_ts', 'make_zero'
                ])
            
            ffmpeg_cmd.append(str(output_path))
            
            # Start FFmpeg process
            logger.info(f"🔧 Starting FFmpeg: {' '.join(ffmpeg_cmd[:8])}... {output_path}")
            logger.info(f"📋 Full FFmpeg command: {' '.join(ffmpeg_cmd)}")
            
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8  # Large buffer for video data
            )
            
            # Check if process started successfully
            initial_poll = process.poll()
            if initial_poll is not None:
                logger.error(f"❌ FFmpeg process failed to start, return code: {initial_poll}")
                stdout, stderr = process.communicate()
                if stderr:
                    logger.error(f"FFmpeg startup stderr: {stderr.decode('utf-8', errors='replace')}")
                return False
            
            frame_number = 0
            processed_frames = 0
            
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Check if FFmpeg process is still alive
                    if process.poll() is not None:
                        logger.error(f"FFmpeg process terminated unexpectedly with return code {process.returncode}")
                        break
                    
                    # Get corresponding analysis if available
                    if frame_number < len(analysis.frame_analyses):
                        frame_analysis = analysis.frame_analyses[frame_number]
                        
                        if frame_analysis.pose_detected and len(frame_analysis.landmarks) == 33:
                            # Draw pose landmarks directly using OpenCV
                            landmarks = frame_analysis.landmarks
                            
                            # Convert normalized coordinates to pixel coordinates with bounds checking
                            for i, landmark in enumerate(landmarks):
                                if landmark.visibility > 0.5:  # Only draw visible landmarks
                                    # Ensure coordinates are within valid bounds [0, 1]
                                    norm_x = max(0.0, min(1.0, landmark.x))
                                    norm_y = max(0.0, min(1.0, landmark.y))
                                    
                                    # Convert to pixel coordinates
                                    x = int(norm_x * width)
                                    y = int(norm_y * height)
                                    
                                    # Ensure pixel coordinates are within frame bounds
                                    x = max(0, min(width - 1, x))
                                    y = max(0, min(height - 1, y))
                                    
                                    # Draw landmark point
                                    cv2.circle(frame, (x, y), 4, (66, 230, 245), -1)  # BGR format
                                    cv2.circle(frame, (x, y), 6, (66, 117, 245), 2)   # BGR format
                            
                            # Draw pose connections (simplified version)
                            connections = [
                                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
                                (11, 23), (12, 24), (23, 24),  # Torso
                                (23, 25), (25, 27), (24, 26), (26, 28),  # Legs
                                (27, 29), (29, 31), (28, 30), (30, 32)  # Feet
                            ]
                            
                            for connection in connections:
                                if (connection[0] < len(landmarks) and connection[1] < len(landmarks) and
                                    landmarks[connection[0]].visibility > 0.5 and landmarks[connection[1]].visibility > 0.5):
                                    
                                    # Convert start point with bounds checking
                                    start_norm_x = max(0.0, min(1.0, landmarks[connection[0]].x))
                                    start_norm_y = max(0.0, min(1.0, landmarks[connection[0]].y))
                                    start_x = max(0, min(width - 1, int(start_norm_x * width)))
                                    start_y = max(0, min(height - 1, int(start_norm_y * height)))
                                    
                                    # Convert end point with bounds checking  
                                    end_norm_x = max(0.0, min(1.0, landmarks[connection[1]].x))
                                    end_norm_y = max(0.0, min(1.0, landmarks[connection[1]].y))
                                    end_x = max(0, min(width - 1, int(end_norm_x * width)))
                                    end_y = max(0, min(height - 1, int(end_norm_y * height)))
                                    
                                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (245, 117, 66), 2)
                            
                            # Add confidence score text
                            cv2.putText(
                                frame,
                                f"Confidence: {frame_analysis.confidence_score:.2f}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 255),
                                2
                            )
                    
                    # Write frame directly to FFmpeg stdin
                    try:
                        if process.stdin and not process.stdin.closed:
                            process.stdin.write(frame.tobytes())
                            processed_frames += 1
                        else:
                            logger.error("FFmpeg stdin is closed or None, stopping")
                            break
                    except BrokenPipeError:
                        logger.error("FFmpeg pipe broken, stopping frame processing")
                        break
                    except OSError as e:
                        logger.error(f"FFmpeg write error: {e}, stopping frame processing")
                        break
                    
                    frame_number += 1
                    
                    if frame_number % 100 == 0:
                        logger.info(f"Piped {frame_number} frames to FFmpeg...")
            
            finally:
                # Close stdin to signal end of input
                try:
                    if hasattr(process, 'stdin') and process.stdin and not process.stdin.closed:
                        process.stdin.close()
                except (OSError, BrokenPipeError) as e:
                    logger.warning(f"⚠️ Expected error closing FFmpeg stdin: {e}")
                except Exception as e:
                    logger.error(f"❌ Unexpected error closing FFmpeg stdin: {e}")
                
                cap.release()
            
            # Wait for FFmpeg to finish
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                logger.info(f"✅ Direct FFmpeg processing completed successfully")
                logger.info(f"📊 Processed {processed_frames} frames via direct piping")
                return True
            else:
                logger.error(f"❌ FFmpeg failed with return code {process.returncode}")
                logger.error(f"📊 Processed {processed_frames} frames before failure")
                if stderr:
                    # Log full stderr for debugging
                    stderr_text = stderr.decode('utf-8', errors='replace')
                    logger.error(f"FFmpeg stderr (full): {stderr_text}")
                if stdout:
                    stdout_text = stdout.decode('utf-8', errors='replace')
                    logger.info(f"FFmpeg stdout: {stdout_text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error in direct pose overlay creation: {e}")
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