"""
Tests for pose analyzer functionality.

Following patterns from tests/test_evaluator.py for comprehensive pose analysis testing.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.ai_coach.pose_analyzer import PoseAnalyzer
from src.ai_coach.models import (
    PoseLandmark,
    FrameAnalysis,
    VideoAnalysis,
    VideoMetadata,
    ProcessingStatus,
)


class TestPoseAnalyzer:
    """Test suite for PoseAnalyzer class."""
    
    @pytest.fixture
    def pose_analyzer(self):
        """Create PoseAnalyzer instance for testing."""
        return PoseAnalyzer(use_gpu=False, model_complexity=1)
    
    @pytest.fixture
    def mock_landmarks(self):
        """Create mock pose landmarks (33 landmarks)."""
        landmarks = []
        for i in range(33):
            landmarks.append(PoseLandmark(
                x=0.5 + (i * 0.01),
                y=0.5 + (i * 0.01), 
                z=0.0,
                visibility=0.9
            ))
        return landmarks
    
    @pytest.fixture
    def sample_frame(self):
        """Create sample video frame as numpy array."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    @pytest.fixture
    def mock_video_path(self, tmp_path):
        """Create temporary video file path."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b"fake video content")
        return str(video_file)
    
    def test_pose_analyzer_initialization(self, pose_analyzer):
        """Test PoseAnalyzer initialization."""
        assert pose_analyzer is not None
        assert pose_analyzer.model_complexity == 1
        assert pose_analyzer.use_gpu == False
        assert hasattr(pose_analyzer, 'pose')
        assert hasattr(pose_analyzer, 'processing_stats')
    
    def test_check_gpu_availability_no_torch(self):
        """Test GPU availability check when torch is not available."""
        with patch('src.ai_coach.pose_analyzer.TORCH_AVAILABLE', False):
            analyzer = PoseAnalyzer(use_gpu=True)
            assert analyzer.use_gpu == False
    
    def test_check_gpu_availability_no_cuda(self):
        """Test GPU availability check when CUDA is not available."""
        with patch('src.ai_coach.pose_analyzer.TORCH_AVAILABLE', True), \
             patch('torch.cuda.is_available', return_value=False):
            analyzer = PoseAnalyzer(use_gpu=True)
            assert analyzer.use_gpu == False
    
    def test_extract_landmarks_no_pose(self, pose_analyzer):
        """Test landmark extraction when no pose is detected."""
        mock_results = Mock()
        mock_results.pose_world_landmarks = None
        
        landmarks = pose_analyzer._extract_landmarks(mock_results)
        assert landmarks == []
    
    def test_extract_landmarks_with_pose(self, pose_analyzer):
        """Test landmark extraction with detected pose."""
        # Mock MediaPipe results with landmarks
        mock_results = Mock()
        mock_landmarks = []
        
        # Create 33 mock landmarks
        for i in range(33):
            mock_lm = Mock()
            mock_lm.x = 0.5 + (i * 0.01)
            mock_lm.y = 0.5 + (i * 0.01)
            mock_lm.z = 0.0
            mock_lm.visibility = 0.9
            mock_landmarks.append(mock_lm)
        
        mock_results.pose_world_landmarks.landmark = mock_landmarks
        
        landmarks = pose_analyzer._extract_landmarks(mock_results)
        
        assert len(landmarks) == 33
        assert all(isinstance(lm, PoseLandmark) for lm in landmarks)
        assert landmarks[0].x == 0.5
        assert landmarks[0].visibility == 0.9
    
    def test_calculate_confidence_no_landmarks(self, pose_analyzer):
        """Test confidence calculation with no landmarks."""
        confidence = pose_analyzer._calculate_confidence([])
        assert confidence == 0.0
    
    def test_calculate_confidence_with_landmarks(self, pose_analyzer, mock_landmarks):
        """Test confidence calculation with valid landmarks."""
        confidence = pose_analyzer._calculate_confidence(mock_landmarks)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be high given good visibility scores
    
    def test_calculate_confidence_wrong_landmark_count(self, pose_analyzer):
        """Test confidence calculation with wrong number of landmarks."""
        # Only 10 landmarks instead of 33
        landmarks = [PoseLandmark(x=0, y=0, z=0, visibility=0.9) for _ in range(10)]
        confidence = pose_analyzer._calculate_confidence(landmarks)
        assert confidence == 0.0
    
    @patch('cv2.cvtColor')
    def test_analyze_frame_success(self, mock_cvtColor, pose_analyzer, sample_frame):
        """Test successful frame analysis."""
        # Mock cv2.cvtColor
        mock_cvtColor.return_value = sample_frame
        
        # Mock MediaPipe pose processing
        mock_results = Mock()
        mock_landmarks = []
        for i in range(33):
            mock_lm = Mock()
            mock_lm.x = 0.5
            mock_lm.y = 0.5
            mock_lm.z = 0.0
            mock_lm.visibility = 0.9
            mock_landmarks.append(mock_lm)
        
        mock_results.pose_world_landmarks.landmark = mock_landmarks
        
        with patch.object(pose_analyzer.pose, 'process', return_value=mock_results):
            result = pose_analyzer.analyze_frame(sample_frame, 10, 1000.0)
        
        assert isinstance(result, FrameAnalysis)
        assert result.frame_number == 10
        assert result.timestamp_ms == 1000.0
        assert len(result.landmarks) == 33
        assert result.pose_detected == True
        assert result.confidence_score > 0
    
    @patch('cv2.cvtColor')
    def test_analyze_frame_no_pose(self, mock_cvtColor, pose_analyzer, sample_frame):
        """Test frame analysis when no pose is detected."""
        mock_cvtColor.return_value = sample_frame
        
        # Mock MediaPipe with no pose detected
        mock_results = Mock()
        mock_results.pose_world_landmarks = None
        
        with patch.object(pose_analyzer.pose, 'process', return_value=mock_results):
            result = pose_analyzer.analyze_frame(sample_frame, 5, 500.0)
        
        assert isinstance(result, FrameAnalysis)
        assert result.frame_number == 5
        assert result.timestamp_ms == 500.0
        assert len(result.landmarks) == 0
        assert result.pose_detected == False
        assert result.confidence_score == 0.0
    
    @patch('cv2.cvtColor')
    def test_analyze_frame_error_handling(self, mock_cvtColor, pose_analyzer, sample_frame):
        """Test frame analysis error handling."""
        mock_cvtColor.side_effect = Exception("CV2 error")
        
        result = pose_analyzer.analyze_frame(sample_frame, 0, 0.0)
        
        assert isinstance(result, FrameAnalysis)
        assert result.pose_detected == False
        assert result.confidence_score == 0.0
    
    @pytest.mark.asyncio
    @patch('cv2.VideoCapture')
    @patch('src.ai_coach.pose_analyzer.Path')
    async def test_analyze_video_success(self, mock_path, mock_cv2_cap, pose_analyzer):
        """Test successful video analysis."""
        # Mock file path
        mock_file = Mock()
        mock_file.stat.return_value.st_size = 1024 * 1024  # 1MB
        mock_file.name = "test.mp4"
        mock_file.suffix = ".mp4"
        mock_path.return_value = mock_file
        
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            'CAP_PROP_FPS': 30.0,
            'CAP_PROP_FRAME_COUNT': 100,
            'CAP_PROP_FRAME_WIDTH': 640,
            'CAP_PROP_FRAME_HEIGHT': 480
        }.get(prop, 0)
        
        # Mock frame reading
        mock_frames = [True] * 10 + [False]  # 10 frames then end
        frame_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        mock_cap.read.side_effect = [(frame_available, frame_data if frame_available else None) 
                                   for frame_available in mock_frames]
        mock_cap.release.return_value = None
        mock_cv2_cap.return_value = mock_cap
        
        # Mock pose detection
        with patch.object(pose_analyzer, 'analyze_frame') as mock_analyze_frame:
            mock_analyze_frame.return_value = FrameAnalysis(
                frame_number=0,
                timestamp_ms=0.0,
                landmarks=[PoseLandmark(x=0.5, y=0.5, z=0.0, visibility=0.9) for _ in range(33)],
                confidence_score=0.9,
                pose_detected=True
            )
            
            result = await pose_analyzer.analyze_video("test.mp4", "test_video_id")
        
        assert isinstance(result, VideoAnalysis)
        assert result.video_id == "test_video_id"
        assert result.status == ProcessingStatus.COMPLETED
        assert result.metadata.fps == 30.0
        assert result.metadata.total_frames == 100
        assert len(result.frame_analyses) == 10
        assert result.poses_detected_count > 0
        assert result.processing_time_seconds > 0
    
    @pytest.mark.asyncio
    @patch('cv2.VideoCapture')
    async def test_analyze_video_invalid_file(self, mock_cv2_cap, pose_analyzer):
        """Test video analysis with invalid file."""
        # Mock failed video opening
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_cv2_cap.return_value = mock_cap
        
        result = await pose_analyzer.analyze_video("invalid.mp4", "test_id")
        
        assert isinstance(result, VideoAnalysis)
        assert result.status == ProcessingStatus.FAILED
        assert result.error_message is not None
        assert result.poses_detected_count == 0
    
    @patch('cv2.VideoCapture')
    @patch('cv2.VideoWriter') 
    def test_create_pose_overlay_video_success(self, mock_writer, mock_cap, pose_analyzer):
        """Test successful pose overlay video creation."""
        # Mock video capture
        mock_cap_instance = Mock()
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.get.side_effect = lambda prop: {
            'CAP_PROP_FPS': 30.0,
            'CAP_PROP_FRAME_WIDTH': 640,
            'CAP_PROP_FRAME_HEIGHT': 480
        }.get(prop, 0)
        
        # Mock frame reading  
        frame_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_cap_instance.read.side_effect = [(True, frame_data), (False, None)]
        mock_cap.return_value = mock_cap_instance
        
        # Mock video writer
        mock_writer_instance = Mock()
        mock_writer.return_value = mock_writer_instance
        
        # Create mock analysis
        analysis = VideoAnalysis(
            video_id="test_id",
            status=ProcessingStatus.COMPLETED,
            metadata=VideoMetadata(
                video_id="test_id",
                filename="test.mp4",
                file_size_mb=1.0,
                duration_seconds=1.0,
                fps=30.0,
                total_frames=1,
                resolution_width=640,
                resolution_height=480,
                format="mp4"
            ),
            frame_analyses=[
                FrameAnalysis(
                    frame_number=0,
                    timestamp_ms=0.0,
                    landmarks=[PoseLandmark(x=0.5, y=0.5, z=0.0, visibility=0.9) for _ in range(33)],
                    confidence_score=0.9,
                    pose_detected=True
                )
            ],
            processing_time_seconds=1.0,
            gpu_memory_used_mb=100.0,
            poses_detected_count=1
        )
        
        with patch.object(pose_analyzer.pose, 'process') as mock_process:
            mock_results = Mock()
            mock_results.pose_landmarks = Mock()
            mock_process.return_value = mock_results
            
            result = pose_analyzer.create_pose_overlay_video("input.mp4", "output.mp4", analysis)
        
        assert result == True
        mock_writer_instance.write.assert_called()
        mock_writer_instance.release.assert_called()
    
    def test_get_performance_stats(self, pose_analyzer):
        """Test performance statistics retrieval."""
        stats = pose_analyzer.get_performance_stats()
        
        assert isinstance(stats, dict)
        assert 'frames_processed' in stats
        assert 'poses_detected' in stats
        assert 'total_processing_time' in stats
        assert 'gpu_memory_peak' in stats
    
    def test_reset_stats(self, pose_analyzer):
        """Test statistics reset."""
        # Set some stats
        pose_analyzer.processing_stats['frames_processed'] = 100
        pose_analyzer.processing_stats['poses_detected'] = 80
        
        pose_analyzer.reset_stats()
        
        assert pose_analyzer.processing_stats['frames_processed'] == 0
        assert pose_analyzer.processing_stats['poses_detected'] == 0
    
    def test_cleanup_resources(self, pose_analyzer):
        """Test proper resource cleanup."""
        # Mock the pose object
        mock_pose = Mock()
        pose_analyzer.pose = mock_pose
        
        # Trigger cleanup
        pose_analyzer.__del__()
        
        mock_pose.close.assert_called_once()


class TestPoseAnalyzerIntegration:
    """Integration tests for PoseAnalyzer with real MediaPipe components."""
    
    @pytest.fixture
    def real_pose_analyzer(self):
        """Create real PoseAnalyzer for integration tests."""
        return PoseAnalyzer(use_gpu=False, model_complexity=0)  # Use simplest model for tests
    
    def test_real_mediapipe_integration(self, real_pose_analyzer):
        """Test with real MediaPipe components (no pose detected on noise)."""
        # Create random noise frame (no actual pose)
        noise_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        
        result = real_pose_analyzer.analyze_frame(noise_frame, 0, 0.0)
        
        assert isinstance(result, FrameAnalysis)
        assert result.frame_number == 0
        # Most likely no pose detected in random noise
        assert result.pose_detected in [True, False]  # Either is valid
        assert 0.0 <= result.confidence_score <= 1.0
    
    @pytest.mark.slow
    def test_performance_under_load(self, real_pose_analyzer):
        """Test analyzer performance with multiple frames."""
        frames = []
        for i in range(10):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            frames.append((frame, i, i * 33.33))  # ~30 FPS timestamps
        
        results = []
        for frame, frame_num, timestamp in frames:
            result = real_pose_analyzer.analyze_frame(frame, frame_num, timestamp)
            results.append(result)
        
        assert len(results) == 10
        assert all(isinstance(r, FrameAnalysis) for r in results)
        assert all(r.frame_number == i for i, r in enumerate(results))
    
    def test_memory_usage_tracking(self, real_pose_analyzer):
        """Test memory usage tracking during processing."""
        initial_stats = real_pose_analyzer.get_performance_stats()
        
        # Process some frames
        for i in range(5):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            real_pose_analyzer.analyze_frame(frame, i, i * 33.33)
        
        final_stats = real_pose_analyzer.get_performance_stats()
        
        assert final_stats['frames_processed'] > initial_stats['frames_processed']
        assert final_stats['frames_processed'] == 5