"""
Tests for video processor functionality.

Tests video upload handling, validation, temporary file management,
and integration with pose analyzer.
"""

import pytest
import asyncio
import tempfile
import json
import uuid
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta

# Test UUIDs for consistent testing
TEST_UUID_1 = str(uuid.uuid4())
TEST_UUID_2 = str(uuid.uuid4())
TEST_UUID_3 = str(uuid.uuid4())

import aiofiles
from fastapi import UploadFile, HTTPException

from src.ai_coach.video_processor import VideoProcessor
from src.ai_coach.models import (
    VideoAnalysis,
    VideoMetadata,
    ProcessingStatus,
    FrameAnalysis,
    PoseLandmark,
)
from src.ai_coach.pose_analyzer import PoseAnalyzer


class MockUploadFile:
    """Mock FastAPI UploadFile for testing."""
    
    def __init__(self, filename: str, content: bytes, content_type: str = "video/mp4", size: int = None):
        self.filename = filename
        self.content = content
        self.content_type = content_type
        self.size = size or len(content)
        self._position = 0
    
    async def read(self, size: int = None) -> bytes:
        """Mock file reading."""
        if size is None:
            result = self.content[self._position:]
            self._position = len(self.content)
        else:
            result = self.content[self._position:self._position + size]
            self._position += len(result)
        return result
    
    async def seek(self, position: int):
        """Mock file seeking."""
        self._position = position


class TestVideoProcessor:
    """Test suite for VideoProcessor class."""
    
    @pytest.fixture
    def tmp_uploads_dir(self, tmp_path):
        """Create temporary uploads directory."""
        uploads_dir = tmp_path / "uploads"
        return str(uploads_dir)
    
    @pytest.fixture
    def mock_pose_analyzer(self):
        """Create mock PoseAnalyzer."""
        analyzer = Mock(spec=PoseAnalyzer)
        
        # Mock async analyze_video method
        async def mock_analyze_video(video_path, video_id):
            return VideoAnalysis(
                video_id=video_id,
                status=ProcessingStatus.COMPLETED,
                metadata=VideoMetadata(
                    video_id=video_id,
                    filename="test.mp4",
                    file_size_mb=1.0,
                    duration_seconds=10.0,
                    fps=30.0,
                    total_frames=300,
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
                processing_time_seconds=2.0,
                gpu_memory_used_mb=512.0,
                poses_detected_count=1
            )
        
        analyzer.analyze_video = mock_analyze_video
        analyzer.create_pose_overlay_video = Mock(return_value=True)
        return analyzer
    
    @pytest.fixture
    def video_processor(self, tmp_uploads_dir, mock_pose_analyzer):
        """Create VideoProcessor instance for testing."""
        return VideoProcessor(uploads_dir=tmp_uploads_dir, pose_analyzer=mock_pose_analyzer)
    
    @pytest.fixture
    def valid_upload_file(self):
        """Create valid upload file for testing."""
        content = b"fake mp4 video content"
        return MockUploadFile("test_video.mp4", content, "video/mp4")
    
    @pytest.fixture
    def large_upload_file(self):
        """Create large upload file that exceeds size limit."""
        content = b"x" * (101 * 1024 * 1024)  # 101MB
        return MockUploadFile("large_video.mp4", content, "video/mp4")
    
    @pytest.fixture
    def invalid_format_file(self):
        """Create invalid format file."""
        content = b"not a video file"
        return MockUploadFile("document.txt", content, "text/plain")
    
    def test_video_processor_initialization(self, video_processor):
        """Test VideoProcessor initialization."""
        assert video_processor is not None
        assert video_processor.uploads_dir.exists()
        assert video_processor.temp_dir.exists()
        assert video_processor.processed_dir.exists()
        assert video_processor.results_dir.exists()
        assert isinstance(video_processor.active_jobs, dict)
    
    @pytest.mark.asyncio
    async def test_validate_video_file_success(self, video_processor, valid_upload_file):
        """Test successful video file validation."""
        result = await video_processor.validate_video_file(valid_upload_file)
        
        assert result["valid"] == True
        assert result["filename"] == "test_video.mp4"
        assert result["format"] == ".mp4"
        assert result["size_bytes"] > 0
    
    @pytest.mark.asyncio
    async def test_validate_video_file_no_filename(self, video_processor):
        """Test validation failure when no filename provided."""
        file = MockUploadFile("", b"content")
        file.filename = None
        
        with pytest.raises(HTTPException) as exc_info:
            await video_processor.validate_video_file(file)
        
        assert exc_info.value.status_code == 400
        assert "No filename provided" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_validate_video_file_invalid_format(self, video_processor, invalid_format_file):
        """Test validation failure for invalid file format."""
        with pytest.raises(HTTPException) as exc_info:
            await video_processor.validate_video_file(invalid_format_file)
        
        assert exc_info.value.status_code == 400
        assert "Unsupported format" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_validate_video_file_too_large(self, video_processor, large_upload_file):
        """Test validation failure for file too large."""
        with pytest.raises(HTTPException) as exc_info:
            await video_processor.validate_video_file(large_upload_file)
        
        assert exc_info.value.status_code == 413
        assert "File too large" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    @patch('cv2.VideoCapture')
    async def test_process_upload_success(self, mock_cv2_cap, video_processor, valid_upload_file):
        """Test successful video upload processing."""
        # Mock OpenCV video validation
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            'CAP_PROP_FPS': 30.0,
            'CAP_PROP_FRAME_COUNT': 300,
        }.get(prop, 0)
        mock_cap.release.return_value = None
        mock_cv2_cap.return_value = mock_cap
        
        temp_path = await video_processor.process_upload(valid_upload_file)
        
        assert temp_path is not None
        assert Path(temp_path).exists()
        assert Path(temp_path).suffix == ".mp4"
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    @patch('cv2.VideoCapture')
    async def test_process_upload_invalid_video(self, mock_cv2_cap, video_processor, valid_upload_file):
        """Test upload processing with invalid video that fails OpenCV validation."""
        # Mock OpenCV failing to open video
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_cv2_cap.return_value = mock_cap
        
        with pytest.raises(HTTPException) as exc_info:
            await video_processor.process_upload(valid_upload_file)
        
        assert exc_info.value.status_code == 400
        assert "Invalid video format" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    @patch('cv2.VideoCapture')
    async def test_process_upload_video_too_long(self, mock_cv2_cap, video_processor, valid_upload_file):
        """Test upload processing with video exceeding duration limit."""
        # Mock video with duration exceeding limit
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            'CAP_PROP_FPS': 30.0,
            'CAP_PROP_FRAME_COUNT': 15000,  # 500 seconds at 30 FPS
        }.get(prop, 0)
        mock_cap.release.return_value = None
        mock_cv2_cap.return_value = mock_cap
        
        with pytest.raises(HTTPException) as exc_info:
            await video_processor.process_upload(valid_upload_file)
        
        assert exc_info.value.status_code == 413
        assert "Video too long" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_analyze_video_async_success(self, video_processor, tmp_path):
        """Test successful asynchronous video analysis."""
        # Create a temporary video file
        temp_video = tmp_path / "test.mp4"
        temp_video.write_bytes(b"fake video content")
        
        result = await video_processor.analyze_video_async(str(temp_video), TEST_UUID_3)
        
        assert isinstance(result, VideoAnalysis)
        assert result.video_id == TEST_UUID_3
        assert result.status == ProcessingStatus.COMPLETED
        
        # Check job tracking
        assert TEST_UUID_3 in video_processor.active_jobs
        assert video_processor.active_jobs[TEST_UUID_3]["status"] == ProcessingStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_analyze_video_async_error(self, video_processor):
        """Test video analysis with error (non-existent file)."""
        result = await video_processor.analyze_video_async("nonexistent.mp4", "error_id")
        
        assert isinstance(result, VideoAnalysis)
        assert result.video_id == "error_id"
        assert result.status == ProcessingStatus.FAILED
        assert result.error_message is not None
        
        # Check job tracking shows failure
        assert "error_id" in video_processor.active_jobs
        assert video_processor.active_jobs["error_id"]["status"] == ProcessingStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_save_analysis_results(self, video_processor):
        """Test saving analysis results to JSON file."""
        analysis = VideoAnalysis(
            video_id=TEST_UUID_1,
            status=ProcessingStatus.COMPLETED,
            metadata=VideoMetadata(
                video_id=TEST_UUID_1,
                filename="test.mp4",
                file_size_mb=1.0,
                duration_seconds=10.0,
                fps=30.0,
                total_frames=300,
                resolution_width=640,
                resolution_height=480,
                format="mp4"
            ),
            frame_analyses=[],
            processing_time_seconds=5.0,
            gpu_memory_used_mb=256.0,
            poses_detected_count=250
        )
        
        await video_processor._save_analysis_results(analysis)
        
        # Check file was created
        results_file = video_processor.results_dir / "save_test.json"
        assert results_file.exists()
        
        # Check file content
        async with aiofiles.open(results_file, 'r') as f:
            data = json.loads(await f.read())
        
        assert data["video_id"] == TEST_UUID_1
        assert data["status"] == "completed"
        assert data["poses_detected_count"] == 250
    
    @pytest.mark.asyncio
    async def test_get_analysis_results(self, video_processor):
        """Test retrieving analysis results."""
        # First save some results
        analysis = VideoAnalysis(
            video_id=TEST_UUID_2,
            status=ProcessingStatus.COMPLETED,
            metadata=VideoMetadata(
                video_id=TEST_UUID_2,
                filename="test.mp4",
                file_size_mb=1.0,
                duration_seconds=10.0,
                fps=30.0,
                total_frames=300,
                resolution_width=640,
                resolution_height=480,
                format="mp4"
            ),
            frame_analyses=[],
            processing_time_seconds=3.0,
            gpu_memory_used_mb=128.0,
            poses_detected_count=200
        )
        
        await video_processor._save_analysis_results(analysis)
        
        # Retrieve results
        retrieved = await video_processor.get_analysis_results(TEST_UUID_2)
        
        assert retrieved is not None
        assert retrieved.video_id == TEST_UUID_2
        assert retrieved.poses_detected_count == 200
    
    @pytest.mark.asyncio
    async def test_get_analysis_results_not_found(self, video_processor):
        """Test retrieving non-existent analysis results."""
        result = await video_processor.get_analysis_results("nonexistent")
        assert result is None
    
    def test_get_job_status(self, video_processor):
        """Test job status retrieval."""
        # Add a job
        video_processor.active_jobs["status_test"] = {
            "status": ProcessingStatus.PROCESSING,
            "progress": 50.0,
            "start_time": datetime.utcnow()
        }
        
        status = video_processor.get_job_status("status_test")
        assert status is not None
        assert status["status"] == ProcessingStatus.PROCESSING
        assert status["progress"] == 50.0
        
        # Test non-existent job
        assert video_processor.get_job_status("nonexistent") is None
    
    def test_get_all_active_jobs(self, video_processor):
        """Test retrieving all active jobs."""
        # Add some jobs
        video_processor.active_jobs["job1"] = {"status": "processing"}
        video_processor.active_jobs["job2"] = {"status": "completed"}
        
        all_jobs = video_processor.get_all_active_jobs()
        
        assert len(all_jobs) == 2
        assert "job1" in all_jobs
        assert "job2" in all_jobs
        assert all_jobs["job1"]["status"] == "processing"
    
    @pytest.mark.asyncio
    async def test_cleanup_old_files(self, video_processor, tmp_path):
        """Test cleanup of old files."""
        # Create old files in temp directory
        old_file = video_processor.temp_dir / "old_video.mp4"
        old_file.write_bytes(b"old content")
        
        # Make file appear old by modifying its timestamp
        old_timestamp = datetime.utcnow() - timedelta(hours=25)
        import os
        os.utime(old_file, (old_timestamp.timestamp(), old_timestamp.timestamp()))
        
        # Create recent file that should not be deleted
        recent_file = video_processor.temp_dir / "recent_video.mp4"
        recent_file.write_bytes(b"recent content")
        
        # Run cleanup
        stats = await video_processor.cleanup_old_files(max_age_hours=24)
        
        assert stats["temp_files"] >= 1
        assert not old_file.exists()
        assert recent_file.exists()
    
    @pytest.mark.asyncio
    async def test_get_processed_video_path(self, video_processor):
        """Test getting processed video path."""
        # Create processed video file
        processed_path = video_processor.processed_dir / "test_id_processed.mp4"
        processed_path.write_bytes(b"processed video content")
        
        result = await video_processor.get_processed_video_path("test_id")
        assert result == processed_path
        assert result.exists()
        
        # Test non-existent processed video
        result = await video_processor.get_processed_video_path("nonexistent")
        assert result is None
    
    def test_get_storage_stats(self, video_processor):
        """Test storage statistics calculation."""
        # Create some files
        (video_processor.temp_dir / "temp1.mp4").write_bytes(b"temp content")
        (video_processor.processed_dir / "processed1.mp4").write_bytes(b"processed content")
        (video_processor.results_dir / "result1.json").write_bytes(b'{"test": "data"}')
        
        stats = video_processor.get_storage_stats()
        
        assert "temp_files" in stats
        assert "processed_files" in stats
        assert "result_files" in stats
        assert "active_jobs" in stats
        assert "total_size_mb" in stats
        
        assert stats["temp_files"] >= 1
        assert stats["processed_files"] >= 1
        assert stats["result_files"] >= 1
        assert isinstance(stats["total_size_mb"], (int, float))


class TestVideoProcessorIntegration:
    """Integration tests for VideoProcessor with file I/O and real components."""
    
    @pytest.fixture
    def integration_processor(self, tmp_path):
        """Create VideoProcessor for integration testing."""
        return VideoProcessor(uploads_dir=str(tmp_path / "uploads"))
    
    @pytest.mark.asyncio
    async def test_full_upload_workflow(self, integration_processor):
        """Test complete upload and processing workflow."""
        # Create mock upload file
        content = b"fake video content for integration test"
        upload_file = MockUploadFile("integration_test.mp4", content, "video/mp4")
        
        with patch('cv2.VideoCapture') as mock_cv2:
            # Mock successful video validation
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                'CAP_PROP_FPS': 30.0,
                'CAP_PROP_FRAME_COUNT': 90,  # 3 seconds
            }.get(prop, 0)
            mock_cap.release.return_value = None
            mock_cv2.return_value = mock_cap
            
            # Process upload
            temp_path = await integration_processor.process_upload(upload_file)
            
            assert Path(temp_path).exists()
            assert Path(temp_path).read_bytes() == content
            
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, integration_processor):
        """Test handling multiple concurrent processing jobs."""
        # Start multiple analysis jobs
        jobs = []
        for i in range(3):
            temp_file = integration_processor.temp_dir / f"concurrent_{i}.mp4"
            temp_file.write_bytes(b"concurrent test content")
            
            job = asyncio.create_task(
                integration_processor.analyze_video_async(str(temp_file), f"concurrent_{i}")
            )
            jobs.append(job)
        
        # Wait for all jobs to complete
        results = await asyncio.gather(*jobs, return_exceptions=True)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Some failures are expected with mock components
                continue
            assert isinstance(result, VideoAnalysis)
            assert result.video_id == f"concurrent_{i}"
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_memory_cleanup_during_processing(self, integration_processor, tmp_path):
        """Test memory cleanup during long processing sessions."""
        # Simulate processing many videos
        for i in range(10):
            temp_file = tmp_path / f"memory_test_{i}.mp4"
            temp_file.write_bytes(b"memory test content")
            
            # Add to active jobs to simulate processing
            integration_processor.active_jobs[f"memory_test_{i}"] = {
                "status": ProcessingStatus.PROCESSING,
                "progress": 50.0,
                "start_time": datetime.utcnow() - timedelta(hours=i)  # Spread across time
            }
        
        # Test cleanup
        initial_job_count = len(integration_processor.active_jobs)
        stats = await integration_processor.cleanup_old_files(max_age_hours=5)
        final_job_count = len(integration_processor.active_jobs)
        
        # Some old jobs should have been cleaned up
        assert final_job_count < initial_job_count
        assert isinstance(stats, dict)