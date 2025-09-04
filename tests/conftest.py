"""
Pytest configuration and fixtures for AI coach tests.
"""

import pytest
import uuid
import numpy as np
from unittest.mock import Mock, AsyncMock
from pathlib import Path
from datetime import datetime

from src.ai_coach.models import (
    PoseLandmark,
    FrameAnalysis,
    VideoAnalysis,
    VideoMetadata,
    ProcessingStatus,
    CoachingFeedback,
    CoachingMetrics,
    ChatSession,
    ChatMessage,
)


@pytest.fixture
def sample_pose_landmarks():
    """Create sample pose landmarks for testing."""
    landmarks = []
    for i in range(33):  # MediaPipe has 33 pose landmarks
        landmarks.append(PoseLandmark(
            x=0.5 + (i * 0.01),
            y=0.5 + (i * 0.01),
            z=0.0,
            visibility=0.9
        ))
    return landmarks


@pytest.fixture
def sample_frame_analysis(sample_pose_landmarks):
    """Create sample frame analysis."""
    return FrameAnalysis(
        frame_number=1,
        timestamp_ms=33.33,  # ~30 FPS
        landmarks=sample_pose_landmarks,
        confidence_score=0.95,
        pose_detected=True
    )


@pytest.fixture
def sample_video_metadata():
    """Create sample video metadata."""
    return VideoMetadata(
        video_id=str(uuid.uuid4()),
        filename="test_video.mp4",
        file_size_mb=10.5,
        duration_seconds=30.0,
        fps=30.0,
        total_frames=900,
        resolution_width=1920,
        resolution_height=1080,
        format="mp4"
    )


@pytest.fixture
def sample_video_analysis(sample_video_metadata, sample_frame_analysis):
    """Create sample video analysis."""
    return VideoAnalysis(
        video_id=sample_video_metadata.video_id,
        status=ProcessingStatus.COMPLETED,
        metadata=sample_video_metadata,
        frame_analyses=[sample_frame_analysis] * 10,  # 10 sample frames
        processing_time_seconds=5.2,
        gpu_memory_used_mb=2048.0,
        poses_detected_count=10
    )


@pytest.fixture
def sample_coaching_feedback(sample_video_metadata):
    """Create sample coaching feedback."""
    return CoachingFeedback(
        video_id=sample_video_metadata.video_id,
        movement_type="squat",
        analysis_summary="Good form overall with room for improvement in depth.",
        priority_areas=["squat_depth", "knee_alignment"],
        specific_recommendations=[
            "Focus on reaching parallel or below",
            "Keep knees aligned with toes"
        ],
        confidence_score=0.85
    )


@pytest.fixture
def sample_coaching_metrics():
    """Create sample coaching metrics."""
    return CoachingMetrics(
        smoothness_score=0.8,
        stability_score=0.75,
        balance_score=0.9,
        symmetry_score=0.85,
        range_of_motion_score=0.7
    )


@pytest.fixture
def sample_chat_session():
    """Create sample chat session."""
    return ChatSession(
        session_id="test_session",
        video_ids=[str(uuid.uuid4())],
        messages=[
            ChatMessage(
                role="user",
                content="How's my squat form?",
                timestamp=datetime.utcnow()
            ),
            ChatMessage(
                role="assistant", 
                content="Your squat form looks good overall. I noticed you could go a bit deeper.",
                timestamp=datetime.utcnow()
            )
        ],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )


@pytest.fixture
def mock_video_file(tmp_path):
    """Create mock video file for testing."""
    video_file = tmp_path / "test_video.mp4"
    video_file.write_bytes(b"fake video content for testing")
    return video_file


@pytest.fixture
def mock_pose_analyzer():
    """Mock pose analyzer for testing."""
    analyzer = Mock()
    analyzer.analyze_frame = Mock(return_value=Mock(
        frame_number=1,
        timestamp_ms=33.33,
        landmarks=[],
        confidence_score=0.8,
        pose_detected=True
    ))
    analyzer.analyze_video = AsyncMock(return_value=Mock(
        video_id="test_id",
        status=ProcessingStatus.COMPLETED,
        poses_detected_count=10
    ))
    return analyzer


@pytest.fixture
def mock_coach_agent():
    """Mock coach agent for testing."""
    agent = Mock()
    agent.generate_feedback = AsyncMock(return_value=Mock(
        video_id="test_id",
        movement_type="general",
        analysis_summary="Good movement pattern detected",
        confidence_score=0.8
    ))
    agent.handle_chat_message = AsyncMock(return_value="Thank you for your question!")
    return agent


# Pytest configuration
def pytest_configure(config):
    """Pytest configuration for AI coach tests."""
    # Add custom markers
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark GPU tests
        if "gpu" in item.nodeid.lower() or "cuda" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Mark integration tests
        if "integration" in item.nodeid.lower() or "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if "slow" in item.nodeid.lower() or "mediapipe" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)