"""
Tests for AI coach agent functionality.

Tests coaching feedback generation, technical metrics calculation,
chat functionality, and movement pattern analysis.
"""

import pytest
import numpy as np
import uuid
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

# Test UUIDs for consistent testing
TEST_UUID_1 = str(uuid.uuid4())
TEST_UUID_2 = str(uuid.uuid4()) 
TEST_UUID_3 = str(uuid.uuid4())

from src.ai_coach.coach_agent import CoachAgent, CoachingKnowledge
from src.ai_coach.models import (
    VideoAnalysis,
    VideoMetadata,
    FrameAnalysis,
    PoseLandmark,
    CoachingFeedback,
    CoachingMetrics,
    ChatSession,
    ChatMessage,
    ProcessingStatus,
)


class TestCoachAgent:
    """Test suite for CoachAgent class."""
    
    @pytest.fixture
    def coach_agent(self):
        """Create CoachAgent instance for testing."""
        return CoachAgent()
    
    @pytest.fixture
    def sample_landmarks(self):
        """Create sample pose landmarks for testing."""
        landmarks = []
        for i in range(33):
            landmarks.append(PoseLandmark(
                x=0.5 + (i * 0.01),
                y=0.5 + (i * 0.01),
                z=0.0 + (i * 0.001),
                visibility=0.9 - (i * 0.01)  # Varying visibility
            ))
        return landmarks
    
    @pytest.fixture
    def high_quality_frame_analysis(self, sample_landmarks):
        """Create high-quality frame analysis with good pose detection."""
        return FrameAnalysis(
            frame_number=0,
            timestamp_ms=0.0,
            landmarks=sample_landmarks,
            confidence_score=0.9,
            pose_detected=True
        )
    
    @pytest.fixture
    def low_quality_frame_analysis(self):
        """Create low-quality frame analysis with poor pose detection."""
        return FrameAnalysis(
            frame_number=0,
            timestamp_ms=0.0,
            landmarks=[],
            confidence_score=0.3,
            pose_detected=False
        )
    
    @pytest.fixture
    def high_quality_video_analysis(self, high_quality_frame_analysis):
        """Create high-quality video analysis for testing."""
        frame_analyses = [high_quality_frame_analysis] * 100  # 100 good frames
        
        return VideoAnalysis(
            video_id=TEST_UUID_1,
            status=ProcessingStatus.COMPLETED,
            metadata=VideoMetadata(
                video_id=TEST_UUID_1,
                filename="test.mp4",
                file_size_mb=5.0,
                duration_seconds=10.0,
                fps=30.0,
                total_frames=300,
                resolution_width=640,
                resolution_height=480,
                format="mp4"
            ),
            frame_analyses=frame_analyses,
            processing_time_seconds=15.0,
            gpu_memory_used_mb=1024.0,
            poses_detected_count=100
        )
    
    @pytest.fixture
    def low_quality_video_analysis(self, low_quality_frame_analysis):
        """Create low-quality video analysis for testing."""
        frame_analyses = [low_quality_frame_analysis] * 10  # Only 10 poor frames
        
        return VideoAnalysis(
            video_id=TEST_UUID_2,
            status=ProcessingStatus.COMPLETED,
            metadata=VideoMetadata(
                video_id=TEST_UUID_2,
                filename="poor_test.mp4",
                file_size_mb=1.0,
                duration_seconds=5.0,
                fps=30.0,
                total_frames=150,
                resolution_width=320,
                resolution_height=240,
                format="mp4"
            ),
            frame_analyses=frame_analyses,
            processing_time_seconds=8.0,
            gpu_memory_used_mb=256.0,
            poses_detected_count=2
        )
    
    def test_coach_agent_initialization(self, coach_agent):
        """Test CoachAgent initialization."""
        assert coach_agent is not None
        assert hasattr(coach_agent, 'knowledge')
        assert hasattr(coach_agent, 'coaching_sessions')
        assert hasattr(coach_agent, 'movement_patterns')
        assert isinstance(coach_agent.knowledge, CoachingKnowledge)
    
    @pytest.mark.asyncio
    async def test_generate_feedback_high_quality(self, coach_agent, high_quality_video_analysis):
        """Test coaching feedback generation for high-quality analysis."""
        feedback = await coach_agent.generate_feedback(high_quality_video_analysis, "general")
        
        assert isinstance(feedback, CoachingFeedback)
        assert feedback.video_id == TEST_UUID_1
        assert feedback.confidence_score > 0.7
        assert len(feedback.analysis_summary) > 10
        assert isinstance(feedback.key_issues, list)
        assert isinstance(feedback.improvement_suggestions, list)
        assert isinstance(feedback.coaching_metrics, CoachingMetrics)
        assert isinstance(feedback.priority_areas, list)
    
    @pytest.mark.asyncio
    async def test_generate_feedback_low_quality(self, coach_agent, low_quality_video_analysis):
        """Test coaching feedback generation for low-quality analysis."""
        feedback = await coach_agent.generate_feedback(low_quality_video_analysis, "general")
        
        assert isinstance(feedback, CoachingFeedback)
        assert feedback.video_id == TEST_UUID_2
        assert feedback.confidence_score <= 0.5
        assert "quality insufficient" in feedback.analysis_summary.lower() or "poor" in feedback.analysis_summary.lower()
        assert "video_quality" in feedback.priority_areas or "video quality" in feedback.analysis_summary
    
    @pytest.mark.asyncio
    async def test_generate_feedback_movement_types(self, coach_agent, high_quality_video_analysis):
        """Test coaching feedback for different movement types."""
        movement_types = ["squat", "deadlift", "overhead_press", "general"]
        
        for movement_type in movement_types:
            feedback = await coach_agent.generate_feedback(high_quality_video_analysis, movement_type)
            
            assert isinstance(feedback, CoachingFeedback)
            assert feedback.video_id == TEST_UUID_1
            assert feedback.confidence_score > 0.0
    
    @pytest.mark.asyncio
    async def test_calculate_coaching_metrics(self, coach_agent, high_quality_video_analysis):
        """Test technical metrics calculation."""
        metrics = await coach_agent._calculate_coaching_metrics(high_quality_video_analysis)
        
        assert isinstance(metrics, CoachingMetrics)
        
        # Check that metrics are calculated and within valid ranges
        if metrics.movement_smoothness is not None:
            assert 0.0 <= metrics.movement_smoothness <= 1.0
        if metrics.posture_stability is not None:
            assert 0.0 <= metrics.posture_stability <= 1.0
        if metrics.balance_score is not None:
            assert 0.0 <= metrics.balance_score <= 1.0
        if metrics.symmetry_score is not None:
            assert 0.0 <= metrics.symmetry_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_calculate_coaching_metrics_insufficient_data(self, coach_agent, low_quality_video_analysis):
        """Test metrics calculation with insufficient data."""
        metrics = await coach_agent._calculate_coaching_metrics(low_quality_video_analysis)
        
        assert isinstance(metrics, CoachingMetrics)
        # Most metrics should be None or default due to insufficient data
    
    def test_calculate_movement_smoothness(self, coach_agent, high_quality_video_analysis):
        """Test movement smoothness calculation."""
        valid_frames = [f for f in high_quality_video_analysis.frame_analyses if f.has_valid_pose]
        
        smoothness = coach_agent._calculate_movement_smoothness(valid_frames)
        
        assert isinstance(smoothness, float)
        assert 0.0 <= smoothness <= 1.0
    
    def test_calculate_posture_stability(self, coach_agent, high_quality_video_analysis):
        """Test posture stability calculation."""
        valid_frames = [f for f in high_quality_video_analysis.frame_analyses if f.has_valid_pose]
        
        stability = coach_agent._calculate_posture_stability(valid_frames)
        
        assert isinstance(stability, float)
        assert 0.0 <= stability <= 1.0
    
    def test_calculate_joint_angle(self, coach_agent):
        """Test joint angle calculation between three points."""
        # Create three points forming a right angle
        p1 = PoseLandmark(x=0.0, y=0.0, z=0.0, visibility=1.0)  # Origin
        p2 = PoseLandmark(x=1.0, y=0.0, z=0.0, visibility=1.0)  # Right
        p3 = PoseLandmark(x=1.0, y=1.0, z=0.0, visibility=1.0)  # Up
        
        angle = coach_agent._calculate_joint_angle(p1, p2, p3)
        
        assert angle is not None
        assert isinstance(angle, float)
        # Should be approximately π/2 (90 degrees) for right angle
        assert abs(angle - np.pi/2) < 0.1
    
    def test_calculate_joint_angle_error_handling(self, coach_agent):
        """Test joint angle calculation error handling."""
        # Create degenerate points (same position)
        p1 = PoseLandmark(x=0.0, y=0.0, z=0.0, visibility=1.0)
        p2 = PoseLandmark(x=0.0, y=0.0, z=0.0, visibility=1.0)  # Same as p1
        p3 = PoseLandmark(x=1.0, y=1.0, z=0.0, visibility=1.0)
        
        angle = coach_agent._calculate_joint_angle(p1, p2, p3)
        
        # Should handle division by zero gracefully
        assert angle is None or isinstance(angle, float)
    
    def test_analyze_squat_pattern(self, coach_agent, high_quality_video_analysis):
        """Test squat movement pattern analysis."""
        # Mock metrics with poor symmetry to trigger squat-specific feedback
        metrics = CoachingMetrics(
            movement_smoothness=0.8,
            posture_stability=0.5,  # Poor stability
            symmetry_score=0.6,    # Poor symmetry
            balance_score=0.7
        )
        
        analysis = coach_agent._analyze_squat_pattern(high_quality_video_analysis, metrics)
        
        assert isinstance(analysis, dict)
        assert "summary" in analysis
        assert "issues" in analysis
        assert "suggestions" in analysis
        assert "priorities" in analysis
        
        # Should identify stability and symmetry issues
        assert any("stability" in issue.lower() for issue in analysis["issues"])
        assert any("knee" in issue.lower() or "symmetry" in issue.lower() for issue in analysis["issues"])
    
    def test_analyze_deadlift_pattern(self, coach_agent, high_quality_video_analysis):
        """Test deadlift movement pattern analysis."""
        metrics = CoachingMetrics(
            posture_stability=0.6,   # Poor back stability
            symmetry_score=0.7,     # Decent symmetry
            balance_score=0.8
        )
        
        analysis = coach_agent._analyze_deadlift_pattern(high_quality_video_analysis, metrics)
        
        assert isinstance(analysis, dict)
        assert "summary" in analysis
        
        # Should identify back stability issues
        assert any("back" in issue.lower() or "stability" in issue.lower() 
                  for issue in analysis["issues"])
    
    def test_analyze_general_movement(self, coach_agent, high_quality_video_analysis):
        """Test general movement pattern analysis."""
        metrics = CoachingMetrics(
            movement_smoothness=0.5,  # Poor smoothness
            balance_score=0.5,        # Poor balance
            symmetry_score=0.6        # Decent symmetry
        )
        
        analysis = coach_agent._analyze_general_movement(high_quality_video_analysis, metrics)
        
        assert isinstance(analysis, dict)
        assert "summary" in analysis
        assert len(analysis["issues"]) >= 2  # Should identify multiple issues
        assert "control" in analysis["priorities"]  # Poor smoothness
        assert "balance" in analysis["priorities"]   # Poor balance
    
    def test_calculate_feedback_confidence(self, coach_agent, high_quality_video_analysis):
        """Test feedback confidence calculation."""
        metrics = CoachingMetrics(
            movement_smoothness=0.8,
            posture_stability=0.9,
            balance_score=0.7,
            symmetry_score=0.8
        )
        
        confidence = coach_agent._calculate_feedback_confidence(high_quality_video_analysis, metrics)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        # Should be high for quality analysis
        assert confidence > 0.6
    
    @pytest.mark.asyncio
    async def test_handle_chat_message_greeting(self, coach_agent):
        """Test chat message handling for greetings."""
        response = await coach_agent.handle_chat_message("session_1", "Hello!", None)
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert any(greeting in response.lower() for greeting in ["hello", "hi", "coach"])
    
    @pytest.mark.asyncio
    async def test_handle_chat_message_analysis_request(self, coach_agent):
        """Test chat message handling for analysis requests."""
        response = await coach_agent.handle_chat_message("session_2", "Can you analyze my form?", "video_123")
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert any(word in response.lower() for word in ["analyze", "analyzing", "movement"])
    
    @pytest.mark.asyncio
    async def test_handle_chat_message_improvement_help(self, coach_agent):
        """Test chat message handling for improvement requests."""
        response = await coach_agent.handle_chat_message("session_3", "How can I improve my technique?", None)
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert any(word in response.lower() for word in ["improve", "better", "upload", "feedback"])
    
    @pytest.mark.asyncio
    async def test_handle_chat_message_creates_session(self, coach_agent):
        """Test that chat messages create and maintain sessions."""
        session_id = "new_session"
        
        # Session should not exist initially
        assert session_id not in coach_agent.coaching_sessions
        
        await coach_agent.handle_chat_message(session_id, "Hello", None)
        
        # Session should now exist
        assert session_id in coach_agent.coaching_sessions
        session = coach_agent.coaching_sessions[session_id]
        assert isinstance(session, ChatSession)
        assert len(session.messages) == 2  # User message + assistant response
    
    def test_get_session(self, coach_agent):
        """Test session retrieval."""
        # Create a session
        session = ChatSession(
            session_id="test_session",
            messages=[],
            video_ids=[]
        )
        coach_agent.coaching_sessions["test_session"] = session
        
        retrieved = coach_agent.get_session("test_session")
        assert retrieved is session
        
        # Test non-existent session
        assert coach_agent.get_session("nonexistent") is None
    
    def test_cleanup_old_sessions(self, coach_agent):
        """Test cleanup of old chat sessions."""
        # Create old and recent sessions
        old_session = ChatSession(
            session_id="old_session",
            messages=[],
            video_ids=[]
        )
        old_session.last_activity = datetime.utcnow() - timedelta(hours=25)
        
        recent_session = ChatSession(
            session_id="recent_session", 
            messages=[],
            video_ids=[]
        )
        recent_session.last_activity = datetime.utcnow() - timedelta(hours=1)
        
        coach_agent.coaching_sessions["old_session"] = old_session
        coach_agent.coaching_sessions["recent_session"] = recent_session
        
        # Cleanup sessions older than 24 hours
        cleaned = coach_agent.cleanup_old_sessions(max_age_hours=24)
        
        assert cleaned == 1
        assert "old_session" not in coach_agent.coaching_sessions
        assert "recent_session" in coach_agent.coaching_sessions


class TestCoachingKnowledge:
    """Test suite for CoachingKnowledge dataclass."""
    
    def test_coaching_knowledge_constants(self):
        """Test that CoachingKnowledge has required constants."""
        knowledge = CoachingKnowledge()
        
        # Test landmark indices are defined
        assert hasattr(knowledge, 'NOSE')
        assert hasattr(knowledge, 'LEFT_SHOULDER')
        assert hasattr(knowledge, 'RIGHT_SHOULDER')
        assert hasattr(knowledge, 'LEFT_HIP')
        assert hasattr(knowledge, 'RIGHT_HIP')
        
        # Test landmark indices are valid (0-32 for 33 landmarks)
        assert 0 <= knowledge.NOSE <= 32
        assert 0 <= knowledge.LEFT_SHOULDER <= 32
        assert 0 <= knowledge.RIGHT_SHOULDER <= 32
        
        # Test angle ranges are defined
        assert hasattr(knowledge, 'SQUAT_KNEE_ANGLE_RANGE')
        assert hasattr(knowledge, 'DEADLIFT_BACK_ANGLE_RANGE')
        assert hasattr(knowledge, 'OVERHEAD_ARM_ANGLE_MIN')
        
        # Test thresholds are defined
        assert hasattr(knowledge, 'STABILITY_THRESHOLD')
        assert hasattr(knowledge, 'SYMMETRY_THRESHOLD')


class TestCoachAgentIntegration:
    """Integration tests for CoachAgent with realistic scenarios."""
    
    @pytest.fixture
    def integration_coach(self):
        """Create coach agent for integration testing."""
        return CoachAgent()
    
    @pytest.mark.asyncio
    async def test_full_coaching_workflow(self, integration_coach, high_quality_video_analysis):
        """Test complete coaching workflow from analysis to chat."""
        # Generate feedback
        feedback = await integration_coach.generate_feedback(high_quality_video_analysis, "squat")
        
        assert isinstance(feedback, CoachingFeedback)
        assert feedback.confidence_score > 0.0
        
        # Start chat session about the analysis
        session_id = "integration_test"
        response1 = await integration_coach.handle_chat_message(
            session_id, 
            "What are the main issues you see in my squat?",
            high_quality_video_analysis.video_id
        )
        
        assert isinstance(response1, str)
        assert len(response1) > 0
        
        # Follow up question
        response2 = await integration_coach.handle_chat_message(
            session_id,
            "How can I improve my balance?",
            high_quality_video_analysis.video_id
        )
        
        assert isinstance(response2, str)
        assert len(response2) > 0
        
        # Check session was maintained
        session = integration_coach.get_session(session_id)
        assert session is not None
        assert len(session.messages) == 4  # 2 user + 2 assistant messages
        assert high_quality_video_analysis.video_id in session.video_ids
    
    @pytest.mark.asyncio
    async def test_coaching_consistency(self, integration_coach, high_quality_video_analysis):
        """Test that coaching feedback is consistent across multiple calls."""
        feedback1 = await integration_coach.generate_feedback(high_quality_video_analysis, "general")
        feedback2 = await integration_coach.generate_feedback(high_quality_video_analysis, "general")
        
        # Should be consistent (same video, same movement type)
        assert feedback1.video_id == feedback2.video_id
        assert abs(feedback1.confidence_score - feedback2.confidence_score) < 0.1
        # Summary might vary slightly but should be similar length
        assert abs(len(feedback1.analysis_summary) - len(feedback2.analysis_summary)) < 100
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, integration_coach):
        """Test error handling and recovery in coaching feedback."""
        # Create analysis with corrupted data
        corrupted_analysis = VideoAnalysis(
            video_id=TEST_UUID_3,
            status=ProcessingStatus.COMPLETED,
            metadata=VideoMetadata(
                video_id=TEST_UUID_3,
                filename="corrupted.mp4",
                file_size_mb=0.1,
                duration_seconds=0.1,
                fps=1.0,
                total_frames=1,
                resolution_width=1,
                resolution_height=1,
                format="unknown"
            ),
            frame_analyses=[],  # No frame data
            processing_time_seconds=0.0,
            gpu_memory_used_mb=0.0,
            poses_detected_count=0
        )
        
        # Should handle gracefully without crashing
        feedback = await integration_coach.generate_feedback(corrupted_analysis, "general")
        
        assert isinstance(feedback, CoachingFeedback)
        assert feedback.video_id == TEST_UUID_3
        assert feedback.confidence_score <= 0.5  # Should be low confidence
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_performance_under_load(self, integration_coach, high_quality_video_analysis):
        """Test coach agent performance with multiple concurrent requests."""
        import asyncio
        
        # Generate multiple feedback requests concurrently
        tasks = []
        for i in range(5):
            task = integration_coach.generate_feedback(high_quality_video_analysis, "general")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed
        assert len(results) == 5
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Coaching feedback generation failed: {result}")
            assert isinstance(result, CoachingFeedback)
            assert result.confidence_score > 0.0