"""
Pydantic models for AI coach pose analysis system.

This module defines the core data models used throughout the AI coach system,
ensuring type safety and data validation for pose analysis, video processing,
and coaching feedback.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid


class ProcessingStatus(str, Enum):
    """Video processing status enum."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PoseLandmark(BaseModel):
    """Single 3D pose landmark point."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "x": 0.5,
                "y": 0.3, 
                "z": -0.1,
                "visibility": 0.95
            }
        }
    )
    
    x: float = Field(description="X coordinate in meters (world) or normalized (image)")
    y: float = Field(description="Y coordinate in meters (world) or normalized (image)")
    z: float = Field(description="Z coordinate in meters (world) or depth (image)")
    visibility: float = Field(
        ge=0, le=1, description="Landmark visibility score (0-1)"
    )
    
    @field_validator("visibility")
    @classmethod
    def validate_visibility(cls, v: float) -> float:
        """Ensure visibility is within valid range."""
        if not 0 <= v <= 1:
            raise ValueError("Visibility must be between 0 and 1")
        return v


class FrameAnalysis(BaseModel):
    """Analysis of a single video frame."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "frame_number": 100,
                "timestamp_ms": 3333.33,
                "landmarks": [],  # Would contain 33 PoseLandmark objects
                "confidence_score": 0.92,
                "pose_detected": True
            }
        }
    )
    
    frame_number: int = Field(ge=0, description="Frame number in video sequence")
    timestamp_ms: float = Field(ge=0, description="Timestamp in milliseconds")
    landmarks: List[PoseLandmark] = Field(
        max_items=33, 
        min_items=0, 
        description="33 MediaPipe pose landmarks (empty if no pose detected)"
    )
    confidence_score: float = Field(
        ge=0, le=1, description="Overall pose detection confidence"
    )
    pose_detected: bool = Field(description="Whether a pose was detected in this frame")
    
    @field_validator("landmarks")
    @classmethod
    def validate_landmarks(cls, v: List[PoseLandmark]) -> List[PoseLandmark]:
        """Validate landmarks - either empty or exactly 33 landmarks."""
        if len(v) not in [0, 33]:
            raise ValueError("Must have either 0 landmarks (no pose) or exactly 33 landmarks")
        return v
    
    @property
    def has_valid_pose(self) -> bool:
        """Check if frame has a valid pose with reasonable confidence."""
        return self.pose_detected and self.confidence_score >= 0.3 and len(self.landmarks) == 33


class VideoMetadata(BaseModel):
    """Video file metadata."""
    
    video_id: str = Field(description="Unique video identifier")
    filename: str = Field(description="Original filename")
    file_size_mb: float = Field(gt=0, description="File size in megabytes")
    duration_seconds: float = Field(gt=0, description="Video duration in seconds")
    fps: float = Field(gt=0, description="Frames per second")
    total_frames: int = Field(gt=0, description="Total number of frames")
    resolution_width: int = Field(gt=0, description="Video width in pixels")
    resolution_height: int = Field(gt=0, description="Video height in pixels")
    format: str = Field(description="Video format (e.g., 'mp4', 'avi')")
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator("video_id")
    @classmethod
    def validate_video_id(cls, v: str) -> str:
        """Ensure video_id is a valid UUID string."""
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError("video_id must be a valid UUID")
        return v


class VideoAnalysis(BaseModel):
    """Complete video pose analysis results."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "video_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "completed",
                "metadata": {},  # VideoMetadata object
                "frame_analyses": [],  # List of FrameAnalysis objects
                "processing_time_seconds": 45.2,
                "gpu_memory_used_mb": 2048.5,
                "poses_detected_count": 850,
                "pose_detection_rate": 0.85,
                "created_at": "2024-01-01T12:00:00Z"
            }
        }
    )
    
    video_id: str = Field(description="Unique video identifier")
    status: ProcessingStatus = Field(description="Processing status")
    metadata: VideoMetadata = Field(description="Video file metadata")
    frame_analyses: List[FrameAnalysis] = Field(
        default_factory=list,
        description="Analysis results for each frame"
    )
    processing_time_seconds: float = Field(
        ge=0, description="Time taken for pose analysis"
    )
    gpu_memory_used_mb: float = Field(
        ge=0, description="Peak GPU memory usage during processing"
    )
    poses_detected_count: int = Field(
        ge=0, description="Number of frames with detected poses"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if processing failed"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def pose_detection_rate(self) -> float:
        """Calculate the rate of successful pose detections."""
        if not self.frame_analyses:
            return 0.0
        return self.poses_detected_count / len(self.frame_analyses)
    
    @property
    def average_confidence(self) -> float:
        """Calculate average confidence across all frames with detected poses."""
        if not self.frame_analyses:
            return 0.0
        
        confident_frames = [
            frame for frame in self.frame_analyses 
            if frame.pose_detected and frame.confidence_score > 0
        ]
        
        if not confident_frames:
            return 0.0
        
        return sum(frame.confidence_score for frame in confident_frames) / len(confident_frames)
    
    @property
    def is_high_quality_analysis(self) -> bool:
        """Determine if the analysis is high quality for coaching."""
        return (
            self.pose_detection_rate >= 0.7 and 
            self.average_confidence >= 0.3 and
            self.poses_detected_count >= 10
        )


class CoachingMetrics(BaseModel):
    """Technical metrics extracted from pose analysis."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "movement_smoothness": 0.85,
                "posture_stability": 0.92,
                "joint_angles_consistency": 0.78,
                "movement_range": 1.2,
                "balance_score": 0.88
            }
        }
    )
    
    movement_smoothness: Optional[float] = Field(
        None, ge=0, le=1, description="Smoothness of movement (0-1)"
    )
    posture_stability: Optional[float] = Field(
        None, ge=0, le=1, description="Overall posture stability (0-1)"
    )
    joint_angles_consistency: Optional[float] = Field(
        None, ge=0, le=1, description="Consistency of joint angles (0-1)"
    )
    movement_range: Optional[float] = Field(
        None, ge=0, description="Range of movement in meters"
    )
    balance_score: Optional[float] = Field(
        None, ge=0, le=1, description="Balance and coordination score (0-1)"
    )
    symmetry_score: Optional[float] = Field(
        None, ge=0, le=1, description="Left-right body symmetry (0-1)"
    )
    tempo_consistency: Optional[float] = Field(
        None, ge=0, le=1, description="Consistency of movement tempo (0-1)"
    )


class CoachingFeedback(BaseModel):
    """AI coach feedback response."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "video_id": "123e4567-e89b-12d3-a456-426614174000",
                "analysis_summary": "Good overall form with room for improvement in balance.",
                "key_issues": ["Slight forward lean during movement", "Inconsistent arm positioning"],
                "improvement_suggestions": ["Focus on core engagement", "Practice mirror work for symmetry"],
                "confidence_score": 0.87,
                "coaching_metrics": {},
                "priority_areas": ["balance", "posture"],
                "created_at": "2024-01-01T12:00:00Z"
            }
        }
    )
    
    video_id: str = Field(description="Associated video identifier")
    analysis_summary: str = Field(
        min_length=10, max_length=500, 
        description="Brief summary of the pose analysis"
    )
    key_issues: List[str] = Field(
        max_items=10, description="Main issues identified in the pose/movement"
    )
    improvement_suggestions: List[str] = Field(
        max_items=10, description="Specific suggestions for improvement"
    )
    confidence_score: float = Field(
        ge=0, le=1, description="AI confidence in the feedback quality"
    )
    coaching_metrics: CoachingMetrics = Field(
        default_factory=CoachingMetrics,
        description="Technical metrics supporting the feedback"
    )
    priority_areas: List[str] = Field(
        max_items=5, 
        description="Priority areas for athlete to focus on"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator("analysis_summary")
    @classmethod
    def validate_analysis_summary(cls, v: str) -> str:
        """Ensure analysis summary is meaningful."""
        if len(v.strip()) < 10:
            raise ValueError("Analysis summary must be at least 10 characters")
        return v.strip()
    
    @field_validator("key_issues", "improvement_suggestions")
    @classmethod
    def validate_feedback_lists(cls, v: List[str]) -> List[str]:
        """Ensure feedback items are meaningful."""
        return [item.strip() for item in v if item.strip()]


class ChatMessage(BaseModel):
    """Chat message between user and AI coach."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message_id": "msg_123",
                "session_id": "session_456", 
                "role": "user",
                "content": "Can you analyze my squat form?",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }
    )
    
    message_id: str = Field(description="Unique message identifier")
    session_id: str = Field(description="Chat session identifier")
    role: str = Field(description="Message role: 'user' or 'assistant'")
    content: str = Field(
        min_length=1, max_length=2000,
        description="Message content"
    )
    video_id: Optional[str] = Field(
        None, description="Associated video ID if message references a video"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Ensure role is valid."""
        if v not in ["user", "assistant"]:
            raise ValueError("Role must be 'user' or 'assistant'")
        return v


class ChatSession(BaseModel):
    """Chat session with AI coach."""
    
    session_id: str = Field(description="Unique session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    messages: List[ChatMessage] = Field(
        default_factory=list, description="Messages in this session"
    )
    video_ids: List[str] = Field(
        default_factory=list, description="Videos discussed in this session"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    
    def add_message(self, role: str, content: str, video_id: Optional[str] = None) -> ChatMessage:
        """Add a new message to the session."""
        message = ChatMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            session_id=self.session_id,
            role=role,
            content=content,
            video_id=video_id
        )
        self.messages.append(message)
        self.last_activity = datetime.utcnow()
        
        if video_id and video_id not in self.video_ids:
            self.video_ids.append(video_id)
        
        return message