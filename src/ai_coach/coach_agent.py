"""
AI coaching agent for pose analysis feedback.

This module provides intelligent coaching feedback based on pose analysis results,
offering context-aware suggestions and technical analysis for athletic performance.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from .models import (
    VideoAnalysis,
    FrameAnalysis,
    CoachingFeedback,
    CoachingMetrics,
    PoseLandmark,
    ChatMessage,
    ChatSession,
)

logger = logging.getLogger(__name__)


@dataclass
class CoachingKnowledge:
    """Coaching knowledge base for different movement patterns."""
    
    # MediaPipe landmark indices for key body parts
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    
    # Ideal ranges for various movements (in radians)
    SQUAT_KNEE_ANGLE_RANGE = (1.57, 2.79)  # 90-160 degrees
    DEADLIFT_BACK_ANGLE_RANGE = (1.22, 1.75)  # 70-100 degrees
    OVERHEAD_ARM_ANGLE_MIN = 2.79  # 160 degrees
    
    # Balance and stability thresholds
    STABILITY_THRESHOLD = 0.05  # meters
    SYMMETRY_THRESHOLD = 0.1  # 10% difference


class CoachAgent:
    """
    AI coaching agent that provides intelligent feedback on athletic movements.
    
    This agent analyzes pose data and provides contextual coaching feedback
    based on movement patterns, biomechanics, and established best practices.
    """
    
    def __init__(self):
        """Initialize the coaching agent."""
        self.knowledge = CoachingKnowledge()
        self.coaching_sessions: Dict[str, ChatSession] = {}
        
        # Movement pattern templates
        self.movement_patterns = {
            "squat": self._analyze_squat_pattern,
            "deadlift": self._analyze_deadlift_pattern,
            "overhead_press": self._analyze_overhead_pattern,
            "general": self._analyze_general_movement
        }
        
        logger.info("CoachAgent initialized")
    
    async def generate_feedback(self, analysis: VideoAnalysis, movement_type: str = "general") -> CoachingFeedback:
        """
        Generate comprehensive coaching feedback from video analysis.
        
        Args:
            analysis: VideoAnalysis results from pose detection
            movement_type: Type of movement being analyzed
            
        Returns:
            CoachingFeedback with analysis and suggestions
        """
        try:
            # Check if analysis has sufficient quality for coaching
            if not analysis.is_high_quality_analysis:
                return self._generate_low_quality_feedback(analysis)
            
            # Calculate coaching metrics
            metrics = await self._calculate_coaching_metrics(analysis)
            
            # Analyze movement pattern
            pattern_analyzer = self.movement_patterns.get(movement_type, self.movement_patterns["general"])
            movement_analysis = pattern_analyzer(analysis, metrics)
            
            # Generate contextual feedback
            feedback = CoachingFeedback(
                video_id=analysis.video_id,
                analysis_summary=movement_analysis["summary"],
                key_issues=movement_analysis["issues"],
                improvement_suggestions=movement_analysis["suggestions"],
                confidence_score=self._calculate_feedback_confidence(analysis, metrics),
                coaching_metrics=metrics,
                priority_areas=movement_analysis["priorities"]
            )
            
            logger.info(f"Generated coaching feedback for {analysis.video_id}: {feedback.confidence_score:.2f} confidence")
            return feedback
            
        except Exception as e:
            logger.error(f"Error generating coaching feedback: {e}")
            return self._generate_error_feedback(analysis, str(e))
    
    async def _calculate_coaching_metrics(self, analysis: VideoAnalysis) -> CoachingMetrics:
        """Calculate technical metrics for coaching analysis."""
        try:
            if not analysis.frame_analyses:
                return CoachingMetrics()
            
            # Get frames with valid poses
            valid_frames = [f for f in analysis.frame_analyses if f.has_valid_pose]
            
            if len(valid_frames) < 5:  # Need minimum frames for analysis
                return CoachingMetrics()
            
            # Calculate movement smoothness
            smoothness = self._calculate_movement_smoothness(valid_frames)
            
            # Calculate posture stability  
            stability = self._calculate_posture_stability(valid_frames)
            
            # Calculate joint angle consistency
            consistency = self._calculate_joint_consistency(valid_frames)
            
            # Calculate movement range
            movement_range = self._calculate_movement_range(valid_frames)
            
            # Calculate balance score
            balance = self._calculate_balance_score(valid_frames)
            
            # Calculate symmetry
            symmetry = self._calculate_symmetry_score(valid_frames)
            
            # Calculate tempo consistency
            tempo = self._calculate_tempo_consistency(valid_frames)
            
            return CoachingMetrics(
                movement_smoothness=smoothness,
                posture_stability=stability,
                joint_angles_consistency=consistency,
                movement_range=movement_range,
                balance_score=balance,
                symmetry_score=symmetry,
                tempo_consistency=tempo
            )
            
        except Exception as e:
            logger.error(f"Error calculating coaching metrics: {e}")
            return CoachingMetrics()
    
    def _calculate_movement_smoothness(self, frames: List[FrameAnalysis]) -> float:
        """Calculate smoothness of movement based on landmark trajectories."""
        try:
            if len(frames) < 10:
                return 0.5
            
            # Track key landmarks over time
            key_landmarks = [self.knowledge.LEFT_WRIST, self.knowledge.RIGHT_WRIST, 
                           self.knowledge.LEFT_KNEE, self.knowledge.RIGHT_KNEE]
            
            smoothness_scores = []
            
            for landmark_idx in key_landmarks:
                # Extract trajectory for this landmark
                trajectory = []
                for frame in frames:
                    if landmark_idx < len(frame.landmarks):
                        landmark = frame.landmarks[landmark_idx]
                        trajectory.append([landmark.x, landmark.y, landmark.z])
                
                if len(trajectory) > 5:
                    trajectory = np.array(trajectory)
                    
                    # Calculate smoothness as inverse of acceleration variance
                    velocities = np.diff(trajectory, axis=0)
                    accelerations = np.diff(velocities, axis=0)
                    
                    if len(accelerations) > 0:
                        acc_variance = np.var(accelerations)
                        smoothness = 1.0 / (1.0 + acc_variance * 100)  # Scale and invert
                        smoothness_scores.append(smoothness)
            
            return np.mean(smoothness_scores) if smoothness_scores else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating movement smoothness: {e}")
            return 0.5
    
    def _calculate_posture_stability(self, frames: List[FrameAnalysis]) -> float:
        """Calculate posture stability based on core landmark positions."""
        try:
            if len(frames) < 5:
                return 0.5
            
            # Track core stability landmarks
            core_landmarks = [self.knowledge.NOSE, self.knowledge.LEFT_SHOULDER, 
                            self.knowledge.RIGHT_SHOULDER, self.knowledge.LEFT_HIP, self.knowledge.RIGHT_HIP]
            
            stability_scores = []
            
            for landmark_idx in core_landmarks:
                positions = []
                for frame in frames:
                    if landmark_idx < len(frame.landmarks):
                        landmark = frame.landmarks[landmark_idx]
                        positions.append([landmark.x, landmark.y])
                
                if len(positions) > 3:
                    positions = np.array(positions)
                    # Calculate stability as inverse of position variance
                    pos_variance = np.var(positions, axis=0).sum()
                    stability = 1.0 / (1.0 + pos_variance * 1000)  # Scale and invert
                    stability_scores.append(stability)
            
            return np.mean(stability_scores) if stability_scores else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating posture stability: {e}")
            return 0.5
    
    def _calculate_joint_consistency(self, frames: List[FrameAnalysis]) -> float:
        """Calculate consistency of joint angles throughout movement."""
        try:
            if len(frames) < 10:
                return 0.5
            
            # Calculate knee angles over time (example joint)
            left_knee_angles = []
            right_knee_angles = []
            
            for frame in frames:
                if len(frame.landmarks) >= 29:  # Ensure all needed landmarks exist
                    # Left knee angle
                    left_angle = self._calculate_joint_angle(
                        frame.landmarks[self.knowledge.LEFT_HIP],
                        frame.landmarks[self.knowledge.LEFT_KNEE], 
                        frame.landmarks[self.knowledge.LEFT_ANKLE]
                    )
                    if left_angle is not None:
                        left_knee_angles.append(left_angle)
                    
                    # Right knee angle  
                    right_angle = self._calculate_joint_angle(
                        frame.landmarks[self.knowledge.RIGHT_HIP],
                        frame.landmarks[self.knowledge.RIGHT_KNEE],
                        frame.landmarks[self.knowledge.RIGHT_ANKLE]
                    )
                    if right_angle is not None:
                        right_knee_angles.append(right_angle)
            
            # Calculate consistency as inverse of angle variance
            consistency_scores = []

            for angles in [left_knee_angles, right_knee_angles]:
                if len(angles) > 5:
                    # Filter out any NaN values before calculating variance
                    valid_angles = [angle for angle in angles if not np.isnan(angle)]

                    if len(valid_angles) > 5:
                        angle_variance = np.var(valid_angles)

                        # Ensure variance is not NaN or infinite
                        if np.isfinite(angle_variance):
                            consistency = 1.0 / (1.0 + angle_variance)
                            # Clamp consistency to valid range [0, 1]
                            consistency = max(0.0, min(1.0, consistency))
                            consistency_scores.append(consistency)

            result = np.mean(consistency_scores) if consistency_scores else 0.5

            # Final safety check for NaN/infinite values
            if not np.isfinite(result):
                return 0.5

            return max(0.0, min(1.0, result))  # Clamp to [0, 1] range
            
        except Exception as e:
            logger.error(f"Error calculating joint consistency: {e}")
            return 0.5
    
    def _calculate_joint_angle(self, p1: PoseLandmark, p2: PoseLandmark, p3: PoseLandmark) -> Optional[float]:
        """Calculate angle between three points (p1-p2-p3)."""
        try:
            # Convert to numpy arrays
            point1 = np.array([p1.x, p1.y, p1.z])
            point2 = np.array([p2.x, p2.y, p2.z])
            point3 = np.array([p3.x, p3.y, p3.z])
            
            # Calculate vectors
            v1 = point1 - point2
            v2 = point3 - point2
            
            # Calculate angle
            # Check for zero-length vectors to prevent division by zero
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 == 0 or norm_v2 == 0:
                return None  # Can't calculate angle for zero-length vectors

            cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle floating point errors
            
            return np.arccos(cos_angle)
            
        except Exception as e:
            logger.error(f"Error calculating joint angle: {e}")
            return None
    
    def _calculate_movement_range(self, frames: List[FrameAnalysis]) -> float:
        """Calculate total range of movement."""
        try:
            if len(frames) < 5:
                return 0.0
            
            # Track hand positions for range calculation
            hand_positions = []
            
            for frame in frames:
                if len(frame.landmarks) > max(self.knowledge.LEFT_WRIST, self.knowledge.RIGHT_WRIST):
                    left_hand = frame.landmarks[self.knowledge.LEFT_WRIST]
                    right_hand = frame.landmarks[self.knowledge.RIGHT_WRIST]
                    
                    # Average hand position
                    avg_pos = [(left_hand.x + right_hand.x) / 2, 
                              (left_hand.y + right_hand.y) / 2,
                              (left_hand.z + right_hand.z) / 2]
                    hand_positions.append(avg_pos)
            
            if len(hand_positions) > 3:
                positions = np.array(hand_positions)
                # Calculate 3D range as max distance between any two positions
                distances = []
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        dist = np.linalg.norm(positions[i] - positions[j])
                        distances.append(dist)
                
                return max(distances) if distances else 0.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating movement range: {e}")
            return 0.0
    
    def _calculate_balance_score(self, frames: List[FrameAnalysis]) -> float:
        """Calculate balance and coordination score."""
        try:
            if len(frames) < 5:
                return 0.5
            
            # Calculate center of mass stability
            com_positions = []
            
            for frame in frames:
                if len(frame.landmarks) >= 24:  # Need hip landmarks
                    left_hip = frame.landmarks[self.knowledge.LEFT_HIP]
                    right_hip = frame.landmarks[self.knowledge.RIGHT_HIP]
                    
                    # Approximate center of mass as hip center
                    com_x = (left_hip.x + right_hip.x) / 2
                    com_y = (left_hip.y + right_hip.y) / 2
                    com_positions.append([com_x, com_y])
            
            if len(com_positions) > 3:
                positions = np.array(com_positions)
                com_variance = np.var(positions, axis=0).sum()
                
                # Good balance = low variance in center of mass
                balance_score = 1.0 / (1.0 + com_variance * 100)
                return balance_score
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating balance score: {e}")
            return 0.5
    
    def _calculate_symmetry_score(self, frames: List[FrameAnalysis]) -> float:
        """Calculate left-right body symmetry."""
        try:
            if len(frames) < 5:
                return 0.5
            
            symmetry_scores = []
            
            # Compare symmetrical landmark pairs
            pairs = [
                (self.knowledge.LEFT_SHOULDER, self.knowledge.RIGHT_SHOULDER),
                (self.knowledge.LEFT_ELBOW, self.knowledge.RIGHT_ELBOW),
                (self.knowledge.LEFT_WRIST, self.knowledge.RIGHT_WRIST),
                (self.knowledge.LEFT_HIP, self.knowledge.RIGHT_HIP),
                (self.knowledge.LEFT_KNEE, self.knowledge.RIGHT_KNEE),
                (self.knowledge.LEFT_ANKLE, self.knowledge.RIGHT_ANKLE)
            ]
            
            for left_idx, right_idx in pairs:
                left_positions = []
                right_positions = []
                
                for frame in frames:
                    if len(frame.landmarks) > max(left_idx, right_idx):
                        left_lm = frame.landmarks[left_idx]
                        right_lm = frame.landmarks[right_idx]
                        
                        left_positions.append([left_lm.x, left_lm.y, left_lm.z])
                        right_positions.append([right_lm.x, right_lm.y, right_lm.z])
                
                if len(left_positions) > 3:
                    left_pos = np.array(left_positions)
                    right_pos = np.array(right_positions)
                    
                    # Mirror right side for comparison (flip x-coordinate)
                    right_pos_mirrored = right_pos.copy()
                    right_pos_mirrored[:, 0] *= -1
                    
                    # Calculate similarity
                    differences = np.abs(left_pos - right_pos_mirrored)
                    avg_difference = np.mean(differences)
                    
                    # Convert to symmetry score (0-1)
                    symmetry = 1.0 / (1.0 + avg_difference * 10)
                    symmetry_scores.append(symmetry)
            
            return np.mean(symmetry_scores) if symmetry_scores else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating symmetry score: {e}")
            return 0.5
    
    def _calculate_tempo_consistency(self, frames: List[FrameAnalysis]) -> float:
        """Calculate consistency of movement tempo."""
        try:
            if len(frames) < 10:
                return 0.5
            
            # Calculate movement speeds over time
            speeds = []
            
            for i in range(1, len(frames)):
                prev_frame = frames[i-1]
                curr_frame = frames[i]
                
                if (prev_frame.has_valid_pose and curr_frame.has_valid_pose and
                    len(prev_frame.landmarks) > self.knowledge.RIGHT_WRIST and
                    len(curr_frame.landmarks) > self.knowledge.RIGHT_WRIST):
                    
                    # Calculate speed of right wrist movement
                    prev_pos = np.array([
                        prev_frame.landmarks[self.knowledge.RIGHT_WRIST].x,
                        prev_frame.landmarks[self.knowledge.RIGHT_WRIST].y,
                        prev_frame.landmarks[self.knowledge.RIGHT_WRIST].z
                    ])
                    
                    curr_pos = np.array([
                        curr_frame.landmarks[self.knowledge.RIGHT_WRIST].x,
                        curr_frame.landmarks[self.knowledge.RIGHT_WRIST].y,
                        curr_frame.landmarks[self.knowledge.RIGHT_WRIST].z
                    ])
                    
                    # Time difference in seconds
                    time_diff = (curr_frame.timestamp_ms - prev_frame.timestamp_ms) / 1000.0
                    
                    if time_diff > 0:
                        distance = np.linalg.norm(curr_pos - prev_pos)
                        speed = distance / time_diff
                        speeds.append(speed)
            
            if len(speeds) > 5:
                speed_variance = np.var(speeds)
                # Consistent tempo = low variance in speeds
                tempo_score = 1.0 / (1.0 + speed_variance * 100)
                return tempo_score
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating tempo consistency: {e}")
            return 0.5
    
    def _analyze_squat_pattern(self, analysis: VideoAnalysis, metrics: CoachingMetrics) -> Dict[str, Any]:
        """Analyze squat movement pattern."""
        issues = []
        suggestions = []
        priorities = []
        
        # Check knee tracking
        if metrics.symmetry_score and metrics.symmetry_score < 0.7:
            issues.append("Asymmetrical knee tracking during squat")
            suggestions.append("Focus on keeping knees aligned over toes")
            priorities.append("knee_alignment")
        
        # Check depth and stability
        if metrics.posture_stability and metrics.posture_stability < 0.6:
            issues.append("Instability during squat descent/ascent")
            suggestions.append("Work on core strength and controlled tempo")
            priorities.append("stability")
        
        # Check smoothness
        if metrics.movement_smoothness and metrics.movement_smoothness < 0.6:
            issues.append("Jerky or inconsistent movement pattern")
            suggestions.append("Practice slow, controlled squats to improve form")
            priorities.append("control")
        
        summary = f"Squat analysis: {len(issues)} areas identified for improvement. "
        if not issues:
            summary = "Good squat form overall with consistent movement pattern."
        
        return {
            "summary": summary,
            "issues": issues,
            "suggestions": suggestions,
            "priorities": priorities
        }
    
    def _analyze_deadlift_pattern(self, analysis: VideoAnalysis, metrics: CoachingMetrics) -> Dict[str, Any]:
        """Analyze deadlift movement pattern."""
        issues = []
        suggestions = []
        priorities = []
        
        # Check back stability
        if metrics.posture_stability and metrics.posture_stability < 0.7:
            issues.append("Back instability during lift")
            suggestions.append("Engage core and maintain neutral spine")
            priorities.append("back_safety")
        
        # Check symmetry
        if metrics.symmetry_score and metrics.symmetry_score < 0.8:
            issues.append("Uneven loading or asymmetrical form")
            suggestions.append("Check bar position and ensure balanced grip")
            priorities.append("symmetry")
        
        summary = f"Deadlift analysis: {len(issues)} areas need attention. "
        if not issues:
            summary = "Strong deadlift form with good stability and control."
        
        return {
            "summary": summary,
            "issues": issues,
            "suggestions": suggestions,
            "priorities": priorities
        }
    
    def _analyze_overhead_pattern(self, analysis: VideoAnalysis, metrics: CoachingMetrics) -> Dict[str, Any]:
        """Analyze overhead press movement pattern."""
        issues = []
        suggestions = []
        priorities = []
        
        # Check shoulder mobility and stability
        if metrics.movement_range and metrics.movement_range < 0.5:
            issues.append("Limited overhead range of motion")
            suggestions.append("Work on shoulder mobility and thoracic extension")
            priorities.append("mobility")
        
        # Check balance
        if metrics.balance_score and metrics.balance_score < 0.7:
            issues.append("Balance issues during overhead movement")
            suggestions.append("Strengthen core and practice balance exercises")
            priorities.append("balance")
        
        summary = f"Overhead press analysis: {len(issues)} areas for improvement. "
        if not issues:
            summary = "Excellent overhead mobility and control."
        
        return {
            "summary": summary,
            "issues": issues,
            "suggestions": suggestions,
            "priorities": priorities
        }
    
    def _analyze_general_movement(self, analysis: VideoAnalysis, metrics: CoachingMetrics) -> Dict[str, Any]:
        """Analyze general movement pattern."""
        issues = []
        suggestions = []
        priorities = []
        
        # Check overall movement quality
        if metrics.movement_smoothness and metrics.movement_smoothness < 0.6:
            issues.append("Movement appears jerky or uncontrolled")
            suggestions.append("Focus on slow, controlled movements to build motor patterns")
            priorities.append("control")
        
        if metrics.balance_score and metrics.balance_score < 0.6:
            issues.append("Balance and stability need improvement")
            suggestions.append("Practice single-leg exercises and core strengthening")
            priorities.append("balance")
        
        if metrics.symmetry_score and metrics.symmetry_score < 0.7:
            issues.append("Left-right asymmetries detected")
            suggestions.append("Include unilateral exercises to address imbalances")
            priorities.append("symmetry")
        
        summary = f"Movement analysis complete: {len(issues)} areas identified. "
        if not issues:
            summary = "Overall movement quality is good with consistent patterns."
        elif len(issues) == 1:
            summary += f"Primary focus area: {priorities[0]}"
        else:
            summary += f"Key areas: {', '.join(priorities[:2])}"
        
        return {
            "summary": summary,
            "issues": issues,
            "suggestions": suggestions,
            "priorities": priorities
        }
    
    def _calculate_feedback_confidence(self, analysis: VideoAnalysis, metrics: CoachingMetrics) -> float:
        """Calculate confidence score for the feedback."""
        confidence_factors = []
        
        # Analysis quality factors
        confidence_factors.append(analysis.pose_detection_rate)
        confidence_factors.append(analysis.average_confidence)
        
        # Data sufficiency
        frame_count_factor = min(1.0, len(analysis.frame_analyses) / 100.0)  # Prefer 100+ frames
        confidence_factors.append(frame_count_factor)
        
        # Metrics reliability
        metrics_count = sum(1 for attr in ['movement_smoothness', 'posture_stability', 
                                         'balance_score', 'symmetry_score']
                           if getattr(metrics, attr) is not None)
        metrics_factor = metrics_count / 4.0
        confidence_factors.append(metrics_factor)
        
        return np.mean(confidence_factors)
    
    def _generate_low_quality_feedback(self, analysis: VideoAnalysis) -> CoachingFeedback:
        """Generate feedback for low quality analysis."""
        return CoachingFeedback(
            video_id=analysis.video_id,
            analysis_summary="Video quality insufficient for detailed analysis. Consider recording with better lighting and camera stability.",
            key_issues=["Low pose detection rate", "Poor video quality"],
            improvement_suggestions=[
                "Ensure good lighting conditions",
                "Keep camera steady during recording",
                "Stay within camera frame throughout movement",
                "Record from a side angle for best pose detection"
            ],
            confidence_score=0.3,
            coaching_metrics=CoachingMetrics(),
            priority_areas=["video_quality"]
        )
    
    def _generate_error_feedback(self, analysis: VideoAnalysis, error_msg: str) -> CoachingFeedback:
        """Generate feedback when analysis fails."""
        return CoachingFeedback(
            video_id=analysis.video_id,
            analysis_summary=f"Analysis failed due to technical error: {error_msg}",
            key_issues=["Technical error during analysis"],
            improvement_suggestions=["Please try uploading the video again"],
            confidence_score=0.0,
            coaching_metrics=CoachingMetrics(),
            priority_areas=["technical_error"]
        )
    
    async def handle_chat_message(self, session_id: str, message: str, video_id: Optional[str] = None) -> str:
        """
        Handle chat message and generate contextual response.
        
        Args:
            session_id: Chat session identifier
            message: User message
            video_id: Associated video ID if discussing specific video
            
        Returns:
            AI coach response
        """
        try:
            # Get or create chat session
            if session_id not in self.coaching_sessions:
                self.coaching_sessions[session_id] = ChatSession(
                    session_id=session_id,
                    messages=[],
                    video_ids=[]
                )
            
            session = self.coaching_sessions[session_id]
            
            # Add user message
            session.add_message("user", message, video_id)
            
            # Generate response based on context
            response = await self._generate_chat_response(session, message, video_id)
            
            # Add assistant response
            session.add_message("assistant", response, video_id)
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling chat message: {e}")
            return "I apologize, but I encountered an error processing your message. Please try again."
    
    async def _generate_chat_response(self, session: ChatSession, message: str, video_id: Optional[str]) -> str:
        """Generate contextual chat response."""
        message_lower = message.lower()
        
        # Greeting responses
        if any(greeting in message_lower for greeting in ["hello", "hi", "hey"]):
            return "Hello! I'm your AI coach. Upload a video of your movement and I'll help analyze your form and technique."
        
        # Video analysis requests
        if any(word in message_lower for word in ["analyze", "analysis", "form", "technique"]):
            if video_id:
                return f"I'm analyzing your video now. I'll look at your movement patterns, balance, symmetry, and provide specific feedback on areas for improvement."
            else:
                return "I'd be happy to analyze your movement! Please upload a video and I'll provide detailed feedback on your form and technique."
        
        # Coaching questions
        if any(word in message_lower for word in ["improve", "better", "help", "fix"]):
            return ("Based on pose analysis, I typically look at movement smoothness, balance, symmetry, and joint mechanics. "
                   "Upload a video and I'll give you specific suggestions tailored to your movement patterns.")
        
        # Default response
        return ("I'm here to help you improve your athletic performance through movement analysis. "
               "Upload a video of your exercise or movement, and I'll provide detailed coaching feedback!")
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get chat session by ID."""
        return self.coaching_sessions.get(session_id)
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up old chat sessions."""
        from datetime import timedelta
        
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        sessions_to_remove = []
        
        for session_id, session in self.coaching_sessions.items():
            if session.last_activity < cutoff_time:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.coaching_sessions[session_id]
        
        return len(sessions_to_remove)