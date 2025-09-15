#!/usr/bin/env python3
"""
3D Pose Visualization using matplotlib for RTMPose3D world coordinates.
Adapted from the official RTMPose3D demo visualization approach.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
from typing import List, Tuple, Optional
import logging

from .models import PoseLandmark

logger = logging.getLogger(__name__)


class Pose3DVisualizer:
    """3D pose visualization using matplotlib, similar to RTMPose3D demo."""
    
    def __init__(self, 
                 fig_size: Tuple[int, int] = (8, 6),
                 axis_limit: float = 400.0,
                 axis_azimuth: float = 70,
                 axis_elev: float = 15):
        """
        Initialize 3D pose visualizer.
        
        Args:
            fig_size: Figure size in inches (width, height)
            axis_limit: The axis limit for 3D visualization
            axis_azimuth: Azimuth angle for 3D view
            axis_elev: Elevation angle for 3D view
        """
        self.fig_size = fig_size
        self.axis_limit = axis_limit
        self.axis_azimuth = axis_azimuth
        self.axis_elev = axis_elev
        
        # RTMPose skeleton connections (17 keypoints COCO format)
        # Adapted from RTMPose3D demo and MMPose skeleton definitions
        self.skeleton_connections = [
            # Head connections
            (0, 1), (0, 2),  # nose to eyes
            (1, 3), (2, 4),  # eyes to ears
            
            # Upper body
            (5, 6),   # left_shoulder to right_shoulder
            (5, 7), (7, 9),   # left arm: shoulder -> elbow -> wrist
            (6, 8), (8, 10),  # right arm: shoulder -> elbow -> wrist
            
            # Torso
            (5, 11), (6, 12),  # shoulders to hips
            (11, 12),  # left_hip to right_hip
            
            # Lower body
            (11, 13), (13, 15),  # left leg: hip -> knee -> ankle
            (12, 14), (14, 16),  # right leg: hip -> knee -> ankle
        ]
        
        # Keypoint colors (RGB values 0-255) - adapted from RTMPose
        # Extended to handle more keypoints (33 colors for MediaPipe compatibility)
        self.keypoint_colors = [
            [255, 128, 0], [255, 153, 51], [255, 178, 102],  # nose, left_eye, right_eye
            [255, 51, 255], [102, 178, 255],                 # left_ear, right_ear
            [255, 51, 51], [51, 255, 51],                    # left_shoulder, right_shoulder
            [255, 128, 0], [255, 178, 102],                  # left_elbow, right_elbow
            [255, 153, 255], [102, 255, 178],                # left_wrist, right_wrist
            [255, 51, 51], [51, 255, 51],                    # left_hip, right_hip
            [255, 128, 0], [255, 178, 102],                  # left_knee, right_knee
            [255, 153, 255], [102, 255, 178],                # left_ankle, right_ankle
            # Extended colors for additional keypoints
            [128, 255, 0], [0, 255, 128], [128, 0, 255],     # hand/finger points
            [255, 0, 128], [0, 128, 255], [255, 255, 0],
            [255, 0, 255], [0, 255, 255], [128, 128, 255],
            [255, 128, 128], [128, 255, 128], [255, 255, 128],
            [128, 128, 128], [200, 100, 100], [100, 200, 100],
            [100, 100, 200], [150, 150, 150], [200, 200, 0],
            [200, 0, 200], [0, 200, 200]                     # Extra colors for any additional points
        ]
        
        # Skeleton link colors (RGB values 0-255)
        self.link_colors = [
            [255, 128, 0], [255, 153, 51], [255, 178, 102],  # head connections
            [255, 51, 255], [102, 178, 255],
            [255, 51, 51], [51, 255, 51], [255, 128, 0],     # upper body
            [255, 178, 102], [255, 153, 255], [102, 255, 178],
            [255, 51, 51], [51, 255, 51],                    # torso  
            [255, 128, 0], [255, 178, 102], [255, 153, 255], [102, 255, 178]  # lower body
        ]
    
    def create_3d_pose_frame(self, 
                           landmarks: List[PoseLandmark], 
                           frame_idx: int = 0,
                           kpt_thr: float = 0.3,
                           show_kpt_idx: bool = False) -> np.ndarray:
        """
        Create a 3D pose visualization frame from landmarks.
        
        Args:
            landmarks: List of PoseLandmark objects with 3D coordinates
            frame_idx: Frame index for title
            kpt_thr: Keypoint confidence threshold
            show_kpt_idx: Whether to show keypoint indices
            
        Returns:
            numpy array representing the rendered 3D pose image
        """
        try:
            # Create matplotlib figure with 3D projection
            fig = plt.figure(figsize=self.fig_size)
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract 3D coordinates and confidence scores
            keypoints = []
            scores = []
            
            for landmark in landmarks:
                # Use world coordinates (z is depth in meters)
                keypoints.append([landmark.x, landmark.y, landmark.z])
                scores.append(landmark.visibility)  # Use visibility as confidence
            
            keypoints = np.array(keypoints)
            scores = np.array(scores)
            
            # Filter valid keypoints based on confidence threshold
            valid_mask = scores > kpt_thr
            
            if not np.any(valid_mask):
                # No valid keypoints, return empty frame
                plt.close(fig)
                return self._create_empty_frame()
            
            # Draw keypoints
            self._draw_3d_keypoints(ax, keypoints, scores, valid_mask, show_kpt_idx)
            
            # Draw skeleton connections
            self._draw_3d_skeleton(ax, keypoints, scores, valid_mask, kpt_thr)
            
            # Configure 3D axes
            self._configure_3d_axes(ax, keypoints[valid_mask])
            
            # Set title
            ax.set_title(f'3D Pose Estimation - Frame {frame_idx}', fontsize=12)
            
            # Convert matplotlib figure to numpy array
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            return buf
            
        except Exception as e:
            logger.error(f"3D pose visualization failed: {e}")
            return self._create_empty_frame()
    
    def _draw_3d_keypoints(self, ax, keypoints: np.ndarray, scores: np.ndarray, 
                          valid_mask: np.ndarray, show_kpt_idx: bool):
        """Draw 3D keypoints as scatter points."""
        valid_keypoints = keypoints[valid_mask]
        valid_scores = scores[valid_mask]
        
        # Get colors for valid keypoints
        colors = []
        valid_indices = np.where(valid_mask)[0]
        
        for idx in valid_indices:
            if idx < len(self.keypoint_colors):
                color = np.array(self.keypoint_colors[idx]) / 255.0  # Convert to [0,1]
                colors.append(color)
            else:
                # Default color for keypoints beyond our defined colors
                colors.append([0.5, 0.5, 0.5])  # Gray
        
        if not colors:
            colors = ['red'] * len(valid_keypoints)
        
        # Ensure colors array matches valid_keypoints length
        while len(colors) < len(valid_keypoints):
            colors.append([0.5, 0.5, 0.5])  # Gray for extra keypoints
        
        # Draw scatter plot with larger points for better visibility
        ax.scatter(valid_keypoints[:, 0], 
                  valid_keypoints[:, 1], 
                  valid_keypoints[:, 2],
                  c=colors, s=80, marker='o', alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Add keypoint indices if requested
        if show_kpt_idx:
            valid_indices = np.where(valid_mask)[0]
            for i, (kpt, idx) in enumerate(zip(valid_keypoints, valid_indices)):
                ax.text(kpt[0], kpt[1], kpt[2], str(idx), fontsize=8)
    
    def _draw_3d_skeleton(self, ax, keypoints: np.ndarray, scores: np.ndarray, 
                         valid_mask: np.ndarray, kpt_thr: float):
        """Draw 3D skeleton connections as lines."""
        for connection_idx, (start_idx, end_idx) in enumerate(self.skeleton_connections):
            # Check if both keypoints are valid and above threshold
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                valid_mask[start_idx] and valid_mask[end_idx] and
                scores[start_idx] > kpt_thr and scores[end_idx] > kpt_thr):
                
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                
                # Get link color
                if connection_idx < len(self.link_colors):
                    color = np.array(self.link_colors[connection_idx]) / 255.0
                else:
                    color = [0.5, 0.5, 0.5]  # Default gray
                
                # Draw line between keypoints with thicker lines for better visibility
                ax.plot([start_point[0], end_point[0]],
                       [start_point[1], end_point[1]],
                       [start_point[2], end_point[2]],
                       color=color, linewidth=3, alpha=0.8)
    
    def _configure_3d_axes(self, ax, valid_keypoints: np.ndarray):
        """Configure 3D axes limits and view angles with adaptive scaling."""
        if len(valid_keypoints) == 0:
            # Default axis configuration
            ax.set_xlim([-self.axis_limit, self.axis_limit])
            ax.set_ylim([-self.axis_limit, self.axis_limit])
            ax.set_zlim([0, self.axis_limit])
        else:
            # Calculate pose dimensions for adaptive scaling
            x_range = np.max(valid_keypoints[:, 0]) - np.min(valid_keypoints[:, 0])
            y_range = np.max(valid_keypoints[:, 1]) - np.min(valid_keypoints[:, 1])
            z_range = np.max(valid_keypoints[:, 2]) - np.min(valid_keypoints[:, 2])
            
            # Use adaptive axis limit based on pose dimensions
            max_dimension = max(x_range, y_range, z_range)
            
            # Set a reasonable axis limit: at least 2 meters, or 1.5x the max dimension
            adaptive_limit = max(max_dimension * 1.5, 2.0)
            
            # For very small poses (< 5m), use adaptive scaling
            # For large poses (> 100m), use original fixed scaling
            if max_dimension < 5.0:
                current_limit = adaptive_limit
            elif max_dimension > 100.0:
                current_limit = self.axis_limit  # Original 400m limit
            else:
                # Interpolate between adaptive and fixed for medium poses
                blend_factor = (max_dimension - 5.0) / 95.0  # 0 to 1
                current_limit = adaptive_limit * (1 - blend_factor) + self.axis_limit * blend_factor
            
            # Calculate center of valid keypoints
            center_x = np.mean(valid_keypoints[:, 0])
            center_y = np.mean(valid_keypoints[:, 1])
            center_z = np.mean(valid_keypoints[:, 2])
            
            # Set axis limits around the center
            ax.set_xlim([center_x - current_limit/2, center_x + current_limit/2])
            ax.set_ylim([center_y - current_limit/2, center_y + current_limit/2])
            
            # For Z axis, keep ground level visible but center around pose
            z_min = min(0, center_z - current_limit/2)  # Include ground level
            z_max = max(center_z + current_limit/2, z_min + current_limit)
            ax.set_zlim([z_min, z_max])
        
        # Set axis labels
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        
        # Set view angle
        ax.view_init(elev=self.axis_elev, azim=self.axis_azimuth)
        
        # Make axes equal
        ax.set_box_aspect([1,1,0.8])  # Slightly compress Z axis for better viewing
    
    def _create_empty_frame(self) -> np.ndarray:
        """Create an empty frame when no valid pose is detected."""
        height, width = int(self.fig_size[1] * 100), int(self.fig_size[0] * 100)
        empty_frame = np.full((height, width, 3), 255, dtype=np.uint8)  # White background
        
        # Add "No pose detected" text
        cv2.putText(empty_frame, "No 3D pose detected", 
                   (width//4, height//2), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (128, 128, 128), 2)
        
        return empty_frame
    
    def create_side_by_side_visualization(self, 
                                        original_frame: np.ndarray,
                                        landmarks: List[PoseLandmark],
                                        frame_idx: int = 0) -> np.ndarray:
        """
        Create side-by-side visualization: 2D overlay (left) + 3D animation (right).
        Similar to RTMPose3D demo approach.
        
        Args:
            original_frame: Original video frame with 2D pose overlay
            landmarks: List of PoseLandmark objects with 3D coordinates
            frame_idx: Frame index
            
        Returns:
            Combined side-by-side image
        """
        # Create 3D visualization
        pose_3d_frame = self.create_3d_pose_frame(landmarks, frame_idx)
        
        # Resize frames to match heights
        h1, w1 = original_frame.shape[:2]
        h2, w2 = pose_3d_frame.shape[:2]
        
        target_height = min(h1, h2)
        
        # Resize original frame
        original_resized = cv2.resize(original_frame, 
                                    (int(w1 * target_height / h1), target_height))
        
        # Resize 3D frame
        pose_3d_resized = cv2.resize(pose_3d_frame, 
                                   (int(w2 * target_height / h2), target_height))
        
        # Concatenate side by side
        combined = np.concatenate([original_resized, pose_3d_resized], axis=1)
        
        return combined