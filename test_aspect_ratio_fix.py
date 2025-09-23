#!/usr/bin/env python3
"""
Test script to verify aspect ratio fix for portrait videos in demo visualizer.
"""

import cv2
import numpy as np
import os
import sys
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai_coach.rtm_pose_analyzer import RTMPoseAnalyzer

def create_test_portrait_video(output_path: str, width: int = 480, height: int = 854, fps: int = 10, duration: int = 3):
    """Create a test portrait video (9:16 aspect ratio like phone videos)."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = fps * duration
    logger.info(f"Creating test portrait video: {width}x{height} ({total_frames} frames)")

    for frame_num in range(total_frames):
        # Create a simple test frame with moving elements
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Add some visual elements for pose detection
        # Draw a simple stick figure that moves
        center_x = width // 2
        center_y = height // 2 + int(50 * np.sin(frame_num * 0.1))

        # Head
        cv2.circle(frame, (center_x, center_y - 100), 30, (255, 255, 255), -1)

        # Body
        cv2.line(frame, (center_x, center_y - 70), (center_x, center_y + 80), (255, 255, 255), 5)

        # Arms
        arm_angle = frame_num * 0.1
        arm_x = int(40 * np.cos(arm_angle))
        cv2.line(frame, (center_x, center_y - 20), (center_x + arm_x, center_y), (255, 255, 255), 3)
        cv2.line(frame, (center_x, center_y - 20), (center_x - arm_x, center_y), (255, 255, 255), 3)

        # Legs
        cv2.line(frame, (center_x, center_y + 80), (center_x - 20, center_y + 160), (255, 255, 255), 3)
        cv2.line(frame, (center_x, center_y + 80), (center_x + 20, center_y + 160), (255, 255, 255), 3)

        # Add frame number
        cv2.putText(frame, f"Frame {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(frame)

    out.release()
    logger.info(f"Test video created: {output_path}")

def test_aspect_ratio_preservation():
    """Test that the demo visualizer preserves aspect ratio for portrait videos."""
    try:
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            test_video_path = os.path.join(temp_dir, "test_portrait.mp4")
            output_video_path = os.path.join(temp_dir, "test_output.mp4")

            # Create test portrait video
            create_test_portrait_video(test_video_path)

            # Initialize RTMPose analyzer with demo visualizer
            logger.info("Initializing RTMPose analyzer with demo visualizer...")
            analyzer = RTMPoseAnalyzer(
                use_demo_visualizer=True,
                use_3d=True
            )

            # Load models to initialize the MMPose visualizer
            logger.info("Loading RTMPose models...")
            analyzer._initialize_model()

            # Process one frame to check output dimensions
            logger.info("Testing frame processing dimensions...")
            cap = cv2.VideoCapture(test_video_path)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                logger.error("Failed to read test video frame")
                return False

            original_height, original_width = frame.shape[:2]
            logger.info(f"Original frame dimensions: {original_width}x{original_height}")

            # Process frame through demo visualizer
            processed_frame = analyzer._process_frame_like_demo(frame)
            processed_height, processed_width = processed_frame.shape[:2]

            logger.info(f"Processed frame dimensions: {processed_width}x{processed_height}")

            # Check aspect ratio preservation
            original_ratio = original_width / original_height
            processed_ratio = processed_width / processed_height

            logger.info(f"Original aspect ratio: {original_ratio:.3f}")
            logger.info(f"Processed aspect ratio: {processed_ratio:.3f}")

            # For side-by-side layout, we expect the width to be roughly doubled
            # but height should remain similar
            expected_width_factor = processed_width / original_width
            height_preservation = abs(processed_height - original_height) / original_height

            logger.info(f"Width expansion factor: {expected_width_factor:.2f}x")
            logger.info(f"Height preservation error: {height_preservation:.1%}")

            # Success criteria:
            # 1. Width should be significantly larger (side-by-side layout)
            # 2. Height should be preserved (within 10% tolerance)
            # 3. No squashing should occur

            success = (
                expected_width_factor > 1.5 and  # Width significantly expanded
                height_preservation < 0.1 and    # Height preserved within 10%
                processed_height > 0 and processed_width > 0  # Valid dimensions
            )

            if success:
                logger.info("✅ Aspect ratio preservation test PASSED!")
                logger.info("   - Side-by-side layout correctly implemented")
                logger.info("   - Original video height preserved")
                logger.info("   - No unwanted squashing detected")
            else:
                logger.error("❌ Aspect ratio preservation test FAILED!")
                logger.error(f"   - Width factor: {expected_width_factor:.2f} (expected > 1.5)")
                logger.error(f"   - Height preservation: {height_preservation:.1%} (expected < 10%)")

            # Save a test frame for visual inspection
            test_frame_path = os.path.join(temp_dir, "test_aspect_ratio_output.jpg")
            cv2.imwrite(test_frame_path, processed_frame)
            logger.info(f"Test frame saved for inspection: {test_frame_path}")

            return success

    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger.info("🧪 Starting aspect ratio preservation test...")
    success = test_aspect_ratio_preservation()

    if success:
        logger.info("🎉 All tests passed! Aspect ratio fix is working correctly.")
        sys.exit(0)
    else:
        logger.error("💥 Tests failed! Aspect ratio issue persists.")
        sys.exit(1)