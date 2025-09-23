#!/usr/bin/env python3

import sys
sys.path.insert(0, 'src')

from ai_coach.rtm_pose_analyzer import RTMPoseAnalyzer
import cv2
import logging
import asyncio

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_corrected_coordinate_transformation():
    """Test the corrected RTMPose3D coordinate transformation"""
    
    print("🎯 Testing Corrected RTMPose3D Coordinate Transformation")
    print("=" * 60)
    
    # Test with bounding box debugging enabled
    analyzer = RTMPoseAnalyzer(
        use_3d=True, 
        enable_detection=True, 
        show_detection_bbox=True  # Enable bounding box debugging
    )
    
    # Test with deadlift video
    video_file = 'examples/deadlift.mp4'
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Failed to read test video")
        return False
        
    print(f"✅ Loaded test frame: {frame.shape}")
    
    # Analyze frame with corrected transformation
    frame_analysis = await analyzer._analyze_single_frame(frame, 0, 0.0)
    
    print("\n📊 Analysis Results:")
    print(f"   - Pose detected: {frame_analysis.pose_detected}")
    print(f"   - Confidence: {frame_analysis.confidence_score:.3f}")
    print(f"   - Landmarks count: {len(frame_analysis.landmarks)}")
    print(f"   - Detected bbox: {frame_analysis.detected_bbox}")
    
    if frame_analysis.pose_detected and frame_analysis.landmarks:
        print("\n🎯 Testing Coordinate Transformation:")
        
        # Create overlay frame to verify coordinates
        overlay_frame = frame.copy()
        
        # Draw the 2D pose overlay and bounding box
        analyzer._draw_2d_pose_overlay(overlay_frame, frame_analysis)
        
        # Save the test result
        output_path = 'test_corrected_coordinates.jpg'
        cv2.imwrite(output_path, overlay_frame)
        print(f"✅ Test result saved to: {output_path}")
        
        # Show some coordinate examples
        print("\n📍 Sample Coordinate Transformations:")
        for i in range(min(5, len(frame_analysis.landmarks))):
            landmark = frame_analysis.landmarks[i]
            print(f"   Landmark {i}: ({landmark.x:.1f}, {landmark.y:.1f}, {landmark.z:.3f})")
        
        return True
    else:
        print("❌ No pose detected in test frame")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_corrected_coordinate_transformation())
    if success:
        print("\n🎉 Coordinate transformation test completed!")
        print("Check 'test_corrected_coordinates.jpg' to verify the alignment.")
    else:
        print("\n❌ Coordinate transformation test failed!")
        sys.exit(1)