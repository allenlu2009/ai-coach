#!/usr/bin/env python3
"""
Test script to verify that the bounding box visibility feature works correctly.
"""
import cv2
import sys
import logging
sys.path.insert(0, 'src')

from ai_coach.rtm_pose_analyzer import RTMPoseAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_bbox_visibility():
    """Test the bounding box visibility debugging feature."""
    
    print("🎯 Testing Bounding Box Visibility Feature")
    print("=" * 60)
    
    # Test video file
    video_file = 'examples/deadlift.mp4'
    
    print(f"📹 Using test video: {video_file}")
    
    # Initialize analyzer with bounding box debugging enabled
    print("\n📍 Step 1: Initialize RTMPoseAnalyzer with bbox debugging")
    analyzer = RTMPoseAnalyzer(
        use_3d=True, 
        enable_detection=True, 
        show_detection_bbox=True  # 🔥 Enable bounding box debugging
    )
    
    print(f"✅ Analyzer initialized:")
    print(f"   - use_3d: {analyzer.use_3d}")
    print(f"   - enable_detection: {analyzer.enable_detection}")
    print(f"   - show_detection_bbox: {analyzer.show_detection_bbox}")
    
    # Initialize the model
    print("\n📍 Step 2: Initialize Models")
    if not analyzer._initialize_model():
        print("❌ Model initialization failed")
        return
    print("✅ Models initialized successfully")
    
    # Load and process a single frame
    print("\n📍 Step 3: Process Single Frame")
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Failed to read video frame")
        return
    
    print(f"✅ Frame loaded: {frame.shape}")
    
    # Analyze the frame
    print("\n📍 Step 4: Analyze Frame")
    frame_analysis = await analyzer._analyze_single_frame(frame, 0, 0.0)
    
    if not frame_analysis.pose_detected:
        print("❌ No pose detected in frame")
        return
    
    print(f"✅ Pose detected with confidence: {frame_analysis.confidence_score:.3f}")
    print(f"   - Landmarks: {len(frame_analysis.landmarks)}")
    print(f"   - Detected bbox: {frame_analysis.detected_bbox}")
    
    # Draw the overlay with bounding box
    print("\n📍 Step 5: Generate Overlay with Bounding Box")
    overlay_frame = analyzer._draw_2d_pose_overlay(frame, frame_analysis)
    
    # Save the result
    output_path = 'test_bbox_overlay.jpg'
    cv2.imwrite(output_path, overlay_frame)
    print(f"✅ Overlay saved to: {output_path}")
    
    # Check if bounding box was drawn
    if frame_analysis.detected_bbox:
        print("🎯 SUCCESS: Bounding box information is available!")
        print(f"   - Bbox coordinates: {frame_analysis.detected_bbox}")
        print(f"   - Bounding box should be visible in {output_path}")
    else:
        print("⚠️  WARNING: No bounding box information found")
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_bbox_visibility())