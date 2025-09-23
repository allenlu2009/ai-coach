#!/usr/bin/env python3
"""
Test RTMPose3D coordinate comparison between demo approach and our approach.
This will help determine if there are fundamental differences in coordinate systems.
"""

import cv2
import numpy as np
import sys
import os

# Add project source to path
sys.path.insert(0, 'src')

from ai_coach.rtm_pose_analyzer import RTMPoseAnalyzer


def test_coordinate_comparison():
    """Compare RTMPose3D coordinates from our implementation vs expected demo behavior."""
    
    print("🎯 RTMPose3D Coordinate Comparison Test")
    print("=" * 60)
    
    # Initialize our RTMPose3D analyzer
    print("\n📍 Step 1: Initialize RTMPose3D Analyzer")
    try:
        analyzer = RTMPoseAnalyzer(use_3d=True)
        if not analyzer.is_rtmpose3d:
            print("❌ RTMPose3D not properly initialized!")
            return
        print("✅ RTMPose3D analyzer initialized successfully")
        print(f"   - Model: {analyzer.pose3d_model}")
        print(f"   - Is RTMPose3D: {analyzer.is_rtmpose3d}")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
        
    # Load test frame
    print("\n📍 Step 2: Load Test Frame")
    video_file = 'examples/deadlift.mp4'
    if not os.path.exists(video_file):
        print(f"❌ Video file not found: {video_file}")
        return
        
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Failed to read video frame")
        return
        
    print(f"✅ Loaded frame: {frame.shape}")
    
    # Run RTMPose3D inference
    print("\n📍 Step 3: Run RTMPose3D Inference")
    try:
        results = analyzer._run_rtmpose3d_inference(frame)
        if not results:
            print("❌ No results from RTMPose3D inference")
            return
            
        print(f"✅ RTMPose3D inference successful: {len(results)} results")
        
        # Extract coordinate data
        result = results[0]
        predictions = result['predictions'][0][0]
        original_coords = predictions['keypoints_2d_overlay']  # Original from model
        transformed_coords = predictions['keypoints']  # After -[0,2,1] transformation
        scores = predictions['keypoint_scores']
        
        print(f"   - Keypoints detected: {len(original_coords)}")
        print(f"   - Coordinate format: {len(original_coords[0])}")
        
        # Analyze coordinate ranges
        print("\n📍 Step 4: Analyze Coordinate Ranges")
        
        # Original coordinates (should be image space)
        orig_array = np.array(original_coords)
        print(f"\n🔷 Original Coordinates (from model):")
        print(f"   - X range: [{orig_array[:, 0].min():.3f}, {orig_array[:, 0].max():.3f}]")
        print(f"   - Y range: [{orig_array[:, 1].min():.3f}, {orig_array[:, 1].max():.3f}]")
        if orig_array.shape[1] >= 3:
            print(f"   - Z range: [{orig_array[:, 2].min():.3f}, {orig_array[:, 2].max():.3f}]")
        
        # Check if original coordinates are in pixel space
        height, width = frame.shape[:2]
        pixel_space = (0 <= orig_array[:, 0].min() and orig_array[:, 0].max() <= width and
                      0 <= orig_array[:, 1].min() and orig_array[:, 1].max() <= height)
        print(f"   - Pixel space compatible: {pixel_space}")
        
        # Transformed coordinates (after -[0,2,1])
        trans_array = np.array(transformed_coords)
        print(f"\n🔶 Transformed Coordinates (after -[0,2,1]):")
        print(f"   - X range: [{trans_array[:, 0].min():.3f}, {trans_array[:, 0].max():.3f}]")
        print(f"   - Y range: [{trans_array[:, 1].min():.3f}, {trans_array[:, 1].max():.3f}]")  
        if trans_array.shape[1] >= 3:
            print(f"   - Z range: [{trans_array[:, 2].min():.3f}, {trans_array[:, 2].max():.3f}]")
        
        # Simulate demo's approach: use transformed coordinates as pixel coordinates
        print("\n📍 Step 5: Demo-Style Coordinate Usage")
        print(f"\n🎭 If we use transformed coords as pixels (demo approach):")
        
        # Check if first 2 dimensions of transformed coords are reasonable for pixels
        demo_pixel_space = (0 <= trans_array[:, 0].min() and trans_array[:, 0].max() <= width and
                           0 <= trans_array[:, 1].min() and trans_array[:, 1].max() <= height)
        print(f"   - Demo pixel space compatible: {demo_pixel_space}")
        
        if demo_pixel_space:
            print("✅ Demo approach should work - transformed coords are pixel-compatible!")
        else:
            print("❌ Demo approach needs adjustment - transformed coords not pixel-compatible")
            
        # Sample coordinate comparison
        print(f"\n📊 Sample Coordinate Comparison (first 5 keypoints):")
        print("   Index | Original (x,y,z)      | Transformed (x,y,z)   | Score")
        print("   ------|----------------------|----------------------|------")
        for i in range(min(5, len(original_coords))):
            orig = original_coords[i]
            trans = transformed_coords[i]
            score = scores[i] if i < len(scores) else 0.0
            print(f"   {i:5d} | {orig[0]:7.1f},{orig[1]:7.1f},{orig[2]:7.3f} | {trans[0]:7.1f},{trans[1]:7.1f},{trans[2]:7.3f} | {score:.3f}")
            
    except Exception as e:
        print(f"❌ RTMPose3D inference failed: {e}")
        import traceback
        print(traceback.format_exc())
        return
    
    print(f"\n🎯 Conclusion:")
    print("   If 'Demo pixel space compatible' = True, we can use the demo approach!")
    print("   If False, we need our current coordinate mapping approach.")


if __name__ == "__main__":
    test_coordinate_comparison()