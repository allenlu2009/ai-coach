#!/usr/bin/env python3
"""
Test RTMPose3D coordinate processing to understand the coordinate system
"""
import cv2
import sys
import numpy as np
sys.path.insert(0, 'src')

from ai_coach.rtm_pose_analyzer import RTMPoseAnalyzer

def test_coordinate_processing():
    """Test RTMPose3D coordinate processing step by step."""
    
    video_file = "examples/deadlift.mp4"
    
    print("🔍 Testing RTMPose3D coordinate processing...")
    
    # Create RTMPose analyzer with 3D
    analyzer = RTMPoseAnalyzer(use_3d=True)
    
    # Initialize model
    if not analyzer._initialize_model():
        print("❌ Failed to initialize RTMPose model")
        return
        
    print(f"✅ RTMPose3D model: is_rtmpose3d={analyzer.is_rtmpose3d}")
    
    # Load single frame
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not read frame")
        return
        
    print(f"✅ Frame loaded: {frame.shape}")
    height, width = frame.shape[:2]
    print(f"   Frame size: {width}x{height}")
    
    # Test RTMPose3D inference directly
    if hasattr(analyzer, 'pose3d_model'):
        print("\n🧪 Running RTMPose3D inference directly...")
        results = analyzer._run_rtmpose3d_inference(frame)
        
        if results and len(results) > 0:
            result = results[0]
            if 'predictions' in result:
                pred = result['predictions'][0][0]
                print(f"🎯 Raw RTMPose3D output keys: {list(pred.keys())}")
                
                # Check both coordinate sets
                if 'keypoints' in pred:
                    raw_kpts = pred['keypoints']
                    print(f"📊 Raw 3D keypoints (first 3): {raw_kpts[:3]}")
                    print(f"   Keypoint coordinate ranges:")
                    raw_array = np.array(raw_kpts)
                    print(f"   - X: {raw_array[:, 0].min():.3f} to {raw_array[:, 0].max():.3f}")
                    print(f"   - Y: {raw_array[:, 1].min():.3f} to {raw_array[:, 1].max():.3f}")
                    print(f"   - Z: {raw_array[:, 2].min():.3f} to {raw_array[:, 2].max():.3f}")
                
                if 'keypoints_2d_overlay' in pred:
                    overlay_kpts = pred['keypoints_2d_overlay']
                    print(f"📍 2D overlay keypoints (first 3): {overlay_kpts[:3]}")
                    overlay_array = np.array(overlay_kpts)
                    print(f"   Overlay coordinate ranges:")
                    print(f"   - X: {overlay_array[:, 0].min():.3f} to {overlay_array[:, 0].max():.3f}")
                    print(f"   - Y: {overlay_array[:, 1].min():.3f} to {overlay_array[:, 1].max():.3f}")
                    if overlay_array.shape[1] >= 3:
                        print(f"   - Z: {overlay_array[:, 2].min():.3f} to {overlay_array[:, 2].max():.3f}")
        
        # Test coordinate conversion
        print("\n🔄 Testing coordinate conversion...")
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        frame_analysis = loop.run_until_complete(analyzer._analyze_single_frame(frame, 0, 0.0))
        loop.close()
        
        if frame_analysis.landmarks:
            print(f"🎯 Converted landmarks (first 10):")
            for i, lm in enumerate(frame_analysis.landmarks[:10]):
                # Convert to pixel coordinates like the overlay function does
                x = int(lm.x * width) if lm.x <= 1.0 else int(lm.x)
                y = int(lm.y * height) if lm.y <= 1.0 else int(lm.y)
                print(f"   {i}: norm=({lm.x:.3f},{lm.y:.3f}) -> pixel=({x},{y}) z={lm.z:.3f}")
                
                # Check if coordinates are in valid range
                valid = 0 <= x < width and 0 <= y < height
                print(f"      Valid: {valid} (frame bounds: 0-{width-1}, 0-{height-1})")
    else:
        print("❌ No RTMPose3D model available")

if __name__ == "__main__":
    test_coordinate_processing()