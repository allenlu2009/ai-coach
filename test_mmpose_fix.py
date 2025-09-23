#!/usr/bin/env python3
"""
Test the MMPose visualizer fix directly with raw inference_topdown results.
"""

import cv2
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, 'src')

def main():
    """Test the MMPose visualizer with proper data format."""
    
    # Test video path
    video_path = "examples/deadlift.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return False
    
    # Read first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Failed to read frame from video")
        return False
    
    print(f"✅ Frame loaded: {frame.shape}")
    
    try:
        # Import RTMPose components
        from ai_coach.rtm_pose_analyzer import RTMPoseAnalyzer
        from ai_coach.mmpose_visualizer import MMPoseVisualizer
        from mmpose.apis import inference_topdown
        import logging
        logging.basicConfig(level=logging.DEBUG)
        
        print("🔥 Initializing RTMPose analyzer...")
        analyzer = RTMPoseAnalyzer(use_3d=True, enable_detection=True)
        
        # Initialize models
        print("🔥 Initializing models...")
        success = analyzer._initialize_model()
        if not success:
            print("❌ Model initialization failed")
            return False
        
        print("✅ Models initialized successfully")
        
        # Get bounding boxes for inference
        if analyzer.detector is not None:
            bboxes = analyzer._detect_persons(frame)
            if len(bboxes) == 0:
                height, width = frame.shape[:2]
                bboxes = np.array([[0, 0, width, height]], dtype=np.float32)
        else:
            height, width = frame.shape[:2]
            bboxes = np.array([[0, 0, width, height]], dtype=np.float32)
        
        # Test direct inference_topdown call (what MMPose visualizer expects)
        print("🎯 Testing direct inference_topdown...")
        raw_pose_results = inference_topdown(analyzer.pose3d_model, frame, bboxes)
        
        print(f"\n📊 Raw inference_topdown results structure:")
        print(f"   - Type: {type(raw_pose_results)}")
        print(f"   - Length: {len(raw_pose_results) if raw_pose_results else 0}")
        
        if raw_pose_results and len(raw_pose_results) > 0:
            first_result = raw_pose_results[0]
            print(f"   - First result type: {type(first_result)}")
            print(f"   - Has pred_instances: {hasattr(first_result, 'pred_instances')}")
            
            if hasattr(first_result, 'pred_instances'):
                pred_instances = first_result.pred_instances
                print(f"   - pred_instances type: {type(pred_instances)}")
                if hasattr(pred_instances, 'keypoints'):
                    print(f"   - keypoints shape: {pred_instances.keypoints.shape}")
                if hasattr(pred_instances, 'keypoint_scores'):
                    print(f"   - keypoint_scores shape: {pred_instances.keypoint_scores.shape}")
        
        # Test MMPose visualizer with raw results
        print("\n🎨 Testing MMPose visualizer...")
        mmpose_viz = MMPoseVisualizer()
        
        # Initialize the visualizer
        print("🔧 Initializing MMPose visualizer...")
        viz_success = mmpose_viz.initialize_models(analyzer.pose3d_model, analyzer.detector)
        if not viz_success:
            print("❌ MMPose visualizer initialization failed")
            return False
        
        print("✅ MMPose visualizer initialized")
        
        # Test visualization
        print("🖼️ Testing visualization...")
        vis_result = mmpose_viz.process_and_visualize(
            frame=frame,
            pose_results=raw_pose_results,
            save_path="test_mmpose_fix_output.jpg"
        )
        
        if vis_result is not None:
            print("✅ MMPose visualization successful!")
            print(f"   - Result shape: {vis_result.shape}")
            print("   - Saved to: test_mmpose_fix_output.jpg")
            return True
        else:
            print("❌ MMPose visualization failed")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")