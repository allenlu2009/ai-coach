#!/usr/bin/env python3
"""
Simple RTMPose test to verify the installation works correctly.
"""

import sys
import time
import torch
from pathlib import Path

def test_rtmpose_import():
    """Test if we can import RTMPose components."""
    print("🔍 Testing RTMPose imports...")
    
    try:
        import mmcv
        print(f"✅ MMCV: {mmcv.__version__}")
    except ImportError as e:
        print(f"❌ MMCV import failed: {e}")
        return False
        
    try:
        import mmengine
        print(f"✅ MMEngine: {mmengine.__version__}")
    except ImportError as e:
        print(f"❌ MMEngine import failed: {e}")
        return False
        
    try:
        import mmpose
        print(f"✅ MMPose: {mmpose.__version__}")
    except ImportError as e:
        print(f"❌ MMPose import failed: {e}")
        return False
        
    return True

def test_gpu_availability():
    """Test GPU and CUDA availability."""
    print("\n🔍 Testing GPU availability...")
    
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("⚠️ No GPU available - will use CPU")
        return False

def test_rtmpose_model():
    """Test RTMPose model initialization."""
    print("\n🔍 Testing RTMPose model initialization...")
    
    try:
        from mmpose.apis import MMPoseInferencer
        
        # Try to initialize RTMPose model
        print("Initializing RTMPose inferencer...")
        inferencer = MMPoseInferencer('human')
        print("✅ RTMPose model loaded successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ RTMPose model initialization failed: {e}")
        return False

def test_mediapipe_comparison():
    """Test MediaPipe for comparison."""
    print("\n🔍 Testing MediaPipe for comparison...")
    
    try:
        import mediapipe as mp
        import cv2
        
        print(f"✅ MediaPipe: {mp.__version__}")
        
        # Initialize MediaPipe pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        print("✅ MediaPipe pose model loaded successfully!")
        pose.close()
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe initialization failed: {e}")
        return False

def run_simple_performance_test():
    """Run a simple performance test without file I/O issues."""
    print("\n🚀 Running simple performance test...")
    
    try:
        import cv2
        import numpy as np
        from mmpose.apis import MMPoseInferencer
        import mediapipe as mp
        
        # Create a dummy image for testing
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "RTMPose Test", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Test RTMPose speed
        print("Testing RTMPose inference speed...")
        rtm_inferencer = MMPoseInferencer('human')
        
        start_time = time.time()
        for i in range(10):  # 10 iterations
            result = rtm_inferencer(test_image, show=False, return_vis=False)
        rtm_time = (time.time() - start_time) / 10
        rtm_fps = 1.0 / rtm_time if rtm_time > 0 else 0
        
        print(f"✅ RTMPose: {rtm_fps:.1f} FPS (avg over 10 frames)")
        
        # Test MediaPipe speed
        print("Testing MediaPipe inference speed...")
        mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        start_time = time.time()
        for i in range(10):  # 10 iterations
            results = mp_pose.process(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        mp_time = (time.time() - start_time) / 10
        mp_fps = 1.0 / mp_time if mp_time > 0 else 0
        
        print(f"✅ MediaPipe: {mp_fps:.1f} FPS (avg over 10 frames)")
        
        # Comparison
        if rtm_fps > mp_fps:
            print(f"🏆 RTMPose is {rtm_fps/mp_fps:.1f}x faster than MediaPipe!")
        else:
            print(f"📈 MediaPipe is {mp_fps/rtm_fps:.1f}x faster than RTMPose")
        
        mp_pose.close()
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 RTMPose Installation & Performance Test")
    print("=" * 50)
    
    # Run tests
    tests_passed = 0
    total_tests = 5
    
    if test_rtmpose_import():
        tests_passed += 1
    
    if test_gpu_availability():
        tests_passed += 1
    
    if test_mediapipe_comparison():
        tests_passed += 1
        
    if test_rtmpose_model():
        tests_passed += 1
        
    if run_simple_performance_test():
        tests_passed += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! RTMPose is ready to use.")
    elif tests_passed >= 3:
        print("✅ Basic functionality working, some optimizations needed.")
    else:
        print("❌ Multiple tests failed, troubleshooting needed.")

if __name__ == "__main__":
    main()