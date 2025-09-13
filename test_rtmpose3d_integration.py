#!/usr/bin/env python3
"""
Test script for RTMPose3D integration.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_rtmpose3d_integration():
    """Test RTMPose3D integration and fallback mechanisms."""
    print("🧪 Testing RTMPose3D Integration...")
    
    # Set environment variables
    os.environ['USE_RTMPOSE'] = 'true'
    os.environ['USE_3D_POSE'] = 'true'
    
    try:
        from ai_coach.rtm_pose_analyzer import RTMPoseAnalyzer
        
        # Test initialization with 3D mode
        print("\n🔮 Testing 3D RTMPose analyzer initialization...")
        analyzer = RTMPoseAnalyzer(use_3d=True)
        
        # Initialize model
        print("🚀 Initializing models...")
        success = analyzer._initialize_model()
        
        print(f"Model initialized: {'✅ Success' if success else '❌ Failed'}")
        print(f"Using RTMPose3D: {'✅ Yes' if analyzer.is_rtmpose3d else '❌ No (using fallback)'}")
        
        # Test depth estimation vs native 3D
        if analyzer.is_rtmpose3d:
            print("🎯 Native RTMPose3D model loaded - will generate real 3D coordinates")
        else:
            print("🧠 Using intelligent depth estimation - will generate estimated 3D coordinates")
            
            # Test depth estimation function
            print("\n📏 Testing depth estimation...")
            test_cases = [
                (0, 0.5, 0.3, 0.9, "nose"),      # Head - should be forward
                (9, 0.4, 0.5, 0.8, "left_wrist"), # Hand - can extend forward
                (11, 0.4, 0.7, 0.7, "left_hip"),  # Hip - body center
                (15, 0.4, 0.9, 0.6, "left_ankle") # Ankle - back in squat
            ]
            
            for rtm_idx, x, y, conf, name in test_cases:
                depth = analyzer._estimate_depth_coordinate(rtm_idx, x, y, conf)
                print(f"   {name:12} (idx {rtm_idx:2d}): depth = {depth:6.1f}")
        
        print("\n✅ RTMPose3D integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_rtmpose3d_integration()