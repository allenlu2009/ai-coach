#!/usr/bin/env python3
"""
Test RTMPose3D initialization to debug why it fails in isolation
"""
import sys
import logging
sys.path.insert(0, 'src')

# Set up logging to see errors
logging.basicConfig(level=logging.INFO)

from ai_coach.rtm_pose_analyzer import RTMPoseAnalyzer

def test_rtm3d_initialization():
    """Test RTMPose3D initialization with detailed error logging."""
    
    print("🔍 Testing RTMPose3D initialization...")
    
    try:
        # Create analyzer with 3D enabled
        analyzer = RTMPoseAnalyzer(use_3d=True)
        
        # Force model initialization
        success = analyzer._initialize_model()
        
        print(f"✅ Model initialization: {'SUCCESS' if success else 'FAILED'}")
        print(f"   - is_rtmpose3d: {analyzer.is_rtmpose3d}")
        print(f"   - has pose3d_model: {hasattr(analyzer, 'pose3d_model')}")
        print(f"   - model_initialized: {analyzer.model_initialized}")
        
        if analyzer.is_rtmpose3d and hasattr(analyzer, 'pose3d_model'):
            print("🎯 RTMPose3D model successfully loaded!")
        else:
            print("❌ RTMPose3D model failed to load")
            
    except Exception as e:
        print(f"❌ Exception during initialization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rtm3d_initialization()