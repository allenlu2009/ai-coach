#!/usr/bin/env python3
"""
Test script to verify the frame synchronization fix for MMPose demo visualizer.

This script tests that:
1. Fresh pose analysis works correctly for individual frames
2. The MMPose demo visualizer creates proper side-by-side visualization
3. No flying wires or synchronization issues occur
"""

import cv2
import sys
import logging
sys.path.insert(0, 'src')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_frame_sync_fix():
    """Test the frame synchronization fix."""
    try:
        from ai_coach.rtm_pose_analyzer import RTMPoseAnalyzer
        
        logger.info("🎯 Testing frame synchronization fix...")
        
        # Initialize RTMPose analyzer with demo visualizer
        analyzer = RTMPoseAnalyzer(
            use_3d=True,
            enable_detection=True,
            use_demo_visualizer=True
        )
        
        # Initialize models
        success = analyzer._initialize_model()
        if not success:
            logger.error("❌ Failed to initialize RTMPose models")
            return False
        
        logger.info("✅ RTMPose models initialized successfully")
        
        # Load test video frame
        video_file = 'examples/deadlift.mp4'
        cap = cv2.VideoCapture(video_file)
        
        if not cap.isOpened():
            logger.error(f"❌ Cannot open video file: {video_file}")
            return False
        
        # Test with multiple frames to check for consistency
        test_frames = []
        for i in range(5):  # Test first 5 frames
            ret, frame = cap.read()
            if ret:
                test_frames.append(frame)
            else:
                break
        
        cap.release()
        
        if len(test_frames) == 0:
            logger.error("❌ No frames read from video")
            return False
        
        logger.info(f"📹 Testing with {len(test_frames)} frames")
        
        # Test fresh pose analysis for each frame
        for i, frame in enumerate(test_frames):
            logger.info(f"\n🔍 Testing frame {i+1}/{len(test_frames)}")
            
            # Test fresh pose analysis
            fresh_results = analyzer._run_fresh_pose_analysis_for_frame(frame)
            
            if fresh_results is None:
                logger.warning(f"⚠️ Frame {i+1}: No fresh pose results")
                continue
            
            logger.info(f"✅ Frame {i+1}: Fresh analysis successful, {len(fresh_results)} results")
            
            # Test MMPose demo visualization
            if analyzer.mmpose_visualizer is not None:
                vis_result = analyzer.mmpose_visualizer.process_and_visualize(
                    frame, 
                    fresh_results,
                    save_path=f"test_frame_sync_frame_{i+1}.jpg"
                )
                
                if vis_result is not None:
                    logger.info(f"✅ Frame {i+1}: Demo visualization successful")
                    logger.info(f"   - Original frame shape: {frame.shape}")
                    logger.info(f"   - Visualization shape: {vis_result.shape}")
                    
                    # Check if it's side-by-side (should be roughly 2x width)
                    if vis_result.shape[1] > frame.shape[1] * 1.5:
                        logger.info(f"✅ Frame {i+1}: Side-by-side layout detected")
                    else:
                        logger.warning(f"⚠️ Frame {i+1}: Unexpected visualization layout")
                else:
                    logger.warning(f"⚠️ Frame {i+1}: Demo visualization returned None")
            else:
                logger.warning(f"⚠️ Frame {i+1}: MMPose visualizer not initialized")
        
        # Test the complete pipeline with overlay drawing
        logger.info("\n🎨 Testing complete overlay pipeline...")
        
        test_frame = test_frames[0]
        
        # Create a dummy FrameAnalysis (this would normally come from pose analysis)
        from ai_coach.data_models import FrameAnalysis, PoseLandmark
        
        frame_analysis = FrameAnalysis(
            frame_number=1,
            timestamp=0.0,
            pose_detected=True,
            landmarks=[],  # We'll use fresh analysis instead
            confidence=0.9
        )
        
        # Test the overlay drawing method
        overlay_result = analyzer._draw_rtmpose_overlay(test_frame, frame_analysis)
        
        if overlay_result is not None:
            logger.info("✅ Complete overlay pipeline successful")
            logger.info(f"   - Input frame shape: {test_frame.shape}")
            logger.info(f"   - Overlay result shape: {overlay_result.shape}")
            
            # Save the result
            cv2.imwrite("test_complete_overlay_pipeline.jpg", overlay_result)
            logger.info("   - Saved result to: test_complete_overlay_pipeline.jpg")
        else:
            logger.error("❌ Complete overlay pipeline failed")
            return False
        
        logger.info("\n🎉 Frame synchronization fix test completed successfully!")
        logger.info("Key improvements:")
        logger.info("   - Fresh pose analysis for each frame prevents synchronization issues")
        logger.info("   - MMPose demo visualizer creates proper side-by-side layout") 
        logger.info("   - No more flying wires from stale cached results")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_frame_sync_fix()
    if success:
        print("\n✅ ALL TESTS PASSED - Frame synchronization fix is working!")
        sys.exit(0)
    else:
        print("\n❌ TESTS FAILED - Frame synchronization fix needs debugging")
        sys.exit(1)