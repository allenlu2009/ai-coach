#!/usr/bin/env python3
"""Test the fixed RTMPose analyzer."""

import sys
import asyncio
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_coach.rtm_pose_analyzer import RTMPoseAnalyzer

async def test_fixed_analyzer():
    """Test the fixed RTMPose analyzer."""
    print("🧪 Testing fixed RTMPose analyzer...")
    
    video_path = "examples/deadlift.mp4"
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    try:
        # Initialize RTMPose analyzer
        analyzer = RTMPoseAnalyzer(use_gpu_encoding=False, frame_skip=3)
        
        # Run analysis with proper UUID
        import uuid
        test_id = str(uuid.uuid4())
        analysis = await analyzer.analyze_video(video_path, test_id)
        
        print(f"✅ Status: {analysis.status.value}")
        print(f"🎬 Total Frames: {len(analysis.frame_analyses)}")
        print(f"🤸 Poses Detected: {analysis.poses_detected_count}")
        print(f"🎯 Detection Rate: {analysis.pose_detection_rate:.2%}")
        print(f"🔥 Average Confidence: {analysis.average_confidence:.2f}")
        print(f"⏱️ Processing Time: {analysis.processing_time_seconds:.2f}s")
        
        if analysis.poses_detected_count > 0:
            print("🎉 SUCCESS: RTMPose is now detecting poses!")
            
            # Check first pose analysis
            first_pose = next((frame for frame in analysis.frame_analyses if frame.pose_detected), None)
            if first_pose:
                print(f"🔍 First pose: frame {first_pose.frame_number}")
                print(f"   Landmarks: {len(first_pose.landmarks)}")
                print(f"   Confidence: {first_pose.confidence_score:.2f}")
        else:
            print("❌ Still no poses detected")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_fixed_analyzer())