#!/usr/bin/env python3
"""
Test script for RTMPose analyzer performance comparison.

This script tests both MediaPipe and RTMPose analyzers to compare:
- Speed (FPS performance)
- Accuracy (pose detection rate)
- GPU memory usage
- Overall processing time
"""

import sys
import time
import asyncio
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_coach.pose_analyzer import PoseAnalyzer  # MediaPipe version
from ai_coach.rtm_pose_analyzer import RTMPoseAnalyzer  # RTMPose version


async def benchmark_analyzer(analyzer, video_path: str, analyzer_name: str) -> dict:
    """Benchmark a pose analyzer and return performance metrics."""
    print(f"\n🔥 Testing {analyzer_name}")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Run analysis
        analysis = await analyzer.analyze_video(video_path, f"test_{analyzer_name.lower()}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        total_frames = len(analysis.frame_analyses) if analysis.frame_analyses else 0
        poses_detected = analysis.poses_detected_count
        fps = total_frames / total_time if total_time > 0 else 0
        detection_rate = (poses_detected / total_frames * 100) if total_frames > 0 else 0
        
        metrics = {
            "analyzer": analyzer_name,
            "status": analysis.status.value,
            "total_time": total_time,
            "total_frames": total_frames,
            "poses_detected": poses_detected,
            "fps": fps,
            "detection_rate": detection_rate,
            "gpu_memory_mb": analysis.gpu_memory_used_mb,
            "error": analysis.error_message if hasattr(analysis, 'error_message') and analysis.error_message else None
        }
        
        # Print results
        print(f"✅ Status: {analysis.status.value}")
        print(f"⏱️  Processing Time: {total_time:.2f} seconds")
        print(f"🎬 Total Frames: {total_frames}")
        print(f"🤸 Poses Detected: {poses_detected}")
        print(f"🚀 FPS: {fps:.1f}")
        print(f"🎯 Detection Rate: {detection_rate:.1f}%")
        print(f"💾 GPU Memory: {analysis.gpu_memory_used_mb:.1f} MB")
        
        if metrics["error"]:
            print(f"⚠️  Error: {metrics['error']}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ {analyzer_name} failed: {e}")
        return {
            "analyzer": analyzer_name,
            "status": "failed",
            "total_time": time.time() - start_time,
            "error": str(e),
            "total_frames": 0,
            "poses_detected": 0,
            "fps": 0,
            "detection_rate": 0,
            "gpu_memory_mb": 0
        }


def print_comparison(mediapipe_metrics: dict, rtmpose_metrics: dict):
    """Print performance comparison between MediaPipe and RTMPose."""
    print("\n" + "=" * 70)
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 70)
    
    print(f"{'Metric':<20} {'MediaPipe':<15} {'RTMPose':<15} {'Improvement':<15}")
    print("-" * 70)
    
    # FPS comparison
    mp_fps = mediapipe_metrics.get("fps", 0)
    rtm_fps = rtmpose_metrics.get("fps", 0)
    fps_improvement = f"{rtm_fps / mp_fps:.1f}x" if mp_fps > 0 else "N/A"
    print(f"{'FPS':<20} {mp_fps:<15.1f} {rtm_fps:<15.1f} {fps_improvement:<15}")
    
    # Processing time comparison  
    mp_time = mediapipe_metrics.get("total_time", 0)
    rtm_time = rtmpose_metrics.get("total_time", 0)
    time_improvement = f"{mp_time / rtm_time:.1f}x faster" if rtm_time > 0 else "N/A"
    print(f"{'Processing Time':<20} {mp_time:<15.2f} {rtm_time:<15.2f} {time_improvement:<15}")
    
    # Detection rate comparison
    mp_rate = mediapipe_metrics.get("detection_rate", 0)
    rtm_rate = rtmpose_metrics.get("detection_rate", 0)
    rate_diff = f"+{rtm_rate - mp_rate:.1f}%" if rtm_rate >= mp_rate else f"{rtm_rate - mp_rate:.1f}%"
    print(f"{'Detection Rate':<20} {mp_rate:<15.1f}% {rtm_rate:<15.1f}% {rate_diff:<15}")
    
    # GPU memory comparison
    mp_mem = mediapipe_metrics.get("gpu_memory_mb", 0)
    rtm_mem = rtmpose_metrics.get("gpu_memory_mb", 0)
    print(f"{'GPU Memory (MB)':<20} {mp_mem:<15.1f} {rtm_mem:<15.1f} {'':<15}")
    
    print("-" * 70)
    
    # Overall assessment
    if rtm_fps > mp_fps and rtm_rate >= mp_rate * 0.9:  # Allow 10% accuracy tolerance
        print("🏆 RTMPose WINNER: Faster with comparable accuracy!")
    elif mp_fps > rtm_fps:
        print("📈 MediaPipe faster, but RTMPose may improve with optimization")
    else:
        print("🤔 Mixed results - further analysis needed")


async def main():
    """Run the RTMPose vs MediaPipe benchmark."""
    print("🚀 RTMPose vs MediaPipe Benchmark")
    print("=" * 50)
    
    # Test video path
    test_video = Path("examples/deadlift.mp4")
    if not test_video.exists():
        print(f"❌ Test video not found: {test_video}")
        print("Please ensure deadlift.mp4 exists in the examples/ directory")
        return
    
    # Initialize analyzers
    mediapipe_analyzer = PoseAnalyzer(use_gpu_encoding=False, frame_skip=3)
    rtmpose_analyzer = RTMPoseAnalyzer(use_gpu_encoding=False, frame_skip=3)
    
    # Benchmark MediaPipe
    print("Starting MediaPipe benchmark...")
    mp_metrics = await benchmark_analyzer(mediapipe_analyzer, str(test_video), "MediaPipe")
    
    # Benchmark RTMPose
    print("\nStarting RTMPose benchmark...")
    rtm_metrics = await benchmark_analyzer(rtmpose_analyzer, str(test_video), "RTMPose")
    
    # Print comparison
    print_comparison(mp_metrics, rtm_metrics)
    
    # Save results
    import json
    results = {
        "mediapipe": mp_metrics,
        "rtmpose": rtm_metrics,
        "test_video": str(test_video),
        "timestamp": time.time()
    }
    
    results_file = Path("benchmark_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: {results_file}")


if __name__ == "__main__":
    asyncio.run(main())