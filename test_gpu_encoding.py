#!/usr/bin/env python3
"""
Test script to verify GPU encoding functionality.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from ai_coach.pose_analyzer import PoseAnalyzer

def test_gpu_encoding():
    """Test GPU encoding availability."""
    print("🧪 Testing GPU encoding capabilities...")
    
    # Test CPU-only analyzer
    analyzer_cpu = PoseAnalyzer(use_gpu=True, use_gpu_encoding=False)
    print(f"✅ CPU Encoding - MediaPipe GPU: {analyzer_cpu.use_gpu}, FFmpeg GPU: {analyzer_cpu.use_gpu_encoding}")
    
    # Test GPU encoding analyzer
    analyzer_gpu = PoseAnalyzer(use_gpu=True, use_gpu_encoding=True)
    print(f"🚀 GPU Encoding - MediaPipe GPU: {analyzer_gpu.use_gpu}, FFmpeg GPU: {analyzer_gpu.use_gpu_encoding}")
    
    if analyzer_gpu.use_gpu_encoding:
        print("🎉 NVIDIA NVENC GPU encoding is available and working!")
        return True
    else:
        print("⚠️ NVIDIA NVENC GPU encoding is not available")
        return False

if __name__ == "__main__":
    test_gpu_encoding()