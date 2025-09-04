#!/usr/bin/env python3
"""
Generate pose analysis videos for all example videos.
This script creates pose analysis videos that can be viewed directly in a browser.
"""

import cv2
import mediapipe as mp
import os
import sys
from pathlib import Path

def create_pose_analysis_video(input_path, output_path):
    """
    Create a pose analysis video with landmarks overlay.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video with pose analysis
    """
    print(f"Processing: {input_path} -> {output_path}")
    
    # Initialize MediaPipe Pose (simple configuration)
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # Lower complexity for faster processing
        enable_segmentation=False,  # Disabled to avoid errors
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True
    )
    
    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"❌ Cannot open video: {input_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    
    # Create output video writer (browser-compatible H.264 format)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264/AVC codec for browser compatibility
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ Cannot create output video: {output_path}")
        cap.release()
        return False
    
    frame_count = 0
    poses_detected = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = pose.process(rgb_frame)
        
        # Draw pose landmarks if detected
        if results.pose_landmarks:
            poses_detected += 1
            
            # Draw landmarks and connections
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),  # Orange landmarks
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)   # Pink connections
            )
        
        # Add frame info overlay
        cv2.putText(
            frame,
            f"Frame: {frame_count+1}/{total_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        cv2.putText(
            frame,
            f"Poses Detected: {poses_detected}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if results.pose_landmarks else (0, 0, 255),
            2
        )
        
        # Write frame to output
        out.write(frame)
        frame_count += 1
        
        # Progress indicator
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}% ({poses_detected} poses detected)")
    
    # Cleanup
    cap.release()
    out.release()
    pose.close()
    
    detection_rate = (poses_detected / total_frames) * 100
    print(f"✅ Completed: {poses_detected}/{total_frames} poses detected ({detection_rate:.1f}%)")
    return True

def main():
    """Generate pose analysis videos for all example videos."""
    
    # Set up paths
    examples_dir = Path("examples")
    output_dir = Path("pose_analysis_videos")
    
    if not examples_dir.exists():
        print("❌ Examples directory not found")
        return
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Find all video files in examples directory
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        video_files.extend(examples_dir.glob(ext))
    
    if not video_files:
        print("❌ No video files found in examples directory")
        return
    
    print(f"🎥 Found {len(video_files)} video files:")
    for video_file in video_files:
        print(f"  - {video_file.name}")
    
    # Process each video
    print("\n🏃‍♀️ Starting pose analysis video generation...")
    
    for video_file in video_files:
        output_file = output_dir / f"{video_file.stem}_pose_analysis.mp4"
        success = create_pose_analysis_video(video_file, output_file)
        
        if success:
            print(f"📹 Generated: {output_file}")
        else:
            print(f"❌ Failed: {video_file.name}")
        print()
    
    print("🎯 All videos processed!")
    print(f"📂 Check the '{output_dir}' directory for generated videos")
    print("💡 You can now open these videos directly in any browser!")

if __name__ == "__main__":
    main()