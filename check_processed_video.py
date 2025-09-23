#!/usr/bin/env python3
"""
Check processed video to see if pose overlays are present
"""
import cv2
import numpy as np

def check_video_frames(video_path, num_frames=5):
    """Check if processed video has pose overlays by examining pixel differences."""
    print(f"🎥 Analyzing processed video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video")
        return
        
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📋 Video info: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Check several frames for overlays
    frame_positions = [0, total_frames//4, total_frames//2, 3*total_frames//4, total_frames-1]
    
    for i, frame_pos in enumerate(frame_positions[:num_frames]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        
        if not ret:
            print(f"❌ Could not read frame {frame_pos}")
            continue
            
        # Look for green lines/circles typical of pose overlays
        # Convert to HSV to detect green color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Green color range in HSV
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_pixels = np.sum(green_mask > 0)
        
        # Look for bright colors typical of overlays
        bright_mask = cv2.inRange(hsv, np.array([0, 50, 200]), np.array([179, 255, 255]))
        bright_pixels = np.sum(bright_mask > 0)
        
        print(f"Frame {frame_pos:4d}: Green pixels: {green_pixels:6d}, Bright pixels: {bright_pixels:6d}")
        
        # Save a sample frame for manual inspection
        if i == 2:  # Middle frame
            sample_path = f"/tmp/processed_video_frame_{frame_pos}.png"
            cv2.imwrite(sample_path, frame)
            print(f"💾 Sample frame saved: {sample_path}")
    
    cap.release()
    
    # Summary
    print("\n📊 Analysis Summary:")
    if green_pixels > 1000 or bright_pixels > 5000:
        print("✅ Likely contains pose overlays (detected colored pixels)")
    else:
        print("❌ May not contain visible pose overlays (minimal colored pixels)")

if __name__ == "__main__":
    video_path = "uploads/processed/6be6bedb-f18e-46b2-84c8-13e72d5916a8_processed.mp4"
    check_video_frames(video_path)