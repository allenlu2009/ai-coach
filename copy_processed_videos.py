#!/usr/bin/env python3
"""
Copy all processed videos from uploads/processed to pose_analysis_videos folder
with descriptive names for easy access and testing.
"""

import os
import shutil
from pathlib import Path

def copy_processed_videos():
    """Copy all processed videos to pose_analysis_videos folder."""
    
    # Paths
    processed_dir = Path("uploads/processed")
    output_dir = Path("pose_analysis_videos")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    if not processed_dir.exists():
        print("❌ uploads/processed directory not found")
        return
    
    # Find all processed video files
    processed_videos = list(processed_dir.glob("*_processed.mp4"))
    
    if not processed_videos:
        print("❌ No processed videos found")
        return
    
    print(f"📹 Found {len(processed_videos)} processed videos")
    
    for video_file in processed_videos:
        # Create descriptive filename
        video_id = video_file.stem.replace("_processed", "")
        output_filename = f"{video_id}_pose_analysis_h264.mp4"
        output_path = output_dir / output_filename
        
        try:
            # Copy the video
            shutil.copy2(str(video_file), str(output_path))
            
            # Get file size for display
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            
            print(f"✅ Copied: {video_file.name} -> {output_filename} ({file_size_mb:.1f}MB)")
            
        except Exception as e:
            print(f"❌ Failed to copy {video_file.name}: {e}")
    
    print(f"🎯 Completed copying videos to {output_dir}")

if __name__ == "__main__":
    copy_processed_videos()