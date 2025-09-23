#!/usr/bin/env python3
"""
Simple script to examine stored 3D coordinates from analysis results.
"""

import json
import numpy as np
import os
import glob

def examine_stored_coordinates():
    """Examine the 3D coordinates stored in analysis results."""
    
    # Find latest analysis result
    pattern = "uploads/results/*.json"
    analysis_files = glob.glob(pattern)
    
    if not analysis_files:
        print("❌ No analysis files found")
        return
    
    # Get the most recent analysis file
    latest_file = max(analysis_files, key=os.path.getctime)
    print(f"📁 Reading analysis from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    print(f"📊 Analysis status: {data['status']}")
    print(f"🎯 Total frames analyzed: {len(data['frame_analyses'])}")
    
    # Find first frame with 3D data
    for i, frame in enumerate(data['frame_analyses']):
        if frame.get('raw_rtmpose_keypoints'):
            print(f"\n🎯 Frame {i} has 3D keypoints:")
            keypoints = np.array(frame['raw_rtmpose_keypoints'])
            
            print(f"Keypoint count: {len(keypoints)}")
            print(f"First 5 raw keypoints:")
            for j in range(min(5, len(keypoints))):
                x, y, z = keypoints[j]
                print(f"  {j}: x={x:.6f}, y={y:.6f}, z={z:.6f}")
            
            # Calculate statistics
            print(f"\n📈 Coordinate Statistics:")
            for axis, name in enumerate(['X', 'Y', 'Z']):
                coords = keypoints[:, axis]
                print(f"  {name} axis: min={np.min(coords):.6f}, max={np.max(coords):.6f}, "
                      f"range={np.max(coords) - np.min(coords):.6f}, std={np.std(coords):.6f}")
            
            # Apply official RTMPose3D transformations to see the result
            print(f"\n🔄 After RTMPose3D transformations:")
            
            # Step 1: Official coordinate transformation: -keypoints[..., [0, 2, 1]]
            transformed_keypoints = -keypoints[:, [0, 2, 1]]
            
            # Step 2: Height rebasing (ground contact)
            if len(transformed_keypoints) > 0:
                min_z = np.min(transformed_keypoints[:, 2])
                transformed_keypoints[:, 2] -= min_z
            
            for axis, name in enumerate(['X', 'Z', 'Y']):
                coords = transformed_keypoints[:, axis]
                print(f"  {name} axis: min={np.min(coords):.6f}, max={np.max(coords):.6f}, "
                      f"range={np.max(coords) - np.min(coords):.6f}")
            
            # Check pose dimensions
            x_range = np.max(transformed_keypoints[:, 0]) - np.min(transformed_keypoints[:, 0])
            y_range = np.max(transformed_keypoints[:, 1]) - np.min(transformed_keypoints[:, 1]) 
            z_range = np.max(transformed_keypoints[:, 2]) - np.min(transformed_keypoints[:, 2])
            
            print(f"\n🎯 Pose dimensions after transformation:")
            print(f"  Width (X): {x_range:.6f} meters")
            print(f"  Depth (Y): {y_range:.6f} meters") 
            print(f"  Height (Z): {z_range:.6f} meters")
            
            # Check against current axis limit
            current_limit = 400.0
            max_dimension = max(x_range, y_range, z_range)
            
            print(f"\n📏 Scale Analysis:")
            print(f"Current axis limit: ±{current_limit} meters")
            print(f"Max pose dimension: {max_dimension:.6f} meters")
            print(f"Scale ratio: {max_dimension / current_limit:.8f}")
            
            if max_dimension < 5.0:
                print("⚠️  SCALING ISSUE CONFIRMED!")
                print("   Pose is tiny compared to axis limit.")
                suggested_limit = max(max_dimension * 2, 2.0)
                print(f"💡 Suggested axis limit: ±{suggested_limit:.2f} meters")
            
            break
    else:
        print("❌ No frames with 3D keypoints found")

if __name__ == "__main__":
    examine_stored_coordinates()