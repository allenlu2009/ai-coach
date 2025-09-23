#!/usr/bin/env python3
"""
Simple test to check RTMPose3D coordinate behavior
"""

import sys
sys.path.insert(0, 'src')

def test_coordinate_logic():
    """Test the coordinate transformation logic directly."""
    import numpy as np
    
    print("🎯 RTMPose3D Coordinate Logic Test")
    print("=" * 50)
    
    # Simulate RTMPose3D raw output (world coordinates)
    # These are example coordinates from our earlier debugging
    raw_coords = np.array([
        [2.240, -0.540, 0.100],  # Example keypoint 1
        [2.100, -0.300, 0.050],  # Example keypoint 2  
        [1.900, -0.200, 0.080],  # Example keypoint 3
        [1.800, -0.100, 0.120],  # Example keypoint 4
        [2.500, -0.800, 0.200],  # Example keypoint 5
    ])
    
    print(f"📊 Raw RTMPose3D coordinates (world space):")
    print(f"   Shape: {raw_coords.shape}")
    print(f"   X range: [{raw_coords[:, 0].min():.3f}, {raw_coords[:, 0].max():.3f}]")
    print(f"   Y range: [{raw_coords[:, 1].min():.3f}, {raw_coords[:, 1].max():.3f}]")
    print(f"   Z range: [{raw_coords[:, 2].min():.3f}, {raw_coords[:, 2].max():.3f}]")
    
    # Apply RTMPose3D demo transformation: -keypoints[..., [0, 2, 1]]
    print(f"\n🔄 Apply Demo Transformation: -keypoints[..., [0, 2, 1]]")
    demo_coords = -raw_coords[..., [0, 2, 1]]
    
    print(f"   Transformed coordinates:")
    print(f"   X range: [{demo_coords[:, 0].min():.3f}, {demo_coords[:, 0].max():.3f}]")
    print(f"   Y range: [{demo_coords[:, 1].min():.3f}, {demo_coords[:, 1].max():.3f}]")
    print(f"   Z range: [{demo_coords[:, 2].min():.3f}, {demo_coords[:, 2].max():.3f}]")
    
    # Check if these could work as pixel coordinates for a typical video
    typical_width, typical_height = 640, 480
    print(f"\n📺 Pixel Space Analysis (example {typical_width}x{typical_height} video):")
    
    pixel_compatible = (0 <= demo_coords[:, 0].min() and demo_coords[:, 0].max() <= typical_width and
                       0 <= demo_coords[:, 1].min() and demo_coords[:, 1].max() <= typical_height)
    
    print(f"   Pixel space compatible: {pixel_compatible}")
    
    if not pixel_compatible:
        print(f"   ❌ Transformed coordinates are NOT pixel-compatible")
        print(f"   ❌ This explains why the demo approach doesn't work directly for us!")
        
        # Show what coordinates would look like
        print(f"\n📍 Sample transformed coordinates:")
        for i, coord in enumerate(demo_coords[:3]):  # First 3 keypoints
            print(f"   Keypoint {i}: x={coord[0]:.1f}, y={coord[1]:.1f}, z={coord[2]:.3f}")
            
        print(f"\n💡 The issue:")
        print(f"   - Raw RTMPose3D coordinates are in world space (meters)")
        print(f"   - Demo transformation: -[x,z,y] still gives world coordinates")
        print(f"   - These are NOT pixel coordinates!")
        print(f"   - Demo must use a different pipeline that produces pixel coordinates")
        
    else:
        print(f"   ✅ Coordinates are pixel-compatible!")
        
    # Test rebase height (z-axis)
    print(f"\n🏔️ Height Rebasing (z-axis to ground level):")
    rebased_coords = demo_coords.copy()
    rebased_coords[:, 2] -= np.min(rebased_coords[:, 2], axis=0, keepdims=True)
    
    print(f"   Original Z range: [{demo_coords[:, 2].min():.3f}, {demo_coords[:, 2].max():.3f}]")
    print(f"   Rebased Z range: [{rebased_coords[:, 2].min():.3f}, {rebased_coords[:, 2].max():.3f}]")
    
    print(f"\n🎯 Conclusion:")
    if not pixel_compatible:
        print(f"   The demo uses a different pipeline that produces pixel coordinates.")
        print(f"   Our current coordinate mapping approach is the correct solution!")
        print(f"   We need to map from world coordinates to pixel coordinates.")
    else:
        print(f"   We can use the demo approach directly!")


if __name__ == "__main__":
    test_coordinate_logic()