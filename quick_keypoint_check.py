#!/usr/bin/env python3
"""Quick check of RTMPose keypoints from deadlift video."""

import cv2
import numpy as np
from pathlib import Path

video_path = "examples/deadlift.mp4"
if Path(video_path).exists():
    from mmpose.apis import MMPoseInferencer
    inferencer = MMPoseInferencer('human')
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        results = list(inferencer(frame, show=False, return_vis=False))
        print(f"📊 Results structure found!")
        
        pred_dict = results[0]['predictions'][0][0]
        keypoints = pred_dict['keypoints']
        keypoint_scores = pred_dict['keypoint_scores']
        bbox = pred_dict['bbox']
        bbox_score = pred_dict['bbox_score']
        
        print(f"🔍 Keypoints: {len(keypoints)} items")
        print(f"🔍 Keypoint scores: {len(keypoint_scores)} items")
        print(f"🔍 BBox: {bbox} (score: {bbox_score})")
        
        if len(keypoints) > 0:
            print(f"✅ First 3 keypoints: {keypoints[:3]}")
            print(f"✅ First 3 scores: {keypoint_scores[:3]}")
            print("🎉 RTMPose IS detecting poses! The structure parsing was wrong.")
        else:
            print("❌ Keypoints list is empty - no poses detected")
    else:
        print("❌ Could not read frame")
else:
    print("❌ Video not found")