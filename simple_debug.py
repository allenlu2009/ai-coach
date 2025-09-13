#!/usr/bin/env python3
"""Simple RTMPose debug to understand the data structure."""

import cv2
import numpy as np
from pathlib import Path

def main():
    print("🔍 Simple RTMPose structure debug...")
    
    video_path = "examples/deadlift.mp4"
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    try:
        from mmpose.apis import MMPoseInferencer
        inferencer = MMPoseInferencer('human')
        
        # Load first frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Could not read frame")
            return
            
        print(f"📹 Frame shape: {frame.shape}")
        
        # Get results and convert generator to list
        results = list(inferencer(frame, show=False, return_vis=False))
        print(f"📊 Results: {len(results)} items")
        
        if len(results) > 0:
            result = results[0]
            print(f"   Type: {type(result)}")
            
            if isinstance(result, dict):
                print(f"   Keys: {list(result.keys())}")
                for key, value in result.items():
                    print(f"   {key}: {type(value)}")
                    
                    # Check predictions list
                    if key == 'predictions' and isinstance(value, list):
                        print(f"      predictions list length: {len(value)}")
                        
                        for i, prediction in enumerate(value):
                            print(f"      prediction[{i}]: {type(prediction)}")
                            
                            # If it's a list (nested structure), inspect its contents
                            if isinstance(prediction, list):
                                print(f"         nested list length: {len(prediction)}")
                                
                                for j, pred_item in enumerate(prediction):
                                    print(f"         pred_item[{j}]: {type(pred_item)}")
                                    
                                    # Check if pred_item has pred_instances
                                    if hasattr(pred_item, 'pred_instances'):
                                        pred_instances = pred_item.pred_instances
                                        print(f"            pred_instances: {type(pred_instances)}")
                                        
                                        # List all attributes
                                        attrs = [attr for attr in dir(pred_instances) if not attr.startswith('_')]
                                        print(f"            pred_instances attrs: {attrs}")
                                        
                                        if hasattr(pred_instances, 'keypoints'):
                                            keypoints = pred_instances.keypoints
                                            print(f"            KEYPOINTS FOUND! Shape: {keypoints.shape}")
                                            print(f"            Keypoints type: {type(keypoints)}")
                                            
                                            if hasattr(keypoints, 'shape') and len(keypoints.shape) >= 2:
                                                print(f"            Number of poses: {keypoints.shape[0]}")
                                                print(f"            Keypoints per pose: {keypoints.shape[1]}")
                                                if len(keypoints.shape) > 2:
                                                    print(f"            Coordinates per keypoint: {keypoints.shape[2]}")
                                                
                                                # Sample a few keypoints
                                                if keypoints.shape[0] > 0:
                                                    print(f"            Sample keypoints: {keypoints[0, :3]}")  # First 3 keypoints of first pose
                                            return True
                                    
                                    # If pred_item is a dict, check its contents
                                    if isinstance(pred_item, dict):
                                        print(f"            Dict keys: {list(pred_item.keys())}")
                                        
                                        for dict_key, dict_value in pred_item.items():
                                            print(f"            {dict_key}: {type(dict_value)}")
                                            
                                            if hasattr(dict_value, 'shape'):
                                                print(f"            {dict_key} shape: {dict_value.shape}")
                                                
                                                # Check if this is keypoints data
                                                if 'keypoint' in dict_key.lower() or dict_key == 'pred_instances':
                                                    print(f"            POTENTIAL KEYPOINTS in {dict_key}!")
                                                    
                                                    if len(dict_value.shape) >= 2:
                                                        print(f"            Number of poses: {dict_value.shape[0]}")
                                                        print(f"            Keypoints per pose: {dict_value.shape[1]}")
                                                        if len(dict_value.shape) > 2:
                                                            print(f"            Coordinates per keypoint: {dict_value.shape[2]}")
                                                        
                                                        # Sample keypoints
                                                        if dict_value.shape[0] > 0:
                                                            print(f"            Sample: {dict_value[0, :3] if len(dict_value.shape) >= 2 else dict_value[:3]}")
                                                        return True
                                            
                                            # Check if dict_value has pred_instances
                                            if hasattr(dict_value, 'pred_instances'):
                                                pred_instances = dict_value.pred_instances
                                                print(f"            pred_instances in {dict_key}: {type(pred_instances)}")
                                                
                                                if hasattr(pred_instances, 'keypoints'):
                                                    keypoints = pred_instances.keypoints
                                                    print(f"            KEYPOINTS FOUND in {dict_key}.pred_instances! Shape: {keypoints.shape}")
                                                    return True
                                    
                                    # Check direct attributes on pred_item
                                    else:
                                        pred_item_attrs = [attr for attr in dir(pred_item) if not attr.startswith('_')]
                                        print(f"            pred_item attrs: {pred_item_attrs}")
                                        
                                        # Check common pose attributes
                                        for attr in ['keypoints', 'pred_instances', 'instances']:
                                            if hasattr(pred_item, attr):
                                                attr_value = getattr(pred_item, attr)
                                                print(f"            {attr}: {type(attr_value)}")
                                                
                                                if hasattr(attr_value, 'shape'):
                                                    print(f"            {attr} shape: {attr_value.shape}")
                                                    if attr == 'keypoints' and len(attr_value.shape) >= 2:
                                                        print(f"            KEYPOINTS FOUND in {attr}!")
                                                        print(f"            Number of poses: {attr_value.shape[0]}")
                                                        return True
                                        
                            # Check if prediction has pred_instances (direct case)
                            elif hasattr(prediction, 'pred_instances'):
                                pred_instances = prediction.pred_instances
                                print(f"         pred_instances: {type(pred_instances)}")
                                
                                # List all attributes
                                attrs = [attr for attr in dir(pred_instances) if not attr.startswith('_')]
                                print(f"         pred_instances attrs: {attrs}")
                                
                                if hasattr(pred_instances, 'keypoints'):
                                    keypoints = pred_instances.keypoints
                                    print(f"         KEYPOINTS FOUND! Shape: {keypoints.shape}")
                                    print(f"         Keypoints type: {type(keypoints)}")
                                    
                                    if hasattr(keypoints, 'shape') and len(keypoints.shape) >= 2:
                                        print(f"         Number of poses: {keypoints.shape[0]}")
                                        print(f"         Keypoints per pose: {keypoints.shape[1]}")
                                        if len(keypoints.shape) > 2:
                                            print(f"         Coordinates per keypoint: {keypoints.shape[2]}")
                                        
                                        # Sample a few keypoints
                                        if keypoints.shape[0] > 0:
                                            print(f"         Sample keypoints: {keypoints[0, :3]}")  # First 3 keypoints of first pose
                                    return True
                                    
                            # Check direct attributes on prediction
                            else:
                                prediction_attrs = [attr for attr in dir(prediction) if not attr.startswith('_')]
                                print(f"         prediction attrs: {prediction_attrs}")
                                
                                # Check common pose attributes
                                for attr in ['keypoints', 'pred_instances', 'instances']:
                                    if hasattr(prediction, attr):
                                        attr_value = getattr(prediction, attr)
                                        print(f"         {attr}: {type(attr_value)}")
                                        
                                        if hasattr(attr_value, 'shape'):
                                            print(f"         {attr} shape: {attr_value.shape}")
                                            if attr == 'keypoints' and len(attr_value.shape) >= 2:
                                                print(f"         KEYPOINTS FOUND in {attr}!")
                                                return True
                    
                    # Check if it has pred_instances
                    elif hasattr(value, 'pred_instances'):
                        pred_instances = value.pred_instances
                        print(f"      pred_instances: {type(pred_instances)}")
                        
                        # List all attributes of pred_instances
                        attrs = [attr for attr in dir(pred_instances) if not attr.startswith('_')]
                        print(f"      pred_instances attrs: {attrs[:10]}")  # First 10 attrs
                        
                        if hasattr(pred_instances, 'keypoints'):
                            keypoints = pred_instances.keypoints
                            print(f"      KEYPOINTS FOUND! Shape: {keypoints.shape}")
                            print(f"      Keypoints type: {type(keypoints)}")
                            
                            if hasattr(keypoints, 'shape') and len(keypoints.shape) >= 2:
                                print(f"      Number of poses: {keypoints.shape[0]}")
                                print(f"      Keypoints per pose: {keypoints.shape[1]}")
                                if len(keypoints.shape) > 2:
                                    print(f"      Coordinates per keypoint: {keypoints.shape[2]}")
                                
                                # Sample a few keypoints
                                if keypoints.shape[0] > 0:
                                    print(f"      Sample keypoints: {keypoints[0, :3]}")  # First 3 keypoints of first pose
                            return True
            
        print("❌ No keypoints found in results structure")
        return False
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()