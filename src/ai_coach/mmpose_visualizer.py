"""
MMPose-compatible visualizer that mimics the official RTMPose3D demo approach.
Uses Pose3dLocalVisualizer to create side-by-side 2D and 3D visualizations.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any, List
import os
import tempfile

logger = logging.getLogger(__name__)


class MMPoseVisualizer:
    """
    Visualizer that follows the exact approach from RTMPose3D demo:
    - Uses MMPose's Pose3dLocalVisualizer 
    - Creates side-by-side 2D and 3D visualization panels
    - Follows the official demo's add_datasample pattern
    """
    
    def __init__(self, 
                 thickness: int = 1,
                 radius: int = 3,
                 kpt_thr: float = 0.3,
                 axis_limit: int = 400,
                 axis_azimuth: int = 70,
                 axis_elev: int = 15):
        """
        Initialize MMPose visualizer with demo settings.
        
        Args:
            thickness: Line thickness for skeleton connections
            radius: Radius for keypoint markers
            kpt_thr: Keypoint confidence threshold
            axis_limit: 3D plot axis limits
            axis_azimuth: 3D plot azimuth angle
            axis_elev: 3D plot elevation angle
        """
        self.thickness = thickness
        self.radius = radius
        self.kpt_thr = kpt_thr
        self.axis_limit = axis_limit
        self.axis_azimuth = axis_azimuth
        self.axis_elev = axis_elev
        
        self.visualizer = None
        self.pose_estimator = None
        self.detector = None
        self.initialized = False
        
        logger.info(f"MMPoseVisualizer initialized with demo settings")
    
    def initialize_models(self, pose_estimator, detector=None):
        """Initialize the visualizer with the pose estimator and detector models."""
        try:
            from mmpose.registry import VISUALIZERS
            
            self.pose_estimator = pose_estimator
            self.detector = detector
            
            # Get dataset metadata from pose estimator
            det_kpt_color = pose_estimator.dataset_meta.get('keypoint_colors', None)
            det_dataset_skeleton = pose_estimator.dataset_meta.get('skeleton_links', None)
            det_dataset_link_color = pose_estimator.dataset_meta.get('skeleton_link_colors', None)
            
            # Configure pose estimator for visualization (following demo lines 286-294)
            pose_estimator.cfg.model.test_cfg.mode = 'vis'
            pose_estimator.cfg.visualizer.radius = self.radius
            pose_estimator.cfg.visualizer.line_width = self.thickness
            pose_estimator.cfg.visualizer.det_kpt_color = det_kpt_color
            pose_estimator.cfg.visualizer.det_dataset_skeleton = det_dataset_skeleton
            pose_estimator.cfg.visualizer.det_dataset_link_color = det_dataset_link_color
            pose_estimator.cfg.visualizer.skeleton = det_dataset_skeleton
            pose_estimator.cfg.visualizer.link_color = det_dataset_link_color
            pose_estimator.cfg.visualizer.kpt_color = det_kpt_color
            
            # Build the visualizer (following demo line 295)
            self.visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
            
            self.initialized = True
            logger.info("✅ MMPose visualizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MMPose visualizer: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def visualize_frame(self, 
                       frame: np.ndarray,
                       pose_data_samples,
                       dataset_name: str = "coco",
                       num_instances: int = 1,
                       save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Visualize frame with 2D and 3D pose using MMPose's approach.
        
        Args:
            frame: Input image frame
            pose_data_samples: Pose estimation results from MMPose
            dataset_name: Dataset name for visualization
            num_instances: Number of pose instances to visualize
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image or None if failed
        """
        if not self.initialized or self.visualizer is None:
            logger.error("MMPose visualizer not initialized")
            return None
            
        try:
            # Convert frame to RGB for visualizer
            visualize_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use the exact same visualization call as the demo (lines 239-256)
            self.visualizer.add_datasample(
                'result',
                visualize_frame,
                data_sample=pose_data_samples,
                det_data_sample=pose_data_samples,
                draw_gt=False,
                draw_2d=True,  # This creates the side-by-side 2D panel
                dataset_2d=dataset_name,
                dataset_3d=dataset_name,
                show=False,  # Don't show window, just generate image
                draw_bbox=True,
                kpt_thr=self.kpt_thr,
                convert_keypoint=False,
                axis_limit=self.axis_limit,
                axis_azimuth=self.axis_azimuth,
                axis_elev=self.axis_elev,
                num_instances=num_instances,
                wait_time=0
            )
            
            # Get the visualization result
            vis_frame = self.visualizer.get_image()
            
            # Convert back to BGR for OpenCV
            if vis_frame is not None:
                vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)

                # Preserve the natural side-by-side aspect ratio from MMPose demo visualizer
                # Do not force resize to original frame dimensions as this squashes the layout
                logger.debug(f"MMPose demo visualization output shape: {vis_frame_bgr.shape} (preserving aspect ratio)")

                # Save if requested
                if save_path:
                    cv2.imwrite(save_path, vis_frame_bgr)
                    logger.debug(f"Saved visualization to: {save_path}")

                return vis_frame_bgr
            else:
                logger.warning("Visualizer returned None image")
                return None
                
        except Exception as e:
            logger.error(f"Failed to visualize frame: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def create_pose_data_samples(self, pose_results: List) -> Any:
        """
        Convert pose results to MMPose data samples format.
        
        Args:
            pose_results: Raw results from inference_topdown (MMPose format)
            
        Returns:
            Merged data samples for visualization
        """
        try:
            from mmpose.structures import merge_data_samples
            
            if not pose_results or len(pose_results) == 0:
                logger.warning("No pose results to convert")
                return None
            
            logger.debug(f"Processing {len(pose_results)} pose results for MMPose demo visualization")
            
            # Apply the demo's post-processing (lines 206-234)
            processed_results = []
            
            for idx, pose_result in enumerate(pose_results):
                logger.debug(f"Processing pose result {idx}: type={type(pose_result)}")
                
                # Check if we have proper MMPose result objects with pred_instances
                if not hasattr(pose_result, 'pred_instances'):
                    logger.warning(f"Pose result {idx} missing pred_instances attribute - this indicates a data format issue")
                    logger.debug(f"Available attributes: {dir(pose_result) if hasattr(pose_result, '__dict__') else 'N/A'}")
                    continue
                    
                # Set track_id for sorting (following demo line 225)
                pose_result.track_id = getattr(pose_result, 'track_id', 1e4)
                pred_instances = pose_result.pred_instances
                
                # Extract keypoints and scores
                keypoints = pred_instances.keypoints
                keypoint_scores = pred_instances.keypoint_scores
                
                logger.debug(f"Original keypoints shape: {keypoints.shape}")
                logger.debug(f"Original keypoint_scores shape: {keypoint_scores.shape}")
                
                # Convert tensors to numpy if needed
                if hasattr(keypoints, 'cpu'):
                    keypoints = keypoints.cpu().numpy()
                if hasattr(keypoint_scores, 'cpu'):
                    keypoint_scores = keypoint_scores.cpu().numpy()
                
                # Handle dimension squeezing (following demo lines 212-217)
                if keypoint_scores.ndim == 3:
                    keypoint_scores = np.squeeze(keypoint_scores, axis=1)
                    pose_result.pred_instances.keypoint_scores = keypoint_scores
                    logger.debug(f"Squeezed keypoint_scores: {keypoint_scores.shape}")
                    
                if keypoints.ndim == 4:
                    keypoints = np.squeeze(keypoints, axis=1)
                    logger.debug(f"Squeezed keypoints: {keypoints.shape}")
                
                # Apply the exact coordinate transformation from demo (line 219)
                # This is essential for proper 3D visualization
                keypoints = -keypoints[..., [0, 2, 1]]

                logger.debug(f"Transformed keypoints shape: {keypoints.shape}")
                logger.debug(f"Transformed coordinate ranges - X: [{keypoints[..., 0].min():.1f}, {keypoints[..., 0].max():.1f}], Y: [{keypoints[..., 1].min():.1f}, {keypoints[..., 1].max():.1f}], Z: [{keypoints[..., 2].min():.3f}, {keypoints[..., 2].max():.3f}]")

                # Rebase Z-axis to ground level (following demo lines 222-224)
                # Always rebase for better visualization (demo default behavior)
                if keypoints.shape[-1] >= 3:
                    keypoints[..., 2] -= np.min(keypoints[..., 2], axis=-1, keepdims=True)
                    logger.debug(f"Rebased keypoints z-range: [{np.min(keypoints[..., 2]):.3f}, {np.max(keypoints[..., 2]):.3f}]")
                
                # Update the pose result with transformed keypoints
                pose_result.pred_instances.keypoints = keypoints
                processed_results.append(pose_result)
                
                logger.debug(f"Successfully processed pose result {idx}")
            
            if not processed_results:
                logger.warning("No valid pose results after processing")
                return None
            
            # Sort by track_id (following demo lines 228-229)
            def get_track_id(x):
                if isinstance(x, dict):
                    return x.get('track_id', 1e4)
                else:
                    return getattr(x, 'track_id', 1e4)
            
            processed_results = sorted(processed_results, key=get_track_id)
            logger.debug(f"Sorted {len(processed_results)} processed results by track_id")
            
            # Merge data samples (following demo line 231)
            merged_samples = merge_data_samples(processed_results)
            logger.debug(f"Successfully merged pose data samples: {type(merged_samples)}")
            
            return merged_samples
            
        except Exception as e:
            logger.error(f"Failed to create pose data samples: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def process_and_visualize(self,
                            frame: np.ndarray,
                            pose_results: List,
                            save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Complete pipeline: process pose results and create visualization.
        
        Args:
            frame: Input image frame
            pose_results: Raw results from inference_topdown
            save_path: Optional path to save result
            
        Returns:
            Visualization image with side-by-side 2D and 3D views
        """
        if not pose_results or len(pose_results) == 0:
            logger.warning("No pose results to visualize")
            return frame
        
        # Convert to MMPose data samples
        pose_data_samples = self.create_pose_data_samples(pose_results)
        if pose_data_samples is None:
            logger.warning("Failed to create pose data samples")
            return frame
        
        # Get dataset name from pose estimator
        dataset_name = "coco"  # Default
        if self.pose_estimator and hasattr(self.pose_estimator, 'dataset_meta'):
            dataset_name = self.pose_estimator.dataset_meta.get('dataset_name', 'coco')
        
        # Visualize
        vis_result = self.visualize_frame(
            frame=frame,
            pose_data_samples=pose_data_samples,
            dataset_name=dataset_name,
            num_instances=1,
            save_path=save_path
        )
        
        return vis_result if vis_result is not None else frame