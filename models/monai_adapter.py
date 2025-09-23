"""
MONAI Model Adapter for Frequency Offset Selection
Adapts 3D MONAI segmentation models for 2D frequency scout image analysis
"""

import torch
import torch.nn as nn
import numpy as np
from monai.networks.nets import UNet, AttentionUnet, SegResNet, DynUNet, UNETR
from scipy import ndimage
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MONAISegmentationAdapter(nn.Module):
    """
    Adapter to use 3D MONAI segmentation models for 2D heart region detection
    in frequency scout images.
    """
    
    def __init__(self, model_name="AttentionUnet", model_path=None, device="cpu"):
        super().__init__()
        self.model_name = model_name
        self.device = device
        
        # Load the MONAI model
        self.model_3d = self._load_monai_model(model_name, model_path)
        
        # Set to evaluation mode
        self.model_3d.eval()
        
        logger.info(f"Loaded MONAI {model_name} model for 2D adaptation")
    
    def _load_monai_model(self, model_name, model_path):
        """Load the appropriate MONAI model with weights."""
        
        # Define model architectures (matching your model_factory.py)
        if model_name == "UNet3D":
            model = UNet(
                spatial_dims=3, in_channels=1, out_channels=1,
                channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2
            )
        elif model_name == "AttentionUnet":
            model = AttentionUnet(
                spatial_dims=3, in_channels=1, out_channels=1,
                channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2)
            )
        elif model_name == "SegResNet":
            model = SegResNet(
                spatial_dims=3, in_channels=1, out_channels=1,
                init_filters=32, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1)
            )
        elif model_name == "DynUNet":
            model = DynUNet(
                spatial_dims=3, in_channels=1, out_channels=1,
                kernel_size=[3, 3, 3, 3, 3, 3],
                strides=[1, 2, 2, 2, 2, 2],
                upsample_kernel_size=[1, 2, 2, 2, 2],
                filters=[16, 32, 64, 128, 256, 320]
            )
        elif model_name == "UNETR":
            # For UNETR, we need the ROI size - using default cardiac size
            roi_size = (128, 128, 128)
            model = UNETR(
                in_channels=1, out_channels=1, img_size=roi_size,
                feature_size=16, hidden_size=768, mlp_dim=3072,
                num_heads=12, norm_name="instance", res_block=True,
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Load weights if path provided
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                logger.info(f"Loaded model weights from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load weights from {model_path}: {e}")
                logger.info("Using randomly initialized weights")
        else:
            logger.warning(f"Model path not found: {model_path}")
            logger.info("Using randomly initialized weights")
        
        return model.to(self.device)
    
    def get_heart_mask(self, image: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Generate heart mask from 2D frequency scout image using 3D MONAI model.
        
        Args:
            image: 2D image tensor (B, C, H, W) or (C, H, W)
            threshold: Threshold for binary mask generation
            
        Returns:
            Binary heart mask tensor
        """
        original_shape = image.shape
        
        # Ensure proper input format
        if len(image.shape) == 3:  # (C, H, W)
            image = image.unsqueeze(0)  # (1, C, H, W)
        
        batch_size, channels, height, width = image.shape
        
        # Convert 2D to 3D by replicating the slice
        # This simulates a single-slice 3D volume
        depth = 16  # Use multiple slices for better 3D processing
        image_3d = image.unsqueeze(2).repeat(1, 1, depth, 1, 1)  # (B, C, D, H, W)
        
        # Resize to model's expected input size if needed
        target_size = (128, 128, 128)  # Standard MONAI input size
        if image_3d.shape[2:] != target_size:
            image_3d = torch.nn.functional.interpolate(
                image_3d, size=target_size, mode='trilinear', align_corners=False
            )
        
        # Run inference
        with torch.no_grad():
            try:
                output_3d = self.model_3d(image_3d)
                
                # Take the middle slice as representative
                middle_slice_idx = output_3d.shape[2] // 2
                output_2d = output_3d[:, :, middle_slice_idx, :, :]
                
                # Resize back to original dimensions
                if output_2d.shape[2:] != (height, width):
                    output_2d = torch.nn.functional.interpolate(
                        output_2d, size=(height, width), mode='bilinear', align_corners=False
                    )
                
                # Apply sigmoid and threshold
                heart_mask = torch.sigmoid(output_2d) > threshold
                
                # Convert to float
                heart_mask = heart_mask.float()
                
                # Post-process the mask
                heart_mask = self._postprocess_mask(heart_mask)
                
            except Exception as e:
                logger.warning(f"MONAI model inference failed: {e}")
                # Fallback to simple heart detector
                heart_mask = self._simple_heart_fallback(image)
        
        # Restore original shape
        if len(original_shape) == 3:
            heart_mask = heart_mask.squeeze(0)
        
        return heart_mask
    
    def _postprocess_mask(self, mask_tensor):
        """Post-process the segmentation mask."""
        # Convert to numpy for processing
        mask_np = mask_tensor.squeeze().cpu().numpy()
        
        # Remove small components
        labeled_mask, num_features = ndimage.label(mask_np)
        if num_features > 1:
            # Keep only the largest component
            component_sizes = [(labeled_mask == i).sum() for i in range(1, num_features + 1)]
            largest_component = np.argmax(component_sizes) + 1
            mask_np = (labeled_mask == largest_component).astype(np.float32)
        
        # Morphological operations
        mask_np = ndimage.binary_closing(mask_np, structure=np.ones((5, 5)))
        mask_np = ndimage.binary_opening(mask_np, structure=np.ones((3, 3)))
        
        # Convert back to tensor
        mask_tensor = torch.from_numpy(mask_np.astype(np.float32))
        
        # Restore batch and channel dimensions
        if len(mask_tensor.shape) == 2:
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        
        return mask_tensor
    
    def _simple_heart_fallback(self, image):
        """Simple fallback heart detector if MONAI model fails."""
        img_np = image.squeeze().cpu().numpy()
        h, w = img_np.shape
        
        # Create center-focused elliptical mask
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        
        # Heart-shaped elliptical region
        radius_y = h * 0.3
        radius_x = w * 0.35
        
        elliptical_mask = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2 <= 1
        
        # Intensity-based refinement
        img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        intensity_mask = (img_norm > 0.2) & (img_norm < 0.8)
        
        # Combine masks
        combined_mask = elliptical_mask & intensity_mask
        
        # Convert to tensor
        mask_tensor = torch.from_numpy(combined_mask.astype(np.float32))
        
        # Add batch and channel dimensions
        if len(image.shape) == 4:
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        elif len(image.shape) == 3:
            mask_tensor = mask_tensor.unsqueeze(0)
        
        return mask_tensor


def create_segmentation_model(in_channels=1, out_channels=2, pretrained=False, checkpoint_path=None):
    """
    Factory function to create segmentation model compatible with frequency selection system.
    
    Args:
        in_channels: Input channels (ignored for MONAI models)
        out_channels: Output channels (ignored for MONAI models)
        pretrained: Whether to load pretrained weights
        checkpoint_path: Path to model checkpoint
    
    Returns:
        MONAISegmentationAdapter instance
    """
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Try to find MONAI model in common locations
    possible_paths = []
    
    if checkpoint_path:
        possible_paths.append(checkpoint_path)
    
    # Add common MONAI model locations
    base_paths = [
        "/kaggle/working/",
        "/Users/abdulrehman/fyp/Comparative-Analysis-of-MONAI-Models-on-EMIDEC-Dataset/",
        "/Users/abdulrehman/fyp/Comparative-Analysis-of-MONAI-Models-on-EMIDEC-Dataset/outputs/",
        "/Users/abdulrehman/fyp/Comparative-Analysis-of-MONAI-Models-on-EMIDEC-Dataset/checkpoints/",
        "/Users/abdulrehman/fyp/Comparative-Analysis-of-MONAI-Models-on-EMIDEC-Dataset/saved_models/",
    ]
    
    model_names = ["AttentionUnet", "AttentionUNet", "attention_unet"]
    extensions = [".pth", ".pt", "_best.pth", "_final.pth"]
    
    for base_path in base_paths:
        for model_name in model_names:
            for ext in extensions:
                possible_paths.append(os.path.join(base_path, f"{model_name}{ext}"))
    
    # Find the first existing model
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            logger.info(f"Found MONAI model at: {model_path}")
            break
    
    if not model_path:
        logger.info("Trying to use model without pretrained weights")
    
    # Create the adapter
    return MONAISegmentationAdapter(
        model_name="AttentionUnet",
        model_path=model_path,
        device=device
    )