"""
EMIDEC Segmentation Model Adapter

This module provides functionality to adapt EMIDEC-trained 3D AttentionUNet models
for 2D frequency scout cardiac segmentation.
"""

import torch
import torch.nn as nn
from monai.networks.nets import AttentionUnet
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EMIDECSegmentationAdapter(nn.Module):
    """
    Adapter to use EMIDEC-trained 3D AttentionUNet for 2D frequency scout segmentation.
    
    This adapter:
    1. Loads EMIDEC 3D AttentionUNet weights
    2. Adapts the architecture for 2D input
    3. Provides heart mask generation for frequency analysis
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,  # Background + Heart for frequency scout
        emidec_checkpoint_path: Optional[str] = None
    ):
        super().__init__()
        
        # Create 2D AttentionUNet for frequency scout images
        self.attention_unet_2d = AttentionUnet(
            spatial_dims=2,  # 2D for frequency scout
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2)
        )
        
        # If EMIDEC checkpoint provided, adapt weights
        if emidec_checkpoint_path:
            self._adapt_emidec_weights(emidec_checkpoint_path)
        
        self.sigmoid = nn.Sigmoid()
        
    def _adapt_emidec_weights(self, checkpoint_path: str):
        """
        Adapt EMIDEC 3D AttentionUNet weights for 2D frequency scout segmentation.
        
        Strategy:
        1. Load EMIDEC 3D model checkpoint
        2. Extract compatible 2D weights from 3D convolution layers
        3. Initialize 2D model with adapted weights
        """
        try:
            logger.info(f"Loading EMIDEC checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract model state dict
            if 'model_state_dict' in checkpoint:
                emidec_state_dict = checkpoint['model_state_dict']
            else:
                emidec_state_dict = checkpoint
            
            # Create temporary 3D model to match EMIDEC architecture
            emidec_3d_model = AttentionUnet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,  # EMIDEC uses single output channel
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2)
            )
            
            # Load EMIDEC weights into 3D model
            emidec_3d_model.load_state_dict(emidec_state_dict, strict=False)
            
            # Extract and adapt weights for 2D model
            adapted_state_dict = self._extract_2d_weights_from_3d(
                emidec_3d_model.state_dict()
            )
            
            # Load adapted weights into 2D model
            missing_keys, unexpected_keys = self.attention_unet_2d.load_state_dict(
                adapted_state_dict, strict=False
            )
            
            logger.info("✅ Successfully adapted EMIDEC weights for 2D segmentation")
            if missing_keys:
                logger.info(f"Missing keys (will use random initialization): {len(missing_keys)}")
            if unexpected_keys:
                logger.info(f"Unexpected keys (ignored): {len(unexpected_keys)}")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to adapt EMIDEC weights: {str(e)}")
            logger.warning("Using random initialization for 2D segmentation model")
    
    def _extract_2d_weights_from_3d(self, state_dict_3d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract 2D weights from 3D convolution layers.
        
        Strategy: Take the middle slice of 3D kernels for 2D adaptation
        """
        adapted_state_dict = {}
        
        for key, tensor in state_dict_3d.items():
            if 'conv' in key and 'weight' in key and tensor.dim() == 5:
                # 3D conv weight: [out_channels, in_channels, d, h, w]
                # Take middle slice for 2D: [out_channels, in_channels, h, w]
                middle_idx = tensor.shape[2] // 2
                adapted_tensor = tensor[:, :, middle_idx, :, :].clone()
                adapted_state_dict[key] = adapted_tensor
                
            elif 'conv' in key and 'bias' in key:
                # Bias terms can be used directly
                adapted_state_dict[key] = tensor.clone()
                
            elif tensor.dim() <= 4:  # Non-conv layers (BN, etc.)
                adapted_state_dict[key] = tensor.clone()
        
        # Adapt final layer for 2-class output (background + heart)
        final_conv_keys = [k for k in adapted_state_dict.keys() 
                          if 'final' in k or 'out' in k]
        
        for key in final_conv_keys:
            if 'weight' in key:
                original_weight = adapted_state_dict[key]
                if original_weight.shape[0] == 1:  # EMIDEC single output
                    # Duplicate for background/heart binary segmentation
                    background_weight = -original_weight  # Invert for background
                    heart_weight = original_weight
                    adapted_state_dict[key] = torch.cat([background_weight, heart_weight], dim=0)
            elif 'bias' in key:
                original_bias = adapted_state_dict[key]
                if original_bias.shape[0] == 1:
                    background_bias = -original_bias
                    heart_bias = original_bias
                    adapted_state_dict[key] = torch.cat([background_bias, heart_bias], dim=0)
        
        return adapted_state_dict
    
    def forward(self, x):
        """Forward pass through 2D AttentionUNet."""
        return self.attention_unet_2d(x)
    
    def get_heart_mask(self, x, threshold: float = 0.5):
        """
        Generate heart mask from frequency scout image.
        
        Args:
            x: Input image tensor [B, C, H, W]
            threshold: Threshold for binary mask
            
        Returns:
            Heart mask tensor [B, 1, H, W]
        """
        with torch.no_grad():
            # Get segmentation output
            logits = self.forward(x)
            
            # Apply sigmoid and get heart channel (channel 1)
            probs = self.sigmoid(logits)
            heart_prob = probs[:, 1:2, :, :]  # Keep heart channel
            
            # Apply threshold
            heart_mask = (heart_prob > threshold).float()
            
            return heart_mask
    
    def predict_heart_segmentation(self, x):
        """
        Get full segmentation prediction.
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            Segmentation logits [B, 2, H, W]
        """
        return self.forward(x)

def create_emidec_segmentation_model(
    in_channels: int = 1,
    out_channels: int = 2,
    emidec_checkpoint_path: Optional[str] = None,
    device: torch.device = torch.device('cpu')
) -> EMIDECSegmentationAdapter:
    """
    Create EMIDEC-adapted segmentation model.
    
    Args:
        in_channels: Input channels (1 for grayscale)
        out_channels: Output channels (2 for background+heart)
        emidec_checkpoint_path: Path to EMIDEC checkpoint
        device: Device to load model on
        
    Returns:
        EMIDEC segmentation adapter model
    """
    model = EMIDECSegmentationAdapter(
        in_channels=in_channels,
        out_channels=out_channels,
        emidec_checkpoint_path=emidec_checkpoint_path
    )
    
    model = model.to(device)
    
    logger.info(f"✅ Created EMIDEC segmentation adapter on {device}")
    
    return model