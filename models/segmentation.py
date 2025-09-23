import torch
import torch.nn as nn
from monai.networks.nets import AttentionUnet
from typing import Tuple, Optional
import os
import sys

# Import MONAI adapter and EMIDEC adapter
try:
    from .monai_adapter import MONAISegmentationAdapter
    MONAI_ADAPTER_AVAILABLE = True
except ImportError:
    MONAI_ADAPTER_AVAILABLE = False

try:
    from .emidec_adapter import create_emidec_segmentation_model
    EMIDEC_ADAPTER_AVAILABLE = True
except ImportError:
    EMIDEC_ADAPTER_AVAILABLE = False

class HeartSegmentationModel(nn.Module):
    """
    Heart segmentation model using MONAI AttentionUNet.
    
    This model is specifically designed for whole heart segmentation on 
    frequency scout cardiac MRI images to localize the ROI where artifacts 
    should be minimized.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,  # Background + Heart
        spatial_dims: int = 2,
        channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
        strides: Tuple[int, ...] = (2, 2, 2, 2),
        kernel_size: int = 3,
        up_kernel_size: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize the heart segmentation model.
        
        Args:
            in_channels: Number of input channels (1 for grayscale MRI)
            out_channels: Number of output classes (2 for binary segmentation)
            spatial_dims: Number of spatial dimensions (2 for 2D images)
            channels: Feature channels for each encoder level
            strides: Stride values for each encoder level
            kernel_size: Convolution kernel size
            up_kernel_size: Upsampling kernel size
            dropout: Dropout probability
        """
        super().__init__()
        
        self.model = AttentionUnet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size,
            dropout=dropout
        )
        
        # Add sigmoid activation for binary segmentation
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the segmentation model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Segmentation logits of shape (batch_size, out_channels, height, width)
        """
        logits = self.model(x)
        return logits
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Generate binary predictions from input images.
        
        Args:
            x: Input tensor
            threshold: Threshold for binary prediction
            
        Returns:
            Binary mask predictions
        """
        with torch.no_grad():
            logits = self.forward(x)
            # Take the heart class (channel 1) and apply sigmoid + threshold
            probs = torch.softmax(logits, dim=1)
            heart_probs = probs[:, 1:2]  # Keep channel dimension
            predictions = (heart_probs > threshold).float()
            return predictions
    
    def get_heart_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Extract heart region mask for ROI analysis.
        
        Args:
            x: Input tensor
            threshold: Threshold for mask generation
            
        Returns:
            Binary heart mask
        """
        return self.predict(x, threshold)


def create_segmentation_model(
    in_channels: int = 1,
    out_channels: int = 2,
    pretrained: bool = False,
    checkpoint_path: str = None
) -> HeartSegmentationModel:
    """
    Factory function to create heart segmentation model.
    Prioritizes EMIDEC-trained model, then MONAI adapter if available.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output classes
        pretrained: Whether to load pretrained weights
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Initialized segmentation model (EMIDEC adapter, MONAI adapter, or standard model)
    """
    
    # First priority: Use EMIDEC-trained model if checkpoint exists
    if pretrained and checkpoint_path and os.path.exists(checkpoint_path):
        if EMIDEC_ADAPTER_AVAILABLE:
            try:
                print(f"✅ Using EMIDEC-trained segmentation model from: {checkpoint_path}")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                return create_emidec_segmentation_model(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    emidec_checkpoint_path=checkpoint_path,
                    device=device
                )
            except Exception as e:
                print(f"⚠️ Failed to load EMIDEC adapter: {str(e)}")
                print("Falling back to standard model...")
    
    # Second priority: Try to use MONAI model from Comparative Analysis project
    if MONAI_ADAPTER_AVAILABLE:
        try:
            
            # Define possible MONAI model paths
            monai_paths = [
                "/kaggle/working/AttentionUnet_best.pth",
                "/kaggle/working/AttentionUnet.pth",
                "/kaggle/working/checkpoints/AttentionUnet_best.pth",
                "/Users/abdulrehman/fyp/Comparative-Analysis-of-MONAI-Models-on-EMIDEC-Dataset/AttentionUnet_best.pth",
                "/Users/abdulrehman/fyp/Comparative-Analysis-of-MONAI-Models-on-EMIDEC-Dataset/outputs/AttentionUnet_best.pth",
                "/Users/abdulrehman/fyp/Comparative-Analysis-of-MONAI-Models-on-EMIDEC-Dataset/checkpoints/AttentionUnet_best.pth",
                "/Users/abdulrehman/fyp/Comparative-Analysis-of-MONAI-Models-on-EMIDEC-Dataset/saved_models/AttentionUnet_best.pth"
            ]
            
            # Check if any MONAI model exists
            monai_model_path = None
            for path in monai_paths:
                if os.path.exists(path):
                    monai_model_path = path
                    break
            
            if monai_model_path:
                print(f"✅ Found MONAI model at: {monai_model_path}")
                return MONAISegmentationAdapter(
                    model_name="AttentionUnet",
                    model_path=monai_model_path,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
            else:
                print("ℹ️ No pre-trained MONAI model found, using standard model")
                
        except Exception as e:
            print(f"⚠️ MONAI adapter failed: {str(e)}")
    
    # Fallback: Use standard model
    print("🔄 Using standard HeartSegmentationModel")
    model = HeartSegmentationModel(
        in_channels=in_channels,
        out_channels=out_channels
    )
    
    if pretrained and checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded pretrained model from {checkpoint_path}")
        except Exception as e:
            print("")
            # print(f"Error loading checkpoint: {e}")
    
    return model


class SegmentationLoss(nn.Module):
    """
    Combined loss function for heart segmentation.
    
    Combines Dice loss and Cross-Entropy loss for robust training.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.7,
        ce_weight: float = 0.3,
        smooth: float = 1e-6
    ):
        """
        Initialize the segmentation loss.
        
        Args:
            dice_weight: Weight for Dice loss component
            ce_weight: Weight for Cross-Entropy loss component
            smooth: Smoothing factor for Dice loss
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        
        self.ce_loss = nn.CrossEntropyLoss()
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss for binary segmentation.
        
        Args:
            pred: Predicted logits (batch_size, num_classes, height, width)
            target: Target mask (batch_size, height, width)
            
        Returns:
            Dice loss value
        """
        # Convert logits to probabilities
        pred_probs = torch.softmax(pred, dim=1)
        
        # Get heart class probabilities
        pred_heart = pred_probs[:, 1]  # Heart class
        
        # Convert target to binary heart mask
        target_heart = (target == 1).float()
        
        # Calculate Dice coefficient
        intersection = (pred_heart * target_heart).sum(dim=(1, 2))
        union = pred_heart.sum(dim=(1, 2)) + target_heart.sum(dim=(1, 2))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        
        return dice_loss
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            pred: Predicted logits
            target: Target mask
            
        Returns:
            Combined loss value
        """
        # Cross-entropy loss
        ce_loss = self.ce_loss(pred, target.long())
        
        # Dice loss
        dice_loss = self.dice_loss(pred, target)
        
        # Combined loss
        total_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        
        return total_loss


def calculate_dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate Dice score for evaluation.
    
    Args:
        pred: Predicted logits (batch_size, num_classes, height, width)
        target: Target mask (batch_size, height, width)
        threshold: Threshold for binary prediction
        
    Returns:
        Dice score
    """
    with torch.no_grad():
        # Convert predictions to binary
        pred_probs = torch.softmax(pred, dim=1)
        pred_binary = (pred_probs[:, 1] > threshold).float()
        
        # Convert target to binary
        target_binary = (target == 1).float()
        
        # Calculate Dice score
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum()
        
        if union == 0:
            return 1.0  # Perfect score if both masks are empty
        
        dice = (2.0 * intersection) / union
        return dice.item()


def calculate_iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate IoU (Intersection over Union) score for evaluation.
    
    Args:
        pred: Predicted logits
        target: Target mask
        threshold: Threshold for binary prediction
        
    Returns:
        IoU score
    """
    with torch.no_grad():
        # Convert predictions to binary
        pred_probs = torch.softmax(pred, dim=1)
        pred_binary = (pred_probs[:, 1] > threshold).float()
        
        # Convert target to binary
        target_binary = (target == 1).float()
        
        # Calculate IoU score
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection
        
        if union == 0:
            return 1.0  # Perfect score if both masks are empty
        
        iou = intersection / union
        return iou.item()