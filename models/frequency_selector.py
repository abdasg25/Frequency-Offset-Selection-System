#!/usr/bin/env python3
"""
Machine Learning-based frequency selector using CNN architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class FrequencySelectionCNN(nn.Module):
    """
    CNN-based frequency selector that processes frequency series to predict optimal frequency index.
    
    Architecture:
    - Input: Frequency series (N_freq, H, W) + Heart mask (H, W)
    - Feature Extraction: 3D CNN for temporal-spatial features
    - Attention: Channel and spatial attention mechanisms
    - Output: Both classification (discrete index) and regression (continuous index)
    """
    
    def __init__(
        self,
        max_frequencies: int = 15,
        image_size: Tuple[int, int] = (256, 256),
        dropout_rate: float = 0.3,
        use_heart_mask: bool = True
    ):
        """
        Initialize the frequency selection CNN.
        
        Args:
            max_frequencies: Maximum number of frequencies in series
            image_size: Input image dimensions (H, W)
            dropout_rate: Dropout probability
            use_heart_mask: Whether to use heart mask as additional input
        """
        super(FrequencySelectionCNN, self).__init__()
        
        self.max_frequencies = max_frequencies
        self.image_size = image_size
        self.dropout_rate = dropout_rate
        self.use_heart_mask = use_heart_mask
        
        # Input channels: frequency series + optional heart mask
        input_channels = 1 + (1 if use_heart_mask else 0)
        
        # 3D CNN for frequency-spatial feature extraction
        self.freq_conv1 = nn.Conv3d(input_channels, 32, kernel_size=(3, 7, 7), padding=(1, 3, 3))
        self.freq_conv2 = nn.Conv3d(32, 64, kernel_size=(3, 5, 5), padding=(1, 2, 2))
        self.freq_conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        
        # Max pooling layers
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        # Attention mechanism
        self.channel_attention = ChannelAttention(128)
        self.spatial_attention = SpatialAttention()
        
        # Global average pooling to handle variable frequency series lengths
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classification head (discrete frequency index)
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, max_frequencies)  # Output logits for each possible frequency
        )
        
        # Regression head (continuous frequency index)
        self.regressor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)  # Output continuous index
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, frequency_series: torch.Tensor, heart_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the frequency selection CNN.
        
        Args:
            frequency_series: Input frequency series (batch_size, n_frequencies, H, W)
            heart_mask: Heart segmentation mask (batch_size, H, W)
            
        Returns:
            Dictionary containing classification logits and regression output
        """
        batch_size, n_frequencies, H, W = frequency_series.shape
        
        # Prepare input
        if self.use_heart_mask and heart_mask is not None:
            # Expand heart mask to match frequency series dimensions
            heart_mask_expanded = heart_mask.unsqueeze(1).expand(-1, n_frequencies, -1, -1)
            # Concatenate along channel dimension
            x = torch.cat([frequency_series.unsqueeze(1), heart_mask_expanded.unsqueeze(1)], dim=1)
        else:
            x = frequency_series.unsqueeze(1)  # Add channel dimension
        
        # Transpose to (batch_size, channels, n_frequencies, H, W) for 3D conv
        x = x.transpose(1, 2)  # (batch_size, n_frequencies, channels, H, W)
        x = x.transpose(1, 2)  # (batch_size, channels, n_frequencies, H, W)
        
        # 3D CNN feature extraction
        x = F.relu(self.bn1(self.freq_conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.freq_conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.freq_conv3(x)))
        x = self.pool3(x)
        
        # Apply attention mechanisms
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        
        # Global average pooling
        x = self.global_avg_pool(x)  # (batch_size, 128, 1, 1, 1)
        x = x.view(batch_size, -1)   # (batch_size, 128)
        
        # Prediction heads
        classification_logits = self.classifier(x)  # (batch_size, max_frequencies)
        regression_output = self.regressor(x)       # (batch_size, 1)
        
        return {
            'classification_logits': classification_logits,
            'regression_output': regression_output.squeeze(-1),  # (batch_size,)
            'features': x
        }
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ChannelAttention(nn.Module):
    """Channel attention mechanism."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
    def forward(self, x):
        b, c, d, h, w = x.size()
        
        # Average pooling path
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_out).view(b, c, 1, 1, 1)
        
        # Max pooling path
        max_out = self.max_pool(x).view(b, c)
        max_out = self.fc(max_out).view(b, c, 1, 1, 1)
        
        # Combine and apply sigmoid
        out = avg_out + max_out
        return torch.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention mechanism."""
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        
    def forward(self, x):
        # Channel-wise average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(x_cat)
        
        return torch.sigmoid(attention)


class MLFrequencySelector:
    """
    Machine Learning-based frequency selector using the FrequencySelectionCNN.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        use_heart_mask: bool = True,
        ensemble_voting: bool = False
    ):
        """
        Initialize the ML frequency selector.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            use_heart_mask: Whether to use heart mask as input
            ensemble_voting: Whether to use ensemble prediction
        """
        self.device = device
        self.use_heart_mask = use_heart_mask
        self.ensemble_voting = ensemble_voting
        
        # Load trained model
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _load_model(self, model_path: str) -> FrequencySelectionCNN:
        """Load trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model parameters from checkpoint
        model_config = checkpoint.get('model_config', {})
        model = FrequencySelectionCNN(**model_config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
        
    def select_optimal_frequency(
        self,
        frequency_series: np.ndarray,
        heart_mask: Optional[np.ndarray] = None
    ) -> Tuple[int, Dict]:
        """
        Select optimal frequency using ML model.
        
        Args:
            frequency_series: Frequency series array (n_frequencies, H, W)
            heart_mask: Heart mask array (H, W)
            
        Returns:
            optimal_index: Predicted optimal frequency index
            analysis_results: Dictionary with prediction details
        """
        # Convert to tensors
        freq_tensor = torch.from_numpy(frequency_series).float().unsqueeze(0).to(self.device)
        
        if heart_mask is not None and self.use_heart_mask:
            mask_tensor = torch.from_numpy(heart_mask).float().unsqueeze(0).to(self.device)
        else:
            mask_tensor = None
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(freq_tensor, mask_tensor)
            
            # Get predictions
            classification_logits = outputs['classification_logits'][0]  # Remove batch dim
            regression_output = outputs['regression_output'][0].item()
            
            # Classification prediction (discrete)
            class_probs = F.softmax(classification_logits, dim=0)
            class_prediction = torch.argmax(class_probs).item()
            
            # Regression prediction (continuous, then round)
            regression_prediction = max(0, min(len(frequency_series) - 1, int(round(regression_output))))
            
            # Ensemble prediction (combine both approaches)
            if self.ensemble_voting:
                # Weight classification and regression predictions
                optimal_index = int(round(0.7 * class_prediction + 0.3 * regression_prediction))
            else:
                # Use classification prediction as primary
                optimal_index = class_prediction
            
            # Ensure valid index
            optimal_index = max(0, min(len(frequency_series) - 1, optimal_index))
        
        # Compile analysis results
        analysis_results = {
            'optimal_frequency_index': optimal_index,
            'classification_prediction': class_prediction,
            'regression_prediction': regression_prediction,
            'classification_probabilities': class_probs.cpu().numpy().tolist(),
            'regression_raw_output': regression_output,
            'model_confidence': float(torch.max(class_probs).item())
        }
        
        return optimal_index, analysis_results


def create_ml_frequency_selector(
    model_path: str,
    device: str = 'cuda',
    use_heart_mask: bool = True
) -> MLFrequencySelector:
    """
    Factory function to create ML frequency selector.
    
    Args:
        model_path: Path to trained model
        device: Device for inference
        use_heart_mask: Whether to use heart mask
        
    Returns:
        MLFrequencySelector instance
    """
    return MLFrequencySelector(
        model_path=model_path,
        device=device,
        use_heart_mask=use_heart_mask
    )
