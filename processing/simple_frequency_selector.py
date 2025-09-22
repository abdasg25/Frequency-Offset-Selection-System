#!/usr/bin/env python3
"""
Simple, effective frequency selection based on intensity statistics and uniformity.
"""

import os
import sys
import numpy as np
import torch
from typing import Tuple, List, Union

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SimpleFrequencySelector:
    """
    Simple frequency offset selector based on image quality metrics.
    """
    
    def __init__(self):
        self.roi_threshold = 0.5
    
    def select_optimal_frequency(
        self, 
        frequency_series: Union[np.ndarray, torch.Tensor],
        heart_mask: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[int, dict]:
        """
        Select optimal frequency using simple image quality metrics.
        
        Args:
            frequency_series: Complete frequency series
            heart_mask: Heart ROI mask
            
        Returns:
            optimal_index: Index of optimal frequency offset
            analysis_results: Dictionary with analysis results
        """
        # Convert tensors to numpy if needed
        if torch.is_tensor(frequency_series):
            frequency_series = frequency_series.cpu().numpy()
        if torch.is_tensor(heart_mask):
            heart_mask = heart_mask.cpu().numpy()
        
        n_frequencies = frequency_series.shape[0]
        quality_scores = []
        
        # Analyze each frequency offset
        for i in range(n_frequencies):
            image = frequency_series[i]
            score = self._compute_image_quality_score(image, heart_mask)
            quality_scores.append(score)
        
        # Select optimal frequency (maximum quality score)
        optimal_index = int(np.argmax(quality_scores))
        
        # Compile results
        analysis_results = {
            'optimal_frequency_index': optimal_index,
            'quality_scores': quality_scores
        }
        
        return optimal_index, analysis_results
    
    def _compute_image_quality_score(self, image, heart_mask):
        """
        Compute image quality score based on multiple metrics.
        Higher score indicates better quality (less artifacts).
        """
        # Focus on heart region only
        roi_pixels = image[heart_mask > self.roi_threshold]
        
        if len(roi_pixels) < 100:
            return 0.0
        
        # Metric 1: Inverse of coefficient of variation (lower CV = more uniform = better)
        mean_intensity = np.mean(roi_pixels)
        std_intensity = np.std(roi_pixels)
        
        if mean_intensity > 0:
            cv = std_intensity / mean_intensity
            uniformity_score = 1.0 / (1.0 + cv)
        else:
            uniformity_score = 0.0
        
        # Metric 2: Edge consistency (gradient-based)
        edge_score = self._compute_edge_consistency_score(image, heart_mask)
        
        # Metric 3: Local contrast stability
        contrast_score = self._compute_contrast_stability_score(image, heart_mask)
        
        # Metric 4: Signal-to-noise estimation
        snr_score = self._compute_snr_score(roi_pixels)
        
        # Combine metrics
        total_score = (
            0.3 * uniformity_score +
            0.3 * edge_score +
            0.2 * contrast_score +
            0.2 * snr_score
        )
        
        return total_score
    
    def _compute_edge_consistency_score(self, image, heart_mask):
        """Compute edge consistency within ROI."""
        # Compute gradients
        grad_y, grad_x = np.gradient(image)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Apply heart mask
        roi_gradients = gradient_magnitude[heart_mask > self.roi_threshold]
        
        if len(roi_gradients) == 0:
            return 0.0
        
        # Consistent edges have predictable gradient distribution
        # Less variation in gradient = more consistent edges
        gradient_cv = np.std(roi_gradients) / (np.mean(roi_gradients) + 1e-8)
        edge_consistency = 1.0 / (1.0 + gradient_cv)
        
        return edge_consistency
    
    def _compute_contrast_stability_score(self, image, heart_mask):
        """Compute local contrast stability."""
        # Use sliding window to compute local standard deviation
        from scipy import ndimage
        
        # Local standard deviation with 3x3 kernel
        local_std = ndimage.generic_filter(image, np.std, size=3)
        
        # Apply heart mask
        roi_local_std = local_std[heart_mask > self.roi_threshold]
        
        if len(roi_local_std) == 0:
            return 0.0
        
        # Stable contrast means consistent local standard deviation
        contrast_variation = np.std(roi_local_std)
        mean_contrast = np.mean(roi_local_std)
        
        if mean_contrast > 0:
            contrast_stability = 1.0 / (1.0 + contrast_variation / mean_contrast)
        else:
            contrast_stability = 0.0
        
        return contrast_stability
    
    def _compute_snr_score(self, roi_pixels):
        """Estimate signal-to-noise ratio."""
        if len(roi_pixels) < 10:
            return 0.0
        
        # Estimate signal as mean intensity
        signal = np.mean(roi_pixels)
        
        # Estimate noise as standard deviation of high-frequency components
        # Simple high-pass: difference from local mean
        local_mean = np.median(roi_pixels)  # Robust estimate
        noise_estimate = np.std(roi_pixels - local_mean)
        
        if noise_estimate > 0:
            snr = signal / noise_estimate
            # Normalize SNR score
            snr_score = snr / (snr + 10.0)  # Sigmoid-like normalization
        else:
            snr_score = 1.0
        
        return snr_score