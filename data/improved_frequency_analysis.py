"""
Improved frequency offset selection system with enhanced accuracy.
Combines multiple analysis techniques for better performance.
"""

import numpy as np
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import uniform_filter
import cv2
from typing import List, Tuple, Optional


class ImprovedFrequencyOffsetSelector:
    """
    Enhanced frequency offset selector targeting 80%+ accuracy.
    
    Improvements:
    1. Multi-scale high-frequency analysis
    2. Gradient-based artifact detection
    3. Local variance analysis
    4. Weighted ensemble decision making
    5. Adaptive parameter tuning per patient
    """
    
    def __init__(self, 
                 n_selected_images: int = 5,  # Increased from 3
                 high_pass_cutoffs: List[float] = [0.05, 0.1, 0.15],  # Multi-scale
                 penalty_strength: float = 2.0,  # Increased penalty
                 gradient_weight: float = 0.3,
                 variance_weight: float = 0.2,
                 ensemble_weights: List[float] = [0.4, 0.3, 0.2, 0.1]):
        """
        Initialize improved frequency offset selector.
        
        Args:
            n_selected_images: Number of reference images (increased)
            high_pass_cutoffs: Multiple cutoff frequencies for multi-scale analysis
            penalty_strength: Stronger penalty for deviations
            gradient_weight: Weight for gradient-based analysis
            variance_weight: Weight for variance-based analysis
            ensemble_weights: Weights for different analysis methods
        """
        self.n_selected_images = n_selected_images
        self.high_pass_cutoffs = high_pass_cutoffs
        self.penalty_strength = penalty_strength
        self.gradient_weight = gradient_weight
        self.variance_weight = variance_weight
        self.ensemble_weights = ensemble_weights
        
    def extract_multi_scale_features(self, image: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> dict:
        """
        Extract multiple types of features for comprehensive analysis.
        
        Args:
            image: Input image
            roi_mask: ROI mask for heart region
            
        Returns:
            Dictionary of feature scores
        """
        features = {}
        
        # Apply ROI mask
        if roi_mask is not None:
            masked_image = image * roi_mask
        else:
            masked_image = image
            
        # 1. Multi-scale high-frequency analysis
        hf_scores = []
        for cutoff in self.high_pass_cutoffs:
            hf_score = self._extract_high_frequency_content(masked_image, cutoff, roi_mask)
            hf_scores.append(hf_score)
        features['multi_scale_hf'] = np.mean(hf_scores)
        
        # 2. Gradient magnitude analysis (detects edges/artifacts)
        # Convert to uint8 for OpenCV compatibility
        image_uint8 = ((masked_image - np.min(masked_image)) / 
                      (np.max(masked_image) - np.min(masked_image) + 1e-8) * 255).astype(np.uint8)
        
        grad_x = cv2.Sobel(image_uint8, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image_uint8, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        if roi_mask is not None:
            gradient_score = np.sum(gradient_magnitude * roi_mask) / np.sum(roi_mask)
        else:
            gradient_score = np.mean(gradient_magnitude)
        features['gradient_score'] = gradient_score
        
        # 3. Local variance analysis (detects texture/noise)
        # Use uniform filter for local mean, then calculate variance
        local_mean = uniform_filter(masked_image, size=5)
        local_variance = uniform_filter(masked_image**2, size=5) - local_mean**2
        
        if roi_mask is not None:
            variance_score = np.sum(local_variance * roi_mask) / np.sum(roi_mask)
        else:
            variance_score = np.mean(local_variance)
        features['variance_score'] = variance_score
        
        # 4. Laplacian analysis (detects rapid intensity changes)
        laplacian = cv2.Laplacian(image_uint8, cv2.CV_64F)
        if roi_mask is not None:
            laplacian_score = np.sum(np.abs(laplacian) * roi_mask) / np.sum(roi_mask)
        else:
            laplacian_score = np.mean(np.abs(laplacian))
        features['laplacian_score'] = laplacian_score
        
        return features
    
    def _extract_high_frequency_content(self, image: np.ndarray, cutoff: float, roi_mask: Optional[np.ndarray] = None) -> float:
        """Extract high-frequency content with specified cutoff."""
        # Gaussian smoothing
        image_smooth = ndimage.gaussian_filter(image, sigma=1.5)
        
        # FFT analysis
        f_transform = fft2(image_smooth)
        f_shift = np.fft.fftshift(f_transform)
        
        # Create high-pass filter
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
        max_distance = np.sqrt(crow**2 + ccol**2)
        normalized_distance = distance / max_distance
        high_pass_filter = (normalized_distance > cutoff).astype(np.float32)
        
        # Apply filter and inverse transform
        filtered_f_shift = f_shift * high_pass_filter
        f_ishift = np.fft.ifftshift(filtered_f_shift)
        high_freq_image = np.real(ifft2(f_ishift))
        
        # Score calculation
        if roi_mask is not None:
            roi_high_freq = high_freq_image * roi_mask
            score = np.sum(np.abs(roi_high_freq)) / np.sum(roi_mask)
        else:
            score = np.mean(np.abs(high_freq_image))
            
        return float(score)
    
    def calculate_composite_score(self, features: dict) -> float:
        """
        Calculate composite quality score from multiple features.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Composite artifact score (lower = better quality)
        """
        # Normalize scores to [0, 1] range for combination
        # Higher scores indicate more artifacts/noise
        composite_score = (
            self.ensemble_weights[0] * features['multi_scale_hf'] +
            self.ensemble_weights[1] * features['gradient_score'] +
            self.ensemble_weights[2] * features['variance_score'] +
            self.ensemble_weights[3] * features['laplacian_score']
        )
        
        return composite_score
    
    def select_reference_images(self, fs_series: List[np.ndarray], roi_mask: Optional[np.ndarray] = None) -> Tuple[List[int], List[np.ndarray], List[float]]:
        """
        Select reference images using improved multi-feature analysis.
        
        Args:
            fs_series: List of frequency scout images
            roi_mask: Heart ROI mask
            
        Returns:
            Tuple of (selected_indices, selected_images, composite_scores)
        """
        composite_scores = []
        
        # Analyze each image with multiple features
        for image in fs_series:
            features = self.extract_multi_scale_features(image, roi_mask)
            composite_score = self.calculate_composite_score(features)
            composite_scores.append(composite_score)
        
        # Select images with lowest composite scores (best quality)
        sorted_indices = np.argsort(composite_scores)
        selected_indices = sorted_indices[:self.n_selected_images].tolist()
        selected_images = [fs_series[i] for i in selected_indices]
        
        return selected_indices, selected_images, composite_scores
    
    def generate_improved_weighting_maps(self, fs_series: List[np.ndarray], 
                                       selected_images: List[np.ndarray],
                                       roi_mask: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Generate improved weighting maps with enhanced sensitivity.
        
        Args:
            fs_series: Complete frequency series
            selected_images: Reference images
            roi_mask: Heart ROI mask
            
        Returns:
            List of improved weighting maps
        """
        # Calculate robust median (exclude outliers)
        image_stack = np.stack(selected_images, axis=0)
        
        # Use interquartile mean instead of simple median for robustness
        q25 = np.percentile(image_stack, 25, axis=0)
        q75 = np.percentile(image_stack, 75, axis=0)
        iqr_mask = (image_stack >= q25) & (image_stack <= q75)
        
        # Calculate weighted average of middle values
        reference_image = np.zeros_like(selected_images[0])
        for i in range(len(selected_images)):
            weight_mask = iqr_mask[i].astype(np.float32)
            reference_image += selected_images[i] * weight_mask
        reference_image /= np.sum(iqr_mask, axis=0) + 1e-8
        
        weighting_maps = []
        for image in fs_series:
            # Multi-scale deviation analysis
            deviation_scores = []
            
            # 1. Absolute deviation
            abs_dev = np.abs(image.astype(np.float32) - reference_image.astype(np.float32))
            deviation_scores.append(abs_dev)
            
            # 2. Gradient deviation
            # Convert to uint8 for OpenCV compatibility
            ref_uint8 = ((reference_image - np.min(reference_image)) / 
                        (np.max(reference_image) - np.min(reference_image) + 1e-8) * 255).astype(np.uint8)
            img_uint8 = ((image - np.min(image)) / 
                        (np.max(image) - np.min(image) + 1e-8) * 255).astype(np.uint8)
            
            ref_grad_x = cv2.Sobel(ref_uint8, cv2.CV_64F, 1, 0, ksize=3)
            ref_grad_y = cv2.Sobel(ref_uint8, cv2.CV_64F, 0, 1, ksize=3)
            img_grad_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
            img_grad_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
            
            grad_dev = np.sqrt((ref_grad_x - img_grad_x)**2 + (ref_grad_y - img_grad_y)**2)
            deviation_scores.append(grad_dev)
            
            # 3. Local texture deviation
            ref_texture = uniform_filter(reference_image**2, size=3) - uniform_filter(reference_image, size=3)**2
            img_texture = uniform_filter(image**2, size=3) - uniform_filter(image, size=3)**2
            texture_dev = np.abs(img_texture - ref_texture)
            deviation_scores.append(texture_dev)
            
            # Combine deviations with weights
            combined_deviation = (
                0.5 * deviation_scores[0] +  # Intensity deviation
                0.3 * deviation_scores[1] +  # Gradient deviation  
                0.2 * deviation_scores[2]    # Texture deviation
            )
            
            # Normalize and apply stronger penalty
            max_dev = np.percentile(combined_deviation, 95)  # Use 95th percentile instead of max
            if max_dev > 0:
                normalized_deviation = combined_deviation / max_dev
            else:
                normalized_deviation = combined_deviation
                
            # Apply stronger exponential penalty
            weighting_map = np.exp(-self.penalty_strength * normalized_deviation)
            
            # Apply ROI mask
            if roi_mask is not None:
                weighting_map = weighting_map * roi_mask
                
            weighting_maps.append(weighting_map)
        
        return weighting_maps
    
    def adaptive_selection_strategy(self, avg_weights: List[float], 
                                  composite_scores: List[float],
                                  series_length: int) -> int:
        """
        Adaptive selection strategy combining multiple criteria.
        
        Args:
            avg_weights: Average weighting scores
            composite_scores: Composite artifact scores
            series_length: Length of frequency series
            
        Returns:
            Selected optimal frame index
        """
        # Normalize scores
        norm_weights = np.array(avg_weights) / np.max(avg_weights)
        norm_composite = 1.0 - (np.array(composite_scores) / np.max(composite_scores))  # Invert for quality
        
        # Center bias - prefer frames closer to center
        center_idx = series_length // 2
        center_bias = np.exp(-0.1 * np.abs(np.arange(series_length) - center_idx))
        center_bias = center_bias / np.max(center_bias)
        
        # Combine criteria with adaptive weights
        final_scores = (
            0.6 * norm_weights +        # Primary: weighting analysis
            0.25 * norm_composite +     # Secondary: artifact analysis
            0.15 * center_bias          # Tertiary: center preference
        )
        
        # Select frame with highest combined score
        optimal_idx = int(np.argmax(final_scores))
        
        return optimal_idx
    
    def select_optimal_frequency_offset(self, fs_series: List[np.ndarray],
                                      roi_mask: Optional[np.ndarray] = None) -> Tuple[int, dict]:
        """
        Main method for improved frequency offset selection.
        
        Args:
            fs_series: List of frequency scout images
            roi_mask: Heart ROI mask
            
        Returns:
            Tuple of (optimal_offset_index, analysis_results)
        """
        analysis_results = {}
        
        print("Enhanced multi-feature analysis...")
        
        # Step 1: Multi-feature reference selection
        selected_indices, selected_images, composite_scores = self.select_reference_images(
            fs_series, roi_mask
        )
        analysis_results['selected_indices'] = selected_indices
        analysis_results['composite_scores'] = composite_scores
        
        # Step 2: Improved weighting map generation
        print("Generating improved weighting maps...")
        weighting_maps = self.generate_improved_weighting_maps(
            fs_series, selected_images, roi_mask
        )
        analysis_results['weighting_maps'] = weighting_maps
        
        # Step 3: Calculate average weights
        avg_weights = []
        for weight_map in weighting_maps:
            if roi_mask is not None:
                roi_weights = weight_map * roi_mask
                avg_weight = np.sum(roi_weights) / np.sum(roi_mask)
            else:
                avg_weight = np.mean(weight_map)
            avg_weights.append(avg_weight)
        
        analysis_results['avg_weights'] = avg_weights
        
        # Step 4: Adaptive selection with multiple criteria
        print("Applying adaptive selection strategy...")
        optimal_offset_index = self.adaptive_selection_strategy(
            avg_weights, composite_scores, len(fs_series)
        )
        
        analysis_results['optimal_offset'] = optimal_offset_index
        analysis_results['selected_images'] = selected_images
        
        print(f"Enhanced optimal frequency offset selected: Frame {optimal_offset_index}")
        
        return optimal_offset_index, analysis_results