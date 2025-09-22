"""
High-frequency content analysis for frequency scout images.
Implementation based on FFT analysis and high-pass filtering.
"""

import numpy as np
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftfreq
import cv2
from typing import List, Tuple, Optional


class HighFrequencyExtractor:
    """
    Extracts high-frequency components from frequency scout images using FFT analysis.
    
    Based on the methodology:
    1. Fourier transformation
    2. High-pass filtering
    3. Inverse Fourier transformation
    4. Subtraction over series
    """
    
    def __init__(self, high_pass_cutoff: float = 0.1, gaussian_sigma: float = 2.0):
        """
        Initialize the high-frequency extractor.
        
        Args:
            high_pass_cutoff: Cutoff frequency for high-pass filter (fraction of Nyquist)
            gaussian_sigma: Sigma for Gaussian smoothing before analysis
        """
        self.high_pass_cutoff = high_pass_cutoff
        self.gaussian_sigma = gaussian_sigma
    
    def create_high_pass_filter(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a high-pass filter in frequency domain.
        
        Args:
            shape: Shape of the image (height, width)
            
        Returns:
            High-pass filter mask
        """
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        
        # Create coordinate arrays
        y, x = np.ogrid[:rows, :cols]
        
        # Calculate distance from center
        distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
        
        # Normalize distance
        max_distance = np.sqrt(crow**2 + ccol**2)
        normalized_distance = distance / max_distance
        
        # Create high-pass filter (1 - low-pass)
        high_pass_filter = normalized_distance > self.high_pass_cutoff
        
        return high_pass_filter.astype(np.float32)
    
    def extract_high_frequency_content(self, image: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> float:
        """
        Extract high-frequency content from a single image.
        
        Args:
            image: Input image (2D array)
            roi_mask: Optional ROI mask to focus analysis on specific region
            
        Returns:
            High-frequency content score
        """
        # Normalize image
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Apply Gaussian smoothing to reduce noise
        if self.gaussian_sigma > 0:
            image = ndimage.gaussian_filter(image, sigma=self.gaussian_sigma)
        
        # Apply ROI mask if provided
        if roi_mask is not None:
            image = image * roi_mask
        
        # Fourier transformation
        f_transform = fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        
        # Create high-pass filter
        high_pass_filter = self.create_high_pass_filter(image.shape)
        
        # Apply high-pass filter
        filtered_f_shift = f_shift * high_pass_filter
        
        # Inverse Fourier transformation
        f_ishift = np.fft.ifftshift(filtered_f_shift)
        high_freq_image = np.real(ifft2(f_ishift))
        
        # Calculate high-frequency content score
        if roi_mask is not None:
            # Focus on ROI for scoring
            roi_high_freq = high_freq_image * roi_mask
            hf_score = np.sum(np.abs(roi_high_freq)) / np.sum(roi_mask)
        else:
            hf_score = np.mean(np.abs(high_freq_image))
        
        return float(hf_score)
    
    def process_frequency_series(self, 
                                fs_series: List[np.ndarray], 
                                roi_mask: Optional[np.ndarray] = None) -> List[float]:
        """
        Process entire frequency scout series to extract high-frequency content.
        
        Args:
            fs_series: List of frequency scout images
            roi_mask: Optional ROI mask (e.g., heart segmentation)
            
        Returns:
            List of high-frequency content scores for each image
        """
        hf_scores = []
        
        for i, image in enumerate(fs_series):
            score = self.extract_high_frequency_content(image, roi_mask)
            hf_scores.append(score)
        
        return hf_scores
    
    def select_lowest_hf_images(self, 
                               fs_series: List[np.ndarray], 
                               hf_scores: List[float], 
                               n_images: int = 3) -> Tuple[List[int], List[np.ndarray]]:
        """
        Select N images with lowest high-frequency content.
        
        Args:
            fs_series: List of frequency scout images
            hf_scores: High-frequency content scores
            n_images: Number of images to select
            
        Returns:
            Tuple of (selected_indices, selected_images)
        """
        # Get indices of N lowest scores
        sorted_indices = np.argsort(hf_scores)
        selected_indices = sorted_indices[:n_images].tolist()
        
        # Get corresponding images
        selected_images = [fs_series[i] for i in selected_indices]
        
        return selected_indices, selected_images


class AdaptiveWeightingMap:
    """
    Generates adaptive weighting maps based on pixel-wise median calculation.
    
    Based on the methodology:
    Penalizes signal deviations from pixel-wise median calculated from selected images.
    """
    
    def __init__(self, penalty_strength: float = 1.0):
        """
        Initialize adaptive weighting map generator.
        
        Args:
            penalty_strength: Strength of penalty for deviations from median
        """
        self.penalty_strength = penalty_strength
    
    def calculate_pixelwise_median(self, selected_images: List[np.ndarray]) -> np.ndarray:
        """
        Calculate pixel-wise median from selected low high-frequency images.
        
        Args:
            selected_images: List of selected images with low high-frequency content
            
        Returns:
            Pixel-wise median image
        """
        # Stack images along new axis
        image_stack = np.stack(selected_images, axis=0)
        
        # Calculate median along image axis
        median_image = np.median(image_stack, axis=0)
        
        return median_image
    
    def generate_weighting_map(self, 
                              image: np.ndarray, 
                              median_reference: np.ndarray,
                              roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate adaptive weighting map for a single image.
        
        Args:
            image: Input frequency scout image
            median_reference: Pixel-wise median reference
            roi_mask: Optional ROI mask to focus on heart region
            
        Returns:
            Adaptive weighting map
        """
        # Calculate absolute deviation from median
        deviation = np.abs(image.astype(np.float32) - median_reference.astype(np.float32))
        
        # Normalize deviation
        max_deviation = np.max(deviation)
        if max_deviation > 0:
            normalized_deviation = deviation / max_deviation
        else:
            normalized_deviation = deviation
        
        # Generate weighting map (higher weight for lower deviation)
        weighting_map = np.exp(-self.penalty_strength * normalized_deviation)
        
        # Apply ROI mask if provided
        if roi_mask is not None:
            weighting_map = weighting_map * roi_mask
        
        return weighting_map
    
    def process_series_weighting(self, 
                                fs_series: List[np.ndarray],
                                selected_images: List[np.ndarray],
                                roi_mask: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Generate weighting maps for entire frequency scout series.
        
        Args:
            fs_series: Complete frequency scout series
            selected_images: Selected images with low high-frequency content
            roi_mask: Optional ROI mask
            
        Returns:
            List of weighting maps for each image in series
        """
        # Calculate pixel-wise median from selected images
        median_reference = self.calculate_pixelwise_median(selected_images)
        
        # Generate weighting maps for all images
        weighting_maps = []
        for image in fs_series:
            weight_map = self.generate_weighting_map(image, median_reference, roi_mask)
            weighting_maps.append(weight_map)
        
        return weighting_maps


class FrequencyOffsetSelector:
    """
    Main class for selecting optimal frequency offset based on research methodology.
    
    Combines high-frequency analysis and adaptive weighting to select optimal offset.
    """
    
    def __init__(self, 
                 n_selected_images: int = 3,
                 high_pass_cutoff: float = 0.1,
                 penalty_strength: float = 1.0):
        """
        Initialize frequency offset selector.
        
        Args:
            n_selected_images: Number of low high-frequency images to select
            high_pass_cutoff: High-pass filter cutoff frequency
            penalty_strength: Penalty strength for adaptive weighting
        """
        self.n_selected_images = n_selected_images
        self.hf_extractor = HighFrequencyExtractor(high_pass_cutoff=high_pass_cutoff)
        self.weighting_generator = AdaptiveWeightingMap(penalty_strength=penalty_strength)
    
    def select_optimal_frequency_offset(self, 
                                      fs_series: List[np.ndarray],
                                      roi_mask: Optional[np.ndarray] = None) -> Tuple[int, dict]:
        """
        Select optimal frequency offset from frequency scout series.
        
        Args:
            fs_series: List of frequency scout images
            roi_mask: Optional ROI mask (heart segmentation)
            
        Returns:
            Tuple of (optimal_offset_index, analysis_results)
        """
        analysis_results = {}
        
        # Step 1: Extract high-frequency content for all images
        print("Extracting high-frequency content...")
        hf_scores = self.hf_extractor.process_frequency_series(fs_series, roi_mask)
        analysis_results['hf_scores'] = hf_scores
        
        # Step 2: Select N images with lowest high-frequency content
        print(f"Selecting {self.n_selected_images} images with lowest high-frequency content...")
        selected_indices, selected_images = self.hf_extractor.select_lowest_hf_images(
            fs_series, hf_scores, self.n_selected_images
        )
        analysis_results['selected_indices'] = selected_indices
        analysis_results['selected_images'] = selected_images
        
        # Step 3: Generate adaptive weighting maps
        print("Generating adaptive weighting maps...")
        weighting_maps = self.weighting_generator.process_series_weighting(
            fs_series, selected_images, roi_mask
        )
        analysis_results['weighting_maps'] = weighting_maps
        
        # Step 4: Calculate average weighting for each image and select maximum
        print("Calculating optimal frequency offset...")
        avg_weights = []
        for weight_map in weighting_maps:
            if roi_mask is not None:
                # Calculate average within ROI
                roi_weights = weight_map * roi_mask
                avg_weight = np.sum(roi_weights) / np.sum(roi_mask)
            else:
                avg_weight = np.mean(weight_map)
            avg_weights.append(avg_weight)
        
        analysis_results['avg_weights'] = avg_weights
        
        # Select frame with maximum average weight
        optimal_offset_index = int(np.argmax(avg_weights))
        analysis_results['optimal_offset'] = optimal_offset_index
        
        print(f"Optimal frequency offset selected: Frame {optimal_offset_index}")
        
        return optimal_offset_index, analysis_results