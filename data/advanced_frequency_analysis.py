"""
Advanced Frequency Offset Selection System - Targeting 80%+ Accuracy
Incorporates temporal consistency, texture analysis, and improved decision algorithms
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.stats import entropy
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import logging

class AdvancedFrequencyOffsetSelector:
    def __init__(self, num_references=7, temporal_window=3):
        """
        Advanced frequency offset selector with enhanced features for 80%+ accuracy
        
        Args:
            num_references: Number of reference images to use
            temporal_window: Window size for temporal consistency analysis
        """
        self.num_references = num_references
        self.temporal_window = temporal_window
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def extract_advanced_features(self, image):
        """
        Extract comprehensive features for optimal frequency selection
        
        Returns:
            dict: Dictionary of feature values
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Ensure proper data type for OpenCV operations
        image_uint8 = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image
        image_float = image_uint8.astype(np.float32) / 255.0
        
        features = {}
        
        # 1. Multi-scale gradient analysis
        sobel_x = cv2.Sobel(image_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image_uint8, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        features['gradient_mean'] = np.mean(gradient_magnitude)
        features['gradient_std'] = np.std(gradient_magnitude)
        features['gradient_max'] = np.max(gradient_magnitude)
        
        # 2. Edge density analysis
        edges = cv2.Canny(image_uint8, 50, 150)
        features['edge_density'] = np.mean(edges) / 255.0
        
        # 3. Texture analysis using Local Binary Patterns approximation
        laplacian = cv2.Laplacian(image_uint8, cv2.CV_64F)
        features['texture_variance'] = np.var(laplacian)
        features['texture_mean'] = np.mean(np.abs(laplacian))
        
        # 4. Frequency domain analysis
        f_transform = np.fft.fft2(image_float)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # High frequency energy in different bands
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Define frequency bands
        low_freq_mask = np.zeros((h, w))
        cv2.circle(low_freq_mask, (center_w, center_h), min(h, w) // 8, 1, -1)
        
        mid_freq_mask = np.zeros((h, w))
        cv2.circle(mid_freq_mask, (center_w, center_h), min(h, w) // 4, 1, -1)
        mid_freq_mask -= low_freq_mask
        
        high_freq_mask = np.ones((h, w)) - mid_freq_mask - low_freq_mask
        
        features['low_freq_energy'] = np.sum(magnitude_spectrum * low_freq_mask)
        features['mid_freq_energy'] = np.sum(magnitude_spectrum * mid_freq_mask)
        features['high_freq_energy'] = np.sum(magnitude_spectrum * high_freq_mask)
        
        # 5. Statistical features
        features['entropy'] = entropy(image_float.flatten() + 1e-8)
        features['skewness'] = self._calculate_skewness(image_float)
        features['kurtosis'] = self._calculate_kurtosis(image_float)
        
        # 6. Artifact detection features
        features['artifact_score'] = self._detect_artifacts(image_uint8)
        
        return features
    
    def _calculate_skewness(self, image):
        """Calculate skewness of image intensity distribution"""
        mean_val = np.mean(image)
        std_val = np.std(image)
        if std_val == 0:
            return 0
        return np.mean(((image - mean_val) / std_val) ** 3)
    
    def _calculate_kurtosis(self, image):
        """Calculate kurtosis of image intensity distribution"""
        mean_val = np.mean(image)
        std_val = np.std(image)
        if std_val == 0:
            return 0
        return np.mean(((image - mean_val) / std_val) ** 4) - 3
    
    def _detect_artifacts(self, image):
        """Detect motion artifacts and noise"""
        # Detect sudden intensity changes that might indicate artifacts
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Look for areas with abnormally high gradients
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        threshold = np.percentile(gradient_mag, 95)
        artifact_mask = gradient_mag > threshold
        
        return np.mean(artifact_mask)
    
    def analyze_temporal_consistency(self, images, features_list):
        """
        Analyze temporal consistency across the image series
        
        Args:
            images: List of images in the series
            features_list: List of feature dictionaries for each image
            
        Returns:
            list: Temporal consistency scores for each image
        """
        consistency_scores = []
        n_images = len(images)
        
        for i in range(n_images):
            # Calculate consistency with neighboring images
            consistency_score = 0
            count = 0
            
            window_start = max(0, i - self.temporal_window // 2)
            window_end = min(n_images, i + self.temporal_window // 2 + 1)
            
            for j in range(window_start, window_end):
                if i != j:
                    # Calculate feature similarity
                    similarity = self._calculate_feature_similarity(
                        features_list[i], features_list[j]
                    )
                    consistency_score += similarity
                    count += 1
            
            if count > 0:
                consistency_score /= count
            
            consistency_scores.append(consistency_score)
        
        return consistency_scores
    
    def _calculate_feature_similarity(self, features1, features2):
        """Calculate similarity between two feature sets"""
        # Normalize and compare key features
        key_features = ['gradient_mean', 'edge_density', 'texture_variance', 
                       'high_freq_energy', 'entropy']
        
        similarities = []
        for feature in key_features:
            if feature in features1 and feature in features2:
                val1, val2 = features1[feature], features2[feature]
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    similarity = 1 - abs(val1 - val2) / max_val
                else:
                    similarity = 1
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0
    
    def compute_advanced_quality_scores(self, images, heart_masks):
        """
        Compute advanced quality scores using multiple criteria
        
        Args:
            images: List of cardiac images
            heart_masks: List of corresponding heart segmentation masks
            
        Returns:
            tuple: (quality_scores, feature_matrix, temporal_scores)
        """
        n_images = len(images)
        features_list = []
        
        # Extract features for all images
        self.logger.info("Extracting advanced features...")
        for i, (image, mask) in enumerate(zip(images, heart_masks)):
            # Focus analysis on heart region
            masked_image = image * (mask > 0.5)
            features = self.extract_advanced_features(masked_image)
            features_list.append(features)
        
        # Analyze temporal consistency
        self.logger.info("Analyzing temporal consistency...")
        temporal_scores = self.analyze_temporal_consistency(images, features_list)
        
        # Compute comprehensive quality scores
        quality_scores = []
        
        for i in range(n_images):
            features = features_list[i]
            
            # Base quality from multiple criteria
            edge_quality = features['edge_density']
            texture_quality = 1.0 / (1.0 + features['texture_variance'] / 1000.0)
            frequency_quality = features['high_freq_energy'] / (features['low_freq_energy'] + 1e-8)
            artifact_penalty = 1.0 - features['artifact_score']
            
            # Temporal consistency bonus
            temporal_bonus = temporal_scores[i]
            
            # Composite quality score
            quality_score = (
                0.25 * edge_quality +
                0.25 * texture_quality +
                0.20 * frequency_quality +
                0.15 * artifact_penalty +
                0.15 * temporal_bonus
            )
            
            quality_scores.append(quality_score)
        
        # Convert features to matrix for analysis
        feature_names = list(features_list[0].keys())
        feature_matrix = np.array([[f[name] for name in feature_names] for f in features_list])
        
        return quality_scores, feature_matrix, temporal_scores
    
    def select_optimal_frequency_offset(self, images, heart_masks):
        """
        Select optimal frequency offset using advanced analysis
        
        Args:
            images: List of cardiac images across frequency offsets
            heart_masks: List of corresponding heart segmentation masks
            
        Returns:
            tuple: (optimal_index, analysis_results)
        """
        self.logger.info("Starting advanced frequency offset selection...")
        
        # Compute advanced quality scores
        quality_scores, feature_matrix, temporal_scores = self.compute_advanced_quality_scores(
            images, heart_masks
        )
        
        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(feature_matrix)
        
        # Apply intelligent selection strategy
        optimal_index, analysis_results = self._intelligent_selection_strategy(
            quality_scores, normalized_features, temporal_scores
        )
        
        # Add detailed analysis results
        analysis_results.update({
            'quality_scores': quality_scores,
            'temporal_scores': temporal_scores,
            'feature_matrix': feature_matrix.tolist(),
            'selected_index': optimal_index
        })
        
        self.logger.info(f"Advanced selection complete. Optimal index: {optimal_index}")
        
        return optimal_index, analysis_results
    
    def _intelligent_selection_strategy(self, quality_scores, normalized_features, temporal_scores):
        """
        Intelligent selection strategy combining multiple criteria
        
        Args:
            quality_scores: Base quality scores for each image
            normalized_features: Normalized feature matrix
            temporal_scores: Temporal consistency scores
            
        Returns:
            tuple: (optimal_index, analysis_results)
        """
        n_images = len(quality_scores)
        
        # 1. Quality-based ranking
        quality_ranks = np.argsort(quality_scores)[::-1]
        
        # 2. Find stable regions (consecutive images with good quality)
        stability_scores = []
        for i in range(n_images):
            window_start = max(0, i - 2)
            window_end = min(n_images, i + 3)
            window_quality = [quality_scores[j] for j in range(window_start, window_end)]
            stability_score = np.mean(window_quality) - np.std(window_quality)
            stability_scores.append(stability_score)
        
        # 3. Center bias with adaptive strength
        center_index = n_images // 2
        center_bias_scores = []
        for i in range(n_images):
            distance_from_center = abs(i - center_index)
            # Stronger bias for longer series
            bias_strength = 0.1 + 0.1 * (n_images / 13.0)
            bias_score = 1.0 - bias_strength * (distance_from_center / center_index)
            center_bias_scores.append(max(0, bias_score))
        
        # 4. Avoid extreme indices more aggressively
        edge_penalty_scores = []
        for i in range(n_images):
            if i < 2 or i >= n_images - 2:
                penalty = 0.7  # Strong penalty for edge indices
            elif i < 4 or i >= n_images - 4:
                penalty = 0.85  # Moderate penalty
            else:
                penalty = 1.0  # No penalty for central indices
            edge_penalty_scores.append(penalty)
        
        # 5. Combine all scores with optimized weights
        combined_scores = []
        for i in range(n_images):
            combined_score = (
                0.35 * quality_scores[i] +
                0.25 * stability_scores[i] +
                0.15 * temporal_scores[i] +
                0.15 * center_bias_scores[i] +
                0.10 * edge_penalty_scores[i]
            )
            combined_scores.append(combined_score)
        
        # Find optimal index
        optimal_index = np.argmax(combined_scores)
        
        # Prepare analysis results
        analysis_results = {
            'quality_ranks': quality_ranks.tolist(),
            'stability_scores': stability_scores,
            'center_bias_scores': center_bias_scores,
            'edge_penalty_scores': edge_penalty_scores,
            'combined_scores': combined_scores,
            'selection_strategy': 'intelligent_multi_criteria'
        }
        
        return optimal_index, analysis_results

def create_advanced_selector(num_references=7, temporal_window=3):
    """
    Factory function to create an advanced frequency offset selector
    
    Args:
        num_references: Number of reference images to use
        temporal_window: Window size for temporal consistency analysis
        
    Returns:
        AdvancedFrequencyOffsetSelector: Configured selector instance
    """
    return AdvancedFrequencyOffsetSelector(
        num_references=num_references,
        temporal_window=temporal_window
    )