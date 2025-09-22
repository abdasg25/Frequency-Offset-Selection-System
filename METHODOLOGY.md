# Research-Based Frequency Offset Selection Methodology

## Abstract

This document describes the implementation of a fully automated image-based system for selecting the optimal frequency offset on frequency scout (FS) images for cardiac bSSFP MRI sequences. The methodology is based on published research achieving 92.1% accuracy compared to expert annotations.

## Background

In balanced steady-state free precession (bSSFP) sequences commonly used for cardiac MRI, signal modulation (banding artifacts) due to B0 inhomogeneity is often observed, especially at higher field strengths. The spatial position of these artifacts can be shifted by a frequency offset to reduce artifacts in a region of interest (ROI), specifically the heart. To this end, frequency scout (FS) scans are acquired to visually select the optimal frequency offset. Our system automates this selection process using machine learning and image analysis techniques.

## Methodology

The proposed prototype system consists of four main steps:

### Step 1: Heart Segmentation and ROI Localization

**Purpose**: Localize the region of interest (heart) where artifacts should be minimized.

**Implementation**:
- A pre-trained deep-learning-based whole heart segmentation network (AttentionUNet) is applied on a four-chamber view FS image
- The segmentation model uses MONAI framework with attention mechanisms
- Heart mask is generated to focus subsequent analysis on cardiac region
- Reference image (middle frame of FS series) is used for segmentation

**Technical Details**:
```python
# Heart segmentation using pre-trained AttentionUNet
segmentation_model = create_segmentation_model(
    in_channels=1,
    out_channels=NUM_CLASSES,
    pretrained=True
)
heart_mask = segmentation_model.get_heart_mask(reference_image, threshold=0.5)
```

### Step 2: High-Frequency Content Extraction

**Purpose**: Extract high-frequency components within the ROI for each frequency offset in the FS series.

**Process**:
1. **Fourier Transformation**: Apply 2D FFT to each FS image
2. **High-Pass Filtering**: Create and apply high-pass filter in frequency domain
3. **Inverse Fourier Transformation**: Convert back to spatial domain
4. **Content Scoring**: Calculate high-frequency content score within heart ROI

**Technical Implementation**:
```python
class HighFrequencyExtractor:
    def extract_high_frequency_content(self, image, roi_mask):
        # Apply Gaussian smoothing to reduce noise
        image = ndimage.gaussian_filter(image, sigma=self.gaussian_sigma)
        
        # Fourier transformation
        f_transform = fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        
        # Apply high-pass filter
        high_pass_filter = self.create_high_pass_filter(image.shape)
        filtered_f_shift = f_shift * high_pass_filter
        
        # Inverse transformation and scoring
        f_ishift = np.fft.ifftshift(filtered_f_shift)
        high_freq_image = np.real(ifft2(f_ishift))
        
        # Calculate score within ROI
        roi_high_freq = high_freq_image * roi_mask
        hf_score = np.sum(np.abs(roi_high_freq)) / np.sum(roi_mask)
        return hf_score
```

**Parameters**:
- High-pass cutoff frequency: 0.1 (fraction of Nyquist frequency)
- Gaussian smoothing sigma: 2.0
- ROI-focused scoring for cardiac region

### Step 3: Selection of Low High-Frequency Images

**Purpose**: Select N images with the lowest high-frequency content to serve as reference for adaptive weighting.

**Process**:
1. Sort all FS images by their high-frequency content scores
2. Select the N lowest-scoring images (default N=3)
3. These images represent frames with minimal banding artifacts

**Rationale**: Images with low high-frequency content in the heart region are likely to have minimal banding artifacts and can serve as good references for identifying optimal frequency offsets.

### Step 4: Adaptive Weighting Map Generation

**Purpose**: Generate weighting maps that penalize signal deviations from a pixel-wise median.

**Process**:
1. **Pixel-wise Median Calculation**: Calculate median image from selected low high-frequency images
2. **Deviation Analysis**: For each FS image, calculate absolute deviation from median reference
3. **Weight Map Generation**: Generate exponential weighting based on deviation
4. **ROI Application**: Apply heart mask to focus weighting on cardiac region

**Mathematical Formulation**:
```
deviation(x,y) = |I(x,y) - median_reference(x,y)|
weight(x,y) = exp(-α × normalized_deviation(x,y))
```

Where α is the penalty strength parameter (default: 1.0).

**Technical Implementation**:
```python
class AdaptiveWeightingMap:
    def generate_weighting_map(self, image, median_reference, roi_mask):
        # Calculate deviation from median
        deviation = np.abs(image - median_reference)
        
        # Normalize deviation
        normalized_deviation = deviation / np.max(deviation)
        
        # Generate exponential weighting
        weighting_map = np.exp(-self.penalty_strength * normalized_deviation)
        
        # Apply ROI mask
        weighting_map = weighting_map * roi_mask
        return weighting_map
```

### Step 5: Optimal Frequency Offset Selection

**Purpose**: Select the frequency offset that maximizes the average weighting within the heart ROI.

**Process**:
1. **Weight Averaging**: Calculate average weight for each FS image within heart ROI
2. **Maximum Selection**: Select frame with highest average weight
3. **Frequency Conversion**: Convert frame index to corresponding frequency offset

**Final Selection**:
```python
# Calculate average weights for each image
avg_weights = []
for weight_map in weighting_maps:
    roi_weights = weight_map * roi_mask
    avg_weight = np.sum(roi_weights) / np.sum(roi_mask)
    avg_weights.append(avg_weight)

# Select optimal offset
optimal_offset_index = np.argmax(avg_weights)
```

## System Architecture

### Class Structure

1. **HighFrequencyExtractor**: Handles FFT analysis and high-frequency content extraction
2. **AdaptiveWeightingMap**: Manages weighting map generation and median calculation
3. **FrequencyOffsetSelector**: Main orchestrator combining all methodology steps

### Integration Pipeline

```python
def select_optimal_frequency_offset(fs_series, roi_mask):
    # Step 1: Extract high-frequency content
    hf_scores = extract_high_frequency_content(fs_series, roi_mask)
    
    # Step 2: Select low HF images
    selected_indices, selected_images = select_lowest_hf_images(fs_series, hf_scores, n=3)
    
    # Step 3: Generate weighting maps
    weighting_maps = generate_weighting_maps(fs_series, selected_images, roi_mask)
    
    # Step 4: Select optimal offset
    optimal_offset = select_maximum_weight_frame(weighting_maps, roi_mask)
    
    return optimal_offset
```

## Performance Characteristics

### Current Implementation Results
- **Accuracy**: 42.3% (11/26 patients correct)
- **Tolerance**: ±2 frames
- **Processing Speed**: ~7.4 patients/second
- **Target Accuracy**: 92.1% (published research benchmark)

### Analysis Output
For each patient, the system provides:
- High-frequency content scores for all frames
- Selected low high-frequency frame indices
- Average weights for all frames
- Optimal frequency offset prediction
- Detailed analysis results

### Failed Cases Analysis
Common failure patterns observed:
- Large differences (6-8 frames) in 7 cases
- Moderate differences (3-5 frames) in 8 cases
- Suggests need for parameter tuning or additional constraints

## Technical Parameters

### Configurable Parameters
- **n_selected_images**: Number of low HF images to select (default: 3)
- **high_pass_cutoff**: High-pass filter cutoff frequency (default: 0.1)
- **penalty_strength**: Adaptive weighting penalty strength (default: 1.0)
- **gaussian_sigma**: Noise reduction smoothing parameter (default: 2.0)

### System Requirements
- Python 3.7+
- NumPy, SciPy for numerical computations
- OpenCV for image processing
- MONAI for heart segmentation
- PyTorch for deep learning components

## Advantages of This Methodology

1. **Physically Motivated**: Based on frequency domain analysis of MRI artifacts
2. **ROI-Focused**: Concentrates analysis on heart region where artifacts matter most
3. **Robust Reference**: Uses multiple low-artifact images for stable median calculation
4. **Adaptive**: Weighting maps adapt to individual image characteristics
5. **Interpretable**: Each step has clear physical/clinical meaning

## Future Improvements

1. **Parameter Optimization**: Systematic tuning of high-pass cutoff and penalty strength
2. **Multi-Scale Analysis**: Incorporate multiple frequency bands
3. **Temporal Consistency**: Consider temporal relationships in FS series
4. **Machine Learning Enhancement**: Combine with learned features for improved accuracy
5. **Validation**: Extensive validation on larger clinical datasets

## Conclusion

This methodology provides a systematic, research-based approach to automated frequency offset selection for cardiac MRI. While current accuracy is below the published target, the framework is solid and provides interpretable results that can guide further optimization efforts.