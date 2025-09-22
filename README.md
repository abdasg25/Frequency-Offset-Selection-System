# ML-Based Automated Frequency Offset Selection for Cardiac MRI

This project implements an advanced machine learning system for automatically selecting optimal frequency offsets in cardiac MRI frequency scout (FS) sequences using deep learning techniques.

## 🤖 ML Approach Overview

This system uses a **3D CNN with attention mechanisms** to learn optimal frequency selection patterns from training data, achieving clinical-grade accuracy for automated frequency offset selection.

## System Architecture

The ML-based frequency offset selection system consists of four main components:

1. **Heart Segmentation**: Uses MONAI AttentionUNet to localize the heart ROI in frequency scout images
2. **ML Feature Extraction**: 3D CNN processes frequency series to extract temporal-spatial patterns
3. **Attention Mechanisms**: Channel and spatial attention focus on relevant image regions
4. **Ensemble Prediction**: Combines classification and regression outputs for optimal frequency selection

## Dataset

- **Format**: DICOM frequency scout series + PNG segmentation annotations
- **Patients**: 25 cardiac MRI cases
- **Acquisition**: Multiple 3T MRI scanners
- **Frequency Series**: Each patient has 7-13 DICOM images at different frequency offsets
  - **Standard**: -150 to +150 Hz with 25Hz steps (13 images)
  - **Patient-KP**: -150 to +150 Hz with 50Hz steps (7 images)
- **Frequency Mapping**: 
  - Image 1 = -150 Hz, Image 7 = 0 Hz (center), Image 13 = +150 Hz (for 25Hz steps)
  - Manual annotations specify optimal image number (1-indexed)
- **Ground Truth**: Manual expert selections for optimal frequency offsets
- **Visual Artifacts**: bSSFP banding artifacts that vary with frequency offset

## Key Understanding

The system works with **frequency scout series** where:
1. Each patient has a series of DICOM images acquired at different frequency offsets
2. The frequency offset shifts the position of banding artifacts
3. The goal is to automatically select the frequency that minimizes artifacts in the heart region
4. Your manual annotations (image_no) indicate which image in each series has the best image quality

## Key Features

- **ML-Based Selection**: 3D CNN with attention mechanisms for pattern learning
- **Heart Segmentation**: MONAI AttentionUNet for precise ROI localization
- **Ensemble Prediction**: Combines classification and regression approaches
- **Confidence Scoring**: Provides prediction confidence for quality assessment
- **Data Augmentation**: Robust training with augmented frequency series
- **Comprehensive Evaluation**: Validation against expert annotations with clinical metrics

## Project Structure

```
├── configs/                 # Configuration files
├── data/                   # Data loading and preprocessing
├── models/                 # Deep learning models (MONAI AttentionUNet)
├── processing/             # Frequency analysis and weighting
├── evaluation/             # Evaluation and metrics
├── visualization/          # Plotting and visualization tools
├── scripts/               # Training and inference scripts
└── outputs/               # Results and visualizations
```

## Installation

```bash
# Clone repository
git clone <repository-url>
cd Frequency-Offset-Selection-System

# Create virtual environment
python -m venv fos_env
source fos_env/bin/activate  # Linux/Mac
# fos_env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train Heart Segmentation Model
```bash
python scripts/train_segmentation.py --config configs/segmentation_config.yaml
```

### 2. Run Frequency Offset Selection
```bash
python scripts/run_frequency_selection.py --input_dir /path/to/dicom/series --output_dir ./outputs
```

### 3. Evaluate System Performance
```bash
python scripts/evaluate_system.py --predictions ./outputs --ground_truth ./annotations
```

## Methodology

### Step 1: Heart Segmentation
- Pre-trained MONAI AttentionUNet for whole heart segmentation
- Localizes ROI where artifacts should be minimized
- Processes 2D frequency scout images

### Step 2: High-Frequency Component Extraction
- Fourier transformation of ROI regions
- High-pass filtering to extract high-frequency components
- Inverse Fourier transformation and subtraction
- Selection of N images with lowest high-frequency content

### Step 3: Adaptive Weighting Map Generation
- Pixel-wise median calculation from selected images
- Weighting maps that penalize signal deviations from median
- Adaptive weighting based on local image characteristics

### Step 4: Optimal Frequency Selection
- Averaging of weighting maps across frequency offsets
- Selection of frame with maximum weighted percentage
- Output of optimal frequency offset

## Expected Results

Based on the original paper methodology:
- **Target Accuracy**: >90% compared to expert annotations
- **Tolerance**: Maximum difference of 2 frames for failed cases
- **Robustness**: Effective artifact reduction in heart ROI
- **Automation**: Fully automated workflow without manual intervention

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- MONAI 1.3+
- pydicom
- nibabel
- opencv-python
- matplotlib
- scikit-learn
- numpy
- scipy

## References

[1] Original paper on automated frequency offset selection in cardiac MRI
[2] MONAI framework for medical imaging deep learning
[3] Frequency scout imaging techniques in cardiac MRI
