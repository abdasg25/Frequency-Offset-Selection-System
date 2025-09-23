# EMIDEC-Powered Frequency Offset Selection for Cardiac MRI

This project implements an advanced machine learning system for automatically selecting optimal frequency offsets in cardiac MRI frequency scout (FS) sequences using EMIDEC-trained deep learning models.

## 🔬 Research Integration

This system integrates with the **Comparative Analysis of MONAI Models on EMIDEC Dataset** project to provide:
- **EMIDEC-trained AttentionUNet** for cardiac segmentation
- **Automatic pipeline** from EMIDEC training to frequency selection
- **Research-grade accuracy** targeting 80%+ performance

## 🧠 Pipeline Overview

### Phase 1: EMIDEC Training
```
EMIDEC Dataset → AttentionUNet (3D) → Cardiac Segmentation Model
```

### Phase 2: Weight Adaptation  
```
3D EMIDEC Weights → Adapter → 2D Frequency Scout Segmentation
```

### Phase 3: Frequency Selection
```
Frequency Series → Heart Masks → Multi-Modal Analysis → Optimal Frequency
```

## Quick Start

Run the complete integrated pipeline:
```bash
python scripts/run_pipeline.py
```

This automatically:
1. Checks EMIDEC dataset availability  
2. Trains AttentionUNet on EMIDEC using integrated comparative analysis code
3. Runs frequency offset selection with EMIDEC weights

## Installation

Install required packages:
```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch>=2.0.0`
- `monai>=1.3.0` 
- `einops>=0.7.0` (required for UNETR in comparative analysis)
- `pydicom`, `nibabel`, `opencv-python`

## 📊 Dataset Requirements

### EMIDEC Dataset (for segmentation training)
- **Location**: `Comparative-Analysis-of-MONAI-Models-on-EMIDEC-Dataset/emidec-dataset-1.0.1`
- **Purpose**: Training cardiac segmentation model
- **Format**: 3D cardiac MRI with expert segmentations

### Cohort Dataset (for frequency selection)
- **Location**: `1st_cohort_SAX/`
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
git clone https://github.com/abdasg25/Frequency-Offset-Selection-System
cd Frequency-Offset-Selection-System

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python scripts/run_pipeline.py
```

## Expected Results

Based on the original paper methodology:
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