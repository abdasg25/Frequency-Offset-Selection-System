# ML-Based Frequency Offset Selection System

This document describes the machine learning approach to automated frequency offset selection for cardiac MRI frequency scout imaging.

## 🎯 Overview

The ML-based system uses a **3D CNN with attention mechanisms** to learn optimal frequency selection patterns from the training data, achieving high accuracy for automated frequency offset selection.

### Key Components:
- **FrequencySelectionCNN**: 3D CNN architecture with channel and spatial attention
- **Combined Loss Function**: Classification + regression with focal loss
- **Ensemble Prediction**: Combines classification and regression outputs
- **Data Augmentation**: Augments training data for better generalization

## 🏗️ Architecture

### FrequencySelectionCNN Model
```python
Input: Frequency Series (N_freq, 256, 256) + Heart Mask (256, 256)
  ↓
3D CNN Feature Extraction:
  - Conv3D layers with batch normalization
  - Max pooling for dimensionality reduction
  - Channel and spatial attention mechanisms
  ↓
Global Average Pooling
  ↓
Dual Prediction Heads:
  - Classification: Discrete frequency index (softmax)
  - Regression: Continuous frequency index (linear)
  ↓
Ensemble Output: Weighted combination of both predictions
```

### Key Features:
- **3D Convolutions**: Process temporal-spatial patterns across frequency series
- **Attention Mechanisms**: Focus on important spatial and channel features
- **Dual Outputs**: Both classification and regression for robust prediction
- **Heart Mask Integration**: Uses segmentation masks as additional input

## 📊 Training Process

### 1. Dataset Preparation
- **FrequencySelectionMLDataset**: Specialized dataset for ML training
- **Data Augmentation**: Random flips, rotations, and color jittering
- **Heart Mask Generation**: Uses pre-trained MONAI AttentionUNet
- **Train/Val Split**: 80/20 split with patient-level separation

### 2. Loss Function
```python
Total Loss = 0.7 × Classification Loss + 0.3 × Regression Loss + 0.1 × Proximity Loss
```
- **Focal Loss**: Handles class imbalance in frequency distribution
- **Smooth L1 Loss**: Robust regression loss
- **Proximity Loss**: Penalizes predictions far from ground truth

### 3. Training Configuration
- **Optimizer**: AdamW with weight decay
- **Learning Rate Scheduler**: ReduceLROnPlateau
- **Gradient Clipping**: Prevents gradient explosion
- **Early Stopping**: Based on validation tolerance accuracy

## 🚀 Usage

### 1. Train the ML Model
```bash
cd /Users/abdulrehman/fyp/Frequency-Offset-Selection-System

# Train with default parameters
python scripts/train_ml_frequency_selector.py

# Custom training
python scripts/train_ml_frequency_selector.py \
    --epochs 150 \
    --batch_size 8 \
    --learning_rate 1e-3 \
    --max_frequencies 15 \
    --validation_split 0.2
```

### 2. Run ML-Based Frequency Selection
```bash
# Use trained model for inference
python scripts/run_ml_frequency_selection.py

# Specify custom model path
python scripts/run_ml_frequency_selection.py \
    --model_path outputs/ml_frequency_models/best_model.pth
```

### 3. Evaluate ML Performance
```bash
# Evaluate trained model performance
python scripts/evaluate_ml_performance.py

# Custom model evaluation
python scripts/evaluate_ml_performance.py \
    --ml_model_path outputs/ml_frequency_models/best_model.pth \
    --save_results
```

## 📁 File Structure

```
Frequency-Offset-Selection-System/
├── models/
│   └── frequency_selector.py          # ML model architecture
├── data/
│   └── ml_dataset.py                  # ML training dataset
├── scripts/
│   ├── train_ml_frequency_selector.py # Training script
│   └── run_ml_frequency_selection.py  # Inference script
└── outputs/
    ├── ml_frequency_models/           # Trained models
    │   ├── best_model.pth            # Best checkpoint
    │   ├── latest_model.pth          # Latest checkpoint
    │   ├── training_history.json     # Training metrics
    │   └── training_history.png      # Training plots
    └── ml_frequency_selection_results/
        └── ml_detailed_results.json   # Inference results
```

## 🎯 Performance Targets

- **Target Accuracy**: >92% (±2 frames tolerance)
- **Expected Performance**: High accuracy across all frequency ranges

### Expected Improvements:
1. **Pattern Learning**: CNN learns complex spatial-temporal patterns
2. **Data-Driven**: Trained on actual frequency selection patterns
3. **Attention Mechanisms**: Focuses on relevant image regions
4. **Ensemble Prediction**: Combines multiple prediction strategies

## 🔧 Hyperparameter Tuning

### Key Parameters:
- **Learning Rate**: 1e-3 to 1e-4 range
- **Batch Size**: 4-8 (limited by GPU memory)
- **Max Frequencies**: 15 (handles variable series lengths)
- **Dropout Rate**: 0.3 for regularization
- **Loss Weights**: 0.7 classification, 0.3 regression

### Optimization Tips:
1. **Start with lower learning rate** if training is unstable
2. **Increase batch size** if GPU memory allows
3. **Adjust loss weights** based on validation performance
4. **Use ensemble voting** for final predictions

## 📈 Expected Results

### Accuracy Improvements:
- **Center Frequencies (0 Hz)**: Maintain 100% accuracy
- **Near-Center (±25 Hz)**: Improve from 85% to >95%
- **Moderate Offset (±50 Hz)**: Improve from 71% to >90%
- **Large Offset (±75-100 Hz)**: Improve from 0-50% to >80%

### Key Success Metrics:
1. **Overall Accuracy**: >92%
2. **Failed Cases**: <3 patients
3. **High Confidence**: >90% accuracy for confident predictions
4. **Consistent Performance**: Robust across all frequency ranges

## 🔄 ML System Features

### Key Components:
- **Neural Network Training**: Supervised learning from frequency selection patterns
- **Enhanced**: Heart mask integration for ROI-focused analysis
- **Improved**: Prediction confidence scoring for quality assessment
- **Robust**: Ensemble prediction combining classification and regression

### Architecture Advantages:
- **MONAI Segmentation**: Same proven heart segmentation model
- **Data Pipeline**: Optimized DICOM loading and preprocessing
- **Evaluation**: Comprehensive ±2 frames tolerance assessment
- **Output Format**: Structured result format with confidence metrics

## 🛠️ Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size
   python scripts/train_ml_frequency_selector.py --batch_size 2
   ```

2. **Model Not Found**:
   ```bash
   # Check model path
   ls outputs/ml_frequency_models/
   ```

3. **Poor Training Performance**:
   - Reduce learning rate: `--learning_rate 5e-4`
   - Increase training epochs: `--epochs 200`
   - Check data augmentation settings

4. **Validation Accuracy Plateau**:
   - Adjust loss weights in training script
   - Modify attention mechanism parameters
   - Increase model capacity (more channels)

## 📚 Technical Details

### Model Architecture:
- **Input Channels**: 1 (frequency series) + 1 (heart mask) = 2
- **3D Conv Layers**: 3 layers with [32, 64, 128] channels
- **Attention**: Channel attention + spatial attention
- **Output Heads**: Classification (15 classes) + Regression (1 value)
- **Parameters**: ~2.5M trainable parameters

### Training Strategy:
- **Loss Function**: Multi-task learning with weighted combination
- **Regularization**: Dropout, weight decay, gradient clipping
- **Optimization**: AdamW with adaptive learning rate
- **Validation**: Patient-level split to prevent data leakage

This ML-based approach represents a comprehensive solution for automated frequency offset selection in cardiac MRI, leveraging deep learning to achieve the target 92% accuracy for clinical-grade performance.