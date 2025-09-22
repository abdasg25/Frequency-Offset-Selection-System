import os
import torch
import random
import numpy as np

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Hardware configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
PIN_MEMORY = torch.cuda.is_available()

# Data paths
DATA_ROOT = "/Users/abdulrehman/fyp/1st_cohort_SAX"
ANNOTATIONS_ROOT = "/Users/abdulrehman/fyp/Annotations"
OUTPUT_DIR = "/Users/abdulrehman/fyp/Frequency-Offset-Selection-System/outputs"

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "visualizations"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "results"), exist_ok=True)

# Image processing parameters
IMAGE_SIZE = (256, 256)  # Standard size for frequency scout images
DICOM_WINDOW_CENTER = 128
DICOM_WINDOW_WIDTH = 256

# Heart segmentation model parameters
SEGMENTATION_MODEL = "AttentionUNet"
NUM_CLASSES = 2  # Background + Heart
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
MAX_EPOCHS = 15
VALIDATION_SPLIT = 0.2

# ML frequency selection parameters
ML_LEARNING_RATE = 1e-3
ML_BATCH_SIZE = 4
ML_MAX_EPOCHS = 100
ML_MAX_FREQUENCIES = 15  # Maximum frequency series length to handle
ML_DROPOUT_RATE = 0.3
ML_WEIGHT_DECAY = 1e-4

# Training parameters
GRADIENT_CLIP_NORM = 1.0
FOCAL_LOSS_ALPHA = 1.0
FOCAL_LOSS_GAMMA = 2.0
CLASSIFICATION_WEIGHT = 0.7
REGRESSION_WEIGHT = 0.3

# Evaluation parameters
TOLERANCE_FRAMES = 2  # Maximum allowed difference for correct prediction
TARGET_ACCURACY = 92.0  # Target accuracy percentage

# Patient information (from your frequency table)
# image_no corresponds to the optimal image in the frequency series (1-indexed)
# manual_freq is the optimal frequency offset in Hz
PATIENT_INFO = {
    'Patient-BEV': {'manual_freq': 0, 'image_no': 7, 'freq_step': 25, 'total_images': 13},
    'Patient-BGT': {'manual_freq': 0, 'image_no': 7, 'freq_step': 25, 'total_images': 13},
    'Patient-BVE': {'manual_freq': 25, 'image_no': 6, 'freq_step': 25, 'total_images': 13},
    'Patient-CEA': {'manual_freq': 0, 'image_no': 7, 'freq_step': 25, 'total_images': 13},
    'Patient-CEJP': {'manual_freq': 75, 'image_no': 4, 'freq_step': 25, 'total_images': 13},
    'Patient-DEM': {'manual_freq': -50, 'image_no': 9, 'freq_step': 25, 'total_images': 13},
    'Patient-FAJ': {'manual_freq': -50, 'image_no': 9, 'freq_step': 25, 'total_images': 13},
    'Patient-FEC': {'manual_freq': -25, 'image_no': 8, 'freq_step': 25, 'total_images': 13},
    'Patient-GJ': {'manual_freq': -50, 'image_no': 9, 'freq_step': 25, 'total_images': 13},
    'Patient-HMJ': {'manual_freq': -25, 'image_no': 8, 'freq_step': 25, 'total_images': 13},
    'Patient-HRJ': {'manual_freq': 0, 'image_no': 7, 'freq_step': 25, 'total_images': 13},
    'Patient-IAM': {'manual_freq': -25, 'image_no': 8, 'freq_step': 25, 'total_images': 13},
    'Patient-INA': {'manual_freq': -100, 'image_no': 11, 'freq_step': 25, 'total_images': 13},
    'Patient-KP': {'manual_freq': -100, 'image_no': 6, 'freq_step': 50, 'total_images': 7},  # Special case: 50Hz steps
    'Patient-MJ': {'manual_freq': 0, 'image_no': 7, 'freq_step': 25, 'total_images': 13},
    'Patient-MS': {'manual_freq': -50, 'image_no': 9, 'freq_step': 25, 'total_images': 13},
    'Patient-MSR': {'manual_freq': -75, 'image_no': 10, 'freq_step': 25, 'total_images': 13},
    'Patient-OAF': {'manual_freq': -50, 'image_no': 9, 'freq_step': 25, 'total_images': 13},
    'Patient-PJ': {'manual_freq': -25, 'image_no': 8, 'freq_step': 25, 'total_images': 13},
    'Patient-RAK': {'manual_freq': 0, 'image_no': 7, 'freq_step': 25, 'total_images': 13},
    'Patient-RK': {'manual_freq': -50, 'image_no': 9, 'freq_step': 25, 'total_images': 13},
    'Patient-SK': {'manual_freq': -25, 'image_no': 8, 'freq_step': 25, 'total_images': 13},
    'Patient-SWJ': {'manual_freq': 0, 'image_no': 7, 'freq_step': 25, 'total_images': 13},
    'Patient-TL': {'manual_freq': -50, 'image_no': 9, 'freq_step': 25, 'total_images': 13},
    'Patient-WBC': {'manual_freq': -25, 'image_no': 8, 'freq_step': 25, 'total_images': 13},
    'Patient-WMR': {'manual_freq': -25, 'image_no': 8, 'freq_step': 25, 'total_images': 13}
}

# Frequency mapping functions
def frequency_to_image_index(frequency_hz: int, freq_step: int = 25) -> int:
    """Convert frequency in Hz to 0-based image index."""
    # Frequency range: -150 to +150 Hz
    # Image index 0 = -150 Hz, Image index 6 = 0 Hz (for 25Hz steps)
    return (frequency_hz + 150) // freq_step

def image_index_to_frequency(image_idx: int, freq_step: int = 25) -> int:
    """Convert 0-based image index to frequency in Hz."""
    return (image_idx * freq_step) - 150

def get_optimal_image_index(patient_id: str) -> int:
    """Get the 0-based optimal image index for a patient."""
    patient_data = PATIENT_INFO[patient_id]
    # Convert 1-based image_no to 0-based index
    return patient_data['image_no'] - 1
