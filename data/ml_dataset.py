#!/usr/bin/env python3
"""
ML Training Dataset for frequency offset selection.
"""

import os
import sys
import glob
import pydicom
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Tuple, Dict, Optional
import cv2
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FrequencySelectionMLDataset(Dataset):
    """
    Dataset for training ML-based frequency selector.
    
    This dataset provides frequency series with corresponding optimal frequency indices
    for supervised learning of frequency offset selection.
    """
    
    def __init__(
        self,
        data_root: str,
        patient_ids: List[str],
        patient_info: Dict[str, Dict],
        segmentation_model: Optional[object] = None,
        image_size: Tuple[int, int] = (256, 256),
        augment_data: bool = True,
        max_frequencies: int = 15,
        device: str = 'cuda'
    ):
        """
        Initialize the ML training dataset.
        
        Args:
            data_root: Root directory containing patient DICOM data
            patient_ids: List of patient IDs to include
            patient_info: Dictionary with patient metadata
            segmentation_model: Pre-trained segmentation model for heart masks
            image_size: Target image size
            augment_data: Whether to apply data augmentation
            max_frequencies: Maximum number of frequencies to handle
            device: Device for segmentation model
        """
        self.data_root = data_root
        self.patient_ids = patient_ids
        self.patient_info = patient_info
        self.segmentation_model = segmentation_model
        self.image_size = image_size
        self.augment_data = augment_data
        self.max_frequencies = max_frequencies
        self.device = device
        
        # Pre-compute all samples
        self.samples = self._prepare_samples()
        
        # Data augmentation transforms
        if augment_data:
            self.augment_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ])
        else:
            self.augment_transforms = None
    
    def _prepare_samples(self) -> List[Dict]:
        """Pre-process and prepare all training samples."""
        samples = []
        
        for patient_id in self.patient_ids:
            try:
                # Load frequency series
                frequency_series, optimal_idx = self._load_frequency_series(patient_id)
                
                # Generate heart mask
                heart_mask = self._generate_heart_mask(frequency_series, patient_id)
                
                # Create sample
                sample = {
                    'patient_id': patient_id,
                    'frequency_series': frequency_series,
                    'heart_mask': heart_mask,
                    'optimal_index': optimal_idx,
                    'series_length': len(frequency_series)
                }
                
                samples.append(sample)
                print(f"Processed {patient_id}: {len(frequency_series)} frequencies, optimal_idx: {optimal_idx}")
                
            except Exception as e:
                print(f"Error processing {patient_id}: {e}")
                continue
        
        print(f"Successfully prepared {len(samples)} samples from {len(self.patient_ids)} patients")
        return samples
    
    def _load_frequency_series(self, patient_id: str) -> Tuple[np.ndarray, int]:
        """Load complete frequency series for a patient."""
        # Find patient directory
        patient_dir = self._find_patient_directory(patient_id)
        if patient_dir is None:
            raise ValueError(f"Patient directory not found for {patient_id}")
        
        # Find all DICOM files
        dicom_files = glob.glob(os.path.join(patient_dir, "**", "*.dcm"), recursive=True)
        dicom_files = sorted(dicom_files)
        
        # Load all images
        series_images = []
        for dicom_path in dicom_files:
            image = self._load_dicom_image(dicom_path)
            series_images.append(image[0])  # Remove channel dimension
        
        if len(series_images) == 0:
            raise ValueError(f"No valid DICOM images found for {patient_id}")
        
        series = np.stack(series_images, axis=0)
        
        # Get optimal index
        patient_data = self.patient_info[patient_id]
        optimal_idx = patient_data['image_no'] - 1  # Convert to 0-based
        
        # Validate index
        if optimal_idx >= len(series_images) or optimal_idx < 0:
            optimal_idx = len(series_images) // 2
        
        return series, optimal_idx
    
    def _generate_heart_mask(self, frequency_series: np.ndarray, patient_id: str) -> np.ndarray:
        """Generate heart mask using segmentation model."""
        if self.segmentation_model is None:
            # Return dummy mask if no segmentation model
            return np.ones(self.image_size, dtype=np.float32)
        
        try:
            # Use middle image as reference
            reference_idx = len(frequency_series) // 2
            reference_image = frequency_series[reference_idx]
            
            # Convert to tensor and add batch/channel dimensions
            image_tensor = torch.from_numpy(reference_image).float()
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Generate mask
            with torch.no_grad():
                mask = self.segmentation_model.get_heart_mask(image_tensor, threshold=0.5)
                mask = mask[0, 0].cpu().numpy()  # Remove batch and channel dims
            
            return mask
            
        except Exception as e:
            print(f"Error generating heart mask for {patient_id}: {e}")
            return np.ones(self.image_size, dtype=np.float32)
    
    def _find_patient_directory(self, patient_id: str) -> Optional[str]:
        """Find patient directory."""
        patient_pattern = os.path.join(self.data_root, patient_id)
        matching_dirs = glob.glob(patient_pattern)
        return matching_dirs[0] if matching_dirs else None
    
    def _load_dicom_image(self, dicom_path: str) -> np.ndarray:
        """Load and preprocess DICOM image."""
        try:
            dicom_data = pydicom.dcmread(dicom_path)
            image = dicom_data.pixel_array.astype(np.float32)
            
            # Normalize
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
            # Resize
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
            
            # Add channel dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            print(f"Error loading DICOM {dicom_path}: {e}")
            return np.zeros((1, *self.image_size), dtype=np.float32)
    
    def _pad_or_crop_series(self, series: np.ndarray) -> np.ndarray:
        """Pad or crop series to max_frequencies length."""
        current_length = series.shape[0]
        
        if current_length == self.max_frequencies:
            return series
        elif current_length < self.max_frequencies:
            # Pad with zeros
            pad_length = self.max_frequencies - current_length
            padding = np.zeros((pad_length, *series.shape[1:]), dtype=series.dtype)
            return np.concatenate([series, padding], axis=0)
        else:
            # Crop to max_frequencies (center crop)
            start_idx = (current_length - self.max_frequencies) // 2
            return series[start_idx:start_idx + self.max_frequencies]
    
    def _apply_augmentation(self, frequency_series: np.ndarray, heart_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to frequency series and heart mask."""
        if self.augment_transforms is None:
            return frequency_series, heart_mask
        
        # Convert to PIL Images for transforms
        augmented_series = []
        
        # Apply same transforms to all images in series and mask
        seed = random.randint(0, 2**32 - 1)
        
        for i in range(len(frequency_series)):
            # Set random seed for consistent transforms
            random.seed(seed)
            torch.manual_seed(seed)
            
            # Convert to PIL and apply transforms
            image_pil = Image.fromarray((frequency_series[i] * 255).astype(np.uint8))
            image_aug = self.augment_transforms(image_pil)
            
            # Convert back to numpy
            image_aug = np.array(image_aug, dtype=np.float32) / 255.0
            augmented_series.append(image_aug)
        
        # Apply same transforms to heart mask
        random.seed(seed)
        torch.manual_seed(seed)
        mask_pil = Image.fromarray((heart_mask * 255).astype(np.uint8))
        mask_aug = self.augment_transforms(mask_pil)
        mask_aug = np.array(mask_aug, dtype=np.float32) / 255.0
        
        return np.stack(augmented_series), mask_aug
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        sample = self.samples[idx]
        
        # Get data
        frequency_series = sample['frequency_series'].copy()
        heart_mask = sample['heart_mask'].copy()
        optimal_index = sample['optimal_index']
        series_length = sample['series_length']
        
        # Apply augmentation
        if self.augment_data and random.random() < 0.7:  # 70% chance of augmentation
            frequency_series, heart_mask = self._apply_augmentation(frequency_series, heart_mask)
        
        # Pad or crop to max_frequencies
        original_length = len(frequency_series)
        frequency_series = self._pad_or_crop_series(frequency_series)
        
        # Adjust optimal index if series was cropped
        if original_length > self.max_frequencies:
            start_idx = (original_length - self.max_frequencies) // 2
            optimal_index = max(0, min(self.max_frequencies - 1, optimal_index - start_idx))
        
        # Convert to tensors
        frequency_series = torch.from_numpy(frequency_series).float()
        heart_mask = torch.from_numpy(heart_mask).float()
        optimal_index = torch.tensor(optimal_index, dtype=torch.long)
        
        # Create one-hot encoding for classification
        optimal_one_hot = torch.zeros(self.max_frequencies)
        if 0 <= optimal_index < self.max_frequencies:
            optimal_one_hot[optimal_index] = 1.0
        
        return {
            'frequency_series': frequency_series,
            'heart_mask': heart_mask,
            'optimal_index': optimal_index,
            'optimal_one_hot': optimal_one_hot,
            'series_length': torch.tensor(series_length, dtype=torch.long),
            'patient_id': sample['patient_id']
        }


def get_ml_training_dataloaders(
    data_root: str,
    patient_ids: List[str],
    patient_info: Dict[str, Dict],
    segmentation_model: Optional[object] = None,
    batch_size: int = 4,
    validation_split: float = 0.2,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (256, 256),
    max_frequencies: int = 15,
    device: str = 'cuda'
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for ML frequency selection training.
    
    Args:
        data_root: Root directory containing patient data
        patient_ids: List of all patient IDs
        patient_info: Patient metadata dictionary
        segmentation_model: Pre-trained segmentation model
        batch_size: Batch size for training
        validation_split: Fraction for validation
        num_workers: Number of data loading workers
        image_size: Target image size
        max_frequencies: Maximum frequency series length
        device: Device for segmentation model
        
    Returns:
        train_loader, val_loader
    """
    # Split patients into train/validation
    n_val = max(1, int(len(patient_ids) * validation_split))
    
    # Shuffle and split
    shuffled_ids = patient_ids.copy()
    random.shuffle(shuffled_ids)
    
    val_patient_ids = shuffled_ids[:n_val]
    train_patient_ids = shuffled_ids[n_val:]
    
    print(f"Training patients: {len(train_patient_ids)}")
    print(f"Validation patients: {len(val_patient_ids)}")
    
    # Create datasets
    train_dataset = FrequencySelectionMLDataset(
        data_root=data_root,
        patient_ids=train_patient_ids,
        patient_info=patient_info,
        segmentation_model=segmentation_model,
        image_size=image_size,
        augment_data=True,  # Augmentation for training
        max_frequencies=max_frequencies,
        device=device
    )
    
    val_dataset = FrequencySelectionMLDataset(
        data_root=data_root,
        patient_ids=val_patient_ids,
        patient_info=patient_info,
        segmentation_model=segmentation_model,
        image_size=image_size,
        augment_data=False,  # No augmentation for validation
        max_frequencies=max_frequencies,
        device=device
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return train_loader, val_loader