import os
import glob
import pydicom
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Tuple, Dict, Optional
import cv2

class FrequencyScoutDataset(Dataset):
    def __init__(
        self, 
        data_root: str,
        annotations_root: str,
        patient_ids: List[str],
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple[int, int] = (256, 256)
    ):
        self.data_root = data_root
        self.annotations_root = annotations_root
        self.patient_ids = patient_ids
        self.transform = transform
        self.image_size = image_size
        
        # Build dataset index
        self.dataset_index = self._build_dataset_index()
        
    def _build_dataset_index(self) -> List[Dict]:
        """Build index of all DICOM files and corresponding annotations."""
        index = []
        
        for patient_id in self.patient_ids:
            patient_dir = self._find_patient_directory(patient_id)
            if patient_dir is None:
                print(f"Warning: Patient directory not found for {patient_id}")
                continue
            dicom_files = self._find_dicom_files(patient_dir)
            annotation_file = self._find_annotation_file(patient_id)
            
            if dicom_files and annotation_file:
                for dicom_file in dicom_files:
                    index.append({
                        'patient_id': patient_id,
                        'dicom_path': dicom_file,
                        'annotation_path': annotation_file,
                        'slice_number': self._extract_slice_number(dicom_file)
                    })
                    
        return index
    
    def _find_patient_directory(self, patient_id: str) -> Optional[str]:
        """Find the directory path for a given patient ID."""
        patient_pattern = os.path.join(self.data_root, patient_id)
        matching_dirs = glob.glob(patient_pattern)
        return matching_dirs[0] if matching_dirs else None
    
    def _find_dicom_files(self, patient_dir: str) -> List[str]:
        """Find all DICOM files in patient directory."""
        # Navigate through the DICOM directory structure
        dicom_pattern = os.path.join(patient_dir, "**", "*.dcm")
        dicom_files = glob.glob(dicom_pattern, recursive=True)
        return sorted(dicom_files)
    
    def _find_annotation_file(self, patient_id: str) -> Optional[str]:
        """Find annotation file for patient."""
        # Extract the short patient ID (e.g., BEV from Patient-BEV)
        short_id = patient_id.replace('Patient-', '')
        annotation_path = os.path.join(self.annotations_root, f"{short_id}.png")
        return annotation_path if os.path.exists(annotation_path) else None
    
    def _extract_slice_number(self, dicom_path: str) -> int:
        """Extract slice number from DICOM filename."""
        filename = os.path.basename(dicom_path)
        # Extract number from filename like "img0001-57.5835_anon.dcm"
        try:
            slice_num = int(filename.split('-')[0].replace('img', '').lstrip('0') or '0')
            return slice_num
        except:
            return 0
    
    def _load_dicom_image(self, dicom_path: str) -> np.ndarray:
        """Load and preprocess DICOM image."""
        try:
            dicom_data = pydicom.dcmread(dicom_path)
            image = dicom_data.pixel_array.astype(np.float32)
            
            # Normalize to 0-1 range
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
            # Resize to target size
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
            
            # Add channel dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            print(f"Error loading DICOM {dicom_path}: {e}")
            # Return zero image as fallback
            return np.zeros((1, *self.image_size), dtype=np.float32)
    
    def _load_annotation(self, annotation_path: str) -> np.ndarray:
        """Load and preprocess PNG annotation."""
        try:
            # Load PNG annotation
            annotation = Image.open(annotation_path)
            annotation = annotation.convert('L')  # Convert to grayscale
            annotation = np.array(annotation)
            
            # Resize to target size
            annotation = cv2.resize(annotation, self.image_size, interpolation=cv2.INTER_NEAREST)
            
            # Convert to binary mask (assuming heart is non-zero)
            mask = (annotation > 0).astype(np.float32)
            
            return mask
            
        except Exception as e:
            print(f"Error loading annotation {annotation_path}: {e}")
            # Return zero mask as fallback
            return np.zeros(self.image_size, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.dataset_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset."""
        sample_info = self.dataset_index[idx]
        
        # Load DICOM image
        image = self._load_dicom_image(sample_info['dicom_path'])
        
        # Load annotation
        mask = self._load_annotation(sample_info['annotation_path'])
        
        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        
        # Apply transforms if provided
        if self.transform:
            # Note: transforms should handle both image and mask
            sample = {'image': image, 'mask': mask}
            sample = self.transform(sample)
            image, mask = sample['image'], sample['mask']
        
        return {
            'image': image,
            'mask': mask,
            'patient_id': sample_info['patient_id'],
            'slice_number': sample_info['slice_number'],
            'dicom_path': sample_info['dicom_path']
        }


class FrequencySeriesDataset(Dataset):
    """
    Dataset for complete frequency scout series (multiple frequency offsets per patient).
    
    This dataset is used for the frequency offset selection task, where each sample
    contains the complete frequency series for a patient.
    """
    
    def __init__(
        self,
        data_root: str,
        patient_ids: List[str],
        patient_info: Dict[str, Dict],
        image_size: Tuple[int, int] = (256, 256)
    ):
        """
        Initialize the frequency series dataset.
        
        Args:
            data_root: Root directory containing patient DICOM data
            patient_ids: List of patient IDs to include
            patient_info: Dictionary with patient metadata (manual frequency, image count)
            image_size: Target image size
        """
        self.data_root = data_root
        self.patient_ids = patient_ids
        self.patient_info = patient_info
        self.image_size = image_size
        
    def _load_frequency_series(self, patient_id: str) -> Tuple[np.ndarray, int]:
        """
        Load complete frequency series for a patient in frequency order.
        
        Returns:
            series: Array of shape (n_frequencies, height, width) ordered by frequency
            optimal_frequency_idx: Index of manually selected optimal frequency (0-based)
        """
        # Find patient directory
        patient_dir = self._find_patient_directory(patient_id)
        if patient_dir is None:
            raise ValueError(f"Patient directory not found for {patient_id}")
        
        # Find all DICOM files
        dicom_files = glob.glob(os.path.join(patient_dir, "**", "*.dcm"), recursive=True)
        dicom_files = sorted(dicom_files)  # Sort by filename
        
        print(f"Found {len(dicom_files)} DICOM files for {patient_id}")
        
        # Load all images in the series
        series_images = []
        for dicom_path in dicom_files:
            image = self._load_dicom_image(dicom_path)
            series_images.append(image[0])  # Remove channel dimension
        
        if len(series_images) == 0:
            raise ValueError(f"No valid DICOM images found for {patient_id}")
        
        series = np.stack(series_images, axis=0)
        
        # Get optimal image index (convert from 1-based to 0-based)
        patient_data = self.patient_info[patient_id]
        optimal_idx = patient_data['image_no'] - 1
        
        # Validate the optimal index
        if optimal_idx >= len(series_images) or optimal_idx < 0:
            print(f"Warning: Invalid optimal index {optimal_idx} for {patient_id}, using middle image")
            optimal_idx = len(series_images) // 2
        
        print(f"Patient {patient_id}: {len(series_images)} images, optimal index: {optimal_idx}")
        
        return series, int(optimal_idx)
    
    def _find_patient_directory(self, patient_id: str) -> Optional[str]:
        """Find the directory path for a given patient ID."""
        patient_pattern = os.path.join(self.data_root, patient_id)
        matching_dirs = glob.glob(patient_pattern)
        return matching_dirs[0] if matching_dirs else None
    
    def _load_dicom_image(self, dicom_path: str) -> np.ndarray:
        """Load and preprocess DICOM image."""
        try:
            dicom_data = pydicom.dcmread(dicom_path)
            image = dicom_data.pixel_array.astype(np.float32)
            
            # Normalize to 0-1 range
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
            # Resize to target size
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
            
            # Add channel dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            print(f"Error loading DICOM {dicom_path}: {e}")
            return np.zeros((1, *self.image_size), dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.patient_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a complete frequency series for a patient."""
        patient_id = self.patient_ids[idx]
        
        # Load frequency series
        series, optimal_idx = self._load_frequency_series(patient_id)
        
        return {
            'patient_id': patient_id,
            'frequency_series': torch.from_numpy(series).float(),
            'optimal_frequency_idx': optimal_idx,
            'manual_frequency': self.patient_info[patient_id]['manual_freq'],
            'n_images': self.patient_info[patient_id]['image_no']
        }


def get_segmentation_dataloaders(
    data_root: str,
    annotations_root: str,
    patient_ids: List[str],
    batch_size: int = 8,
    validation_split: float = 0.2,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (256, 256)
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for heart segmentation.
    
    Args:
        data_root: Root directory containing patient DICOM data
        annotations_root: Directory containing PNG annotations
        patient_ids: List of all patient IDs
        batch_size: Batch size for dataloaders
        validation_split: Fraction of data to use for validation
        num_workers: Number of worker processes
        image_size: Target image size
        
    Returns:
        train_loader, val_loader
    """
    # Split patients into train/validation
    n_val = int(len(patient_ids) * validation_split)
    val_patient_ids = patient_ids[:n_val]
    train_patient_ids = patient_ids[n_val:]
    
    # Define transforms (can be extended with augmentations)
    transform = transforms.Compose([
        # Add data augmentations here if needed
    ])
    
    # Create datasets
    train_dataset = FrequencyScoutDataset(
        data_root=data_root,
        annotations_root=annotations_root,
        patient_ids=train_patient_ids,
        transform=transform,
        image_size=image_size
    )
    
    val_dataset = FrequencyScoutDataset(
        data_root=data_root,
        annotations_root=annotations_root,
        patient_ids=val_patient_ids,
        transform=None,  # No augmentation for validation
        image_size=image_size
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


def get_frequency_series_dataloader(
    data_root: str,
    patient_ids: List[str],
    patient_info: Dict[str, Dict],
    batch_size: int = 1,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (256, 256)
) -> DataLoader:
    """
    Create dataloader for frequency series analysis.
    
    Args:
        data_root: Root directory containing patient DICOM data
        patient_ids: List of patient IDs
        patient_info: Dictionary with patient metadata
        batch_size: Batch size (usually 1 for frequency series)
        num_workers: Number of worker processes
        image_size: Target image size
        
    Returns:
        DataLoader for frequency series
    """
    dataset = FrequencySeriesDataset(
        data_root=data_root,
        patient_ids=patient_ids,
        patient_info=patient_info,
        image_size=image_size
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )