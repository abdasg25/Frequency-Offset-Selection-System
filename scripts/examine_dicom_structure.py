#!/usr/bin/env python3
"""
Script to examine the exact structure of your DICOM frequency scout data.
"""

import os
import sys
import glob
import pydicom
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import DATA_ROOT, PATIENT_INFO

def examine_patient_structure(patient_id="Patient-BEV"):
    """Examine the detailed structure of a patient's DICOM data."""
    print(f"=== Examining {patient_id} Structure ===")
    
    # Find patient directory
    patient_pattern = os.path.join(DATA_ROOT, patient_id)
    matching_dirs = glob.glob(patient_pattern)
    
    if not matching_dirs:
        print(f"❌ Patient directory not found: {patient_pattern}")
        return
    
    patient_dir = matching_dirs[0]
    print(f"✓ Patient directory: {patient_dir}")
    
    # Walk through directory structure
    print("\nDirectory structure:")
    for root, dirs, files in os.walk(patient_dir):
        level = root.replace(patient_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.dcm'):
                print(f"{subindent}{file}")
    
    # Find all DICOM files
    dicom_files = glob.glob(os.path.join(patient_dir, "**", "*.dcm"), recursive=True)
    dicom_files = sorted(dicom_files)
    
    print(f"\n✓ Found {len(dicom_files)} DICOM files")
    
    # Examine DICOM metadata
    print("\nDICOM Analysis:")
    for i, dicom_path in enumerate(dicom_files):
        try:
            ds = pydicom.dcmread(dicom_path)
            
            # Extract useful metadata
            filename = os.path.basename(dicom_path)
            image_shape = ds.pixel_array.shape if hasattr(ds, 'pixel_array') else 'N/A'
            
            # Try to extract frequency information if available
            freq_info = "N/A"
            
            # Check common DICOM tags that might contain frequency info
            if hasattr(ds, 'ImageComments'):
                freq_info = ds.ImageComments
            elif hasattr(ds, 'SeriesDescription'):
                freq_info = ds.SeriesDescription
            elif hasattr(ds, 'ProtocolName'):
                freq_info = ds.ProtocolName
            
            print(f"  Image {i+1:2d}: {filename}")
            print(f"    Shape: {image_shape}")
            print(f"    Info: {freq_info}")
            
            # Try to extract frequency from filename
            if 'img' in filename and '-' in filename:
                parts = filename.split('-')
                if len(parts) > 1:
                    freq_part = parts[1].split('_')[0]
                    print(f"    Filename freq: {freq_part}")
            
            print()
            
        except Exception as e:
            print(f"  Error reading {dicom_path}: {e}")
    
    # Show patient info from config
    if patient_id in PATIENT_INFO:
        patient_data = PATIENT_INFO[patient_id]
        print(f"Configuration data:")
        print(f"  Manual frequency: {patient_data['manual_freq']} Hz")
        print(f"  Optimal image number: {patient_data['image_no']}")
        print(f"  Frequency step: {patient_data['freq_step']} Hz")
        
        optimal_idx = patient_data['image_no'] - 1
        if optimal_idx < len(dicom_files):
            print(f"  Optimal image file: {os.path.basename(dicom_files[optimal_idx])}")
        else:
            print(f"  ⚠️  Optimal index {optimal_idx} exceeds available files ({len(dicom_files)})")

def examine_frequency_series_pattern():
    """Try to understand the frequency series pattern across patients."""
    print("\n=== Frequency Series Pattern Analysis ===")
    
    series_lengths = {}
    
    for patient_id in list(PATIENT_INFO.keys())[:5]:  # First 5 patients
        patient_pattern = os.path.join(DATA_ROOT, patient_id)
        matching_dirs = glob.glob(patient_pattern)
        
        if matching_dirs:
            patient_dir = matching_dirs[0]
            dicom_files = glob.glob(os.path.join(patient_dir, "**", "*.dcm"), recursive=True)
            series_lengths[patient_id] = len(dicom_files)
            
            patient_data = PATIENT_INFO[patient_id]
            print(f"{patient_id}:")
            print(f"  DICOM files: {len(dicom_files)}")
            print(f"  Freq step: {patient_data['freq_step']} Hz")
            print(f"  Optimal image: {patient_data['image_no']}")
            print(f"  Manual freq: {patient_data['manual_freq']} Hz")
            
            # Calculate expected series length
            if patient_data['freq_step'] == 25:
                expected_length = 13  # -150 to +150 in 25Hz steps = 13 images
            elif patient_data['freq_step'] == 50:
                expected_length = 7   # -150 to +150 in 50Hz steps = 7 images
            else:
                expected_length = "Unknown"
            
            print(f"  Expected length: {expected_length}")
            
            if len(dicom_files) != expected_length and expected_length != "Unknown":
                print(f"  ⚠️  Length mismatch!")
            
            print()
    
    print("Series length summary:")
    for patient_id, length in series_lengths.items():
        freq_step = PATIENT_INFO[patient_id]['freq_step']
        print(f"  {patient_id}: {length} images ({freq_step}Hz steps)")

def main():
    """Run DICOM structure examination."""
    print("DICOM Frequency Scout Data Structure Analysis")
    print("=" * 50)
    
    # Examine one patient in detail
    examine_patient_structure("Patient-BEV")
    
    # Look at frequency series patterns
    examine_frequency_series_pattern()
    
    print("\n=== Recommendations ===")
    print("1. Verify that DICOM files are ordered by frequency offset")
    print("2. Check if frequency information is encoded in filenames or DICOM metadata")
    print("3. Ensure the 'image_no' in your table corresponds to the correct frequency")
    print("4. The system assumes filenames are sorted alphabetically by frequency")

if __name__ == "__main__":
    main()