#!/usr/bin/env python3
"""
Test script to verify data loading and frequency mapping for the frequency offset selection system.
"""

import os
import sys
import glob

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import *

def test_patient_data_structure():
    """Test that patient data structure matches expectations."""
    print("=== Testing Patient Data Structure ===")
    
    for patient_id in list(PATIENT_INFO.keys())[:3]:  # Test first 3 patients
        print(f"\nTesting {patient_id}:")
        
        # Find patient directory
        patient_pattern = os.path.join(DATA_ROOT, patient_id)
        matching_dirs = glob.glob(patient_pattern)
        
        if not matching_dirs:
            print(f"  ❌ Patient directory not found: {patient_pattern}")
            continue
            
        patient_dir = matching_dirs[0]
        print(f"  ✓ Found patient directory: {patient_dir}")
        
        # Find DICOM files
        dicom_pattern = os.path.join(patient_dir, "**", "*.dcm")
        dicom_files = glob.glob(dicom_pattern, recursive=True)
        
        print(f"  ✓ Found {len(dicom_files)} DICOM files")
        
        # Check patient info
        patient_data = PATIENT_INFO[patient_id]
        print(f"  ✓ Manual frequency: {patient_data['manual_freq']} Hz")
        print(f"  ✓ Optimal image number: {patient_data['image_no']}")
        print(f"  ✓ Frequency step: {patient_data['freq_step']} Hz")
        
        # Check frequency mapping
        optimal_idx = get_optimal_image_index(patient_id)
        print(f"  ✓ Optimal image index (0-based): {optimal_idx}")
        
        if optimal_idx >= len(dicom_files):
            print(f"  ⚠️  Warning: Optimal index {optimal_idx} >= series length {len(dicom_files)}")
        
        # Test a few DICOM files
        for i, dicom_path in enumerate(dicom_files[:3]):
            filename = os.path.basename(dicom_path)
            print(f"    DICOM {i+1}: {filename}")

def test_frequency_mapping():
    """Test frequency to image index mapping."""
    print("\n=== Testing Frequency Mapping ===")
    
    # Test standard 25Hz steps (-150 to +150 Hz)
    print("\nStandard 25Hz steps:")
    test_frequencies = [-150, -100, -50, -25, 0, 25, 50, 100, 150]
    
    for freq in test_frequencies:
        idx = frequency_to_image_index(freq, 25)
        back_freq = image_index_to_frequency(idx, 25)
        print(f"  {freq:4d} Hz -> Index {idx:2d} -> {back_freq:4d} Hz")
    
    # Test Patient-KP special case (50Hz steps)
    print("\nPatient-KP 50Hz steps:")
    for freq in [-150, -100, -50, 0, 50, 100, 150]:
        idx = frequency_to_image_index(freq, 50)
        back_freq = image_index_to_frequency(idx, 50)
        print(f"  {freq:4d} Hz -> Index {idx:2d} -> {back_freq:4d} Hz")

def test_annotation_files():
    """Test that annotation files exist for patients."""
    print("\n=== Testing Annotation Files ===")
    
    for patient_id in list(PATIENT_INFO.keys())[:5]:  # Test first 5 patients
        # Extract short ID (e.g., BEV from Patient-BEV)
        short_id = patient_id.replace('Patient-', '')
        annotation_path = os.path.join(ANNOTATIONS_ROOT, f"{short_id}.png")
        
        if os.path.exists(annotation_path):
            print(f"  ✓ {patient_id}: {annotation_path}")
        else:
            print(f"  ❌ {patient_id}: Missing annotation {annotation_path}")

def main():
    """Run all tests."""
    print("Frequency Offset Selection System - Data Verification")
    print("=" * 55)
    
    print(f"Data Root: {DATA_ROOT}")
    print(f"Annotations Root: {ANNOTATIONS_ROOT}")
    print(f"Total Patients: {len(PATIENT_INFO)}")
    
    # Run tests
    test_patient_data_structure()
    test_frequency_mapping()
    test_annotation_files()
    
    print("\n=== Summary ===")
    print("✓ Data structure verification complete")
    print("✓ If no errors above, the system should work correctly")
    print("\nNext steps:")
    print("1. Train segmentation model: python scripts/train_segmentation.py")
    print("2. Run frequency selection: python scripts/run_frequency_selection.py --visualize")

if __name__ == "__main__":
    main()