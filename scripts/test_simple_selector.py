#!/usr/bin/env python3
"""
Test the simple frequency selector on a subset of patients.
"""

import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import *
from data.dataset import get_frequency_series_dataloader
from models.segmentation import create_segmentation_model
from processing.simple_frequency_selector import SimpleFrequencySelector

def test_simple_selector():
    """Test the simple frequency selector."""
    print("=== Testing Simple Frequency Selector ===")
    
    # Load segmentation model
    segmentation_model = create_segmentation_model(
        in_channels=1,
        out_channels=NUM_CLASSES,
        pretrained=True,
        checkpoint_path=os.path.join(OUTPUT_DIR, "checkpoints", "segmentation_best.pth")
    )
    segmentation_model = segmentation_model.to(DEVICE)
    segmentation_model.eval()
    
    # Test on a subset of patients first
    test_patients = ["Patient-BEV", "Patient-HRJ", "Patient-SK", "Patient-CEJP", "Patient-FAJ"]
    
    # Create dataloader
    dataloader = get_frequency_series_dataloader(
        data_root=DATA_ROOT,
        patient_ids=test_patients,
        patient_info=PATIENT_INFO,
        batch_size=1,
        num_workers=0,
        image_size=IMAGE_SIZE
    )
    
    # Initialize selector
    selector = SimpleFrequencySelector()
    
    results = []
    correct_predictions = 0
    total_patients = 0
    
    print(f"Testing on {len(test_patients)} patients...")
    
    for batch in tqdm(dataloader, desc="Testing patients"):
        patient_id = batch['patient_id'][0]
        frequency_series = batch['frequency_series'][0]  # Remove batch dimension
        patient_data = PATIENT_INFO[patient_id]
        
        optimal_image_idx = patient_data['image_no'] - 1  # Convert to 0-based index
        
        print(f"\nProcessing {patient_id}...")
        print(f"Manual optimal: Image {patient_data['image_no']} ({patient_data['manual_freq']} Hz)")
        print(f"Frequency step: {patient_data['freq_step']} Hz")
        print(f"Series length: {frequency_series.shape[0]} images")
        
        # Generate heart mask from reference image
        reference_idx = optimal_image_idx if optimal_image_idx < frequency_series.shape[0] else frequency_series.shape[0] // 2
        reference_image = frequency_series[reference_idx:reference_idx+1].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            heart_mask = segmentation_model.get_heart_mask(reference_image, threshold=0.5)
            heart_mask = heart_mask[0, 0].cpu().numpy()
        
        # Run frequency selection
        frequency_series_np = frequency_series.numpy()
        predicted_idx, analysis_results = selector.select_optimal_frequency(
            frequency_series_np, heart_mask
        )
        
        # Convert predicted index to frequency
        predicted_freq = index_to_frequency(
            predicted_idx, 
            patient_data['freq_step'], 
            frequency_series.shape[0]
        )
        
        predicted_image_no = predicted_idx + 1  # Convert to 1-based
        
        print(f"Predicted optimal: Image {predicted_image_no} ({predicted_freq} Hz)")
        
        # Evaluate prediction
        difference = abs(predicted_idx - optimal_image_idx)
        is_correct = difference <= TOLERANCE_FRAMES
        
        if is_correct:
            print(f"✓ Correct prediction (difference: {difference})")
            correct_predictions += 1
        else:
            print(f"✗ Incorrect prediction (difference: {difference})")
        
        total_patients += 1
        
        # Show quality scores
        quality_scores = analysis_results['quality_scores']
        print(f"Quality scores: {[f'{s:.3f}' for s in quality_scores]}")
        print(f"Max score at index {predicted_idx} = {quality_scores[predicted_idx]:.3f}")
        
        # Store results
        result = {
            'patient_id': patient_id,
            'ground_truth_index': optimal_image_idx,
            'predicted_index': predicted_idx,
            'ground_truth_frequency': patient_data['manual_freq'],
            'predicted_frequency': predicted_freq,
            'difference': difference,
            'is_correct': is_correct,
            'quality_scores': quality_scores
        }
        results.append(result)
    
    # Calculate accuracy
    accuracy = (correct_predictions / total_patients) * 100
    
    print(f"\n=== Test Results ===")
    print(f"Correct predictions: {correct_predictions}/{total_patients}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Tolerance: ±{TOLERANCE_FRAMES} frames")
    
    return results, accuracy

def index_to_frequency(index, freq_step, series_length):
    """Convert image index to frequency offset."""
    center_index = series_length // 2
    frequency_offset = (index - center_index) * freq_step
    return frequency_offset

def main():
    """Run test."""
    results, accuracy = test_simple_selector()
    
    # Save results
    output_path = os.path.join(OUTPUT_DIR, "simple_selector_test_results.json")
    with open(output_path, 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()