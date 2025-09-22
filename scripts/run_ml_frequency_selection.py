#!/usr/bin/env python3
"""
Research-based automated frequency offset selection system.
Implementation of the published methodology using high-frequency analysis and adaptive weighting.
"""

import os
import sys
import json
import numpy as np
import cv2
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import *
from data.dataset import get_frequency_series_dataloader
from models.segmentation import create_segmentation_model
from data.advanced_frequency_analysis import AdvancedFrequencyOffsetSelector

def convert_to_json_serializable(obj):
    """Convert numpy types to JSON serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    else:
        return obj

def index_to_frequency(index, freq_step, series_length):
    """Convert image index to frequency offset."""
    center_index = series_length // 2
    frequency_offset = (index - center_index) * freq_step
    return frequency_offset

def run_frequency_selection_system():
    """Run the complete research-based frequency offset selection system."""
    
    print("=== Improved Research-Based Automated Frequency Offset Selection System ===")
    print(f"Input directory: {DATA_ROOT}")
    print(f"Output directory: {os.path.join(OUTPUT_DIR, 'frequency_selection_results')}")
    print("🎯 Target: 80%+ accuracy with improved multi-feature analysis")
    
    # Create output directory
    output_dir = os.path.join(OUTPUT_DIR, "frequency_selection_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load heart segmentation model
    print("Loading heart segmentation model...")
    segmentation_model = create_segmentation_model(
        in_channels=1,
        out_channels=NUM_CLASSES,
        pretrained=True,
        checkpoint_path=os.path.join(OUTPUT_DIR, "checkpoints", "segmentation_best.pth")
    )
    segmentation_model.eval()
    
    # Initialize advanced frequency offset selector for higher accuracy
    print("Initializing advanced frequency offset selector...")
    frequency_selector = AdvancedFrequencyOffsetSelector(
        num_references=7,  # Increased for better analysis
        temporal_window=3  # Temporal consistency analysis
    )
    
    # Get all patient IDs
    patient_ids = list(PATIENT_INFO.keys())
    print(f"Processing {len(patient_ids)} patients...")
    
    # Create dataloader for frequency series
    dataloader = get_frequency_series_dataloader(
        data_root=DATA_ROOT,
        patient_ids=patient_ids,
        patient_info=PATIENT_INFO,
        batch_size=1,
        num_workers=0,
        image_size=IMAGE_SIZE
    )
    
    # Process all patients
    patient_results = []
    correct_predictions = 0
    total_patients = 0
    
    for batch in tqdm(dataloader, desc="Processing patients"):
        patient_id = batch['patient_id'][0]
        frequency_series = batch['frequency_series'][0]  # Remove batch dimension
        patient_data = PATIENT_INFO[patient_id]
        
        # Ground truth information
        optimal_image_idx = patient_data['image_no'] - 1  # Convert to 0-based index
        ground_truth_freq = patient_data['manual_freq']
        freq_step = patient_data['freq_step']
        series_length = frequency_series.shape[0]
        
        print(f"\nProcessing {patient_id}...")
        print(f"Manual optimal: Image {patient_data['image_no']} ({ground_truth_freq} Hz)")
        print(f"Frequency step: {freq_step} Hz")
        print(f"Series length: {series_length} images")
        
        # Generate heart mask using segmentation model
        # Use middle image as reference for heart mask generation
        reference_idx = series_length // 2
        reference_image = frequency_series[reference_idx:reference_idx+1].unsqueeze(0)
        
        heart_mask = segmentation_model.get_heart_mask(reference_image, threshold=0.5)
        heart_mask = heart_mask[0, 0].numpy()
        
        # Convert frequency series to numpy for processing
        frequency_series_np = frequency_series.numpy()
        
        # Convert to list of 2D images
        fs_images = [frequency_series_np[i] for i in range(series_length)]
        
        # Run advanced frequency offset selection
        print(f"Analyzing patient {patient_id}...")
        print("Advanced multi-criteria analysis...")
        print("Extracting comprehensive features...")
        print("Applying intelligent selection strategy...")
        predicted_idx, analysis_results = frequency_selector.select_optimal_frequency_offset(
            fs_images, heart_mask
        )
        
        # Convert predicted index to frequency
        predicted_freq = index_to_frequency(predicted_idx, freq_step, series_length)
        predicted_image_no = predicted_idx + 1  # Convert to 1-based
        
        print(f"Advanced prediction: Image {predicted_image_no} ({predicted_freq} Hz)")
        print(f"Quality scores: {[f'{score:.4f}' for score in analysis_results['quality_scores']]}")
        print(f"Combined scores: {[f'{score:.4f}' for score in analysis_results['combined_scores']]}")
        print(f"Selection strategy: {analysis_results['selection_strategy']}")
        
        # Evaluate prediction
        difference = abs(predicted_idx - optimal_image_idx)
        is_correct = difference <= TOLERANCE_FRAMES
        
        if is_correct:
            print(f"✓ Correct prediction (difference: {difference})")
            correct_predictions += 1
        else:
            print(f"✗ Incorrect prediction (difference: {difference})")
        
        total_patients += 1
        
        # Store results
        result = {
            'patient_id': patient_id,
            'ground_truth_frequency': ground_truth_freq,
            'ground_truth_image_no': patient_data['image_no'],
            'ground_truth_index': optimal_image_idx,
            'predicted_frequency': predicted_freq,
            'predicted_image_no': predicted_image_no,
            'predicted_index': predicted_idx,
            'difference': difference,
            'is_correct': is_correct,
            'frequency_step': freq_step,
            'series_length': series_length,
            'quality_scores': analysis_results.get('quality_scores', []),
            'temporal_scores': analysis_results.get('temporal_scores', []),
            'combined_scores': analysis_results.get('combined_scores', []),
            'stability_scores': analysis_results.get('stability_scores', []),
            'analysis_details': {
                'selected_index': analysis_results['selected_index'],
                'selection_strategy': analysis_results['selection_strategy'],
                'feature_count': len(analysis_results.get('feature_matrix', [])[0]) if analysis_results.get('feature_matrix') else 0
            }
        }
        
        patient_results.append(result)
    
    # Calculate final accuracy
    accuracy = (correct_predictions / total_patients) * 100
    
    print(f"\n=== Advanced Frequency Selection Results ===")
    print(f"Total patients processed: {total_patients}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Tolerance: ±{TOLERANCE_FRAMES} frames")
    

    
    # Show failed cases
    failed_cases = [result for result in patient_results if not result['is_correct']]
    if failed_cases:
        print(f"\nFailed cases ({len(failed_cases)}):")
        for result in failed_cases:
            print(f"  {result['patient_id']}: difference = {result['difference']} frames")
    
    # Save detailed results
    final_results = {
        'accuracy': accuracy,
        'correct_predictions': correct_predictions,
        'total_patients': total_patients,
        'tolerance_frames': TOLERANCE_FRAMES,
        'approach': 'Advanced research-based methodology with multi-criteria analysis and intelligent selection',
        'methodology': {
            'step1': 'Heart segmentation using pre-trained AttentionUNet',
            'step2': 'Advanced feature extraction (gradient, texture, frequency, artifacts)',
            'step3': 'Temporal consistency analysis across image series',
            'step4': 'Multi-criteria quality assessment with normalization',
            'step5': 'Intelligent selection strategy with stability and bias considerations',
            'step6': 'Adaptive center bias and edge penalty for optimal selection'
        },
        'patient_results': patient_results
    }
    
    # Convert to JSON serializable format
    final_results = convert_to_json_serializable(final_results)
    
    # Save results
    results_path = os.path.join(output_dir, "advanced_methodology_results.json")
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")
    
    return final_results

def main():
    """Main function."""
    results = run_frequency_selection_system()
    
    # if results is not None:
    #     # print(f"\n🎯 Target accuracy: 92.1% (published research benchmark)")
    #     if results['accuracy'] >= 92.1:
    #         print("🎉 Target accuracy achieved!")
    #     else:
    #         print(f"📈 Need {92.1 - results['accuracy']:.1f} more percentage points to reach target")
    
    return results

if __name__ == "__main__":
    main()