import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import *
from data.dataset import get_frequency_series_dataloader
from models.segmentation import create_segmentation_model
from processing.frequency_analysis import FrequencyOffsetSelector
from visualization.plotting import FrequencyVisualizationManager

def run_frequency_selection_system(args):
    """
    Run the complete frequency offset selection system.
    
    This function implements the full pipeline:
    1. Load frequency scout series
    2. Generate heart segmentation masks
    3. Perform frequency domain analysis
    4. Generate weighting maps
    5. Select optimal frequency offset
    """
    print("=== Automated Frequency Offset Selection System ===")
    print(f"Input directory: {args.input_dir or DATA_ROOT}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {DEVICE}")
    
    # Setup output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "weighting_maps"), exist_ok=True)
    
    # Load segmentation model
    print("Loading heart segmentation model...")
    segmentation_model = create_segmentation_model(
        in_channels=1,
        out_channels=NUM_CLASSES,
        pretrained=True,
        checkpoint_path=args.segmentation_checkpoint
    )
    segmentation_model = segmentation_model.to(DEVICE)
    segmentation_model.eval()
    
    # Create frequency offset selector
    frequency_selector = FrequencyOffsetSelector()
    
    # Create visualization manager
    if args.visualize:
        viz_manager = FrequencyVisualizationManager(
            output_dir=os.path.join(args.output_dir, "visualizations")
        )
    
    # Get patient list
    if args.patient_id:
        patient_ids = [args.patient_id]
    else:
        patient_ids = list(PATIENT_INFO.keys())
    
    # Create dataloader for frequency series
    dataloader = get_frequency_series_dataloader(
        data_root=args.input_dir or DATA_ROOT,
        patient_ids=patient_ids,
        patient_info=PATIENT_INFO,
        batch_size=1,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE
    )
    
    print(f"Processing {len(patient_ids)} patients...")
    
    results = []
    correct_predictions = 0
    
    # Process each patient
    for batch in tqdm(dataloader, desc="Processing patients"):
        patient_id = batch['patient_id'][0]
        frequency_series = batch['frequency_series'][0]  # Remove batch dimension
        optimal_frequency_idx = batch['optimal_frequency_idx'][0].item()
        manual_frequency = batch['manual_frequency'][0].item()
        
        print(f"\nProcessing {patient_id}...")
        
        # Get patient-specific information
        patient_data = PATIENT_INFO[patient_id]
        freq_step = patient_data['freq_step']
        optimal_image_idx = patient_data['image_no'] - 1  # Convert to 0-based
        manual_frequency = patient_data['manual_freq']
        
        print(f"Manual optimal: Image {patient_data['image_no']} ({manual_frequency} Hz)")
        print(f"Frequency step: {freq_step} Hz")
        print(f"Series length: {frequency_series.shape[0]} images")
        
        # Step 1: Generate heart segmentation for reference image
        # Use the manually selected optimal image for segmentation reference
        if optimal_image_idx < frequency_series.shape[0]:
            reference_idx = optimal_image_idx
        else:
            reference_idx = frequency_series.shape[0] // 2
            print(f"Warning: Using middle image {reference_idx} for segmentation reference")
            
        reference_image = frequency_series[reference_idx:reference_idx+1].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            heart_mask = segmentation_model.get_heart_mask(reference_image, threshold=0.5)
            heart_mask = heart_mask[0, 0].cpu().numpy()  # Remove batch and channel dims
        
        # Step 2: Run frequency offset selection
        frequency_series_np = frequency_series.numpy()
        predicted_idx, analysis_results = frequency_selector.select_optimal_frequency(
            frequency_series_np, heart_mask
        )
        
        # Convert indices to frequencies for display
        def idx_to_freq(idx, step=freq_step, series_length=frequency_series.shape[0]):
            # For frequency series from -150 to +150 Hz:
            # Index 0 = -150 Hz, Index (series_length-1) = +150 Hz
            # Center index = (series_length-1) // 2 corresponds to 0 Hz
            total_range = 300  # -150 to +150 = 300 Hz range
            freq_per_step = total_range / (series_length - 1)
            return int(-150 + (idx * freq_per_step))
        
        predicted_frequency = idx_to_freq(predicted_idx)
        
        print(f"Predicted optimal: Image {predicted_idx + 1} ({predicted_frequency} Hz)")
        print(f"Weighting scores: {[f'{score:.3f}' for score in analysis_results['weighting_scores']]}")
        print(f"Max score at index {predicted_idx} = {analysis_results['weighting_scores'][predicted_idx]:.3f}")
        
        # Step 3: Evaluate prediction
        evaluation = frequency_selector.evaluate_selection(
            predicted_idx, optimal_image_idx, tolerance=TOLERANCE_FRAMES
        )
        
        if evaluation['is_correct']:
            correct_predictions += 1
            print(f"✓ Correct prediction (difference: {evaluation['difference']})")
        else:
            print(f"✗ Incorrect prediction (difference: {evaluation['difference']})")
        
        # Step 4: Save results
        patient_results = {
            'patient_id': patient_id,
            'ground_truth_frequency': manual_frequency,
            'ground_truth_image_no': patient_data['image_no'],
            'ground_truth_index': optimal_image_idx,
            'predicted_frequency': predicted_frequency,
            'predicted_image_no': predicted_idx + 1,
            'predicted_index': predicted_idx,
            'difference': evaluation['difference'],
            'is_correct': evaluation['is_correct'],
            'frequency_step': freq_step,
            'series_length': frequency_series.shape[0],
            'weighting_scores': analysis_results['weighting_scores'],
            'frequency_energies': analysis_results['high_frequency_energies']
        }
        results.append(patient_results)
        
        # Step 5: Generate visualizations
        if args.visualize:
            print("Generating visualizations...")
            
            # Save weighting maps
            weighting_maps = analysis_results['weighting_maps']
            weighting_maps_path = os.path.join(
                args.output_dir, "weighting_maps", f"{patient_id}_weighting_maps.npy"
            )
            np.save(weighting_maps_path, weighting_maps)
            
            # Create visualization plots
            viz_manager.plot_frequency_selection_results(
                patient_id=patient_id,
                frequency_series=frequency_series_np,
                heart_mask=heart_mask,
                analysis_results=analysis_results,
                ground_truth_idx=optimal_image_idx,
                predicted_idx=predicted_idx
            )
    
    # Calculate overall accuracy
    accuracy = correct_predictions / len(results) * 100
    
    print(f"\n=== Final Results ===")
    print(f"Total patients processed: {len(results)}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Tolerance: ±{TOLERANCE_FRAMES} frames")
    
    # Save detailed results
    import json
    results_path = os.path.join(args.output_dir, "detailed_results.json")
    with open(results_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_json_serializable(value) for key, value in obj.items()}
            else:
                return obj
        
        json_results = []
        for result in results:
            json_result = convert_to_json_serializable(result)
            json_results.append(json_result)
        
        json.dump({
            'accuracy': float(accuracy),
            'correct_predictions': int(correct_predictions),
            'total_patients': int(len(results)),
            'tolerance_frames': int(TOLERANCE_FRAMES),
            'patient_results': json_results
        }, f, indent=2)
    
    print(f"Detailed results saved to: {results_path}")
    
    # Print failed cases
    failed_cases = [r for r in results if not r['is_correct']]
    if failed_cases:
        print(f"\nFailed cases ({len(failed_cases)}):")
        for case in failed_cases:
            print(f"  {case['patient_id']}: difference = {case['difference']} frames")


def main():
    parser = argparse.ArgumentParser(description="Run frequency offset selection system")
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Input directory containing DICOM data')
    parser.add_argument('--output_dir', type=str, 
                        default=os.path.join(OUTPUT_DIR, "frequency_selection_results"),
                        help='Output directory for results')
    parser.add_argument('--segmentation_checkpoint', type=str,
                        default=os.path.join(OUTPUT_DIR, "checkpoints", "segmentation_best.pth"),
                        help='Path to segmentation model checkpoint')
    parser.add_argument('--patient_id', type=str, default=None,
                        help='Process specific patient only')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # Validate segmentation checkpoint
    if not os.path.exists(args.segmentation_checkpoint):
        print(f"Error: Segmentation checkpoint not found: {args.segmentation_checkpoint}")
        print("Please train the segmentation model first using train_segmentation.py")
        sys.exit(1)
    
    run_frequency_selection_system(args)


if __name__ == "__main__":
    main()