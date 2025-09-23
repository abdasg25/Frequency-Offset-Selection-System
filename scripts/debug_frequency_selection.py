#!/usr/bin/env python3
"""
Debug script to analyze why the frequency selection algorithm is not working well.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import *
from data.dataset import get_frequency_series_dataloader
from models.segmentation import create_segmentation_model
from processing.frequency_analysis import FrequencyOffsetSelector

def debug_single_patient(patient_id="Patient-BEV"):
    """Debug the frequency selection for a single patient."""
    print(f"=== Debugging {patient_id} ===")
    
    # Load segmentation model
    segmentation_model = create_segmentation_model(
        in_channels=1,
        out_channels=NUM_CLASSES,
        pretrained=True,
        checkpoint_path=os.path.join(OUTPUT_DIR, "checkpoints", "segmentation_best.pth")
    )
    segmentation_model = segmentation_model.to(DEVICE)
    segmentation_model.eval()
    
    # Create dataloader for single patient
    dataloader = get_frequency_series_dataloader(
        data_root=DATA_ROOT,
        patient_ids=[patient_id],
        patient_info=PATIENT_INFO,
        batch_size=1,
        num_workers=0,
        image_size=IMAGE_SIZE
    )
    
    # Get patient data
    batch = next(iter(dataloader))
    frequency_series = batch['frequency_series'][0]  # Remove batch dimension
    patient_data = PATIENT_INFO[patient_id]
    optimal_image_idx = patient_data['image_no'] - 1
    
    print(f"Series shape: {frequency_series.shape}")
    print(f"Ground truth optimal index: {optimal_image_idx}")
    print(f"Ground truth frequency: {patient_data['manual_freq']} Hz")
    
    # Generate heart mask from reference image
    reference_idx = optimal_image_idx if optimal_image_idx < frequency_series.shape[0] else frequency_series.shape[0] // 2
    reference_image = frequency_series[reference_idx:reference_idx+1].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        heart_mask = segmentation_model.get_heart_mask(reference_image, threshold=0.5)
        heart_mask = heart_mask[0, 0].cpu().numpy()
    
    print(f"Heart mask coverage: {np.sum(heart_mask)/heart_mask.size*100:.1f}% of image")
    
    # Run frequency analysis
    frequency_selector = FrequencyOffsetSelector()
    frequency_series_np = frequency_series.numpy()
    
    predicted_idx, analysis_results = frequency_selector.select_optimal_frequency(
        frequency_series_np, heart_mask
    )
    
    print(f"Predicted optimal index: {predicted_idx}")
    
    # Analyze results
    weighting_scores = analysis_results['weighting_scores']
    energy_values = analysis_results['high_frequency_energies']
    selected_indices = analysis_results['selected_low_freq_indices']
    
    print(f"\nSelected low HF indices: {selected_indices}")
    print(f"Energy values: {[f'{e:.2f}' for e in energy_values]}")
    print(f"Weighting scores: {[f'{w:.3f}' for w in weighting_scores]}")
    
    # Create debug visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Original images (first few)
    for i in range(min(3, frequency_series.shape[0])):
        axes[0, i].imshow(frequency_series_np[i], cmap='gray')
        axes[0, i].set_title(f'Image {i+1}' + (' (GT)' if i == optimal_image_idx else '') + (' (Pred)' if i == predicted_idx else ''))
        axes[0, i].axis('off')
    
    # 2. Heart mask
    axes[1, 0].imshow(heart_mask, cmap='red', alpha=0.7)
    axes[1, 0].imshow(frequency_series_np[reference_idx], cmap='gray', alpha=0.5)
    axes[1, 0].set_title('Heart Mask Overlay')
    axes[1, 0].axis('off')
    
    # 3. Energy plot
    axes[1, 1].plot(energy_values, 'b-o', label='HF Energy')
    axes[1, 1].axvline(optimal_image_idx, color='green', linestyle='--', label='GT')
    axes[1, 1].axvline(predicted_idx, color='red', linestyle='--', label='Predicted')
    axes[1, 1].set_title('High-Frequency Energy')
    axes[1, 1].set_xlabel('Image Index')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 4. Weighting scores
    axes[1, 2].plot(weighting_scores, 'r-o', label='Weighting Score')
    axes[1, 2].axvline(optimal_image_idx, color='green', linestyle='--', label='GT')
    axes[1, 2].axvline(predicted_idx, color='red', linestyle='--', label='Predicted')
    axes[1, 2].set_title('Weighting Scores')
    axes[1, 2].set_xlabel('Image Index')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    # Save debug plot
    debug_path = os.path.join(OUTPUT_DIR, f"debug_{patient_id}.png")
    plt.savefig(debug_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Debug visualization saved: {debug_path}")
    
    # Check for potential issues
    print(f"\n=== Potential Issues ===")
    
    
    # Show which images were selected for median calculation
    print(f"\nImages selected for median (low HF): {[i+1 for i in selected_indices]}")
    print(f"Their HF energies: {[energy_values[i] for i in selected_indices]}")

def main():
    """Run debug analysis."""
    print("Frequency Selection Algorithm Debug Analysis")
    print("=" * 50)
    
    # Debug a few patients
    debug_patients = ["Patient-BEV", "Patient-CEJP", "Patient-HRJ"]
    
    for patient_id in debug_patients:
        try:
            debug_single_patient(patient_id)
            print("\n" + "="*50 + "\n")
        except Exception as e:
            print(f"Error debugging {patient_id}: {e}")

if __name__ == "__main__":
    main()