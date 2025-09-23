#!/usr/bin/env python3
"""
Integrated EMIDEC-Powered Frequency Offset Selection Pipeline

This script orchestrates the complete pipeline:
1. Checks for EMIDEC-trained segmentation model
2. Trains AttentionUNet on EMIDEC if needed
3. Runs frequency offset selection with EMIDEC-trained weights
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import OUTPUT_DIR, EMIDEC_DATASET_ROOT, EMIDEC_PROJECT_ROOT

def main():
    """Main pipeline execution."""
    
    print("=" * 80)
    print("Frequency Offset Selection Pipeline")
    print("=" * 80)
    
    try:
        # Check if EMIDEC dataset is available
        if not os.path.exists(EMIDEC_DATASET_ROOT):
            print(f"ERROR: EMIDEC dataset not available at: §{EMIDEC_DATASET_ROOT}")
            print("Please ensure the EMIDEC dataset is properly downloaded.")
            return False
        
        if not os.path.exists(EMIDEC_PROJECT_ROOT):
            print(f"ERROR: EMIDEC Comparative Analysis project not found at: {EMIDEC_PROJECT_ROOT}")
            print("Please ensure the Comparative Analysis project is available.")
            return False
        
        print("SUCCESS: EMIDEC dataset and project found")
        
        # Check if segmentation model already exists
        segmentation_checkpoint = os.path.join(OUTPUT_DIR, "checkpoints", "segmentation_best.pth")
        
        if os.path.exists(segmentation_checkpoint):
            print(f"SUCCESS: Found existing EMIDEC-trained segmentation model: {segmentation_checkpoint}")
            print("Proceeding directly to frequency offset selection...")
        else:
            print("EMIDEC-trained segmentation model not found")
            print("Starting EMIDEC AttentionUNet training...")
            
            # Import and run integrated EMIDEC training
            try:
                from integrated_emidec_training import train_attention_unet_on_emidec, check_emidec_dataset
                
                # Check dataset
                check_emidec_dataset()
                
                # Train model
                checkpoint_path, best_dice = train_attention_unet_on_emidec()
                
                print("Integrated EMIDEC training completed successfully!")
                print(f"   Checkpoint: {checkpoint_path}")
                print(f"   Best Dice: {best_dice:.4f}")
                
            except Exception as e:
                print(f"Integrated EMIDEC training failed: {str(e)}")
                return False
        
        # Run frequency offset selection
        print("\nStarting Frequency Offset Selection with EMIDEC-trained model...")
        
        freq_script = os.path.join(os.path.dirname(__file__), "run_ml_frequency_selection.py")
        result = subprocess.run([sys.executable, freq_script], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ERROR: Frequency offset selection failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
        
        print("SUCCESS: Frequency offset selection completed successfully!")
        print(result.stdout)
        
        # Show results location
        results_dir = os.path.join(OUTPUT_DIR, "frequency_selection_results")
        print(f"\nResults saved to: {results_dir}")
        
        print("\nPipeline completed successfully!")
        print("Summary:")
        print("   SUCCESS: EMIDEC AttentionUNet trained/loaded")
        print("   SUCCESS: Frequency offset selection completed")
        print("   SUCCESS: Results saved and analyzed")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Pipeline failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)