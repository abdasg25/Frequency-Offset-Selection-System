#!/usr/bin/env python3

import os
import sys
import shutil
import subprocess
import torch
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import OUTPUT_DIR, EMIDEC_DATASET_ROOT, EMIDEC_PROJECT_ROOT

def check_emidec_dataset():
    """Check if EMIDEC dataset is available."""
    
    if not os.path.exists(EMIDEC_DATASET_ROOT):
        raise FileNotFoundError(
            f"❌ EMIDEC dataset not available at: {EMIDEC_DATASET_ROOT}\n"
            f"Please ensure the EMIDEC dataset is properly downloaded and placed in the correct location."
        )
    
    # Check if dataset has the required structure
    case_dirs = [d for d in os.listdir(EMIDEC_DATASET_ROOT) if d.startswith('Case_')]
    if len(case_dirs) == 0:
        raise ValueError(
            f"❌ EMIDEC dataset structure invalid at: {EMIDEC_DATASET_ROOT}\n"
            f"No Case_ directories found. Please check dataset integrity."
        )
    
    print(f"✅ EMIDEC dataset found with {len(case_dirs)} cases")
    return EMIDEC_DATASET_ROOT

def train_emidec_attention_unet():
    """Train AttentionUNet on EMIDEC dataset using the comparative analysis pipeline."""
    
    print("🔥 Starting EMIDEC AttentionUNet Training...")
    
    # Check EMIDEC dataset availability
    try:
        emidec_path = check_emidec_dataset()
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return None
    
    # Path to the comparative analysis project
    if not os.path.exists(EMIDEC_PROJECT_ROOT):
        raise FileNotFoundError(
            f"❌ Comparative Analysis project not found at: {EMIDEC_PROJECT_ROOT}"
        )
    
    print(f"📂 Using EMIDEC project at: {EMIDEC_PROJECT_ROOT}")
    
    # Change to the EMIDEC project directory and run training
    original_cwd = os.getcwd()
    
    try:
        os.chdir(EMIDEC_PROJECT_ROOT)
        print(f"📁 Changed working directory to: {EMIDEC_PROJECT_ROOT}")
        
        # Run the exact training command for AttentionUNet
        print("🚀 Running: python -m scripts.train --model AttentionUNet")
        
        result = subprocess.run([
            sys.executable, "-m", "scripts.train", "--model", "AttentionUNet"
        ], capture_output=True, text=True, cwd=EMIDEC_PROJECT_ROOT)
        
        if result.returncode != 0:
            print(f"❌ Training failed with error:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None
        
        print("✅ EMIDEC AttentionUNet training completed successfully!")
        print(f"Training output:\n{result.stdout}")
        
        # Find the trained model checkpoint
        checkpoint_path = os.path.join(EMIDEC_PROJECT_ROOT, "output", "checkpoints", "AttentionUNet", "AttentionUNet_best.pth")
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ Trained checkpoint not found at: {checkpoint_path}")
            return None
        
        return checkpoint_path
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return None
        
    finally:
        # Always return to original directory
        os.chdir(original_cwd)
        print(f"📁 Returned to original directory: {original_cwd}")

def copy_emidec_weights_to_frequency_project(emidec_checkpoint_path):
    """Copy trained EMIDEC weights to frequency offset selection project."""
    
    if emidec_checkpoint_path is None:
        print("❌ No checkpoint path provided")
        return None
    
    # Create frequency project checkpoint directory
    freq_checkpoint_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    os.makedirs(freq_checkpoint_dir, exist_ok=True)
    
    # Destination path for the segmentation model
    dest_checkpoint_path = os.path.join(freq_checkpoint_dir, "segmentation_best.pth")
    
    try:
        # Copy the checkpoint
        shutil.copy2(emidec_checkpoint_path, dest_checkpoint_path)
        print(f"✅ Copied EMIDEC weights from: {emidec_checkpoint_path}")
        print(f"✅ To frequency project at: {dest_checkpoint_path}")
        
        # Verify the copy
        if os.path.exists(dest_checkpoint_path):
            checkpoint = torch.load(dest_checkpoint_path, map_location='cpu')
            print(f"✅ Checkpoint verification successful")
            print(f"   - Epoch: {checkpoint.get('epoch', 'Unknown')}")
            print(f"   - Validation Dice: {checkpoint.get('val_dice', 'Unknown')}")
            
            return dest_checkpoint_path
        else:
            print(f"❌ Copy verification failed")
            return None
            
    except Exception as e:
        print(f"❌ Error copying checkpoint: {str(e)}")
        return None

def main():
    """Main function to orchestrate EMIDEC training and weight transfer."""
    
    print("=" * 80)
    print("🧠 EMIDEC AttentionUNet Training for Frequency Offset Selection")
    print("=" * 80)
    
    # Step 1: Train AttentionUNet on EMIDEC dataset
    print("\n📋 Step 1: Training AttentionUNet on EMIDEC Dataset")
    emidec_checkpoint = train_emidec_attention_unet()
    
    if emidec_checkpoint is None:
        print("❌ EMIDEC training failed. Cannot proceed with frequency offset selection.")
        return False
    
    # Step 2: Copy weights to frequency offset selection project
    print("\n📋 Step 2: Copying Trained Weights to Frequency Project")
    freq_checkpoint = copy_emidec_weights_to_frequency_project(emidec_checkpoint)
    
    if freq_checkpoint is None:
        print("❌ Weight transfer failed. Cannot proceed with frequency offset selection.")
        return False
    
    print("\n🎉 SUCCESS! EMIDEC-trained segmentation model is ready for frequency offset selection!")
    print(f"✅ Segmentation model checkpoint: {freq_checkpoint}")
    print("\n📝 Next Steps:")
    print("   1. Run: python scripts/run_ml_frequency_selection.py")
    print("   2. The pipeline will now use EMIDEC-trained AttentionUNet for heart segmentation")
    
    return True

if __name__ == "__main__":
    main()