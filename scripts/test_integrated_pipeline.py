#!/usr/bin/env python3
"""
Test script for integrated EMIDEC training
Tests the pipeline without running the full training
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_integrated_pipeline():
    """Test the integrated pipeline components."""
    
    print("=" * 60)
    print("Testing Integrated EMIDEC Pipeline")
    print("=" * 60)
    
    try:
        # Test imports
        print("Testing imports...")
        
        from configs.config import OUTPUT_DIR, EMIDEC_DATASET_ROOT
        print(f"SUCCESS: Config imported - EMIDEC dataset: {EMIDEC_DATASET_ROOT}")
        
        from scripts.integrated_emidec_training import check_emidec_dataset
        print("SUCCESS: Integrated EMIDEC training module imported")
        
        # Test dataset check
        print("\nTesting EMIDEC dataset check...")
        try:
            check_emidec_dataset()
            print("SUCCESS: EMIDEC dataset check passed")
        except Exception as e:
            print(f"ERROR: EMIDEC dataset check failed: {str(e)}")
            return False
        
        # Test model imports
        print("\nTesting MONAI model imports...")
        from monai.networks.nets import AttentionUnet
        from monai.losses import DiceLoss
        from monai.metrics import DiceMetric
        print("SUCCESS: MONAI imports successful")
        
        # Test model creation
        print("\nTesting AttentionUNet model creation...")
        import torch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        model = AttentionUnet(
            spatial_dims=3, 
            in_channels=1, 
            out_channels=1,
            channels=(16, 32, 64, 128, 256), 
            strides=(2, 2, 2, 2)
        ).to(device)
        
        print(f"SUCCESS: AttentionUNet created successfully on {device}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test checkpoint directory
        print("\nTesting checkpoint directory...")
        checkpoint_dir = os.path.join(OUTPUT_DIR, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"SUCCESS: Checkpoint directory ready: {checkpoint_dir}")
        
        print("\nAll tests passed! Integrated pipeline is ready.")
        print("\nTo run the full pipeline:")
        print("   python scripts/run_pipeline.py")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_integrated_pipeline()
    sys.exit(0 if success else 1)