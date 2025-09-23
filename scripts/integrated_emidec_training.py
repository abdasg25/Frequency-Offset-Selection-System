"""
Integrated EMIDEC Training Module
Directly integrates the MONAI Comparative Analysis code into the frequency pipeline
"""

import os
import sys
import csv
import torch
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# MONAI imports
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    Resized, EnsureTyped, Lambdad, Compose, AsDiscrete
)
from monai.networks.nets import AttentionUnet
from monai.losses import DiceLoss
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric

# Add frequency project to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import OUTPUT_DIR, EMIDEC_DATASET_ROOT, EMIDEC_PROJECT_ROOT

# EMIDEC Training Configuration (copied from comparative analysis)
EMIDEC_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EMIDEC_SEED = 42
EMIDEC_NUM_WORKERS = 0
EMIDEC_BATCH_SIZE = 1
EMIDEC_ROI = (128, 128, 128)
EMIDEC_MAX_EPOCHS = 5
EMIDEC_LEARNING_RATE = 1e-4

# Set reproducibility
random.seed(EMIDEC_SEED)
np.random.seed(EMIDEC_SEED)
torch.manual_seed(EMIDEC_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(EMIDEC_SEED)

def resolve_file(path):
    """If path is a dir, find the .nii file inside it (copied from comparative analysis utils)"""
    if os.path.isdir(path):
        import glob
        nii_files = glob.glob(os.path.join(path, "*.nii"))
        if len(nii_files) == 0:
            raise FileNotFoundError(f"No .nii file found inside {path}")
        return nii_files[0]
    return path

def get_emidec_case_paths(root, case_name):
    """Get image and label paths for a case (copied from comparative analysis utils)."""
    
    # EMIDEC structure: Case_XXX/Images/Case_XXX.nii/ and Case_XXX/Contours/Case_XXX.nii/
    img_dir = os.path.join(root, case_name, "Images", f"{case_name}.nii")
    lbl_dir = os.path.join(root, case_name, "Contours", f"{case_name}.nii")
    
    try:
        img_path = resolve_file(img_dir)
        lbl_path = resolve_file(lbl_dir)
        
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            return {"image": img_path, "label": lbl_path}
        else:
            print(f"Warning: Missing files for {case_name}")
            print(f"  Image path: {img_path} (exists: {os.path.exists(img_path)})")
            print(f"  Label path: {lbl_path} (exists: {os.path.exists(lbl_path)})")
            return None
            
    except Exception as e:
        print(f"Error processing {case_name}: {str(e)}")
        return None

def _nanfix(x):
    """Fix NaN values in arrays."""
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def get_emidec_dataloaders(split=0.8):
    """Create EMIDEC dataloaders (copied from comparative analysis)."""
    
    # Check EMIDEC dataset
    if not os.path.exists(EMIDEC_DATASET_ROOT):
        raise FileNotFoundError(f"EMIDEC dataset not found at: {EMIDEC_DATASET_ROOT}")
    
    # Get all cases
    all_cases = sorted([d for d in os.listdir(EMIDEC_DATASET_ROOT) if d.startswith("Case_")])
    random.shuffle(all_cases)
    
    if len(all_cases) == 0:
        raise ValueError(f"No cases found in EMIDEC dataset: {EMIDEC_DATASET_ROOT}")
    
    print(f"Found {len(all_cases)} EMIDEC cases")
    
    # Split train/val
    split_idx = int(split * len(all_cases))
    train_cases = all_cases[:split_idx]
    val_cases = all_cases[split_idx:]
    
    print(f"Train cases: {len(train_cases)}, Val cases: {len(val_cases)}")
    
    # Get file paths
    train_files = []
    for case in train_cases:
        paths = get_emidec_case_paths(EMIDEC_DATASET_ROOT, case)
        if paths:
            train_files.append(paths)
        else:
            print(f"WARNING: Skipping {case} - missing files")
    
    val_files = []
    for case in val_cases:
        paths = get_emidec_case_paths(EMIDEC_DATASET_ROOT, case)
        if paths:
            val_files.append(paths)
        else:
            print(f"WARNING: Skipping {case} - missing files")
    
    print(f"Found {len(train_files)} training files, {len(val_files)} validation files")
    
    if len(train_files) == 0 or len(val_files) == 0:
        raise ValueError(f"No valid training/validation files found. Check EMIDEC dataset structure at {EMIDEC_DATASET_ROOT}")
    
    
    # Define transforms (copied from comparative analysis)
    common_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Lambdad(keys="image", func=_nanfix),
        Lambdad(keys="label", func=lambda x: (x > 0).astype(np.uint8)),
        ScaleIntensityd(keys="image"),
        Resized(keys=["image", "label"], spatial_size=EMIDEC_ROI, mode=("trilinear", "nearest")),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    # Create datasets
    train_ds = CacheDataset(train_files, transform=common_transforms, cache_rate=1.0, num_workers=EMIDEC_NUM_WORKERS)
    val_ds = CacheDataset(val_files, transform=common_transforms, cache_rate=1.0, num_workers=EMIDEC_NUM_WORKERS)
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=EMIDEC_BATCH_SIZE, shuffle=True, num_workers=EMIDEC_NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=EMIDEC_BATCH_SIZE, shuffle=False, num_workers=EMIDEC_NUM_WORKERS)
    
    return train_loader, val_loader, train_ds, val_ds

def create_attention_unet():
    """Create AttentionUNet model (copied from comparative analysis)."""
    model = AttentionUnet(
        spatial_dims=3, 
        in_channels=1, 
        out_channels=1,
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2)
    ).to(EMIDEC_DEVICE)
    
    return model

def train_one_epoch_emidec(model, loader, optimizer, loss_fn):
    """Train one epoch (copied from comparative analysis)."""
    model.train()
    total_loss, it = 0.0, 0
    
    for batch in loader:
        images = batch["image"].to(EMIDEC_DEVICE).float()
        labels = batch["label"].to(EMIDEC_DEVICE).float()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        it += 1
    
    return total_loss / max(it, 1)

@torch.no_grad()
def validate_model_emidec(model, loader, inferer, dice_metric, post_pred, post_label):
    """Validate model (copied from comparative analysis)."""
    model.eval()
    dice_metric.reset()
    
    for batch in loader:
        images = batch["image"].to(EMIDEC_DEVICE).float()
        labels = batch["label"].to(EMIDEC_DEVICE).float()
        
        outputs = inferer(images, model)
        preds = post_pred(outputs.sigmoid())
        labs = post_label(labels)
        dice_metric(y_pred=preds, y=labs)
    
    try:
        return float(dice_metric.aggregate().item())
    except:
        return 0.0

def train_attention_unet_on_emidec():
    """
    Train AttentionUNet on EMIDEC dataset.
    This is the exact integration of the comparative analysis code.
    """
    
    print("=" * 80)
    print("Training AttentionUNet on EMIDEC Dataset")
    print("=" * 80)
    
    try:
        # Load data
        print("Loading EMIDEC dataset...")
        train_loader, val_loader, train_ds, val_ds = get_emidec_dataloaders()
        
        # Create model
        print("Creating AttentionUNet model...")
        model = create_attention_unet()
        optimizer = torch.optim.Adam(model.parameters(), lr=EMIDEC_LEARNING_RATE)
        
        # Training components (copied from comparative analysis)
        loss_fn = DiceLoss(sigmoid=True)
        inferer = SlidingWindowInferer(roi_size=EMIDEC_ROI, sw_batch_size=1, overlap=0.25)
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        post_pred = AsDiscrete(threshold=0.5)
        post_label = AsDiscrete(threshold=0.5)
        
        # Create checkpoint directory
        ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Training loop (copied from comparative analysis)
        best_dice = 0.0
        best_epoch = -1
        train_losses = []
        val_dices = []
        
        print(f"Starting training for {EMIDEC_MAX_EPOCHS} epochs...")
        
        for epoch in tqdm(range(1, EMIDEC_MAX_EPOCHS + 1), desc="Training AttentionUNet"):
            
            # Train
            train_loss = train_one_epoch_emidec(model, train_loader, optimizer, loss_fn)
            
            # Validate
            val_dice = validate_model_emidec(model, val_loader, inferer, dice_metric, post_pred, post_label)
            
            train_losses.append(train_loss)
            val_dices.append(val_dice)
            
            print(f"Epoch {epoch:03d}/{EMIDEC_MAX_EPOCHS} | Loss: {train_loss:.4f} | Dice: {val_dice:.4f}")
            
            # Save best checkpoint
            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch
                
                # Save checkpoint in frequency project format
                checkpoint_path = os.path.join(ckpt_dir, "segmentation_best.pth")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "train_loss": train_loss
                }, checkpoint_path)
                
                print(f"SUCCESS: Saved best checkpoint: {checkpoint_path}")
        
        # Save training plot
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(val_dices, label="Validation Dice")
        plt.title("Validation Dice Score")
        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(OUTPUT_DIR, "emidec_training_metrics.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Training plot saved: {plot_path}")
        
        # Final results
        print(f"Training completed!")
        print(f"   Best Dice: {best_dice:.4f} at epoch {best_epoch}")
        print(f"   Checkpoint: {checkpoint_path}")
        
        return checkpoint_path, best_dice
        
    except Exception as e:
        print(f"ERROR: EMIDEC training failed: {str(e)}")
        raise

def check_emidec_dataset():
    """Check if EMIDEC dataset is available."""
    if not os.path.exists(EMIDEC_DATASET_ROOT):
        raise FileNotFoundError(
            f"ERROR: EMIDEC dataset not available at: {EMIDEC_DATASET_ROOT}\n"
            f"Please ensure the EMIDEC dataset is properly downloaded and placed in the correct location."
        )
    
    # Check if dataset has the required structure
    case_dirs = [d for d in os.listdir(EMIDEC_DATASET_ROOT) if d.startswith('Case_')]
    if len(case_dirs) == 0:
        raise ValueError(
            f"ERROR: EMIDEC dataset structure invalid at: {EMIDEC_DATASET_ROOT}\n"
            f"No Case_ directories found. Please check dataset integrity."
        )
    
    print(f"SUCCESS: EMIDEC dataset found with {len(case_dirs)} cases")
    return True

def main():
    """Main function to run EMIDEC training."""
    try:
        # Check dataset
        check_emidec_dataset()
        
        # Train model
        checkpoint_path, best_dice = train_attention_unet_on_emidec()
        
        print("\nEMIDEC AttentionUNet training completed successfully!")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Best Dice: {best_dice:.4f}")
        
        return checkpoint_path
        
    except Exception as e:
        print(f"ERROR: EMIDEC training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()