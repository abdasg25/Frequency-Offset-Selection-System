import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import *
from data.dataset import get_segmentation_dataloaders
from models.segmentation import (
    create_segmentation_model, 
    SegmentationLoss, 
    calculate_dice_score, 
    calculate_iou_score
)

def train_segmentation_model(args):
    """
    Train the heart segmentation model using MONAI AttentionUNet.
    
    This function implements the training pipeline for the first step
    of the frequency offset selection system - heart segmentation.
    """
    print("=== Heart Segmentation Model Training ===")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Max epochs: {MAX_EPOCHS}")
    
    # Create dataloaders
    print("Loading dataset...")
    patient_ids = list(PATIENT_INFO.keys())
    train_loader, val_loader = get_segmentation_dataloaders(
        data_root=DATA_ROOT,
        annotations_root=ANNOTATIONS_ROOT,
        patient_ids=patient_ids,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating segmentation model...")
    model = create_segmentation_model(
        in_channels=1,
        out_channels=NUM_CLASSES,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint
    )
    model = model.to(DEVICE)
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = SegmentationLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )
    
    # Setup tensorboard logging
    log_dir = os.path.join(OUTPUT_DIR, "logs", "segmentation")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Training variables
    best_dice = 0.0
    best_epoch = 0
    
    print("Starting training...")
    for epoch in range(MAX_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_iou = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(DEVICE)
            masks = batch['mask'].to(DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            batch_dice = calculate_dice_score(outputs, masks)
            batch_iou = calculate_iou_score(outputs, masks)
            
            train_loss += loss.item()
            train_dice += batch_dice
            train_iou += batch_iou
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Dice': f"{batch_dice:.4f}",
                'IoU': f"{batch_iou:.4f}"
            })
        
        # Calculate training averages
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_iou /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Val]")
            for batch in pbar:
                images = batch['image'].to(DEVICE)
                masks = batch['mask'].to(DEVICE)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # Calculate metrics
                batch_dice = calculate_dice_score(outputs, masks)
                batch_iou = calculate_iou_score(outputs, masks)
                
                val_loss += loss.item()
                val_dice += batch_dice
                val_iou += batch_iou
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Dice': f"{batch_dice:.4f}",
                    'IoU': f"{batch_iou:.4f}"
                })
        
        # Calculate validation averages
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Log to tensorboard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Dice/Train', train_dice, epoch)
        writer.add_scalar('Dice/Validation', val_dice, epoch)
        writer.add_scalar('IoU/Train', train_iou, epoch)
        writer.add_scalar('IoU/Validation', val_iou, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{MAX_EPOCHS}")
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            
            checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoints", "segmentation_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'val_iou': val_iou
            }, checkpoint_path)
            
            print(f"New best model saved! Dice: {best_dice:.4f}")
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoints", f"segmentation_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_loss': val_loss
            }, checkpoint_path)
        
        print("-" * 60)
    
    writer.close()
    
    print("Training completed!")
    print(f"Best validation Dice: {best_dice:.4f} at epoch {best_epoch + 1}")


def test_segmentation_model(args):
    """Test the trained segmentation model."""
    print("=== Testing Heart Segmentation Model ===")
    
    # Load test data
    patient_ids = list(PATIENT_INFO.keys())
    _, test_loader = get_segmentation_dataloaders(
        data_root=DATA_ROOT,
        annotations_root=ANNOTATIONS_ROOT,
        patient_ids=patient_ids[-5:],  # Use last 5 patients for testing
        batch_size=1,
        validation_split=0.0,  # Use all as test
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE
    )
    
    # Load model
    model = create_segmentation_model(
        in_channels=1,
        out_channels=NUM_CLASSES,
        pretrained=True,
        checkpoint_path=args.checkpoint
    )
    model = model.to(DEVICE)
    model.eval()
    
    # Test model
    total_dice = 0.0
    total_iou = 0.0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(DEVICE)
            masks = batch['mask'].to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate metrics
            dice = calculate_dice_score(outputs, masks)
            iou = calculate_iou_score(outputs, masks)
            
            total_dice += dice
            total_iou += iou
            
            print(f"Patient: {batch['patient_id'][0]}, Dice: {dice:.4f}, IoU: {iou:.4f}")
    
    # Calculate averages
    avg_dice = total_dice / len(test_loader)
    avg_iou = total_iou / len(test_loader)
    
    print(f"\nTest Results:")
    print(f"Average Dice: {avg_dice:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train heart segmentation model")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help='Mode: train or test')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--pretrained', action='store_true',
                        help='Load pretrained weights')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_segmentation_model(args)
    elif args.mode == 'test':
        if args.checkpoint is None:
            args.checkpoint = os.path.join(OUTPUT_DIR, "checkpoints", "segmentation_best.pth")
        test_segmentation_model(args)


if __name__ == "__main__":
    main()