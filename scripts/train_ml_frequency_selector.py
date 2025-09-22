#!/usr/bin/env python3
"""
Training script for ML-based frequency offset selection.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import *
from models.frequency_selector import FrequencySelectionCNN
from models.segmentation import create_segmentation_model
from data.ml_dataset import get_ml_training_dataloaders


class FrequencySelectionLoss(nn.Module):
    """
    Combined loss function for frequency selection training.
    Combines classification loss and regression loss.
    """
    
    def __init__(
        self,
        classification_weight: float = 0.7,
        regression_weight: float = 0.3,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0
    ):
        """
        Initialize combined loss function.
        
        Args:
            classification_weight: Weight for classification loss
            regression_weight: Weight for regression loss  
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        super(FrequencySelectionLoss, self).__init__()
        
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
    
    def focal_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for handling class imbalance."""
        ce_loss = nn.functional.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return torch.mean(focal_loss)
    
    def forward(
        self,
        classification_logits: torch.Tensor,
        regression_output: torch.Tensor,
        target_indices: torch.Tensor,
        series_lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            classification_logits: Model classification outputs
            regression_output: Model regression outputs  
            target_indices: Ground truth frequency indices
            series_lengths: Actual series lengths for each sample
            
        Returns:
            Dictionary containing individual and total losses
        """
        batch_size = classification_logits.size(0)
        
        # Classification loss (focal loss for better handling of imbalanced data)
        classification_loss = self.focal_loss(classification_logits, target_indices)
        
        # Regression loss (smooth L1 for robustness)
        regression_targets = target_indices.float()
        regression_loss = self.smooth_l1_loss(regression_output, regression_targets)
        
        # Proximity loss (penalize predictions far from target)
        proximity_loss = self._compute_proximity_loss(
            classification_logits, target_indices, series_lengths
        )
        
        # Combined loss
        total_loss = (
            self.classification_weight * classification_loss +
            self.regression_weight * regression_loss +
            0.1 * proximity_loss  # Small weight for proximity
        )
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'regression_loss': regression_loss,
            'proximity_loss': proximity_loss
        }
    
    def _compute_proximity_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        series_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Compute proximity loss to penalize predictions far from target."""
        batch_size, max_freq = logits.shape
        
        # Create distance matrix
        indices = torch.arange(max_freq, device=logits.device).expand(batch_size, -1)
        target_expanded = targets.unsqueeze(1).expand(-1, max_freq)
        distances = torch.abs(indices - target_expanded).float()
        
        # Weight by prediction probabilities
        probs = torch.softmax(logits, dim=1)
        proximity_loss = torch.sum(probs * distances, dim=1).mean()
        
        return proximity_loss


class FrequencySelectionTrainer:
    """Trainer class for frequency selection model."""
    
    def __init__(
        self,
        model: FrequencySelectionCNN,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        """
        Initialize trainer.
        
        Args:
            model: FrequencySelectionCNN model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Training device
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function
        self.criterion = FrequencySelectionLoss()
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # Training history
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        self.best_val_accuracy = 0.0
        self.best_model_state = None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            # Move to device
            frequency_series = batch['frequency_series'].to(self.device)
            heart_mask = batch['heart_mask'].to(self.device)
            target_indices = batch['optimal_index'].to(self.device)
            series_lengths = batch['series_length'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(frequency_series, heart_mask)
            
            # Compute loss
            loss_dict = self.criterion(
                outputs['classification_logits'],
                outputs['regression_output'],
                target_indices,
                series_lengths
            )
            
            # Backward pass
            loss = loss_dict['total_loss']
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Compute accuracy
            predictions = torch.argmax(outputs['classification_logits'], dim=1)
            correct = (predictions == target_indices).sum().item()
            
            # Update metrics
            batch_size = frequency_series.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            correct_predictions += correct
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct/batch_size:.4f}"
            })
        
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        
        return {
            'train_loss': avg_loss,
            'train_accuracy': accuracy
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        exact_matches = 0
        within_tolerance = 0
        tolerance = 2  # ±2 frames tolerance
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                frequency_series = batch['frequency_series'].to(self.device)
                heart_mask = batch['heart_mask'].to(self.device)
                target_indices = batch['optimal_index'].to(self.device)
                series_lengths = batch['series_length'].to(self.device)
                
                # Forward pass
                outputs = self.model(frequency_series, heart_mask)
                
                # Compute loss
                loss_dict = self.criterion(
                    outputs['classification_logits'],
                    outputs['regression_output'],
                    target_indices,
                    series_lengths
                )
                
                # Compute accuracy metrics
                predictions = torch.argmax(outputs['classification_logits'], dim=1)
                differences = torch.abs(predictions - target_indices)
                
                exact_correct = (predictions == target_indices).sum().item()
                tolerance_correct = (differences <= tolerance).sum().item()
                
                # Update metrics
                batch_size = frequency_series.size(0)
                total_loss += loss_dict['total_loss'].item() * batch_size
                total_samples += batch_size
                exact_matches += exact_correct
                within_tolerance += tolerance_correct
        
        avg_loss = total_loss / total_samples
        exact_accuracy = exact_matches / total_samples
        tolerance_accuracy = within_tolerance / total_samples
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': exact_accuracy,
            'val_tolerance_accuracy': tolerance_accuracy
        }
    
    def train(self, num_epochs: int, save_dir: str) -> Dict:
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of training epochs
            save_dir: Directory to save model checkpoints
            
        Returns:
            Training history dictionary
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            prev_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_metrics['val_loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            if current_lr != prev_lr:
                print(f"Learning rate reduced: {prev_lr:.2e} → {current_lr:.2e}")
            
            # Update history
            self.train_history['train_loss'].append(train_metrics['train_loss'])
            self.train_history['val_loss'].append(val_metrics['val_loss'])
            self.train_history['train_accuracy'].append(train_metrics['train_accuracy'])
            self.train_history['val_accuracy'].append(val_metrics['val_accuracy'])
            self.train_history['learning_rates'].append(current_lr)
            
            # Print metrics
            print(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Train Acc: {train_metrics['train_accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_accuracy']:.4f}, "
                  f"Val Tolerance Acc: {val_metrics['val_tolerance_accuracy']:.4f}")
            print(f"Learning Rate: {current_lr:.2e}")
            
            # Save best model
            if val_metrics['val_tolerance_accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['val_tolerance_accuracy']
                self.best_model_state = self.model.state_dict().copy()
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_accuracy': self.best_val_accuracy,
                    'train_history': self.train_history,
                    'model_config': {
                        'max_frequencies': self.model.max_frequencies,
                        'image_size': self.model.image_size,
                        'use_heart_mask': self.model.use_heart_mask
                    }
                }
                
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
                print(f"✓ New best model saved (Val Tolerance Acc: {self.best_val_accuracy:.4f})")
            
            # Save latest checkpoint
            latest_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_history': self.train_history
            }
            torch.save(latest_checkpoint, os.path.join(save_dir, 'latest_model.pth'))
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Save training plots
        self._save_training_plots(save_dir)
        
        print(f"\nTraining completed! Best validation tolerance accuracy: {self.best_val_accuracy:.4f}")
        
        return self.train_history
    
    def _save_training_plots(self, save_dir: str):
        """Save training history plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.train_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.train_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.train_history['train_accuracy'], label='Train Acc')
        axes[0, 1].plot(self.train_history['val_accuracy'], label='Val Acc')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(self.train_history['learning_rates'])
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Remove empty subplot
        axes[1, 1].remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ML frequency selector')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--max_frequencies', type=int, default=15, help='Maximum frequency series length')
    parser.add_argument('--validation_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--save_dir', type=str, default='outputs/ml_frequency_models', help='Model save directory')
    
    args = parser.parse_args()
    
    print("=== ML-Based Frequency Offset Selection Training ===")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max frequencies: {args.max_frequencies}")
    
    # Load segmentation model
    print("\nLoading segmentation model...")
    segmentation_model = create_segmentation_model(
        in_channels=1,
        out_channels=NUM_CLASSES,
        pretrained=True,
        checkpoint_path=os.path.join(OUTPUT_DIR, "checkpoints", "segmentation_best.pth")
    )
    segmentation_model = segmentation_model.to(DEVICE)
    segmentation_model.eval()
    
    # Get patient IDs
    patient_ids = list(PATIENT_INFO.keys())
    print(f"Total patients: {len(patient_ids)}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = get_ml_training_dataloaders(
        data_root=DATA_ROOT,
        patient_ids=patient_ids,
        patient_info=PATIENT_INFO,
        segmentation_model=segmentation_model,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        num_workers=2,  # Reduced for stability
        image_size=IMAGE_SIZE,
        max_frequencies=args.max_frequencies,
        device=DEVICE
    )
    
    # Create model
    print("\nCreating frequency selection model...")
    model = FrequencySelectionCNN(
        max_frequencies=args.max_frequencies,
        image_size=IMAGE_SIZE,
        use_heart_mask=True
    )
    
    # Create trainer
    trainer = FrequencySelectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(
        num_epochs=args.epochs,
        save_dir=args.save_dir
    )
    
    # Save training history
    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed! Models saved to: {args.save_dir}")
    print(f"Best validation accuracy: {trainer.best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()