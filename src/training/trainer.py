"""
Training module for the anomaly detection system.
Implements comprehensive training with validation, early stopping, and model checkpointing.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils import LoggerMixin, TrainingError, handle_exceptions
from ..models import ConvolutionalAutoencoder


class LossCalculator:
    """Calculates various loss components for training."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def reconstruction_loss(
        self, 
        reconstruction: torch.Tensor, 
        target: torch.Tensor,
        loss_type: str = "mse"
    ) -> torch.Tensor:
        """Calculate reconstruction loss."""
        if loss_type == "mse":
            return self.mse_loss(reconstruction, target)
        elif loss_type == "l1":
            return self.l1_loss(reconstruction, target)
        elif loss_type == "combined":
            return 0.7 * self.mse_loss(reconstruction, target) + 0.3 * self.l1_loss(reconstruction, target)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def perceptual_loss(
        self, 
        reconstruction: torch.Tensor, 
        target: torch.Tensor,
        feature_extractor: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """Calculate perceptual loss using feature extractor."""
        if feature_extractor is None:
            return torch.tensor(0.0, device=self.device)
        
        with torch.no_grad():
            target_features = feature_extractor(target)
        
        reconstruction_features = feature_extractor(reconstruction)
        return self.mse_loss(reconstruction_features, target_features)
    
    def total_loss(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        reconstruction_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        feature_extractor: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate total loss with components."""
        recon_loss = self.reconstruction_loss(reconstruction, target, "combined")
        perc_loss = self.perceptual_loss(reconstruction, target, feature_extractor)
        
        total = reconstruction_weight * recon_loss + perceptual_weight * perc_loss
        
        loss_components = {
            'reconstruction_loss': recon_loss.item(),
            'perceptual_loss': perc_loss.item(),
            'total_loss': total.item()
        }
        
        return total, loss_components


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


class ModelCheckpoint:
    """Model checkpointing utility."""
    
    def __init__(
        self, 
        filepath: Path, 
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        mode: str = 'min'
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        
        # Create directory if it doesn't exist
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def __call__(self, metrics: Dict[str, float], model: nn.Module, optimizer: optim.Optimizer, epoch: int):
        """Save model checkpoint if conditions are met."""
        current_score = metrics.get(self.monitor, float('inf'))
        
        should_save = False
        if not self.save_best_only:
            should_save = True
        elif self.mode == 'min' and current_score < self.best_score:
            should_save = True
            self.best_score = current_score
        elif self.mode == 'max' and current_score > self.best_score:
            should_save = True
            self.best_score = current_score
        
        if should_save:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'best_score': self.best_score
            }
            torch.save(checkpoint, self.filepath)


class AnomalyDetectionTrainer(LoggerMixin):
    """
    Comprehensive trainer for anomaly detection models.
    Handles training, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: ConvolutionalAutoencoder,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use for training
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or {}
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_calculator()
        self._setup_callbacks()
        self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        self.logger.info(f"Initialized trainer with device: {self.device}")
    
    def _setup_optimizer(self):
        """Setup optimizer."""
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', 'cosine')
        epochs = self.config.get('epochs', 100)
        
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=epochs // 3, gamma=0.1
            )
        else:
            self.scheduler = None
    
    def _setup_loss_calculator(self):
        """Setup loss calculator."""
        self.loss_calculator = LossCalculator(self.device)
    
    def _setup_callbacks(self):
        """Setup training callbacks."""
        # Early stopping
        patience = self.config.get('early_stopping_patience', 15)
        self.early_stopping = EarlyStopping(patience=patience)
        
        # Model checkpoint
        models_dir = Path(self.config.get('models_dir', 'models'))
        checkpoint_path = models_dir / 'best_model.pth'
        save_best_only = self.config.get('save_best_only', True)
        
        self.checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=save_best_only
        )
    
    def _setup_logging(self):
        """Setup tensorboard logging."""
        log_dir = Path(self.config.get('tensorboard_dir', 'runs'))
        self.writer = SummaryWriter(log_dir / f'experiment_{int(time.time())}')
    
    @handle_exceptions(TrainingError)
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            batch_size = images.size(0)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            loss, loss_components = self.loss_calculator.total_loss(
                outputs['reconstruction'], images
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * batch_size
            total_recon_loss += loss_components['reconstruction_loss'] * batch_size
            total_samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Recon': f'{loss_components["reconstruction_loss"]:.4f}'
            })
            
            # Log batch metrics
            if batch_idx % 100 == 0:
                step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), step)
        
        # Calculate epoch metrics
        epoch_metrics = {
            'train_loss': total_loss / total_samples,
            'train_recon_loss': total_recon_loss / total_samples
        }
        
        return epoch_metrics
    
    @handle_exceptions(TrainingError)
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                batch_size = images.size(0)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss, loss_components = self.loss_calculator.total_loss(
                    outputs['reconstruction'], images
                )
                
                # Update metrics
                total_loss += loss.item() * batch_size
                total_recon_loss += loss_components['reconstruction_loss'] * batch_size
                total_samples += batch_size
        
        # Calculate epoch metrics
        epoch_metrics = {
            'val_loss': total_loss / total_samples,
            'val_recon_loss': total_recon_loss / total_samples
        }
        
        return epoch_metrics
    
    def train(self, epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Train the model for specified number of epochs.
        
        Args:
            epochs: Number of epochs to train
            
        Returns:
            Training history dictionary
        """
        if epochs is None:
            epochs = self.config.get('epochs', 100)
        
        self.logger.info(f"Starting training for {epochs} epochs")
        
        try:
            for epoch in range(epochs):
                self.current_epoch = epoch
                start_time = time.time()
                
                # Train epoch
                train_metrics = self.train_epoch()
                
                # Validate epoch
                val_metrics = self.validate_epoch()
                
                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Log metrics
                self._log_epoch_metrics(epoch_metrics, epoch)
                
                # Store losses
                self.train_losses.append(train_metrics['train_loss'])
                if 'val_loss' in val_metrics:
                    self.val_losses.append(val_metrics['val_loss'])
                
                # Callbacks
                self.checkpoint(epoch_metrics, self.model, self.optimizer, epoch)
                
                if 'val_loss' in val_metrics:
                    if self.early_stopping(val_metrics['val_loss'], self.model):
                        self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        break
                
                # Log epoch time
                epoch_time = time.time() - start_time
                self.logger.info(f"Epoch {epoch + 1}/{epochs} completed in {epoch_time:.2f}s")
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            self.writer.close()
        
        # Return training history
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }
        
        self.logger.info("Training completed")
        return history
    
    def _log_epoch_metrics(self, metrics: Dict[str, float], epoch: int):
        """Log epoch metrics to tensorboard and console."""
        # Console logging
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch + 1} - {metric_str}")
        
        # Tensorboard logging
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'Epoch/{metric_name}', metric_value, epoch)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Epoch/learning_rate', current_lr, epoch)
    
    def save_model(self, filepath: Path, include_optimizer: bool = True):
        """Save model state."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if include_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Path, load_optimizer: bool = True):
        """Load model state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def plot_training_history(self, save_path: Optional[Path] = None):
        """Plot training history."""
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        # Learning rate plot
        plt.subplot(1, 2, 2)
        if self.scheduler is not None:
            lrs = [self.scheduler.get_last_lr()[0] for _ in range(len(self.train_losses))]
            plt.plot(lrs, label='Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
