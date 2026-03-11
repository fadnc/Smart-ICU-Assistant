"""
Training Pipeline for ICU Prediction Models
Handles temporal data splitting, training, and evaluation.
GPU-optimized for RTX 3050 (4GB VRAM) with mixed precision (FP16).
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
import yaml
import logging
import os
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── GPU helpers ──────────────────────────────────────────
def get_device():
    """Get best available device with info logging."""
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU detected: {name} ({vram:.1f} GB VRAM)")
        return dev
    logger.info("No GPU detected — using CPU")
    return torch.device('cpu')


def clear_gpu_memory():
    """Free cached VRAM between training runs."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class ICUDataset(Dataset):
    """PyTorch Dataset for ICU time-series data"""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset
        
        Args:
            sequences: Feature sequences [n_samples, seq_len, n_features]
            labels: Multi-task labels [n_samples, num_tasks]
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


class ModelTrainer:
    """Train and evaluate ICU prediction models"""
    
    def __init__(self, 
                 model: nn.Module,
                 config: dict,
                 device=None):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model to train
            config: Configuration dictionary
            device: Device to use for training (auto-detects GPU)
        """
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.config = config
        self.use_amp = (self.device.type == 'cuda')  # Mixed precision on GPU
        
        # Training parameters — smaller batch for 4GB VRAM
        lstm_config = config.get('LSTM_CONFIG', {})
        gpu_config = config.get('GPU_CONFIG', {})
        self.learning_rate = lstm_config.get('learning_rate', 0.001)
        
        # Auto-select batch size based on device
        if self.device.type == 'cuda':
            self.batch_size = gpu_config.get('batch_size', 32)  # 32 for 4GB VRAM
        else:
            self.batch_size = lstm_config.get('batch_size', 64)
        
        self.epochs = lstm_config.get('epochs', 50)
        self.grad_accum_steps = gpu_config.get('grad_accum_steps', 2)  # simulate larger batch
        
        # Initialize optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()
        
        # Mixed-precision scaler
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0
        
        # Metrics history
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []
        
        logger.info(f"Trainer initialized on {self.device} | "
                    f"batch={self.batch_size} | AMP={'ON' if self.use_amp else 'OFF'} | "
                    f"grad_accum={self.grad_accum_steps}")
    
    def temporal_split(self,
                      sequences: np.ndarray,
                      labels: np.ndarray,
                      timestamps: List,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15) -> Tuple:
        """
        Perform temporal train/val/test split (no data leakage)
        
        Args:
            sequences: Feature sequences
            labels: Labels
            timestamps: Corresponding timestamps
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            
        Returns:
            Tuple of (train, val, test) sequences and labels
        """
        # Sort by timestamp
        sorted_indices = np.argsort(timestamps)
        sequences = sequences[sorted_indices]
        labels = labels[sorted_indices]
        
        # Calculate split points
        n_samples = len(sequences)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # Split
        train_X = sequences[:train_end]
        train_y = labels[:train_end]
        
        val_X = sequences[train_end:val_end]
        val_y = labels[train_end:val_end]
        
        test_X = sequences[val_end:]
        test_y = labels[val_end:]
        
        logger.info(f"Temporal split: Train={len(train_X)}, Val={len(val_X)}, Test={len(test_X)}")
        
        return (train_X, train_y), (val_X, val_y), (test_X, test_y)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch with mixed precision and gradient accumulation.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()
        
        for step, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass with optional AMP
            if self.use_amp:
                with autocast('cuda'):
                    predictions = self.model(sequences)
                    loss = self.criterion(predictions, labels) / self.grad_accum_steps
                self.scaler.scale(loss).backward()
            else:
                predictions = self.model(sequences)
                loss = self.criterion(predictions, labels) / self.grad_accum_steps
                loss.backward()
            
            total_loss += loss.item() * self.grad_accum_steps
            
            # Gradient accumulation step
            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """
        Evaluate model on validation set (uses AMP for speed).
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (avg_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with autocast('cuda'):
                        predictions = self.model(sequences)
                        loss = self.criterion(predictions, labels)
                else:
                    predictions = self.model(sequences)
                    loss = self.criterion(predictions, labels)
                
                total_loss += loss.item()
                all_predictions.append(predictions.float().cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Concatenate all predictions and labels
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)
        
        # Compute metrics
        metrics = self.compute_metrics(all_predictions, all_labels)
        
        return avg_loss, metrics
    
    def compute_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Compute evaluation metrics for all tasks
        
        Args:
            predictions: Model predictions [n_samples, num_tasks]
            labels: Ground truth labels [n_samples, num_tasks]
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'auroc': [],
            'auprc': [],
            'brier': []
        }
        
        num_tasks = labels.shape[1]
        
        for task_idx in range(num_tasks):
            y_true = labels[:, task_idx]
            y_pred = predictions[:, task_idx]
            
            # Skip if task has no positive examples
            if len(np.unique(y_true)) < 2:
                metrics['auroc'].append(np.nan)
                metrics['auprc'].append(np.nan)
                metrics['brier'].append(np.nan)
                continue
            
            # AUROC
            try:
                auroc = roc_auc_score(y_true, y_pred)
                metrics['auroc'].append(auroc)
            except:
                metrics['auroc'].append(np.nan)
            
            # AUPRC
            try:
                auprc = average_precision_score(y_true, y_pred)
                metrics['auprc'].append(auprc)
            except:
                metrics['auprc'].append(np.nan)
            
            # Brier Score
            try:
                brier = brier_score_loss(y_true, y_pred)
                metrics['brier'].append(brier)
            except:
                metrics['brier'].append(np.nan)
        
        # Compute averages
        metrics['mean_auroc'] = np.nanmean(metrics['auroc'])
        metrics['mean_auprc'] = np.nanmean(metrics['auprc'])
        metrics['mean_brier'] = np.nanmean(metrics['brier'])
        
        return metrics
    
    def train(self, 
              train_X: np.ndarray,
              train_y: np.ndarray,
              val_X: np.ndarray,
              val_y: np.ndarray,
              verbose: bool = True) -> Dict:
        """
        Full training loop with early stopping and GPU optimizations.
        
        Args:
            train_X: Training sequences
            train_y: Training labels
            val_X: Validation sequences
            val_y: Validation labels
            verbose: Print progress
            
        Returns:
            Training history
        """
        # Create datasets and loaders
        train_dataset = ICUDataset(train_X, train_y)
        val_dataset = ICUDataset(val_X, val_y)
        
        pin = (self.device.type == 'cuda')  # Faster CPU→GPU transfers
        num_workers = 2 if pin else 0
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            pin_memory=pin, num_workers=num_workers, persistent_workers=(num_workers > 0)
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            pin_memory=pin, num_workers=num_workers, persistent_workers=(num_workers > 0)
        )
        
        eff_batch = self.batch_size * self.grad_accum_steps
        logger.info(f"Training on {self.device} | "
                    f"batch={self.batch_size}×{self.grad_accum_steps}={eff_batch} | "
                    f"AMP={'ON' if self.use_amp else 'OFF'} | epochs={self.epochs}")
        
        for epoch in range(self.epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Evaluate
            val_loss, metrics = self.evaluate(val_loader)
            self.val_losses.append(val_loss)
            self.metrics_history.append(metrics)
            
            if verbose and (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Mean AUROC: {metrics['mean_auroc']:.4f}"
                )
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                self.save_checkpoint('best_model.pth')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.load_checkpoint('best_model.pth')
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions (uses AMP for speed).
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions
        """
        self.model.eval()
        dataset = ICUDataset(X, np.zeros((len(X), 1)))  # Dummy labels
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for sequences, _ in loader:
                sequences = sequences.to(self.device, non_blocking=True)
                if self.use_amp:
                    with autocast('cuda'):
                        pred = self.model(sequences)
                else:
                    pred = self.model(sequences)
                predictions.append(pred.float().cpu().numpy())
        
        return np.vstack(predictions)
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.metrics_history = checkpoint.get('metrics_history', [])
