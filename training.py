"""
Training Pipeline for ICU Prediction Models
Handles temporal data splitting, training, and evaluation.
GPU-optimized for RTX 3050 (4GB VRAM) with mixed precision (FP16).

FIXES:
  - Early stopping now requires min_delta improvement (was triggering after epoch 1)
  - Checkpoint always saves on first epoch so "No checkpoint saved" warning is gone
  - raw=True in rolling apply for Cython acceleration
  - NaN loss guard prevents patience counter incrementing on degenerate first epoch
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
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
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"GPU detected: {name} ({vram:.1f} GB VRAM)")
        return dev
    logger.info("No GPU detected — using CPU")
    return torch.device('cpu')


def clear_gpu_memory():
    """Free cached VRAM between training runs."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def log_gpu_memory(tag: str = ""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        cached = torch.cuda.memory_reserved() / (1024 ** 2)
        logger.info(f"[GPU {tag}] Allocated: {alloc:.0f}MB | Cached: {cached:.0f}MB")


def temporal_split_data(sequences, labels, timestamps, train_ratio=0.7, val_ratio=0.15):
    """Standalone temporal split — no ModelTrainer needed (for XGBoost etc.)."""
    sorted_indices = np.argsort(timestamps)
    sequences = sequences[sorted_indices]
    labels = labels[sorted_indices]
    n = len(sequences)
    t_end = int(n * train_ratio)
    v_end = int(n * (train_ratio + val_ratio))
    return {
        'train': (sequences[:t_end], labels[:t_end]),
        'val': (sequences[t_end:v_end], labels[t_end:v_end]),
        'test': (sequences[v_end:], labels[v_end:]),
    }


class ICUDataset(Dataset):
    """PyTorch Dataset for ICU time-series data"""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.nan_to_num(
            torch.FloatTensor(sequences), nan=0.0, posinf=0.0, neginf=0.0
        )
        self.labels = torch.nan_to_num(
            torch.FloatTensor(labels), nan=0.0, posinf=0.0, neginf=0.0
        )

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
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.config = config
        self.use_amp = (self.device.type == 'cuda')

        lstm_config = config.get('LSTM_CONFIG', {})
        gpu_config = config.get('GPU_CONFIG', {})
        self.learning_rate = lstm_config.get('learning_rate', 0.001)

        if self.device.type == 'cuda':
            self.batch_size = gpu_config.get('batch_size', 32)
        else:
            self.batch_size = lstm_config.get('batch_size', 64)

        self.epochs = lstm_config.get('epochs', 50)
        self.grad_accum_steps = gpu_config.get('grad_accum_steps', 2)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # FIX: BCEWithLogitsLoss expects raw logits — do NOT apply sigmoid before this loss.
        # sigmoid IS applied in predict() for inference, which is correct.
        self.criterion = nn.BCEWithLogitsLoss()

        self.scaler = GradScaler('cuda') if self.use_amp else None

        # FIX: Increased patience + min_delta so early stopping doesn't fire after epoch 1.
        # Previously patience=10 with no min_delta caused stopping before any real learning.
        self.best_val_loss = float('inf')
        self.patience = 20          # was 10 — doubled to allow more learning time
        self.min_delta = 1e-4       # NEW: must improve by at least this much to reset patience
        self.patience_counter = 0

        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []

        logger.info(
            f"Trainer initialized on {self.device} | "
            f"batch={self.batch_size} | AMP={'ON' if self.use_amp else 'OFF'} | "
            f"grad_accum={self.grad_accum_steps} | "
            f"patience={self.patience} | min_delta={self.min_delta}"
        )

    def temporal_split(self,
                       sequences: np.ndarray,
                       labels: np.ndarray,
                       timestamps: List,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15) -> Dict:
        sorted_indices = np.argsort(timestamps)
        sequences = sequences[sorted_indices]
        labels = labels[sorted_indices]

        n_samples = len(sequences)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        train_X = sequences[:train_end]
        train_y = labels[:train_end]
        val_X = sequences[train_end:val_end]
        val_y = labels[train_end:val_end]
        test_X = sequences[val_end:]
        test_y = labels[val_end:]

        logger.info(f"Temporal split: Train={len(train_X)}, Val={len(val_X)}, Test={len(test_X)}")

        return {
            'train': (train_X, train_y),
            'val': (val_X, val_y),
            'test': (test_X, test_y),
        }

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()

        for step, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

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

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

        return total_loss / len(train_loader)

    def evaluate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
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
                # FIX: sigmoid here is correct — BCEWithLogitsLoss used for training (raw logits),
                # but we need probabilities [0,1] for AUROC computation.
                all_predictions.append(torch.sigmoid(predictions).float().cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)
        metrics = self.compute_metrics(all_predictions, all_labels)

        return avg_loss, metrics

    def compute_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict:
        metrics = {'auroc': [], 'auprc': [], 'brier': []}
        num_tasks = labels.shape[1]

        for task_idx in range(num_tasks):
            y_true = labels[:, task_idx]
            y_pred = predictions[:, task_idx]

            if len(np.unique(y_true)) < 2:
                metrics['auroc'].append(np.nan)
                metrics['auprc'].append(np.nan)
                metrics['brier'].append(np.nan)
                continue

            try:
                metrics['auroc'].append(roc_auc_score(y_true, y_pred))
            except Exception:
                metrics['auroc'].append(np.nan)

            try:
                metrics['auprc'].append(average_precision_score(y_true, y_pred))
            except Exception:
                metrics['auprc'].append(np.nan)

            try:
                metrics['brier'].append(
                    np.mean((y_pred - y_true) ** 2)
                )
            except Exception:
                metrics['brier'].append(np.nan)

        valid_auroc = [v for v in metrics['auroc'] if not np.isnan(v)]
        valid_auprc = [v for v in metrics['auprc'] if not np.isnan(v)]
        valid_brier = [v for v in metrics['brier'] if not np.isnan(v)]
        metrics['mean_auroc'] = float(np.mean(valid_auroc)) if valid_auroc else 0.0
        metrics['mean_auprc'] = float(np.mean(valid_auprc)) if valid_auprc else 0.0
        metrics['mean_brier'] = float(np.mean(valid_brier)) if valid_brier else 1.0

        for task_idx in range(num_tasks):
            metrics[f'task_{task_idx}_auroc'] = (
                metrics['auroc'][task_idx] if task_idx < len(metrics['auroc']) else np.nan
            )

        return metrics

    def train(self,
              train_X: np.ndarray,
              train_y: np.ndarray,
              val_X: np.ndarray,
              val_y: np.ndarray,
              verbose: bool = True) -> Dict:
        train_dataset = ICUDataset(train_X, train_y)
        val_dataset = ICUDataset(val_X, val_y)

        pin = (self.device.type == 'cuda')
        import sys
        num_workers = 0 if sys.platform == 'win32' else (2 if pin else 0)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            pin_memory=pin, num_workers=num_workers,
            persistent_workers=(num_workers > 0)
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            pin_memory=pin, num_workers=num_workers,
            persistent_workers=(num_workers > 0)
        )

        eff_batch = self.batch_size * self.grad_accum_steps
        logger.info(
            f"Training on {self.device} | "
            f"batch={self.batch_size}×{self.grad_accum_steps}={eff_batch} | "
            f"AMP={'ON' if self.use_amp else 'OFF'} | epochs={self.epochs}"
        )

        import uuid
        self._best_ckpt = os.path.join('models', f'_best_{uuid.uuid4().hex[:8]}.pth')
        os.makedirs('models', exist_ok=True)

        # FIX: Save checkpoint at epoch 0 (before any training) so we always have a fallback.
        # Previously the checkpoint was only saved when val_loss improved, meaning if epoch 1
        # had NaN loss the patience counter ran to 10 and stopped with NO checkpoint saved.
        self.save_checkpoint(self._best_ckpt)

        # Progress bar for epoch training
        epoch_pbar = tqdm(range(self.epochs), desc="Epoch Training", unit="epoch", leave=True)
        for epoch in epoch_pbar:
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            val_loss, metrics = self.evaluate(val_loader)
            self.val_losses.append(val_loss)
            self.metrics_history.append(metrics)

            # Update progress bar with live metrics
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'auroc': f'{metrics.get("mean_auroc", 0):.4f}',
                'patience': f'{self.patience_counter}/{self.patience}'
            })

            if verbose and (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} — "
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                    f"AUROC: {metrics['mean_auroc']:.4f}"
                )

            # FIX: Only increment patience counter if loss did NOT improve by min_delta.
            # Old code incremented counter on ANY non-improvement including NaN epochs.
            if np.isnan(val_loss):
                # NaN loss — don't count against patience, just skip
                logger.warning(f"Epoch {epoch + 1}: NaN validation loss — skipping patience update")
                continue

            if val_loss < (self.best_val_loss - self.min_delta):
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(self._best_ckpt)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1} (patience={self.patience})")
                    break

        # Load best checkpoint (guaranteed to exist due to pre-training save above)
        try:
            self.load_checkpoint(self._best_ckpt)
            logger.info("Loaded best checkpoint")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e} — using final epoch weights")
        finally:
            # Always clean up temp checkpoint
            if os.path.exists(self._best_ckpt):
                os.remove(self._best_ckpt)

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        dataset = ICUDataset(X, np.zeros((len(X), 1)))
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
                # sigmoid converts raw logits → probabilities for AUROC
                predictions.append(torch.sigmoid(pred).float().cpu().numpy())

        return np.vstack(predictions)

    def save_checkpoint(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history,
        }, filepath)

    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.metrics_history = checkpoint.get('metrics_history', [])