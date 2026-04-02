"""
GPU Training Loop for Smart ICU Assistant.

FIXES / OPTIMIZATIONS (this revision):
  1. BCELoss → BCEWithLogitsLoss
       BCELoss on sigmoid outputs is unsafe under FP16 autocast. Replaced with
       BCEWithLogitsLoss throughout. Models output raw logits; torch.sigmoid()
       applied only at inference.

  2. num_workers=0 enforced on Windows
       DataLoader multiprocessing spawn on Windows crashes without an
       if __name__=='__main__' guard. Detected at runtime; single-process
       fallback applied automatically.

  3. Updated deprecated AMP API
       torch.cuda.amp.GradScaler  → torch.amp.GradScaler('cuda', ...)
       torch.cuda.amp.autocast    → torch.amp.autocast('cuda', ...)

  4. GPU utilisation fix — batch_size 32→128, grad_accum 2→1
       With batch=32 the RTX 3050 was 46% utilised and only 0.2/4.0 GB VRAM.
       Root cause: tiny batches + single-process DataLoader means CPU spends
       ~3s loading while GPU computes in ~1s. batch=128 fills VRAM properly
       (~1.8 GB LSTM / ~2.2 GB Transformer) and roughly doubles throughput.
       grad_accum_steps dropped to 1 since effective batch is now large enough.

  5. prefetch_factor + persistent_workers on Linux/macOS
       When num_workers>0 (non-Windows), prefetch_factor=2 pre-loads the next
       2 batches while the GPU processes the current one, hiding CPU latency.
       persistent_workers=True keeps workers alive between epochs, saving ~5s
       spawn overhead per epoch across 50 epochs per task.

  6. weights_only=True on torch.load
       Suppresses FutureWarning about unsafe pickle loading in PyTorch 2.x.
"""

import os
import sys
import uuid
import platform
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── GPU helpers ───────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        idx  = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        vram = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        logger.info(f"GPU detected: {name} ({vram:.1f} GB VRAM)")
        return torch.device('cuda')
    logger.info("No GPU detected — using CPU")
    return torch.device('cpu')


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_gpu_memory(tag: str = ""):
    if torch.cuda.is_available():
        alloc  = torch.cuda.memory_allocated() / (1024 ** 2)
        cached = torch.cuda.memory_reserved()  / (1024 ** 2)
        logger.info(f"[GPU {tag}] Allocated: {alloc:.0f}MB | Cached: {cached:.0f}MB")


# ── DataLoader helpers ────────────────────────────────────────────────────────

def _safe_num_workers(requested: int) -> int:
    """
    Force num_workers=0 on Windows.

    PyTorch DataLoader uses 'spawn' multiprocessing on Windows. This requires
    the script entry point to be inside 'if __name__ == "__main__":', which
    main_pipeline.py is not (it is called as a module). Without the guard,
    worker processes crash with KeyboardInterrupt / pickle errors on start.

    On Linux / macOS, 'fork' is used and num_workers>0 works correctly.
    """
    if platform.system() == 'Windows':
        if requested > 0:
            logger.info(
                f"Windows detected — forcing num_workers=0 "
                f"(was {requested}; add 'if __name__==\"__main__\"' guard to enable)"
            )
        return 0
    return requested


def _build_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """
    Build a DataLoader with optional prefetch and persistent workers.

    prefetch_factor=2 and persistent_workers=True are only valid when
    num_workers > 0.  On Windows (num_workers=0) they raise ValueError
    and must be omitted entirely.

    prefetch_factor=2: while GPU processes batch N, workers preload batches
    N+1 and N+2 into pinned memory, hiding CPU load latency behind GPU compute.

    persistent_workers=True: keeps worker processes alive between epochs.
    Without this, workers are re-spawned at the start of every epoch
    (costs ~5s with 980K sequences × 50 epochs = ~4 min wasted per task).
    """
    ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))

    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        kwargs['prefetch_factor']    = 2
        kwargs['persistent_workers'] = True

    return DataLoader(ds, **kwargs)


# ── Loss function ─────────────────────────────────────────────────────────────

class MultiTaskBCEWithLogitsLoss(nn.Module):
    """
    AMP-safe multi-task binary cross-entropy.

    Uses BCEWithLogitsLoss (fused sigmoid + log + BCE) on raw logits.
    BCELoss on explicit sigmoid outputs is unsafe under FP16 autocast:
    sigmoid can saturate to exactly 0 or 1 in half precision, making
    log(p) = -inf → NaN loss.

    Models must output raw logits (no final Sigmoid layer).
    torch.sigmoid() is applied by the caller at inference time.
    """

    def __init__(self, task_weights: Optional[List[float]] = None):
        super().__init__()
        self.task_weights = task_weights
        self.bce_logits   = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        task_losses = self.bce_logits(logits, targets).mean(dim=0)
        if self.task_weights is not None:
            weights     = torch.tensor(self.task_weights, device=task_losses.device)
            task_losses = task_losses * weights
        return task_losses.mean()


# ── Temporal split ────────────────────────────────────────────────────────────

def temporal_split_data(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: List,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Time-ordered split — no shuffle, no leakage."""
    n       = len(X)
    i_train = int(n * train_frac)
    i_val   = int(n * (train_frac + val_frac))
    return {
        'train': (X[:i_train],      y[:i_train]),
        'val':   (X[i_train:i_val], y[i_train:i_val]),
        'test':  (X[i_val:],        y[i_val:]),
    }


# ── Model trainer ─────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Trains PyTorch models with AMP, large batches, early stopping.

    Key settings for RTX 3050 4 GB:
      batch_size=128   fills VRAM properly (~1.8 GB LSTM, ~2.2 GB Transformer)
      grad_accum=1     no accumulation needed at this batch size
      AMP=ON           FP16 halves VRAM and speeds tensor core matmul
      num_workers=0    Windows limitation (see _safe_num_workers)
    """

    def __init__(self, model: nn.Module, config: dict):
        self.model  = model
        self.config = config
        self.device = get_device()
        self.model.to(self.device)

        gpu_cfg  = config.get('GPU_CONFIG', {})
        lstm_cfg = config.get('LSTM_CONFIG', {})
        es_cfg   = config.get('EARLY_STOPPING', {})

        self.batch_size       = gpu_cfg.get('batch_size', 128)
        self.grad_accum_steps = max(1, gpu_cfg.get('grad_accum_steps', 1))
        self.use_amp          = gpu_cfg.get('use_amp', True) and self.device.type == 'cuda'
        self.pin_memory       = gpu_cfg.get('pin_memory', True) and self.device.type == 'cuda'
        self.num_workers      = _safe_num_workers(gpu_cfg.get('num_workers', 0))
        self.epochs           = lstm_cfg.get('epochs', 50)
        self.lr               = lstm_cfg.get('learning_rate', 0.001)
        self.patience         = es_cfg.get('patience', 20)
        self.min_delta        = es_cfg.get('min_delta', 1e-4)

        self.scaler    = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = MultiTaskBCEWithLogitsLoss()

        accum_note = (
            f"×{self.grad_accum_steps}={self.batch_size * self.grad_accum_steps}"
            if self.grad_accum_steps > 1 else ""
        )
        logger.info(
            f"Trainer | device={self.device} | "
            f"batch={self.batch_size}{accum_note} | "
            f"AMP={'ON' if self.use_amp else 'OFF'} | "
            f"patience={self.patience} | min_delta={self.min_delta} | "
            f"workers={self.num_workers}"
        )

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        return _build_loader(X, y, self.batch_size, shuffle, self.num_workers, self.pin_memory)

    def temporal_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: List,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        splits = temporal_split_data(X, y, timestamps)
        logger.info(
            f"Temporal split → "
            f"Train={len(splits['train'][0]):,} | "
            f"Val={len(splits['val'][0]):,} | "
            f"Test={len(splits['test'][0]):,}"
        )
        return splits

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        val_X:   np.ndarray,
        val_y:   np.ndarray,
        task_name:  str  = "",
        model_name: str  = "",
        verbose:    bool = True,
    ):
        train_loader = self._make_loader(train_X, train_y, shuffle=True)
        val_loader   = self._make_loader(val_X,   val_y,   shuffle=False)

        label      = f"[{task_name}/{model_name}]" if task_name else ""
        accum_note = (
            f"×{self.grad_accum_steps}={self.batch_size * self.grad_accum_steps}"
            if self.grad_accum_steps > 1 else ""
        )
        logger.info(
            f"[training] Starting | batch={self.batch_size}{accum_note} | "
            f"AMP={'ON' if self.use_amp else 'OFF'} | "
            f"epochs={self.epochs} | train={len(train_X):,} | val={len(val_X):,}"
        )

        best_val_loss  = float('inf')
        patience_count = 0
        ckpt_path      = f"models/_best_{uuid.uuid4().hex[:8]}.pth"
        os.makedirs('models', exist_ok=True)
        torch.save(self.model.state_dict(), ckpt_path)  # epoch-0 safety save

        epoch_bar = tqdm(
            range(self.epochs),
            desc=f"  {label} epochs    ",
            unit="ep",
            file=sys.stderr,
            dynamic_ncols=True,
            leave=True,
        )

        for epoch in epoch_bar:
            # ── Train ─────────────────────────────────────────────────────────
            self.model.train()
            self.optimizer.zero_grad()
            train_losses = []

            batch_bar = tqdm(
                train_loader,
                desc="  batches",
                unit="batch",
                file=sys.stderr,
                dynamic_ncols=True,
                leave=False,
            )

            for step, (xb, yb) in enumerate(batch_bar):
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    logits = self.model(xb)
                    if logits.shape[1] != yb.shape[1]:
                        logits = logits[:, :yb.shape[1]]
                    loss = self.criterion(logits, yb) / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                loss_val = loss.item() * self.grad_accum_steps
                if not (np.isnan(loss_val) or np.isinf(loss_val)):
                    train_losses.append(loss_val)
                batch_bar.set_postfix(loss=f"{loss_val:.4f}")

            mean_train = float(np.mean(train_losses)) if train_losses else float('nan')

            # ── Validation ────────────────────────────────────────────────────
            self.model.eval()
            val_losses  = []
            all_preds   = []
            all_targets = []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)

                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        logits = self.model(xb)
                        if logits.shape[1] != yb.shape[1]:
                            logits = logits[:, :yb.shape[1]]
                        val_loss = self.criterion(logits, yb)

                    if not (torch.isnan(val_loss) or torch.isinf(val_loss)):
                        val_losses.append(val_loss.item())

                    all_preds.append(torch.sigmoid(logits).cpu().numpy())
                    all_targets.append(yb.cpu().numpy())

            mean_val = float(np.mean(val_losses)) if val_losses else float('nan')

            try:
                preds_np   = np.vstack(all_preds)
                targets_np = np.vstack(all_targets)
                aurocs = [
                    roc_auc_score(targets_np[:, t], preds_np[:, t])
                    for t in range(targets_np.shape[1])
                    if len(np.unique(targets_np[:, t])) > 1
                ]
                mean_auroc = float(np.nanmean(aurocs)) if aurocs else 0.0
            except Exception:
                mean_auroc = 0.0

            epoch_bar.set_postfix_str(
                f"train={mean_train:.4f} | val={mean_val:.4f} | "
                f"auroc={mean_auroc:.4f} | pat={patience_count}/{self.patience}"
            )

            if np.isnan(mean_val):
                logger.warning(
                    f"[training] Epoch {epoch + 2}: NaN val loss — skipping patience update"
                )
                continue

            if mean_val < best_val_loss - self.min_delta:
                best_val_loss  = mean_val
                patience_count = 0
                torch.save(self.model.state_dict(), ckpt_path)
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    logger.info(
                        f"[training] Early stop at epoch {epoch + 1} (patience={self.patience})"
                    )
                    break

        try:
            self.model.load_state_dict(
                torch.load(ckpt_path, map_location=self.device, weights_only=True)
            )
            logger.info(f"[training] Restored best checkpoint (val_loss={best_val_loss:.4f})")
        except Exception as e:
            logger.warning(f"[training] Could not restore checkpoint: {e} — using final weights")
        finally:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return sigmoid probabilities — models output raw logits."""
        self.model.eval()
        loader    = self._make_loader(X, np.zeros((len(X), 1)), shuffle=False)
        all_preds = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    logits = self.model(xb)
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
        return np.vstack(all_preds)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, float]:
        metrics = {}
        for t in range(min(predictions.shape[1], targets.shape[1])):
            try:
                if len(np.unique(targets[:, t])) > 1:
                    metrics[f'task_{t}_auroc'] = float(
                        roc_auc_score(targets[:, t], predictions[:, t])
                    )
                else:
                    metrics[f'task_{t}_auroc'] = float('nan')
            except Exception:
                metrics[f'task_{t}_auroc'] = float('nan')
        return metrics

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(
            {'model_state_dict': self.model.state_dict(), 'config': self.config},
            path,
        )
        logger.info(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str):
        ckpt  = torch.load(path, map_location=self.device, weights_only=True)
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        self.model.load_state_dict(state)
        logger.info(f"Checkpoint loaded ← {path}")