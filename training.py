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

  4. GPU utilisation fix — batch_size 32→64, grad_accum 2→1
       With batch=32 the RTX 3050 was 46% utilised and only 0.2/4.0 GB VRAM.
       Root cause: tiny batches + single-process DataLoader means CPU spends
       ~3s loading while GPU computes in ~1s. batch=64 fills VRAM properly
       (~1.2 GB LSTM / ~1.6 GB Transformer) and roughly doubles throughput.
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
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    recall_score, brier_score_loss,
)
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Suppress torch.compile / inductor errors — fall back to eager mode silently.
# On Windows, Triton is unsupported and inductor crashes with:
#   ImportError: cannot import name 'triton_key' from 'triton.compiler.compiler'
# This MUST be set BEFORE any model forward() call.
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Suppress known harmless warnings
warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')
warnings.filterwarnings('ignore', message='.*Torch was not compiled with flash attention.*')

logger = logging.getLogger(__name__)


# ── GPU helpers ───────────────────────────────────────────────────────────────

_device_logged = False

def get_device() -> torch.device:
    global _device_logged
    if torch.cuda.is_available():
        if not _device_logged:
            idx  = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            vram = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
            logger.info(f"GPU: {name} ({vram:.1f} GB VRAM)")
            # cuDNN auto-tune: finds fastest kernel for fixed-size inputs (seq_len=24)
            torch.backends.cudnn.benchmark = True
            # Allow TF32 on Ampere+ GPUs (RTX 30xx) for ~2x matmul throughput
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            _device_logged = True
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
    AMP-safe multi-task binary cross-entropy with pos_weight support.

    Uses BCEWithLogitsLoss (fused sigmoid + log + BCE) on raw logits.
    BCELoss on explicit sigmoid outputs is unsafe under FP16 autocast:
    sigmoid can saturate to exactly 0 or 1 in half precision, making
    log(p) = -inf → NaN loss.

    pos_weight: tensor of shape (num_tasks,), where each value = n_neg/n_pos.
    Upweights the loss for positive examples to handle class imbalance.
    Example: if mortality rate is 10%, pos_weight = 0.9/0.1 = 9.0, so
    missing a true positive is penalized 9× more than a false positive.
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None,
                 task_weights: Optional[List[float]] = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.task_weights     = task_weights
        self.label_smoothing  = label_smoothing
        # pos_weight is set later via set_pos_weight() once we know the device
        self._pos_weight = pos_weight

    def set_pos_weight(self, pos_weight: torch.Tensor):
        """Set pos_weight tensor (call after moving to device)."""
        self._pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Label smoothing: soften 0→ε, 1→1-ε
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        pw = self._pos_weight
        if pw is not None:
            pw = pw.to(logits.device)
            # BCEWithLogitsLoss with pos_weight applied per-task
            loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pw)
        else:
            loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        task_losses = loss_fn(logits, targets).mean(dim=0)
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
      batch_size=64    fills VRAM properly (~1.2 GB LSTM, ~1.6 GB Transformer)
      grad_accum=1     no accumulation needed at this batch size
      AMP=ON           FP16 halves VRAM and speeds tensor core matmul
      num_workers=0    Windows limitation (see _safe_num_workers)

    Model-specific learning rate:
      LSTM:        lr from LSTM_CONFIG.learning_rate        (default: 0.001)
      Transformer: lr from TRANSFORMER_CONFIG.learning_rate (default: 0.0001)
    """

    def __init__(self, model: nn.Module, config: dict, model_type: str = 'lstm'):
        self.model  = model
        self.config = config
        self.device = get_device()
        self.model.to(self.device)
        self.model_type = model_type.lower()

        gpu_cfg  = config.get('GPU_CONFIG', {})
        es_cfg   = config.get('EARLY_STOPPING', {})
        sched_cfg = config.get('SCHEDULER', {})

        # Model-specific learning rate + epochs
        if self.model_type == 'transformer':
            model_cfg = config.get('TRANSFORMER_CONFIG', {})
            default_lr = 0.0001
        else:
            model_cfg = config.get('LSTM_CONFIG', {})
            default_lr = 0.001
        lstm_cfg = config.get('LSTM_CONFIG', {})  # epochs always from LSTM_CONFIG

        self.batch_size       = gpu_cfg.get('batch_size', 64)
        self.grad_accum_steps = max(1, gpu_cfg.get('grad_accum_steps', 1))
        self.use_amp          = gpu_cfg.get('use_amp', True) and self.device.type == 'cuda'
        self.pin_memory       = gpu_cfg.get('pin_memory', True) and self.device.type == 'cuda'
        self.num_workers      = _safe_num_workers(gpu_cfg.get('num_workers', 0))
        self.epochs           = lstm_cfg.get('epochs', 50)
        self.lr               = model_cfg.get('learning_rate', default_lr)
        self.patience         = es_cfg.get('patience', 20)
        self.min_delta        = es_cfg.get('min_delta', 1e-4)
        self.nan_abort_epochs = es_cfg.get('nan_abort_epochs', 5)
        self.label_smoothing  = sched_cfg.get('label_smoothing', 0.05)

        # Scheduler config
        self.warmup_epochs    = sched_cfg.get('warmup_epochs', 3) if self.model_type == 'transformer' else 0
        self.scheduler_factor = sched_cfg.get('factor', 0.5)
        self.scheduler_patience = sched_cfg.get('patience', 5)

        self.scaler    = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = MultiTaskBCEWithLogitsLoss(label_smoothing=self.label_smoothing)

        # LR scheduler: ReduceLROnPlateau (halves LR when val loss stalls)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=1e-6,
        )

        # torch.compile() — fuses ops and reduces kernel launch overhead.
        # suppress_errors=True is set at module level, so any inductor/Triton
        # crash falls back to eager mode automatically on all platforms.
        if (hasattr(torch, 'compile')
                and self.device.type == 'cuda'
                and platform.system() != 'Windows'):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
            except Exception:
                pass  # silently fall back to eager mode

        # Only log full config once; subsequent inits just show lr
        if not getattr(ModelTrainer, '_config_logged', False):
            logger.info(
                f"Training config | batch={self.batch_size} | "
                f"AMP={'ON' if self.use_amp else 'OFF'} | "
                f"patience={self.patience} | nan_abort={self.nan_abort_epochs} | "
                f"label_smooth={self.label_smoothing}"
            )
            ModelTrainer._config_logged = True

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        return _build_loader(X, y, self.batch_size, shuffle, self.num_workers, self.pin_memory)

    def temporal_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: List,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return temporal_split_data(X, y, timestamps)

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

        label = f"{task_name}/{model_name}" if task_name else ""

        # ── Compute pos_weight from training label frequencies ─────────────
        # pos_weight[t] = n_negative / n_positive for each task column
        # This upweights rare positives (e.g., mortality ~10% → pw=9.0)
        n_pos = train_y.sum(axis=0)            # shape: (num_tasks,)
        n_neg = train_y.shape[0] - n_pos
        # Clamp to avoid div-by-zero and extreme weights
        pos_weight = np.clip(n_neg / np.maximum(n_pos, 1.0), 1.0, 50.0)
        pw_tensor  = torch.FloatTensor(pos_weight).to(self.device)
        self.criterion.set_pos_weight(pw_tensor)

        best_val_loss  = float('inf')
        patience_count = 0
        nan_consec     = 0          # consecutive NaN val loss epochs
        ckpt_path      = f"models/_best_{uuid.uuid4().hex[:8]}.pth"
        os.makedirs('models', exist_ok=True)
        torch.save(self.model.state_dict(), ckpt_path)  # epoch-0 safety save
        train_steps = len(train_loader)
        best_auroc  = 0.0

        epoch_bar = tqdm(
            range(self.epochs),
            desc=f"  {label:<30s}",
            unit="ep",
            file=sys.stderr,
            dynamic_ncols=True,
            leave=True,
            bar_format=(
                "{l_bar}{bar:30}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ),
        )

        for epoch in epoch_bar:
            # ── Warmup LR for Transformer ──────────────────────────────────
            if epoch < self.warmup_epochs:
                warmup_lr = self.lr * (epoch + 1) / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg['lr'] = warmup_lr

            # ── Train ─────────────────────────────────────────────────────────
            self.model.train()
            self.optimizer.zero_grad()
            train_losses = []
            batch_bar = tqdm(
                total=train_steps,
                desc=f"    {label[:26]:<26s}",
                unit="batch",
                file=sys.stderr,
                dynamic_ncols=True,
                leave=False,
                disable=(not verbose or train_steps <= 1),
                bar_format="{l_bar}{bar:24}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
            )

            for step, (xb, yb) in enumerate(train_loader, start=1):
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    logits = self.model(xb)
                    if logits.shape[1] != yb.shape[1]:
                        logits = logits[:, :yb.shape[1]]
                    loss = self.criterion(logits, yb) / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                should_step = (step % self.grad_accum_steps == 0) or (step == train_steps)
                if should_step:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                loss_val = loss.item() * self.grad_accum_steps
                if not (np.isnan(loss_val) or np.isinf(loss_val)):
                    train_losses.append(loss_val)
                if not batch_bar.disable:
                    batch_bar.update(1)
                    batch_bar.set_postfix_str(
                        f"epoch={epoch + 1}/{self.epochs} | loss={loss_val:.4f} | lr={self.optimizer.param_groups[0]['lr']:.1e}"
                    )

            if not batch_bar.disable:
                batch_bar.close()

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

            best_auroc = max(best_auroc, mean_auroc)
            best_val_display = "n/a" if np.isinf(best_val_loss) else f"{best_val_loss:.4f}"

            epoch_bar.set_postfix_str(
                f"train={mean_train:.4f} | val={mean_val:.4f} | "
                f"auroc={mean_auroc:.4f} | "
                f"lr={self.optimizer.param_groups[0]['lr']:.1e} | pat={patience_count}/{self.patience}"
            )

            if np.isnan(mean_val):
                nan_consec += 1
                logger.warning(
                    f"[training] Epoch {epoch + 1}: NaN val loss — "
                    f"{nan_consec}/{self.nan_abort_epochs} consecutive"
                )
                if nan_consec >= self.nan_abort_epochs:
                    logger.error(
                        f"[training] ABORTING: {nan_consec} consecutive NaN val loss epochs. "
                        f"Model has diverged — restoring best checkpoint."
                    )
                    break
                continue
            else:
                nan_consec = 0   # reset on any valid val loss

            # Step LR scheduler (only after warmup phase)
            if epoch >= self.warmup_epochs:
                self.scheduler.step(mean_val)

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
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute comprehensive clinical metrics per task.

        Returns per-task: AUROC, AUPRC, F1, sensitivity, specificity, Brier score
        Plus macro-averaged versions across all tasks.
        """
        metrics = {}
        n_tasks = min(predictions.shape[1], targets.shape[1])

        all_aurocs, all_auprcs, all_f1s = [], [], []
        all_sens, all_specs, all_briers = [], [], []

        for t in range(n_tasks):
            y_true = targets[:, t]
            y_prob = predictions[:, t]
            y_pred = (y_prob >= threshold).astype(int)

            try:
                if len(np.unique(y_true)) < 2:
                    # Degenerate label — can't compute ranking metrics
                    metrics[f'task_{t}_auroc']       = float('nan')
                    metrics[f'task_{t}_auprc']       = float('nan')
                    metrics[f'task_{t}_f1']          = float('nan')
                    metrics[f'task_{t}_sensitivity'] = float('nan')
                    metrics[f'task_{t}_specificity'] = float('nan')
                    metrics[f'task_{t}_brier']       = float('nan')
                    continue

                auroc = roc_auc_score(y_true, y_prob)
                auprc = average_precision_score(y_true, y_prob)
                f1    = f1_score(y_true, y_pred, zero_division=0)
                sens  = recall_score(y_true, y_pred, zero_division=0)      # sensitivity = recall
                # specificity = TN / (TN + FP)
                tn = ((y_pred == 0) & (y_true == 0)).sum()
                fp = ((y_pred == 1) & (y_true == 0)).sum()
                spec  = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
                brier = brier_score_loss(y_true, y_prob)

                metrics[f'task_{t}_auroc']       = float(auroc)
                metrics[f'task_{t}_auprc']       = float(auprc)
                metrics[f'task_{t}_f1']          = float(f1)
                metrics[f'task_{t}_sensitivity'] = float(sens)
                metrics[f'task_{t}_specificity'] = float(spec)
                metrics[f'task_{t}_brier']       = float(brier)

                all_aurocs.append(auroc)
                all_auprcs.append(auprc)
                all_f1s.append(f1)
                all_sens.append(sens)
                all_specs.append(spec)
                all_briers.append(brier)

            except Exception:
                metrics[f'task_{t}_auroc']       = float('nan')
                metrics[f'task_{t}_auprc']       = float('nan')
                metrics[f'task_{t}_f1']          = float('nan')
                metrics[f'task_{t}_sensitivity'] = float('nan')
                metrics[f'task_{t}_specificity'] = float('nan')
                metrics[f'task_{t}_brier']       = float('nan')

        # Macro averages
        metrics['macro_auroc']       = float(np.mean(all_aurocs)) if all_aurocs else 0.0
        metrics['macro_auprc']       = float(np.mean(all_auprcs)) if all_auprcs else 0.0
        metrics['macro_f1']          = float(np.mean(all_f1s))    if all_f1s    else 0.0
        metrics['macro_sensitivity'] = float(np.mean(all_sens))   if all_sens   else 0.0
        metrics['macro_specificity'] = float(np.mean(all_specs))  if all_specs  else 0.0
        metrics['macro_brier']       = float(np.mean(all_briers)) if all_briers else 0.0

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
