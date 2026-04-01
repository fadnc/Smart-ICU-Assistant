"""
models.py — ML Models for ICU Prediction

A100 changes vs previous version:
  - Removed sigmoid from all model forward() methods.
    Trainer now uses BCEWithLogitsLoss (numerically stable, works with BF16).
  - LSTM hidden_size default raised to 256 (was 128).
  - Transformer d_model default raised to 256 (was 128).
  - TCN removed (BatchNorm1d instability under AMP with imbalanced labels).
  - XGBoost uses device='cuda', tree_method='hist' (XGBoost 2.x API).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
import xgboost as xgb
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── LSTM ──────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """
    Bidirectional LSTM for multi-task ICU outcome prediction.
    No sigmoid in forward — use BCEWithLogitsLoss in trainer.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 256,
                 num_layers: int = 3,
                 num_tasks: int = 6,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        enc_size     = hidden_size * 2 if bidirectional else hidden_size
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(enc_size)   # LayerNorm stable under BF16

        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(enc_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
            )
            for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        h = self.norm(self.dropout(lstm_out[:, -1, :]))
        return torch.cat([head(h) for head in self.task_heads], dim=1)


# ── Transformer ───────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TransformerModel(nn.Module):
    """
    Transformer for multi-task ICU prediction.
    No sigmoid in forward — BCEWithLogitsLoss in trainer.
    Uses pre-norm (norm_first=True) for better BF16 gradient stability.
    """

    def __init__(self,
                 input_size: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 num_tasks: int = 6,
                 dropout: float = 0.3):
        super().__init__()

        self.proj    = nn.Linear(input_size, d_model)
        self.pe      = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True,    # pre-norm: more stable under BF16
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_tasks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pe(self.proj(x))
        x = self.encoder(x)
        return self.head(x.mean(dim=1))


# ── MultitaskLSTM ─────────────────────────────────────────────────────────────

class MultitaskLSTM(nn.Module):
    """Shared BiLSTM encoder with per-task heads + composite deterioration score."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 256,
                 num_layers: int = 3,
                 tasks_per_group: List[int] = None,
                 dropout: float = 0.3):
        super().__init__()

        self.shared_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        enc_size = hidden_size * 2
        if tasks_per_group is None:
            tasks_per_group = [3, 3, 6, 3, 2, 3]
        self.tasks_per_group = tasks_per_group

        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(enc_size, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, n)
            )
            for n in tasks_per_group
        ])

        total = sum(tasks_per_group)
        self.composite = nn.Sequential(
            nn.Linear(enc_size + total, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _  = self.shared_lstm(x)
        h       = self.dropout(out[:, -1, :])
        tasks   = torch.cat([head(h) for head in self.task_heads], dim=1)
        comp    = self.composite(torch.cat([h, tasks], dim=1))
        return torch.cat([tasks, comp], dim=1)


# ── XGBoost ───────────────────────────────────────────────────────────────────

class XGBoostPredictor:
    """
    XGBoost multi-task predictor. One XGBClassifier per label.
    Uses XGBoost 2.x API: tree_method='hist', device='cuda'.
    """

    def __init__(self,
                 num_tasks: int = 6,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 n_estimators: int = 100,
                 subsample: float = 0.8,
                 tree_method: str = 'hist',
                 gpu_id: int = 0):
        self.num_tasks = num_tasks
        self.models    = []

        use_gpu = torch.cuda.is_available()
        device  = f'cuda:{gpu_id}' if use_gpu else 'cpu'
        logger.info(f"XGBoost: device={device}, tree_method=hist")

        for _ in range(num_tasks):
            self.models.append(xgb.XGBClassifier(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                objective='binary:logistic',
                eval_metric='auc',
                tree_method='hist',
                device=device,
                random_state=42,
            ))

    def flatten_sequences(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(len(X), -1)

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        Xf = self.flatten_sequences(X)
        for i, model in enumerate(self.models):
            if len(np.unique(y[:, i])) < 2:
                self.models[i] = None
                continue
            model.fit(Xf, y[:, i], verbose=False)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xf = self.flatten_sequences(X)
        return np.column_stack([
            m.predict_proba(Xf)[:, 1] if m is not None else np.full(len(Xf), 0.5)
            for m in self.models
        ])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# ── Multi-task loss (kept for external use) ───────────────────────────────────

class MultiTaskLoss(nn.Module):
    def __init__(self, task_weights: Optional[List[float]] = None):
        super().__init__()
        self.task_weights = task_weights
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = self.bce(predictions, targets).mean(dim=0)
        if self.task_weights is not None:
            w = torch.tensor(self.task_weights, device=losses.device)
            losses = losses * w
        return losses.mean()


# ── Factory ───────────────────────────────────────────────────────────────────

def create_model(model_type: str, config: dict) -> nn.Module:
    input_size = config.get('input_size', 50)
    num_tasks  = config.get('num_tasks', 6)

    if model_type.lower() == 'lstm':
        c = config.get('LSTM_CONFIG', {})
        return LSTMModel(
            input_size=input_size,
            hidden_size=c.get('hidden_size', 256),
            num_layers=c.get('num_layers', 3),
            num_tasks=num_tasks,
            dropout=c.get('dropout', 0.3),
            bidirectional=c.get('bidirectional', True),
        )

    elif model_type.lower() == 'transformer':
        c = config.get('TRANSFORMER_CONFIG', {})
        return TransformerModel(
            input_size=input_size,
            d_model=c.get('d_model', 256),
            nhead=c.get('nhead', 8),
            num_layers=c.get('num_layers', 4),
            dim_feedforward=c.get('dim_feedforward', 512),
            num_tasks=num_tasks,
            dropout=c.get('dropout', 0.3),
        )

    elif model_type.lower() == 'multitask_lstm':
        c = config.get('LSTM_CONFIG', {})
        return MultitaskLSTM(
            input_size=input_size,
            hidden_size=c.get('hidden_size', 256),
            num_layers=c.get('num_layers', 3),
            tasks_per_group=config.get('tasks_per_group', [3, 3, 6, 3, 2, 3]),
            dropout=c.get('dropout', 0.3),
        )

    elif model_type.lower() == 'xgboost':
        c = config.get('XGBOOST_CONFIG', {})
        return XGBoostPredictor(
            num_tasks=num_tasks,
            max_depth=c.get('max_depth', 6),
            learning_rate=c.get('learning_rate', 0.1),
            n_estimators=c.get('n_estimators', 100),
            subsample=c.get('subsample', 0.8),
            tree_method='hist',
            gpu_id=c.get('gpu_id', 0),
        )

    elif model_type.lower() == 'tcn':
        raise ValueError(
            "TCN removed — BatchNorm1d instability under AMP with imbalanced labels. "
            "Use 'lstm' or 'xgboost'."
        )

    else:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            "Choose from: lstm, transformer, multitask_lstm, xgboost"
        )


if __name__ == "__main__":
    import torch
    logger.info("Smoke-testing models (no TCN)...")
    B, T, F, K = 32, 24, 81, 6
    X = torch.randn(B, T, F)

    lstm = LSTMModel(F, num_tasks=K)
    assert lstm(X).shape == (B, K), "LSTM shape mismatch"
    logger.info(f"LSTM OK — output {lstm(X).shape}")

    tf = TransformerModel(F, num_tasks=K)
    assert tf(X).shape == (B, K), "Transformer shape mismatch"
    logger.info(f"Transformer OK — output {tf(X).shape}")

    logger.info("All smoke tests passed.")