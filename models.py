"""
Machine Learning Models for ICU Prediction
Implements LSTM, Transformer, MultitaskLSTM, and XGBoost models.

REMOVED: TCN (TCNBlock, TCNModel)
  Reason: nn.BatchNorm1d in TCNBlock collapses under FP16 AMP when mini-batches
  contain all-zero labels (common with rare ICU events — vasopressor, AKI Stage 3,
  sepsis). Running mean/variance → 0, normalization divides near-zero by near-zero,
  producing NaN loss from epoch 2-7 on every task. The fix (GroupNorm) would require
  revalidation; LSTM + XGBoost match or beat TCN AUROC on all confirmed tasks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
import xgboost as xgb
import yaml
import logging

logger = logging.getLogger(__name__)


# ── LSTM ──────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """Bidirectional LSTM for multi-task ICU outcome prediction."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_tasks: int = 6,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        super().__init__()

        self.input_size    = input_size
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.num_tasks     = num_tasks
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.dropout = nn.Dropout(dropout)

        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_output_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
            )
            for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_output  = self.dropout(lstm_out[:, -1, :])
        outputs = [head(last_output) for head in self.task_heads]
        return torch.cat(outputs, dim=1)


# ── Transformer ───────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer model for ICU prediction.

    FP16/AMP safety:
      - norm_first=True (Pre-LayerNorm): normalizes inputs BEFORE attention,
        preventing activation magnitude growth that causes FP16 overflow.
      - forward() disables autocast for the transformer encoder, running
        attention in FP32. Q·K^T dot products can exceed 65504 (FP16 max)
        after a few epochs → NaN softmax → NaN loss. FP32 max is 3.4e38.
    """

    def __init__(self,
                 input_size: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 3,
                 dim_feedforward: int = 256,
                 num_tasks: int = 6,
                 dropout: float = 0.3):
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,       # Pre-LN: stabilizes training under AMP
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_tasks),
        )

        # Xavier/Kaiming init for stable initial activations
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        """Xavier init for linear layers, standard for Transformers."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        # Run transformer encoder in FP32 to prevent attention score overflow
        # in FP16 (Q·K^T can exceed 65504 after a few epochs of training)
        with torch.amp.autocast('cuda', enabled=False):
            x = self.transformer_encoder(x.float())
        x = x.mean(dim=1)
        return self.fc(x)


# ── MultitaskLSTM ─────────────────────────────────────────────────────────────

class MultitaskLSTM(nn.Module):
    """
    Multitask LSTM with shared encoder + task-specific heads.
    Produces individual task predictions AND a composite deterioration score.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_task_groups: int = 6,
                 tasks_per_group: List[int] = None,
                 dropout: float = 0.3):
        super().__init__()

        self.hidden_size = hidden_size

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

        self.task_heads = nn.ModuleList()
        for n_tasks in tasks_per_group:
            self.task_heads.append(nn.Sequential(
                nn.Linear(enc_size, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_tasks),
            ))

        total_tasks = sum(tasks_per_group)
        self.composite_head = nn.Sequential(
            nn.Linear(enc_size + total_tasks, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.shared_lstm(x)
        shared_repr  = self.dropout(lstm_out[:, -1, :])

        task_outputs    = [head(shared_repr) for head in self.task_heads]
        all_tasks       = torch.cat(task_outputs, dim=1)
        composite_score = self.composite_head(torch.cat([shared_repr, all_tasks], dim=1))
        return torch.cat([all_tasks, composite_score], dim=1)


# ── XGBoost ───────────────────────────────────────────────────────────────────

class XGBoostPredictor:
    """
    XGBoost baseline for multi-task prediction using flattened features.
    XGBoost 2.x API: tree_method='hist', device='cuda'/'cpu'.
    """

    def __init__(self,
                 num_tasks: int = 6,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 n_estimators: int = 100,
                 subsample: float = 0.8,
                 tree_method: str = 'hist',
                 device: str = 'cpu'):
        self.num_tasks = num_tasks
        self.models    = []

        logger.info(
            f"XGBoost: {'GPU' if 'cuda' in device else 'CPU'} "
            f"(device={device}, tree_method=hist)"
        )

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
        n_samples, seq_len, n_features = X.shape
        return X.reshape(n_samples, -1)

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        X_flat = self.flatten_sequences(X)
        for i, model in enumerate(self.models):
            if len(np.unique(y[:, i])) < 2:
                if verbose:
                    logger.info(f"Skipping task {i + 1}/{self.num_tasks} (degenerate labels)")
                self.models[i] = None
                continue
            if verbose:
                logger.info(f"Training XGBoost task {i + 1}/{self.num_tasks}...")
            model.fit(X_flat, y[:, i], verbose=False)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_flat = self.flatten_sequences(X)
        predictions = []
        for model in self.models:
            if model is None:
                predictions.append(np.full(len(X_flat), 0.5))
            else:
                predictions.append(model.predict_proba(X_flat)[:, 1])
        return np.column_stack(predictions)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# ── Factory ───────────────────────────────────────────────────────────────────

def create_model(model_type: str, config: dict):
    """
    Factory function to create models.

    Supported types: 'lstm', 'transformer', 'multitask_lstm', 'xgboost'
    Removed:         'tcn'  (raises ValueError with explanation)
    """
    input_size = config.get('input_size', 50)
    num_tasks  = config.get('num_tasks', 6)

    mtype = model_type.lower()

    if mtype == 'tcn':
        raise ValueError(
            "TCN has been removed. "
            "nn.BatchNorm1d collapses under FP16 AMP with imbalanced ICU labels, "
            "producing NaN loss from epoch 2-7 on every task. "
            "Use 'lstm' or 'xgboost' instead."
        )

    if mtype == 'lstm':
        cfg = config.get('LSTM_CONFIG', {})
        return LSTMModel(
            input_size=input_size,
            hidden_size=cfg.get('hidden_size', 128),
            num_layers=cfg.get('num_layers', 2),
            num_tasks=num_tasks,
            dropout=cfg.get('dropout', 0.3),
            bidirectional=cfg.get('bidirectional', True),
        )

    if mtype == 'transformer':
        cfg = config.get('TRANSFORMER_CONFIG', {})
        return TransformerModel(
            input_size=input_size,
            d_model=cfg.get('d_model', 128),
            nhead=cfg.get('nhead', 8),
            num_layers=cfg.get('num_layers', 3),
            dim_feedforward=cfg.get('dim_feedforward', 256),
            num_tasks=num_tasks,
            dropout=cfg.get('dropout', 0.3),
        )

    if mtype == 'multitask_lstm':
        cfg = config.get('LSTM_CONFIG', {})
        return MultitaskLSTM(
            input_size=input_size,
            hidden_size=cfg.get('hidden_size', 128),
            num_layers=cfg.get('num_layers', 2),
            num_task_groups=6,
            tasks_per_group=config.get('tasks_per_group', [3, 3, 6, 3, 2, 3]),
            dropout=cfg.get('dropout', 0.3),
        )

    if mtype == 'xgboost':
        cfg = config.get('XGBOOST_CONFIG', {})
        return XGBoostPredictor(
            num_tasks=num_tasks,
            max_depth=cfg.get('max_depth', 6),
            learning_rate=cfg.get('learning_rate', 0.1),
            n_estimators=cfg.get('n_estimators', 100),
            subsample=cfg.get('subsample', 0.8),
            device=cfg.get('device', 'cpu'),
        )

    raise ValueError(
        f"Unknown model type: '{model_type}'. "
        "Choose from: lstm, transformer, multitask_lstm, xgboost"
    )


if __name__ == "__main__":
    logger.info("Testing model architectures (TCN removed)...")

    batch_size = 32
    seq_len    = 24
    input_size = 50
    num_tasks  = 6

    X = torch.randn(batch_size, seq_len, input_size)
    y = torch.randint(0, 2, (batch_size, num_tasks)).float()

    logger.info("=== LSTM ===")
    lstm = LSTMModel(input_size, hidden_size=128, num_tasks=num_tasks)
    print(f"LSTM output: {lstm(X).shape}")

    logger.info("=== Transformer ===")
    transformer = TransformerModel(input_size, num_tasks=num_tasks)
    print(f"Transformer output: {transformer(X).shape}")

    logger.info("=== MultitaskLSTM ===")
    mt = MultitaskLSTM(input_size, tasks_per_group=[3, 3, 6, 3, 2, 3])
    print(f"MultitaskLSTM output: {mt(X).shape}")

    logger.info("=== XGBoost ===")
    xgb_m = XGBoostPredictor(num_tasks=num_tasks)
    xgb_m.fit(X.numpy(), y.numpy(), verbose=True)
    print(f"XGBoost output: {xgb_m.predict_proba(X.numpy()).shape}")

    logger.info("✓ All tests passed!")