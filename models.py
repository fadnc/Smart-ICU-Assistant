"""
Machine Learning Models for ICU Prediction
Implements LSTM, TCN, Transformer, MultitaskLSTM, and XGBoost models.

FIXES:
  - XGBoostPredictor: removed deprecated gpu_id, tree_method='gpu_hist',
    use_label_encoder. Now uses tree_method='hist', device='cuda' (XGBoost 2.x API).
  - Auto GPU detection for XGBoost updated to match new API.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import xgboost as xgb
import yaml
import logging

logging.basicConfig(level=logging.INFO)
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
        super(LSTMModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_tasks = num_tasks
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
        last_output = self.dropout(lstm_out[:, -1, :])
        outputs = [head(last_output) for head in self.task_heads]
        return torch.cat(outputs, dim=1)


# ── TCN ───────────────────────────────────────────────────────────────────────

class TCNBlock(nn.Module):
    """Temporal Convolutional Network block with dilated convolutions."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.3):
        super(TCNBlock, self).__init__()

        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=self.padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=self.padding, dilation=dilation)

        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = out[:, :, :-self.padding]
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :-self.padding]
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.relu(out + residual)


class TCNModel(nn.Module):
    """Temporal Convolutional Network for multi-task ICU prediction."""

    def __init__(self,
                 input_size: int,
                 num_channels: List[int] = None,
                 kernel_size: int = 3,
                 num_tasks: int = 6,
                 dropout: float = 0.3):
        super(TCNModel, self).__init__()

        if num_channels is None:
            num_channels = [64, 128, 256]

        self.input_size = input_size
        self.num_tasks = num_tasks
        self.input_proj = nn.Conv1d(input_size, num_channels[0], 1)

        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = num_channels[i - 1] if i > 0 else num_channels[0]
            out_ch = num_channels[i]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))

        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_channels[-1], 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )
            for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        outputs = [head(x) for head in self.task_heads]
        return torch.cat(outputs, dim=1)


# ── Transformer ───────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
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
    """Transformer model for ICU prediction."""

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
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_tasks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
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
        shared_repr = self.dropout(lstm_out[:, -1, :])

        task_outputs = [head(shared_repr) for head in self.task_heads]
        all_tasks = torch.cat(task_outputs, dim=1)

        composite_score = self.composite_head(torch.cat([shared_repr, all_tasks], dim=1))
        return torch.cat([all_tasks, composite_score], dim=1)


# ── XGBoost ───────────────────────────────────────────────────────────────────

class XGBoostPredictor:
    """
    XGBoost baseline for multi-task prediction using flattened features.

    FIX: Updated to XGBoost 2.x API:
      - Removed deprecated tree_method='gpu_hist' → now tree_method='hist'
      - Removed deprecated gpu_id parameter → now device='cuda' or device='cpu'
      - Removed removed use_label_encoder parameter
    """

    def __init__(self,
                 num_tasks: int = 6,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 n_estimators: int = 100,
                 subsample: float = 0.8,
                 tree_method: str = 'hist',
                 gpu_id: int = 0):           # kept for API compat, mapped to device
        self.num_tasks = num_tasks
        self.models = []

        # FIX: Determine device using new XGBoost 2.x API
        import torch as _torch
        use_gpu = _torch.cuda.is_available()
        device = f'cuda:{gpu_id}' if use_gpu else 'cpu'

        if use_gpu:
            logger.info(f"XGBoost: using GPU (device={device}, tree_method=hist)")
        else:
            logger.info("XGBoost: using CPU (tree_method=hist)")

        for _ in range(num_tasks):
            # FIX: XGBoost 2.x params — no gpu_hist, no gpu_id, no use_label_encoder
            model = xgb.XGBClassifier(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                objective='binary:logistic',
                eval_metric='auc',
                tree_method='hist',          # works for both CPU and GPU
                device=device,              # replaces gpu_id + gpu_hist combo
                random_state=42,
            )
            self.models.append(model)

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
                pred = model.predict_proba(X_flat)[:, 1]
                predictions.append(pred)

        return np.column_stack(predictions)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)


# ── Multi-task loss ───────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    def __init__(self, task_weights: Optional[List[float]] = None):
        super(MultiTaskLoss, self).__init__()
        self.task_weights = task_weights
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        task_losses = self.bce(predictions, targets).mean(dim=0)

        if self.task_weights is not None:
            weights = torch.tensor(self.task_weights, device=task_losses.device)
            weighted_losses = task_losses * weights
        else:
            weighted_losses = task_losses

        return weighted_losses.mean()


# ── Factory ───────────────────────────────────────────────────────────────────

def create_model(model_type: str, config: dict):
    """
    Factory function to create models.

    Args:
        model_type: 'lstm', 'tcn', 'transformer', 'multitask_lstm', or 'xgboost'
        config: Configuration dictionary (must contain 'input_size' and 'num_tasks')
    """
    input_size = config.get('input_size', 50)
    num_tasks = config.get('num_tasks', 6)

    if model_type.lower() == 'lstm':
        lstm_config = config.get('LSTM_CONFIG', {})
        return LSTMModel(
            input_size=input_size,
            hidden_size=lstm_config.get('hidden_size', 128),
            num_layers=lstm_config.get('num_layers', 2),
            num_tasks=num_tasks,
            dropout=lstm_config.get('dropout', 0.3),
            bidirectional=lstm_config.get('bidirectional', True),
        )

    elif model_type.lower() == 'tcn':
        tcn_config = config.get('TCN_CONFIG', {})
        return TCNModel(
            input_size=input_size,
            num_channels=tcn_config.get('num_channels', [64, 128, 256]),
            kernel_size=tcn_config.get('kernel_size', 3),
            num_tasks=num_tasks,
            dropout=tcn_config.get('dropout', 0.3),
        )

    elif model_type.lower() == 'transformer':
        tf_config = config.get('TRANSFORMER_CONFIG', {})
        return TransformerModel(
            input_size=input_size,
            d_model=tf_config.get('d_model', 128),
            nhead=tf_config.get('nhead', 8),
            num_layers=tf_config.get('num_layers', 3),
            dim_feedforward=tf_config.get('dim_feedforward', 256),
            num_tasks=num_tasks,
            dropout=tf_config.get('dropout', 0.3),
        )

    elif model_type.lower() == 'multitask_lstm':
        lstm_config = config.get('LSTM_CONFIG', {})
        return MultitaskLSTM(
            input_size=input_size,
            hidden_size=lstm_config.get('hidden_size', 128),
            num_layers=lstm_config.get('num_layers', 2),
            num_task_groups=6,
            tasks_per_group=config.get('tasks_per_group', [3, 3, 6, 3, 2, 3]),
            dropout=lstm_config.get('dropout', 0.3),
        )

    elif model_type.lower() == 'xgboost':
        xgb_config = config.get('XGBOOST_CONFIG', {})
        return XGBoostPredictor(
            num_tasks=num_tasks,
            max_depth=xgb_config.get('max_depth', 6),
            learning_rate=xgb_config.get('learning_rate', 0.1),
            n_estimators=xgb_config.get('n_estimators', 100),
            subsample=xgb_config.get('subsample', 0.8),
            tree_method='hist',   # always hist in 2.x; device param controls GPU
            gpu_id=xgb_config.get('gpu_id', 0),
        )

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            "Choose from: lstm, tcn, transformer, multitask_lstm, xgboost"
        )


if __name__ == "__main__":
    logger.info("Testing model architectures...")

    batch_size = 32
    seq_len = 24
    input_size = 50
    num_tasks = 6

    X = torch.randn(batch_size, seq_len, input_size)
    y = torch.randint(0, 2, (batch_size, num_tasks)).float()

    logger.info("\n=== Testing LSTM Model ===")
    lstm = LSTMModel(input_size, hidden_size=128, num_tasks=num_tasks)
    print(f"LSTM output shape: {lstm(X).shape}")

    logger.info("\n=== Testing TCN Model ===")
    tcn = TCNModel(input_size, num_channels=[64, 128, 256], num_tasks=num_tasks)
    print(f"TCN output shape: {tcn(X).shape}")

    logger.info("\n=== Testing Transformer Model ===")
    transformer = TransformerModel(input_size, num_tasks=num_tasks)
    print(f"Transformer output shape: {transformer(X).shape}")

    logger.info("\n=== Testing MultitaskLSTM ===")
    mt_lstm = MultitaskLSTM(input_size, tasks_per_group=[3, 3, 6, 3, 2, 3])
    print(f"MultitaskLSTM output shape: {mt_lstm(X).shape}")

    logger.info("\n=== Testing XGBoost Model ===")
    xgb_model = XGBoostPredictor(num_tasks=num_tasks)
    X_np = X.numpy()
    y_np = y.numpy()
    xgb_model.fit(X_np, y_np, verbose=True)
    print(f"XGBoost predictions shape: {xgb_model.predict_proba(X_np).shape}")

    logger.info("\n✓ All model tests passed!")