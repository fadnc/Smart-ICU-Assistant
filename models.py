"""
Machine Learning Models for ICU Prediction
Implements LSTM (with attention), Transformer, XGBoost, and LightGBM.

These 4 models are trained per task and combined via ensemble methods
(AUROC²-weighted averaging + stacking meta-learner).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xgboost as xgb
import yaml
import logging
import math

logger = logging.getLogger(__name__)


# ── LSTM with Attention ───────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """
    Bidirectional LSTM with temporal attention pooling.

    Instead of using only the last timestep (lstm_out[:, -1, :]), attention-
    weighted pooling lets the model focus on the most clinically relevant
    timepoints — e.g., the hour where vitals spiked, not just the most recent
    measurement. This helps for tasks like mortality where the critical
    deterioration event may occur hours before the prediction point.
    """

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

        # Temporal attention: learns which timesteps matter most
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_size // 2, 1, bias=False),
        )

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
        lstm_out, _ = self.lstm(x)                          # (B, T, H)

        # Attention-weighted pooling over all timesteps
        attn_weights = self.attention(lstm_out)              # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)        # normalize over T
        context      = (lstm_out * attn_weights).sum(dim=1)  # (B, H)
        context      = self.dropout(context)

        outputs = [head(context) for head in self.task_heads]
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
                 device: str = 'cpu',
                 scale_pos_weight: float = 1.0,
                 min_child_weight: float = 1.0,
                 colsample_bytree: float = 1.0,
                 gamma: float = 0.0,
                 early_stopping_rounds: int = 0):
        self.num_tasks = num_tasks
        self.models    = []
        self.early_stopping_rounds = early_stopping_rounds

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
                scale_pos_weight=scale_pos_weight,
                min_child_weight=min_child_weight,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
            ))

    def flatten_sequences(self, X: np.ndarray) -> np.ndarray:
        n_samples, seq_len, n_features = X.shape
        return X.reshape(n_samples, -1)

    def fit(self, X: np.ndarray, y: np.ndarray,
            val_X: np.ndarray = None, val_y: np.ndarray = None,
            verbose: bool = False):
        X_flat = self.flatten_sequences(X)
        val_X_flat = self.flatten_sequences(val_X) if val_X is not None else None

        for i, model in enumerate(self.models):
            if len(np.unique(y[:, i])) < 2:
                if verbose:
                    logger.info(f"Skipping task {i + 1}/{self.num_tasks} (degenerate labels)")
                self.models[i] = None
                continue

            # Per-task class weight: neg_count / pos_count
            pos = y[:, i].sum()
            neg = len(y[:, i]) - pos
            spw = float(neg / pos) if pos > 0 else 1.0
            model.set_params(scale_pos_weight=spw)

            if verbose:
                logger.info(f"Training XGBoost task {i + 1}/{self.num_tasks} (spw={spw:.1f})...")

            fit_kwargs = {'verbose': False}
            if val_X_flat is not None and val_y is not None and self.early_stopping_rounds > 0:
                fit_kwargs['eval_set'] = [(val_X_flat, val_y[:, i])]
                model.set_params(early_stopping_rounds=self.early_stopping_rounds)

            model.fit(X_flat, y[:, i], **fit_kwargs)

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


# ── LightGBM ─────────────────────────────────────────────────────────────────

class LightGBMPredictor:
    """
    LightGBM for multi-task prediction using flattened features.
    Leaf-wise growth handles imbalanced data differently from XGBoost's
    level-wise approach, adding model diversity to the ensemble.
    """

    def __init__(self,
                 num_tasks: int = 6,
                 max_depth: int = -1,
                 num_leaves: int = 63,
                 learning_rate: float = 0.05,
                 n_estimators: int = 300,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 min_child_samples: int = 20,
                 device: str = 'cpu',
                 scale_pos_weight: float = 1.0,
                 early_stopping_rounds: int = 0):
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM is not installed. Install with: pip install lightgbm"
            )

        self.num_tasks = num_tasks
        self.models    = []
        self.early_stopping_rounds = early_stopping_rounds

        gpu_str = 'gpu' if 'cuda' in device else 'cpu'
        logger.info(f"LightGBM: {gpu_str.upper()} (device={device})")

        for _ in range(num_tasks):
            self.models.append(lgb.LGBMClassifier(
                max_depth=max_depth,
                num_leaves=num_leaves,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_samples=min_child_samples,
                objective='binary',
                metric='auc',
                device=gpu_str,
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                verbose=-1,
            ))

    def flatten_sequences(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        return X.reshape(n_samples, -1)

    def fit(self, X: np.ndarray, y: np.ndarray,
            val_X: np.ndarray = None, val_y: np.ndarray = None,
            verbose: bool = False):
        import lightgbm as lgb

        X_flat = self.flatten_sequences(X)
        val_X_flat = self.flatten_sequences(val_X) if val_X is not None else None

        for i, model in enumerate(self.models):
            if len(np.unique(y[:, i])) < 2:
                if verbose:
                    logger.info(f"Skipping LightGBM task {i + 1}/{self.num_tasks} (degenerate)")
                self.models[i] = None
                continue

            # Per-task class weight
            pos = y[:, i].sum()
            neg = len(y[:, i]) - pos
            spw = float(neg / pos) if pos > 0 else 1.0
            model.set_params(scale_pos_weight=spw)

            if verbose:
                logger.info(f"Training LightGBM task {i + 1}/{self.num_tasks} (spw={spw:.1f})...")

            callbacks = [lgb.log_evaluation(period=-1)]  # suppress logs
            if val_X_flat is not None and val_y is not None and self.early_stopping_rounds > 0:
                callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
                model.fit(
                    X_flat, y[:, i],
                    eval_set=[(val_X_flat, val_y[:, i])],
                    callbacks=callbacks,
                )
            else:
                model.fit(X_flat, y[:, i], callbacks=callbacks)

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

    Supported types: 'lstm', 'transformer', 'xgboost', 'lightgbm'
    """
    input_size = config.get('input_size', 50)
    num_tasks  = config.get('num_tasks', 6)

    mtype = model_type.lower()

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

    if mtype == 'xgboost':
        cfg = config.get('XGBOOST_CONFIG', {})
        return XGBoostPredictor(
            num_tasks=num_tasks,
            max_depth=cfg.get('max_depth', 6),
            learning_rate=cfg.get('learning_rate', 0.1),
            n_estimators=cfg.get('n_estimators', 100),
            subsample=cfg.get('subsample', 0.8),
            device=cfg.get('device', 'cpu'),
            min_child_weight=cfg.get('min_child_weight', 1.0),
            colsample_bytree=cfg.get('colsample_bytree', 1.0),
            gamma=cfg.get('gamma', 0.0),
            early_stopping_rounds=cfg.get('early_stopping_rounds', 0),
        )

    if mtype == 'lightgbm':
        cfg = config.get('LIGHTGBM_CONFIG', {})
        device_str = cfg.get('device', 'cpu')
        return LightGBMPredictor(
            num_tasks=num_tasks,
            max_depth=cfg.get('max_depth', -1),
            num_leaves=cfg.get('num_leaves', 63),
            learning_rate=cfg.get('learning_rate', 0.05),
            n_estimators=cfg.get('n_estimators', 300),
            subsample=cfg.get('subsample', 0.8),
            colsample_bytree=cfg.get('colsample_bytree', 0.8),
            min_child_samples=cfg.get('min_child_samples', 20),
            device=device_str,
            early_stopping_rounds=cfg.get('early_stopping_rounds', 0),
        )

    raise ValueError(
        f"Unknown model type: '{model_type}'. "
        "Choose from: lstm, transformer, xgboost, lightgbm"
    )


if __name__ == "__main__":
    logger.info("Testing model architectures...")

    batch_size = 32
    seq_len    = 24
    input_size = 50
    num_tasks  = 6

    X = torch.randn(batch_size, seq_len, input_size)
    y = torch.randint(0, 2, (batch_size, num_tasks)).float()

    logger.info("=== LSTM (with attention) ===")
    lstm = LSTMModel(input_size, hidden_size=128, num_tasks=num_tasks)
    print(f"LSTM output: {lstm(X).shape}")

    logger.info("=== Transformer ===")
    transformer = TransformerModel(input_size, num_tasks=num_tasks)
    print(f"Transformer output: {transformer(X).shape}")

    logger.info("=== XGBoost ===")
    xgb_m = XGBoostPredictor(num_tasks=num_tasks)
    xgb_m.fit(X.numpy(), y.numpy(), verbose=True)
    print(f"XGBoost output: {xgb_m.predict_proba(X.numpy()).shape}")

    logger.info("✓ All tests passed!")