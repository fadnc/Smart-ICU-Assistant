"""
Machine Learning Models for ICU Prediction
Implements LSTM (with attention), Transformer, MultitaskLSTM, XGBoost,
LightGBM, TabTransformer, and TFT models.

REMOVED: TCN (TCNBlock, TCNModel)
  Reason: nn.BatchNorm1d in TCNBlock collapses under FP16 AMP when mini-batches
  contain all-zero labels (common with rare ICU events — vasopressor, AKI Stage 3,
  sepsis). Running mean/variance → 0, normalization divides near-zero by near-zero,
  producing NaN loss from epoch 2-7 on every task. The fix (GroupNorm) would require
  revalidation; LSTM + XGBoost match or beat TCN AUROC on all confirmed tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
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


# ── TabTransformer ────────────────────────────────────────────────────────────

class TabTransformerModel(nn.Module):
    """
    TabTransformer for time-series tabular ICU data.

    Architecture: applies per-timestep feature attention, then temporal pooling.
    Input shape: (batch, seq_len=24, features=81)

    Design:
      1. Per-timestep: project each feature to an embedding, run column-wise
         self-attention (features attend to each other within each timestep)
      2. Temporal: apply learned temporal attention over the 24 timesteps
      3. Output: task-specific heads producing (batch, num_tasks)

    This captures both inter-feature relationships (which vitals/labs interact)
    and temporal dynamics (how those relationships evolve over time).
    """

    def __init__(self,
                 input_size: int,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 num_tasks: int = 6,
                 dropout: float = 0.3):
        super().__init__()

        self.input_size = input_size
        self.d_model    = d_model

        # Per-feature embedding: each of the 81 features gets its own embedding
        self.feature_embedding = nn.Linear(1, d_model)
        # Learnable feature-type embeddings (like column IDs)
        self.feature_type_embed = nn.Parameter(torch.randn(input_size, d_model) * 0.02)

        # Column-wise attention: features attend to each other per timestep
        col_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.column_transformer = nn.TransformerEncoder(col_layer, num_layers=num_layers)

        # Temporal attention over timesteps
        self.temporal_proj = nn.Linear(input_size * d_model, d_model * 2)
        temp_layer = nn.TransformerEncoderLayer(
            d_model=d_model * 2, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(temp_layer, num_layers=1)

        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_tasks),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape  # (batch, 24, 81)

        # 1. Per-timestep column attention
        # Reshape to (B*T, F, 1) → embed → (B*T, F, d_model)
        x_flat = x.reshape(B * T, F, 1)
        feat_emb = self.feature_embedding(x_flat) + self.feature_type_embed.unsqueeze(0)

        # Run FP32 to avoid attention overflow
        with torch.amp.autocast('cuda', enabled=False):
            col_out = self.column_transformer(feat_emb.float())   # (B*T, F, d_model)

        # 2. Reshape back: flatten features per timestep → (B, T, F*d_model)
        col_out = col_out.reshape(B, T, F * self.d_model)
        temporal_in = self.temporal_proj(col_out)               # (B, T, d_model*2)

        # 3. Temporal attention
        with torch.amp.autocast('cuda', enabled=False):
            temp_out = self.temporal_transformer(temporal_in.float())  # (B, T, d_model*2)

        # 4. Mean pool over time and predict
        pooled = temp_out.mean(dim=1)
        return self.fc(pooled)


# ── Temporal Fusion Transformer (TFT) ─────────────────────────────────────────

class _GatedLinearUnit(nn.Module):
    """GLU: σ(Wx + b) ⊙ (Vx + c)"""
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_out)
        self.fc2 = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc1(x)) * self.fc2(x)


class _GatedResidualNetwork(nn.Module):
    """GRN with optional context vector."""
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.gate = _GatedLinearUnit(d_out, d_out)
        self.layernorm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = F.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        h = self.gate(h)
        return self.layernorm(h + residual)


class _VariableSelectionNetwork(nn.Module):
    """Selects and weights the most relevant input variables."""
    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Per-feature GRN
        self.feature_grns = nn.ModuleList([
            _GatedResidualNetwork(1, d_model, d_model, dropout)
            for _ in range(n_features)
        ])
        # Softmax variable weights
        self.weight_grn = _GatedResidualNetwork(
            n_features * d_model, d_model, n_features, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., n_features)
        original_shape = x.shape[:-1]

        # Transform each feature
        feature_outputs = []
        for i in range(self.n_features):
            fi = x[..., i:i+1]  # (..., 1)
            feature_outputs.append(self.feature_grns[i](fi))  # (..., d_model)
        stacked = torch.stack(feature_outputs, dim=-2)  # (..., n_features, d_model)

        # Compute variable selection weights
        flat = stacked.reshape(*original_shape, -1)  # (..., n_features * d_model)
        weights = F.softmax(self.weight_grn(flat), dim=-1)  # (..., n_features)

        # Weighted sum
        selected = (stacked * weights.unsqueeze(-1)).sum(dim=-2)  # (..., d_model)
        return selected


class TFTModel(nn.Module):
    """
    Temporal Fusion Transformer — designed for multivariate time-series
    with mixed static and temporal inputs.

    Input: (batch, seq_len=24, n_features=81) — 24 hourly timesteps of
    81 features including vitals, labs, and derived rolling features.

    Architecture:
      1. Variable Selection Network — learns which features matter most
      2. LSTM encoder — captures local temporal patterns
      3. Interpretable Multi-Head Attention — attends across timesteps
      4. Gated output — task-specific predictions

    Best suited for sepsis and AKI tasks where temporal trajectory patterns
    (rising creatinine, worsening SIRS criteria) are diagnostically critical.
    """

    def __init__(self,
                 input_size: int,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 1,
                 num_tasks: int = 6,
                 dropout: float = 0.3):
        super().__init__()

        self.d_model = d_model

        # Variable selection (per-timestep)
        self.vsn = _VariableSelectionNetwork(input_size, d_model, dropout)

        # Temporal processing via LSTM
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_model,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.post_lstm_gate = _GatedLinearUnit(d_model, d_model)
        self.post_lstm_norm = nn.LayerNorm(d_model)

        # Interpretable multi-head attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.post_attn_gate = _GatedLinearUnit(d_model, d_model)
        self.post_attn_norm = nn.LayerNorm(d_model)

        # Feed-forward
        self.ff = _GatedResidualNetwork(d_model, d_model * 2, d_model, dropout)

        # Output
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_tasks),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape

        # 1. Variable selection per timestep
        selected = self.vsn(x)                      # (B, T, d_model)

        # 2. LSTM encoding
        lstm_out, _ = self.lstm(selected)            # (B, T, d_model)
        gated = self.post_lstm_gate(lstm_out)
        temporal = self.post_lstm_norm(gated + selected)  # skip connection

        # 3. Interpretable multi-head attention (FP32 for stability)
        with torch.amp.autocast('cuda', enabled=False):
            attn_out, _ = self.self_attn(
                temporal.float(), temporal.float(), temporal.float()
            )
        attn_gated = self.post_attn_gate(attn_out)
        enriched = self.post_attn_norm(attn_gated + temporal)

        # 4. Feed-forward + pool
        output = self.ff(enriched)                  # (B, T, d_model)
        pooled = output.mean(dim=1)                 # (B, d_model)

        return self.fc(pooled)


# ── Factory ───────────────────────────────────────────────────────────────────

def create_model(model_type: str, config: dict):
    """
    Factory function to create models.

    Supported types: 'lstm', 'transformer', 'multitask_lstm', 'xgboost',
                     'lightgbm', 'tabtransformer', 'tft'
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

    if mtype == 'tabtransformer':
        cfg = config.get('TABTRANSFORMER_CONFIG', {})
        return TabTransformerModel(
            input_size=input_size,
            d_model=cfg.get('d_model', 64),
            nhead=cfg.get('nhead', 4),
            num_layers=cfg.get('num_layers', 2),
            num_tasks=num_tasks,
            dropout=cfg.get('dropout', 0.3),
        )

    if mtype == 'tft':
        cfg = config.get('TFT_CONFIG', {})
        return TFTModel(
            input_size=input_size,
            d_model=cfg.get('d_model', 64),
            nhead=cfg.get('nhead', 4),
            num_layers=cfg.get('num_layers', 1),
            num_tasks=num_tasks,
            dropout=cfg.get('dropout', 0.3),
        )

    raise ValueError(
        f"Unknown model type: '{model_type}'. "
        "Choose from: lstm, transformer, multitask_lstm, xgboost, "
        "lightgbm, tabtransformer, tft"
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

    logger.info("=== MultitaskLSTM ===")
    mt = MultitaskLSTM(input_size, tasks_per_group=[3, 3, 6, 3, 2, 3])
    print(f"MultitaskLSTM output: {mt(X).shape}")

    logger.info("=== TabTransformer ===")
    tab = TabTransformerModel(input_size, num_tasks=num_tasks)
    print(f"TabTransformer output: {tab(X).shape}")

    logger.info("=== TFT ===")
    tft = TFTModel(input_size, num_tasks=num_tasks)
    print(f"TFT output: {tft(X).shape}")

    logger.info("=== XGBoost ===")
    xgb_m = XGBoostPredictor(num_tasks=num_tasks)
    xgb_m.fit(X.numpy(), y.numpy(), verbose=True)
    print(f"XGBoost output: {xgb_m.predict_proba(X.numpy()).shape}")

    logger.info("✓ All tests passed!")