"""
Machine Learning Models for ICU Prediction
Implements LSTM, TCN, and XGBoost models for multi-task prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """
    Bidirectional LSTM for multi-task ICU outcome prediction
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_tasks: int = 6,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        """
        Initialize LSTM model
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            num_tasks: Number of prediction tasks
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_tasks = num_tasks
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Task-specific output heads
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_output_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            )
            for _ in range(num_tasks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            
        Returns:
            Output tensor [batch_size, num_tasks]
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last timestep output
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            last_output = lstm_out[:, -1, :]
        else:
            last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Multi-task outputs
        outputs = []
        for head in self.task_heads:
            task_output = head(last_output)
            outputs.append(task_output)
        
        # Concatenate all task outputs
        output = torch.cat(outputs, dim=1)  # [batch_size, num_tasks]
        
        return output


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block with dilated convolutions
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int,
                 dropout: float = 0.3):
        super(TCNBlock, self).__init__()
        
        # Causal padding
        self.padding = (kernel_size - 1) * dilation
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        
        # Normalization and activation
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection"""
        residual = x
        
        # First conv block
        out = self.conv1(x)
        out = out[:, :, :-self.padding]  # Remove future padding (causal)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = out[:, :, :-self.padding]
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        return self.relu(out + residual)


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network for multi-task ICU prediction
    """
    
    def __init__(self,
                 input_size: int,
                 num_channels: List[int] = [64, 128, 256],
                 kernel_size: int = 3,
                 num_tasks: int = 6,
                 dropout: float = 0.3):
        """
        Initialize TCN model
        
        Args:
            input_size: Number of input features
            num_channels: List of channel sizes for each TCN level
            kernel_size: Convolutional kernel size
            num_tasks: Number of prediction tasks
            dropout: Dropout rate
        """
        super(TCNModel, self).__init__()
        
        self.input_size = input_size
        self.num_tasks = num_tasks
        
        # Input projection
        self.input_proj = nn.Conv1d(input_size, num_channels[0], 1)
        
        # TCN blocks with increasing dilation
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = num_channels[i-1] if i > 0 else num_channels[0]
            out_ch = num_channels[i]
            
            layers.append(
                TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
        
        self.tcn = nn.Sequential(*layers)
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Task-specific heads
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_channels[-1], 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            for _ in range(num_tasks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            
        Returns:
            Output tensor [batch_size, num_tasks]
        """
        # Transpose for Conv1d: [batch, features, time]
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_proj(x)
        
        # TCN forward
        x = self.tcn(x)
        
        # Global pooling
        x = self.pool(x).squeeze(-1)  # [batch, channels]
        
        # Multi-task outputs
        outputs = []
        for head in self.task_heads:
            task_output = head(x)
            outputs.append(task_output)
        
        output = torch.cat(outputs, dim=1)  # [batch_size, num_tasks]
        
        return output


class XGBoostPredictor:
    """
    XGBoost baseline for multi-task prediction using flattened features
    """
    
    def __init__(self,
                 num_tasks: int = 6,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 n_estimators: int = 100,
                 subsample: float = 0.8,
                 tree_method: str = 'auto',
                 gpu_id: int = 0):
        """
        Initialize XGBoost predictor
        
        Args:
            num_tasks: Number of prediction tasks
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            n_estimators: Number of boosting rounds
            subsample: Subsample ratio
            tree_method: 'gpu_hist' for GPU, 'auto' for CPU
            gpu_id: GPU device id
        """
        self.num_tasks = num_tasks
        self.models = []
        
        # Auto-detect GPU
        import torch
        if tree_method == 'auto' and torch.cuda.is_available():
            tree_method = 'gpu_hist'
            logger.info(f"XGBoost: using GPU (gpu_hist, device {gpu_id})")
        
        # Create one model per task
        for i in range(num_tasks):
            xgb_params = dict(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                objective='binary:logistic',
                eval_metric='auc',
                use_label_encoder=False,
                random_state=42,
                tree_method=tree_method,
            )
            if tree_method == 'gpu_hist':
                xgb_params['gpu_id'] = gpu_id
            model = xgb.XGBClassifier(**xgb_params)
            self.models.append(model)
    
    def flatten_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Flatten temporal sequences to feature vectors
        
        Args:
            X: Input sequences [n_samples, seq_len, n_features]
            
        Returns:
            Flattened features [n_samples, seq_len * n_features]
        """
        n_samples, seq_len, n_features = X.shape
        return X.reshape(n_samples, -1)
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        """
        Train XGBoost models
        
        Args:
            X: Training sequences [n_samples, seq_len, n_features]
            y: Labels [n_samples, num_tasks]
            verbose: Print training progress
        """
        # Flatten sequences
        X_flat = self.flatten_sequences(X)
        
        # Train each task model
        for i, model in enumerate(self.models):
            # Skip tasks with degenerate labels (all same class)
            if len(np.unique(y[:, i])) < 2:
                if verbose:
                    logger.info(f"Skipping task {i+1}/{self.num_tasks} (degenerate labels)")
                self.models[i] = None  # Mark as untrained
                continue
            
            if verbose:
                logger.info(f"Training task {i+1}/{self.num_tasks}...")
            
            model.fit(X_flat, y[:, i], verbose=False)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities
        
        Args:
            X: Input sequences [n_samples, seq_len, n_features]
            
        Returns:
            Predictions [n_samples, num_tasks]
        """
        X_flat = self.flatten_sequences(X)
        
        predictions = []
        for model in self.models:
            if model is None:
                # Untrained model (degenerate labels) — predict neutral probability
                predictions.append(np.full(len(X_flat), 0.5))
            else:
                pred = model.predict_proba(X_flat)[:, 1]  # Get positive class probability
                predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels
        
        Args:
            X: Input sequences
            threshold: Classification threshold
            
        Returns:
            Binary predictions [n_samples, num_tasks]
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with task weighting
    """
    
    def __init__(self, task_weights: Optional[List[float]] = None):
        """
        Initialize multi-task loss
        
        Args:
            task_weights: Optional weights for each task
        """
        super(MultiTaskLoss, self).__init__()
        self.task_weights = task_weights
        self.bce = nn.BCELoss(reduction='none')
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted multi-task loss
        
        Args:
            predictions: Model predictions [batch_size, num_tasks]
            targets: Ground truth labels [batch_size, num_tasks]
            
        Returns:
            Scalar loss
        """
        # Compute BCE for each task
        task_losses = self.bce(predictions, targets).mean(dim=0)  # [num_tasks]
        
        # Apply task weights
        if self.task_weights is not None:
            weights = torch.tensor(self.task_weights, device=task_losses.device)
            weighted_losses = task_losses * weights
        else:
            weighted_losses = task_losses
        
        # Return mean loss across tasks
        return weighted_losses.mean()


# ==================== NEW ARCHITECTURES ====================


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer model"""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer model for ICU prediction (Fadhi Task 1: Sepsis)
    Multi-head self-attention captures cross-feature interactions
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
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_tasks),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
        Returns:
            [batch_size, num_tasks]
        """
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Use CLS-like aggregation: mean pool across time
        x = x.mean(dim=1)
        return self.fc(x)


class MultitaskLSTM(nn.Module):
    """
    Multitask LSTM with shared encoder + task-specific heads (Fadhi Task 4)
    Produces individual task predictions AND a composite deterioration score
    
    Architecture:
        Shared BiLSTM encoder → shared representation
        → Task-specific heads (mortality, sepsis, AKI, hypotension, vasopressor, ventilation)
        → Composite head (combines shared representation into unified risk score)
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
        
        # Shared encoder
        self.shared_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        enc_size = hidden_size * 2  # bidirectional
        
        # Default tasks_per_group: [3 mortality, 3 sepsis, 6 AKI, 3 hypotension, 2 vasopressor, 3 ventilation]
        if tasks_per_group is None:
            tasks_per_group = [3, 3, 6, 3, 2, 3]
        self.tasks_per_group = tasks_per_group
        
        # Task-specific heads
        self.task_heads = nn.ModuleList()
        for n_tasks in tasks_per_group:
            head = nn.Sequential(
                nn.Linear(enc_size, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_tasks),
                nn.Sigmoid()
            )
            self.task_heads.append(head)
        
        # Composite deterioration score head
        total_tasks = sum(tasks_per_group)
        self.composite_head = nn.Sequential(
            nn.Linear(enc_size + total_tasks, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_size]
        Returns:
            [batch_size, total_tasks + 1]  (last column = composite score)
        """
        # Shared encoding
        lstm_out, _ = self.shared_lstm(x)
        shared_repr = self.dropout(lstm_out[:, -1, :])
        
        # Task-specific predictions
        task_outputs = []
        for head in self.task_heads:
            task_outputs.append(head(shared_repr))
        
        all_tasks = torch.cat(task_outputs, dim=1)
        
        # Composite deterioration score
        composite_input = torch.cat([shared_repr, all_tasks], dim=1)
        composite_score = self.composite_head(composite_input)
        
        # Return all tasks + composite as last column
        return torch.cat([all_tasks, composite_score], dim=1)


def create_model(model_type: str, config: dict) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_type: 'lstm', 'tcn', 'transformer', 'multitask_lstm', or 'xgboost'
        config: Configuration dictionary
        
    Returns:
        Model instance
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
            bidirectional=lstm_config.get('bidirectional', True)
        )
    
    elif model_type.lower() == 'tcn':
        tcn_config = config.get('TCN_CONFIG', {})
        return TCNModel(
            input_size=input_size,
            num_channels=tcn_config.get('num_channels', [64, 128, 256]),
            kernel_size=tcn_config.get('kernel_size', 3),
            num_tasks=num_tasks,
            dropout=tcn_config.get('dropout', 0.3)
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
            dropout=tf_config.get('dropout', 0.3)
        )
    
    elif model_type.lower() == 'multitask_lstm':
        lstm_config = config.get('LSTM_CONFIG', {})
        return MultitaskLSTM(
            input_size=input_size,
            hidden_size=lstm_config.get('hidden_size', 128),
            num_layers=lstm_config.get('num_layers', 2),
            num_task_groups=6,
            tasks_per_group=config.get('tasks_per_group', [3, 3, 6, 3, 2, 3]),
            dropout=lstm_config.get('dropout', 0.3)
        )
    
    elif model_type.lower() == 'xgboost':
        xgb_config = config.get('XGBOOST_CONFIG', {})
        return XGBoostPredictor(
            num_tasks=num_tasks,
            max_depth=xgb_config.get('max_depth', 6),
            learning_rate=xgb_config.get('learning_rate', 0.1),
            n_estimators=xgb_config.get('n_estimators', 100),
            subsample=xgb_config.get('subsample', 0.8)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: lstm, tcn, transformer, multitask_lstm, xgboost")


if __name__ == "__main__":
    # Test models
    logger.info("Testing model architectures...")
    
    # Test data
    batch_size = 32
    seq_len = 24
    input_size = 50
    num_tasks = 6
    
    X = torch.randn(batch_size, seq_len, input_size)
    y = torch.randint(0, 2, (batch_size, num_tasks)).float()
    
    # Test LSTM
    logger.info("\n=== Testing LSTM Model ===")
    lstm = LSTMModel(input_size, hidden_size=128, num_tasks=num_tasks)
    lstm_out = lstm(X)
    print(f"LSTM output shape: {lstm_out.shape}")
    
    # Test TCN
    logger.info("\n=== Testing TCN Model ===")
    tcn = TCNModel(input_size, num_channels=[64, 128, 256], num_tasks=num_tasks)
    tcn_out = tcn(X)
    print(f"TCN output shape: {tcn_out.shape}")
    
    # Test Transformer
    logger.info("\n=== Testing Transformer Model ===")
    transformer = TransformerModel(input_size, num_tasks=num_tasks)
    tf_out = transformer(X)
    print(f"Transformer output shape: {tf_out.shape}")
    
    # Test MultitaskLSTM
    logger.info("\n=== Testing MultitaskLSTM ===")
    mt_lstm = MultitaskLSTM(input_size, tasks_per_group=[3, 3, 6, 3, 2, 3])
    mt_out = mt_lstm(X)
    print(f"MultitaskLSTM output shape: {mt_out.shape}")
    print(f"  (20 task predictions + 1 composite deterioration score = 21)")
    
    # Test XGBoost
    logger.info("\n=== Testing XGBoost Model ===")
    xgb_model = XGBoostPredictor(num_tasks=num_tasks)
    X_np = X.numpy()
    y_np = y.numpy()
    xgb_model.fit(X_np, y_np, verbose=True)
    xgb_pred = xgb_model.predict_proba(X_np)
    print(f"XGBoost predictions shape: {xgb_pred.shape}")
    
    # Test multi-task loss
    logger.info("\n=== Testing Multi-Task Loss ===")
    criterion = MultiTaskLoss(task_weights=[1.0] * num_tasks)
    loss = criterion(lstm_out, y)
    print(f"Multi-task loss: {loss.item():.4f}")
    
    logger.info("\n✓ All model tests passed!")

