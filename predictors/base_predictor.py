"""
Base Predictor — Shared logic for all ICU prediction tasks.
Each task-specific predictor inherits from this class.
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import timedelta
from abc import ABC, abstractmethod
import yaml

import torch
from models import create_model, XGBoostPredictor
from training import ModelTrainer, clear_gpu_memory

logger = logging.getLogger(__name__)


class BasePredictor(ABC):
    """
    Abstract base class for all ICU prediction tasks.

    Each predictor:
      1. Defines its own label generation logic
      2. Trains all 5 model types (LSTM, TCN, Transformer, MultitaskLSTM, XGBoost)
      3. Auto-selects the best model based on validation AUROC
      4. Saves per-task results and model comparison report
    """

    # -- Subclasses MUST override these ----------------------------------------
    TASK_NAME: str = ""                 # e.g. "mortality"
    TASK_DESCRIPTION: str = ""          # human-readable description
    WINDOWS: List[int] = []            # prediction horizons in hours
    LABEL_PREFIX: str = ""             # e.g. "mortality" → mortality_6h, mortality_12h
    MODELS_TO_TRY: List[str] = [
        'lstm', 'tcn', 'transformer', 'xgboost'
    ]

    def __init__(self, config_path: str = 'config.yaml'):
        self.config = self._load_config(config_path)
        self.config_path = config_path
        self.time_windows = self.config.get('TIME_WINDOWS', {})
        self.results: Dict = {}
        self.best_model_name: Optional[str] = None
        self.best_auroc: float = 0.0

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    # -- Label names -----------------------------------------------------------

    def get_label_names(self) -> List[str]:
        """Return list of label column names this predictor generates."""
        return [f'{self.LABEL_PREFIX}_{w}h' for w in self.WINDOWS]

    def get_num_tasks(self) -> int:
        return len(self.get_label_names())

    # -- Abstract methods (subclass MUST implement) ----------------------------

    @abstractmethod
    def generate_labels(self,
                        stay: pd.Series,
                        vitals: pd.DataFrame,
                        labs: pd.DataFrame,
                        current_time: pd.Timestamp,
                        **extra_data) -> Dict[str, int]:
        """
        Generate binary labels for one timepoint.

        Args:
            stay: One row from the merged ICU dataset
            vitals: Raw vitals DataFrame (datetime-indexed, columns like heartrate, meanbp, …)
            labs: Raw labs DataFrame (datetime-indexed, columns like creatinine, wbc, …)
            current_time: The prediction time
            **extra_data: Additional tables (prescriptions, diagnoses, chartevents, …)

        Returns:
            Dict mapping label_name → 0/1
        """
        ...

    # -- Training pipeline (shared) --------------------------------------------

    def train_all_models(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         timestamps: List,
                         output_dir: str = 'output') -> Dict:
        """
        Train every model type on this task's labels and select the best.

        Args:
            X: Feature sequences [n_samples, seq_len, n_features]
            y: Full label matrix [n_samples, total_labels]
               (contains ALL tasks; this method extracts only its own columns)
            timestamps: Corresponding timestamps for temporal splitting
            output_dir: Directory for saving results

        Returns:
            Dict with per-model AUROC scores + best model info
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs('models', exist_ok=True)

        task_labels = self._extract_task_labels(y)
        if task_labels is None:
            logger.warning(f"[{self.TASK_NAME}] Could not extract labels — skipping")
            return {}

        num_tasks = task_labels.shape[1]
        input_size = X.shape[2]

        comparison = {}

        for model_name in self.MODELS_TO_TRY:
            logger.info(f"[{self.TASK_NAME}] Training {model_name.upper()}...")
            try:
                # Clear VRAM before each model (critical for 4GB)
                clear_gpu_memory()
                
                metrics = self._train_single_model(
                    model_name, X, task_labels, timestamps, input_size, num_tasks
                )
                comparison[model_name] = metrics
                mean_auroc = metrics.get('mean_test_auroc', 0)
                logger.info(f"[{self.TASK_NAME}] {model_name.upper()} → Mean AUROC: {mean_auroc:.4f}")

                if mean_auroc > self.best_auroc:
                    self.best_auroc = mean_auroc
                    self.best_model_name = model_name
            except Exception as e:
                logger.error(f"[{self.TASK_NAME}] {model_name} failed: {e}")
                comparison[model_name] = {'error': str(e)}

        self.results = {
            'task': self.TASK_NAME,
            'description': self.TASK_DESCRIPTION,
            'labels': self.get_label_names(),
            'input_size': X.shape[2],
            'best_model': self.best_model_name,
            'best_auroc': self.best_auroc,
            'comparison': comparison,
        }

        # Save per-task report
        report_path = os.path.join(output_dir, f'{self.TASK_NAME}_report.json')
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"[{self.TASK_NAME}] Best model: {self.best_model_name} "
                     f"(AUROC {self.best_auroc:.4f}) — report saved to {report_path}")

        return self.results

    # -- Internal helpers ------------------------------------------------------

    def _extract_task_labels(self, y_full: np.ndarray) -> Optional[np.ndarray]:
        """Extract only this task's columns from the full label matrix."""
        label_names = self.get_label_names()
        num_labels = len(label_names)
        # The label matrix is ordered the same as generate_all_labels output
        # We need to know the column indices for this task in the full label matrix
        # This will be set up by the pipeline
        if hasattr(self, '_label_indices') and self._label_indices is not None:
            return y_full[:, self._label_indices]

        # Fallback: if we have exactly the right number of columns, use all
        if y_full.shape[1] == num_labels:
            return y_full

        logger.warning(f"[{self.TASK_NAME}] Cannot determine label columns "
                       f"(expected {num_labels}, got {y_full.shape[1]})")
        return None

    def set_label_indices(self, all_label_names: List[str]):
        """Set which columns in the full label matrix belong to this task."""
        my_labels = self.get_label_names()
        self._label_indices = []
        for name in my_labels:
            if name in all_label_names:
                self._label_indices.append(all_label_names.index(name))
            else:
                logger.warning(f"[{self.TASK_NAME}] Label '{name}' not found in full label list")
        if self._label_indices:
            self._label_indices = np.array(self._label_indices)
        else:
            self._label_indices = None

    def _train_single_model(self,
                            model_name: str,
                            X: np.ndarray,
                            y: np.ndarray,
                            timestamps: List,
                            input_size: int,
                            num_tasks: int) -> Dict:
        """Train one model type and return evaluation metrics."""
        config = self.config.copy()
        config['input_size'] = input_size
        config['num_tasks'] = num_tasks

        if model_name == 'xgboost':
            return self._train_xgboost(X, y, timestamps, config)
        else:
            return self._train_dl_model(model_name, X, y, timestamps, config)

    def _train_dl_model(self, model_name: str, X, y, timestamps, config) -> Dict:
        """Train a deep learning model (LSTM, TCN, Transformer)."""
        model = create_model(model_name, config)
        trainer = ModelTrainer(model, config)

        # Temporal split
        splits = trainer.temporal_split(X, y, timestamps)
        train_X, train_y = splits['train']
        val_X, val_y = splits['val']
        test_X, test_y = splits['test']

        # Train
        trainer.train(train_X, train_y, val_X, val_y, verbose=False)

        # Evaluate on test set
        predictions = trainer.predict(test_X)
        metrics = trainer.compute_metrics(predictions, test_y)

        # Compute per-task AUROC
        aurocs = [v for k, v in metrics.items()
                  if k.startswith('task_') and k.endswith('_auroc')
                  and not np.isnan(v)]
        mean_auroc = float(np.mean(aurocs)) if aurocs else 0.0

        # Save model checkpoint
        model_path = os.path.join('models', f'{self.TASK_NAME}_{model_name}.pth')
        trainer.save_checkpoint(model_path)

        return {
            'mean_test_auroc': mean_auroc,
            'per_task_metrics': metrics,
            'model_path': model_path,
        }

    def _train_xgboost(self, X, y, timestamps, config) -> Dict:
        """Train XGBoost baseline (GPU-accelerated when available)."""
        xgb_config = config.get('XGBOOST_CONFIG', {})
        predictor = XGBoostPredictor(
            num_tasks=y.shape[1],
            max_depth=xgb_config.get('max_depth', 6),
            learning_rate=xgb_config.get('learning_rate', 0.1),
            n_estimators=xgb_config.get('n_estimators', 100),
            subsample=xgb_config.get('subsample', 0.8),
            tree_method=xgb_config.get('tree_method', 'auto'),
            gpu_id=xgb_config.get('gpu_id', 0),
        )

        # Temporal split (use ModelTrainer just for splitting)
        dummy_model = create_model('lstm', config)
        trainer = ModelTrainer(dummy_model, config)
        splits = trainer.temporal_split(X, y, timestamps)
        train_X, train_y = splits['train']
        val_X, val_y = splits['val']
        test_X, test_y = splits['test']

        # Train
        predictor.fit(train_X, train_y, verbose=False)

        # Evaluate
        predictions = predictor.predict_proba(test_X)
        metrics = trainer.compute_metrics(predictions, test_y)

        aurocs = [v for k, v in metrics.items()
                  if k.startswith('task_') and k.endswith('_auroc')
                  and not np.isnan(v)]
        mean_auroc = float(np.mean(aurocs)) if aurocs else 0.0

        # Save model
        model_path = os.path.join('models', f'{self.TASK_NAME}_xgboost.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(predictor, f)

        return {
            'mean_test_auroc': mean_auroc,
            'per_task_metrics': metrics,
            'model_path': model_path,
        }

    # -- Standalone runner -----------------------------------------------------

    def __repr__(self):
        return (f"<{self.__class__.__name__} task='{self.TASK_NAME}' "
                f"windows={self.WINDOWS} best={self.best_model_name}>")
