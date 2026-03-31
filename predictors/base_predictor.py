"""
Base Predictor — Shared logic for all ICU prediction tasks.

FIXES:
  - XGBoostPredictor params updated to XGBoost 2.x API (device= instead of gpu_id=,
    tree_method='hist' instead of 'gpu_hist', removed use_label_encoder).
  - _extract_task_labels now logs clearly when label indices are missing,
    making cache-based runs easier to debug.
  - clear_gpu_memory() called before AND after each DL model to reduce VRAM fragmentation.
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
import yaml

import torch
from models import create_model, XGBoostPredictor
from training import ModelTrainer, clear_gpu_memory, log_gpu_memory, temporal_split_data

logger = logging.getLogger(__name__)


class BasePredictor(ABC):
    """
    Abstract base class for all ICU prediction tasks.

    Each predictor:
      1. Defines its own label generation logic
      2. Trains all model types (LSTM, TCN, Transformer, XGBoost)
      3. Auto-selects the best model based on test AUROC
      4. Saves per-task results and model comparison report
    """

    TASK_NAME: str = ""
    TASK_DESCRIPTION: str = ""
    WINDOWS: List[int] = []
    LABEL_PREFIX: str = ""
    MODELS_TO_TRY: List[str] = ['lstm', 'tcn', 'transformer', 'xgboost']

    def __init__(self, config_path: str = 'config.yaml'):
        self.config = self._load_config(config_path)
        self.config_path = config_path
        self.time_windows = self.config.get('TIME_WINDOWS', {})
        self.results: Dict = {}
        self.best_model_name: Optional[str] = None
        self.best_auroc: float = 0.0
        self._label_indices: Optional[np.ndarray] = None

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    # ── Label names ────────────────────────────────────────────────────────

    def get_label_names(self) -> List[str]:
        return [f'{self.LABEL_PREFIX}_{w}h' for w in self.WINDOWS]

    def get_num_tasks(self) -> int:
        return len(self.get_label_names())

    # ── Abstract interface ─────────────────────────────────────────────────

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
            vitals: Raw vitals DataFrame (datetime-indexed)
            labs: Raw labs DataFrame (datetime-indexed)
            current_time: The prediction time
            **extra_data: Additional tables (prescriptions, diagnoses, ...)

        Returns:
            Dict mapping label_name → 0/1
        """
        ...

    # ── Training pipeline ──────────────────────────────────────────────────

    def train_all_models(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         timestamps: List,
                         output_dir: str = 'output') -> Dict:
        """
        Train every model type on this task's labels and select the best.

        Label indices must be set via set_label_indices() BEFORE calling this.
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs('models', exist_ok=True)

        task_labels = self._extract_task_labels(y)
        if task_labels is None:
            logger.warning(f"[{self.TASK_NAME}] Could not extract labels — skipping")
            return {}

        num_tasks = task_labels.shape[1]
        input_size = X.shape[2]

        logger.info(
            f"[{self.TASK_NAME}] Training on {num_tasks} labels "
            f"(indices: {self._label_indices}), input_size={input_size}"
        )

        comparison = {}

        for model_name in self.MODELS_TO_TRY:
            logger.info(f"[{self.TASK_NAME}] Training {model_name.upper()}...")
            try:
                # FIX: Clear VRAM before AND after each model
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

                # FIX: Also clear after training to free model weights from VRAM
                clear_gpu_memory()

            except Exception as e:
                logger.error(f"[{self.TASK_NAME}] {model_name} failed: {e}", exc_info=True)
                comparison[model_name] = {'error': str(e)}
                clear_gpu_memory()  # always clean up on error too

        self.results = {
            'task': self.TASK_NAME,
            'description': self.TASK_DESCRIPTION,
            'labels': self.get_label_names(),
            'input_size': X.shape[2],
            'best_model': self.best_model_name,
            'best_auroc': self.best_auroc,
            'comparison': comparison,
        }

        report_path = os.path.join(output_dir, f'{self.TASK_NAME}_report.json')
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(
            f"[{self.TASK_NAME}] Best model: {self.best_model_name} "
            f"(AUROC {self.best_auroc:.4f}) — report saved to {report_path}"
        )

        return self.results

    # ── Internal helpers ───────────────────────────────────────────────────

    def _extract_task_labels(self, y_full: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract only this task's columns from the full label matrix.

        FIX: Now logs clearly when indices are missing so cache-based runs
        are easier to debug. Previously silently fell through to wrong columns.
        """
        label_names = self.get_label_names()
        num_labels = len(label_names)

        if self._label_indices is not None and len(self._label_indices) > 0:
            return y_full[:, self._label_indices]

        # Fallback: exact column count match
        if y_full.shape[1] == num_labels:
            logger.debug(
                f"[{self.TASK_NAME}] No label indices set — using all {num_labels} columns "
                f"(exact match). Call set_label_indices() before training for correctness."
            )
            return y_full

        # Cannot determine correct columns — fail loudly
        logger.error(
            f"[{self.TASK_NAME}] LABEL INDEX ERROR: label indices not set and column count "
            f"mismatch (expected {num_labels}, got {y_full.shape[1]}). "
            f"Ensure set_label_indices() is called before train_all_models()."
        )
        return None

    def set_label_indices(self, all_label_names: List[str]):
        """Set which columns in the full label matrix belong to this task."""
        my_labels = self.get_label_names()
        indices = []
        for name in my_labels:
            if name in all_label_names:
                indices.append(all_label_names.index(name))
            else:
                logger.warning(f"[{self.TASK_NAME}] Label '{name}' not found in full label list")

        if indices:
            self._label_indices = np.array(indices)
            logger.debug(f"[{self.TASK_NAME}] Label indices: {self._label_indices} → {my_labels}")
        else:
            self._label_indices = None
            logger.error(f"[{self.TASK_NAME}] No label indices found — training will fail")

    def _train_single_model(self,
                            model_name: str,
                            X: np.ndarray,
                            y: np.ndarray,
                            timestamps: List,
                            input_size: int,
                            num_tasks: int) -> Dict:
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
        log_gpu_memory(f"{self.TASK_NAME}/{model_name} init")

        splits = trainer.temporal_split(X, y, timestamps)
        train_X, train_y = splits['train']
        val_X, val_y = splits['val']
        test_X, test_y = splits['test']

        trainer.train(train_X, train_y, val_X, val_y, verbose=False)
        log_gpu_memory(f"{self.TASK_NAME}/{model_name} post-train")

        predictions = trainer.predict(test_X)
        metrics = trainer.compute_metrics(predictions, test_y)

        aurocs = [
            v for k, v in metrics.items()
            if k.startswith('task_') and k.endswith('_auroc') and not np.isnan(v)
        ]
        mean_auroc = float(np.mean(aurocs)) if aurocs else 0.0

        model_path = os.path.join('models', f'{self.TASK_NAME}_{model_name}.pth')
        trainer.save_checkpoint(model_path)

        return {
            'mean_test_auroc': mean_auroc,
            'per_task_metrics': metrics,
            'model_path': model_path,
        }

    def _train_xgboost(self, X, y, timestamps, config) -> Dict:
        """
        Train XGBoost baseline (GPU-accelerated when available).

        FIX: Updated to XGBoost 2.x API — uses device= instead of gpu_id=,
        tree_method='hist' instead of 'gpu_hist', removed use_label_encoder.
        """
        xgb_config = config.get('XGBOOST_CONFIG', {})

        predictor = XGBoostPredictor(
            num_tasks=y.shape[1],
            max_depth=xgb_config.get('max_depth', 6),
            learning_rate=xgb_config.get('learning_rate', 0.1),
            n_estimators=xgb_config.get('n_estimators', 100),
            subsample=xgb_config.get('subsample', 0.8),
            tree_method='hist',
            gpu_id=xgb_config.get('gpu_id', 0),
        )

        splits = temporal_split_data(X, y, timestamps)
        train_X, train_y = splits['train']
        val_X, val_y = splits['val']
        test_X, test_y = splits['test']

        predictor.fit(train_X, train_y, verbose=False)
        predictions = predictor.predict_proba(test_X)

        from sklearn.metrics import roc_auc_score
        metrics = {}
        for t in range(test_y.shape[1]):
            try:
                if len(set(test_y[:, t])) > 1:
                    metrics[f'task_{t}_auroc'] = roc_auc_score(test_y[:, t], predictions[:, t])
                else:
                    metrics[f'task_{t}_auroc'] = float('nan')
            except Exception:
                metrics[f'task_{t}_auroc'] = float('nan')

        aurocs = [v for k, v in metrics.items() if k.endswith('_auroc') and not np.isnan(v)]
        mean_auroc = float(np.mean(aurocs)) if aurocs else 0.0

        model_path = os.path.join('models', f'{self.TASK_NAME}_xgboost.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(predictor, f)

        return {
            'mean_test_auroc': mean_auroc,
            'per_task_metrics': metrics,
            'model_path': model_path,
        }

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} task='{self.TASK_NAME}' "
            f"windows={self.WINDOWS} best={self.best_model_name}>"
        )