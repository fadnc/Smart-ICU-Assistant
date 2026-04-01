"""
Base Predictor — Shared logic for all ICU prediction tasks.

A100 changes:
  - MODELS_TO_TRY: ['lstm', 'transformer', 'xgboost']  (TCN removed)
  - All GPU config (batch_size, workers, precision) read from config.yaml GPU_CONFIG
    so tuning one file propagates everywhere.
  - _train_dl_model passes full config including GPU_CONFIG to ModelTrainer.
  - Val batch = train batch × 2 (no backward pass, can fit double).
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
import yaml
from tqdm import tqdm

import torch
from models import create_model, XGBoostPredictor
from training import ModelTrainer, clear_gpu_memory, log_gpu_memory, temporal_split_data

logger = logging.getLogger(__name__)


class BasePredictor(ABC):

    TASK_NAME:        str       = ""
    TASK_DESCRIPTION: str       = ""
    WINDOWS:          List[int] = []
    LABEL_PREFIX:     str       = ""
    MODELS_TO_TRY:    List[str] = ['lstm', 'transformer', 'xgboost']

    def __init__(self, config_path: str = 'config.yaml'):
        self.config       = self._load_config(config_path)
        self.config_path  = config_path
        self.time_windows = self.config.get('TIME_WINDOWS', {})
        self.results:          Dict             = {}
        self.best_model_name:  Optional[str]   = None
        self.best_auroc:       float            = 0.0
        self._label_indices:   Optional[np.ndarray] = None

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_label_names(self) -> List[str]:
        return [f'{self.LABEL_PREFIX}_{w}h' for w in self.WINDOWS]

    def get_num_tasks(self) -> int:
        return len(self.get_label_names())

    @abstractmethod
    def generate_labels(self,
                        stay: pd.Series,
                        vitals: pd.DataFrame,
                        labs: pd.DataFrame,
                        current_time: pd.Timestamp,
                        **extra_data) -> Dict[str, int]:
        ...

    # ── Training pipeline ─────────────────────────────────────────────────────

    def train_all_models(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         timestamps: List,
                         output_dir: str = 'output') -> Dict:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs('models', exist_ok=True)

        task_labels = self._extract_task_labels(y)
        if task_labels is None:
            logger.warning(f"[{self.TASK_NAME}] Could not extract labels — skipping")
            return {}

        num_tasks  = task_labels.shape[1]
        input_size = X.shape[2]

        logger.info(
            f"[{self.TASK_NAME}] Training {len(self.MODELS_TO_TRY)} models | "
            f"{num_tasks} labels | input_size={input_size}"
        )

        comparison = {}

        model_bar = tqdm(
            self.MODELS_TO_TRY,
            desc=f"  [{self.TASK_NAME}] models          ",
            unit="model",
            file=sys.stderr,
            dynamic_ncols=True,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )

        for model_name in model_bar:
            model_bar.set_postfix_str(f"training {model_name.upper()}…")
            try:
                clear_gpu_memory()
                metrics    = self._train_single_model(
                    model_name, X, task_labels, timestamps, input_size, num_tasks
                )
                comparison[model_name] = metrics
                mean_auroc = metrics.get('mean_test_auroc', 0.0)
                model_bar.set_postfix_str(f"{model_name.upper()} AUROC={mean_auroc:.4f}")
                logger.info(f"[{self.TASK_NAME}] {model_name.upper()} → AUROC={mean_auroc:.4f}")

                if mean_auroc > self.best_auroc:
                    self.best_auroc      = mean_auroc
                    self.best_model_name = model_name

                clear_gpu_memory()

            except Exception as e:
                logger.error(f"[{self.TASK_NAME}] {model_name} failed: {e}", exc_info=True)
                comparison[model_name] = {'error': str(e)}
                model_bar.set_postfix_str(f"{model_name.upper()} FAILED")
                clear_gpu_memory()

        model_bar.close()

        self.results = {
            'task':        self.TASK_NAME,
            'description': self.TASK_DESCRIPTION,
            'labels':      self.get_label_names(),
            'input_size':  X.shape[2],
            'best_model':  self.best_model_name,
            'best_auroc':  self.best_auroc,
            'comparison':  comparison,
        }

        report_path = os.path.join(output_dir, f'{self.TASK_NAME}_report.json')
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(
            f"[{self.TASK_NAME}] ★ Best: {self.best_model_name} "
            f"(AUROC {self.best_auroc:.4f}) → {report_path}"
        )
        return self.results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_task_labels(self, y_full: np.ndarray) -> Optional[np.ndarray]:
        label_names = self.get_label_names()
        num_labels  = len(label_names)

        if self._label_indices is not None and len(self._label_indices) > 0:
            return y_full[:, self._label_indices]

        if y_full.shape[1] == num_labels:
            return y_full

        logger.error(
            f"[{self.TASK_NAME}] LABEL INDEX ERROR: expected {num_labels} cols, "
            f"got {y_full.shape[1]}. Call set_label_indices() before training."
        )
        return None

    def set_label_indices(self, all_label_names: List[str]):
        my_labels = self.get_label_names()
        indices   = [all_label_names.index(n) for n in my_labels if n in all_label_names]
        missing   = [n for n in my_labels if n not in all_label_names]
        if missing:
            logger.warning(f"[{self.TASK_NAME}] Labels not found: {missing}")
        self._label_indices = np.array(indices) if indices else None
        if self._label_indices is None:
            logger.error(f"[{self.TASK_NAME}] No label indices — training will fail")

    def _train_single_model(self, model_name, X, y, timestamps, input_size, num_tasks):
        config               = self.config.copy()
        config['input_size'] = input_size
        config['num_tasks']  = num_tasks
        if model_name == 'xgboost':
            return self._train_xgboost(X, y, timestamps, config)
        return self._train_dl_model(model_name, X, y, timestamps, config)

    def _train_dl_model(self, model_name, X, y, timestamps, config):
        model   = create_model(model_name, config)
        trainer = ModelTrainer(model, config)
        log_gpu_memory(f"{self.TASK_NAME}/{model_name} init")

        splits           = trainer.temporal_split(X, y, timestamps)
        train_X, train_y = splits['train']
        val_X,   val_y   = splits['val']
        test_X,  test_y  = splits['test']

        trainer.train(
            train_X, train_y, val_X, val_y,
            task_name=self.TASK_NAME, model_name=model_name,
        )
        log_gpu_memory(f"{self.TASK_NAME}/{model_name} post-train")

        predictions = trainer.predict(test_X)
        metrics     = trainer.compute_metrics(predictions, test_y)
        mean_auroc  = metrics.get('mean_auroc', 0.0)

        model_path = os.path.join('models', f'{self.TASK_NAME}_{model_name}.pth')
        trainer.save_checkpoint(model_path)

        return {
            'mean_test_auroc':  mean_auroc,
            'per_task_metrics': metrics,
            'model_path':       model_path,
        }

    def _train_xgboost(self, X, y, timestamps, config):
        xgb_cfg   = config.get('XGBOOST_CONFIG', {})
        predictor = XGBoostPredictor(
            num_tasks     = y.shape[1],
            max_depth     = xgb_cfg.get('max_depth', 6),
            learning_rate = xgb_cfg.get('learning_rate', 0.1),
            n_estimators  = xgb_cfg.get('n_estimators', 100),
            subsample     = xgb_cfg.get('subsample', 0.8),
            tree_method   = 'hist',
            gpu_id        = xgb_cfg.get('gpu_id', 0),
        )

        splits           = temporal_split_data(X, y, timestamps)
        train_X, train_y = splits['train']
        test_X,  test_y  = splits['test']

        with tqdm(
            total=y.shape[1],
            desc=f"  [{self.TASK_NAME}] XGBoost",
            unit="task", file=sys.stderr, dynamic_ncols=True, leave=False,
        ) as bar:
            predictor.fit(train_X, train_y)
            bar.update(y.shape[1])

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

        valid = [v for v in metrics.values() if not np.isnan(v)]
        metrics['mean_auroc'] = float(np.mean(valid)) if valid else 0.0

        model_path = os.path.join('models', f'{self.TASK_NAME}_xgboost.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(predictor, f)

        return {
            'mean_test_auroc':  metrics['mean_auroc'],
            'per_task_metrics': metrics,
            'model_path':       model_path,
        }

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} task='{self.TASK_NAME}' "
            f"windows={self.WINDOWS} best={self.best_model_name}>"
        )