"""
Base Predictor — Shared logic for all ICU prediction tasks.

CHANGES:
  - MODELS_TO_TRY: removed 'tcn'. Now ['lstm', 'transformer', 'xgboost'].
    TCN removed because BatchNorm1d collapses under FP16 AMP with rare ICU labels,
    producing NaN loss from epoch 2-7 on every task and wasting ~3h per run.
  - task_name + model_name forwarded to ModelTrainer.train() for labeled epoch bars.
  - Overall model-comparison bar per task.
  - XGBoost training wrapped in a tqdm spinner.
  - clear_gpu_memory() called before AND after each model.
  - _extract_task_labels logs clearly when label indices are missing.
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
from training import ModelTrainer, clear_gpu_memory, temporal_split_data

logger = logging.getLogger(__name__)


class BasePredictor(ABC):
    """Abstract base class for all ICU prediction tasks."""

    TASK_NAME:        str       = ""
    TASK_DESCRIPTION: str       = ""
    WINDOWS:          List[int] = []
    LABEL_PREFIX:     str       = ""

    # TCN removed — BatchNorm1d → NaN loss under FP16 AMP with imbalanced labels
    MODELS_TO_TRY: List[str] = ['lstm', 'transformer', 'xgboost']

    def __init__(self, config_path: str = 'config.yaml'):
        self.config      = self._load_config(config_path)
        self.config_path = config_path
        self.time_windows = self.config.get('TIME_WINDOWS', {})
        self.results:         Dict             = {}
        self.best_model_name: Optional[str]   = None
        self.best_auroc:      float            = 0.0
        self._label_indices:  Optional[np.ndarray] = None

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    # ── Label names ───────────────────────────────────────────────────────────

    def get_label_names(self) -> List[str]:
        return [f'{self.LABEL_PREFIX}_{w}h' for w in self.WINDOWS]

    def get_num_tasks(self) -> int:
        return len(self.get_label_names())

    # ── Abstract interface ────────────────────────────────────────────────────

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
        """Train every model type and pick the best by test AUROC."""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs('models', exist_ok=True)

        task_labels = self._extract_task_labels(y)
        if task_labels is None:
            logger.warning(f"[{self.TASK_NAME}] Could not extract labels — skipping")
            return {}

        num_tasks  = task_labels.shape[1]
        input_size = X.shape[2]

        logger.info(
            f"[{self.TASK_NAME}] {len(self.MODELS_TO_TRY)} models × "
            f"{num_tasks} labels | input={input_size}"
        )

        comparison = {}

        model_bar = tqdm(
            self.MODELS_TO_TRY,
            desc=f"  [{self.TASK_NAME}]",
            unit="model",
            file=sys.stderr,
            dynamic_ncols=True,
            leave=True,
            bar_format=(
                "{l_bar}{bar:20}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}] {postfix}"
            ),
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

                model_bar.set_postfix_str(
                    f"{model_name.upper()} done | AUROC={mean_auroc:.4f}"
                )
                logger.info(
                    f"[{self.TASK_NAME}] {model_name.upper()} → AUROC={mean_auroc:.4f}"
                )

                if mean_auroc > self.best_auroc:
                    self.best_auroc      = mean_auroc
                    self.best_model_name = model_name

                clear_gpu_memory()

            except Exception as e:
                logger.error(
                    f"[{self.TASK_NAME}] {model_name} failed: {e}", exc_info=True
                )
                comparison[model_name] = {'error': str(e)}
                model_bar.set_postfix_str(f"{model_name.upper()} FAILED")
                clear_gpu_memory()

        model_bar.close()

        # ── Ensemble predictions ──────────────────────────────────────────
        ensemble_cfg = self.config.get('ENSEMBLE', {})
        if ensemble_cfg.get('enabled', False):
            # Collect predictions from successful models that stored them
            pred_list = [
                mmetrics['_test_predictions']
                for mmetrics in comparison.values()
                if isinstance(mmetrics, dict) and '_test_predictions' in mmetrics
            ]
            if len(pred_list) >= 2:
                # Average predictions from all successful models
                avg_preds = np.mean(pred_list, axis=0)
                test_y_ens = comparison[next(
                    k for k, v in comparison.items()
                    if isinstance(v, dict) and '_test_targets' in v
                )]['_test_targets']

                from sklearn.metrics import (
                    roc_auc_score, average_precision_score, f1_score,
                    recall_score, brier_score_loss,
                )
                ens_metrics = {}
                all_aurocs = []
                for t in range(test_y_ens.shape[1]):
                    y_true = test_y_ens[:, t]
                    y_prob = avg_preds[:, t]
                    y_pred = (y_prob >= 0.5).astype(int)
                    try:
                        if len(np.unique(y_true)) < 2:
                            continue
                        auroc = roc_auc_score(y_true, y_prob)
                        auprc = average_precision_score(y_true, y_prob)
                        f1    = f1_score(y_true, y_pred, zero_division=0)
                        sens  = recall_score(y_true, y_pred, zero_division=0)
                        tn = ((y_pred == 0) & (y_true == 0)).sum()
                        fp = ((y_pred == 1) & (y_true == 0)).sum()
                        spec  = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
                        brier = brier_score_loss(y_true, y_prob)
                        for metric, val in [('auroc', auroc), ('auprc', auprc), ('f1', f1),
                                           ('sensitivity', sens), ('specificity', spec), ('brier', brier)]:
                            ens_metrics[f'task_{t}_{metric}'] = float(val)
                        all_aurocs.append(auroc)
                    except Exception:
                        pass

                mean_ens_auroc = float(np.mean(all_aurocs)) if all_aurocs else 0.0
                ens_metrics['macro_auroc'] = mean_ens_auroc
                comparison['ensemble'] = {
                    'mean_test_auroc':  mean_ens_auroc,
                    'per_task_metrics': ens_metrics,
                    'model_path':       'ensemble (no single checkpoint)',
                    'models_used':      [k for k, v in comparison.items()
                                         if isinstance(v, dict) and '_test_predictions' in v],
                }
                logger.info(
                    f"[{self.TASK_NAME}] ENSEMBLE ({len(pred_list)} models) → "
                    f"AUROC={mean_ens_auroc:.4f}"
                )
                if mean_ens_auroc > self.best_auroc:
                    self.best_auroc      = mean_ens_auroc
                    self.best_model_name = 'ensemble'

        # Clean up stored predictions from comparison dict (large arrays)
        for mname in list(comparison.keys()):
            if isinstance(comparison[mname], dict):
                comparison[mname].pop('_test_predictions', None)
                comparison[mname].pop('_test_targets', None)

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
            logger.debug(
                f"[{self.TASK_NAME}] No indices set — using all {num_labels} cols (exact match)"
            )
            return y_full

        logger.error(
            f"[{self.TASK_NAME}] LABEL INDEX ERROR: expected {num_labels} cols, "
            f"got {y_full.shape[1]}. Call set_label_indices() before training."
        )
        return None

    def set_label_indices(self, all_label_names: List[str]):
        my_labels = self.get_label_names()
        indices   = []
        for name in my_labels:
            if name in all_label_names:
                indices.append(all_label_names.index(name))
            else:
                logger.warning(f"[{self.TASK_NAME}] Label '{name}' not in full label list")

        if indices:
            self._label_indices = np.array(indices)
            logger.debug(f"[{self.TASK_NAME}] Indices: {self._label_indices} → {my_labels}")
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
        config               = self.config.copy()
        config['input_size'] = input_size
        config['num_tasks']  = num_tasks

        if model_name == 'xgboost':
            return self._train_xgboost(X, y, timestamps, config)
        return self._train_dl_model(model_name, X, y, timestamps, config)

    def _train_dl_model(self, model_name: str, X, y, timestamps, config) -> Dict:
        """Train LSTM / Transformer with epoch-level progress bar."""
        model   = create_model(model_name, config)
        trainer = ModelTrainer(model, config, model_type=model_name)

        splits           = trainer.temporal_split(X, y, timestamps)
        train_X, train_y = splits['train']
        val_X,   val_y   = splits['val']
        test_X,  test_y  = splits['test']

        trainer.train(
            train_X, train_y, val_X, val_y,
            task_name=self.TASK_NAME,
            model_name=model_name,
            verbose=False,
        )

        predictions = trainer.predict(test_X)
        metrics     = trainer.compute_metrics(predictions, test_y)

        aurocs = [
            v for k, v in metrics.items()
            if k.startswith('task_') and k.endswith('_auroc') and not np.isnan(v)
        ]
        mean_auroc = float(np.mean(aurocs)) if aurocs else 0.0

        model_path = os.path.join('models', f'{self.TASK_NAME}_{model_name}.pth')
        trainer.save_checkpoint(model_path)

        return {
            'mean_test_auroc':    mean_auroc,
            'per_task_metrics':   metrics,
            'model_path':         model_path,
            '_test_predictions':  predictions,   # stored for ensemble, cleaned up later
            '_test_targets':      test_y,
        }

    def _train_xgboost(self, X, y, timestamps, config) -> Dict:
        """Train XGBoost with a tqdm spinner."""
        xgb_config = config.get('XGBOOST_CONFIG', {})

        predictor = XGBoostPredictor(
            num_tasks     = y.shape[1],
            max_depth     = xgb_config.get('max_depth', 6),
            learning_rate = xgb_config.get('learning_rate', 0.1),
            n_estimators  = xgb_config.get('n_estimators', 100),
            subsample     = xgb_config.get('subsample', 0.8),
            device        = xgb_config.get('device', 'cpu'),
        )

        splits           = temporal_split_data(X, y, timestamps)
        train_X, train_y = splits['train']
        val_X,   val_y   = splits['val']
        test_X,  test_y  = splits['test']

        with tqdm(
            total=y.shape[1],
            desc=f"  [{self.TASK_NAME}] XGBoost",
            unit="task",
            file=sys.stderr,
            dynamic_ncols=True,
            leave=False,
        ) as xgb_bar:
            predictor.fit(train_X, train_y, verbose=False)
            xgb_bar.update(y.shape[1])

        predictions = predictor.predict_proba(test_X)

        from sklearn.metrics import (
            roc_auc_score, average_precision_score, f1_score,
            recall_score, brier_score_loss,
        )
        metrics = {}
        all_aurocs, all_auprcs, all_f1s = [], [], []
        all_sens, all_specs, all_briers = [], [], []

        for t in range(test_y.shape[1]):
            y_true = test_y[:, t]
            y_prob = predictions[:, t]
            y_pred = (y_prob >= 0.5).astype(int)

            try:
                if len(set(y_true)) < 2:
                    for m in ('auroc', 'auprc', 'f1', 'sensitivity', 'specificity', 'brier'):
                        metrics[f'task_{t}_{m}'] = float('nan')
                    continue

                auroc = roc_auc_score(y_true, y_prob)
                auprc = average_precision_score(y_true, y_prob)
                f1    = f1_score(y_true, y_pred, zero_division=0)
                sens  = recall_score(y_true, y_pred, zero_division=0)
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

                all_aurocs.append(auroc); all_auprcs.append(auprc)
                all_f1s.append(f1); all_sens.append(sens)
                all_specs.append(spec); all_briers.append(brier)

            except Exception:
                for m in ('auroc', 'auprc', 'f1', 'sensitivity', 'specificity', 'brier'):
                    metrics[f'task_{t}_{m}'] = float('nan')

        metrics['macro_auroc']       = float(np.mean(all_aurocs)) if all_aurocs else 0.0
        metrics['macro_auprc']       = float(np.mean(all_auprcs)) if all_auprcs else 0.0
        metrics['macro_f1']          = float(np.mean(all_f1s))    if all_f1s    else 0.0
        metrics['macro_sensitivity'] = float(np.mean(all_sens))   if all_sens   else 0.0
        metrics['macro_specificity'] = float(np.mean(all_specs))  if all_specs  else 0.0
        metrics['macro_brier']       = float(np.mean(all_briers)) if all_briers else 0.0

        aurocs     = [v for k, v in metrics.items() if k.endswith('_auroc') and not k.startswith('macro') and not np.isnan(v)]
        mean_auroc = float(np.mean(aurocs)) if aurocs else 0.0

        model_path = os.path.join('models', f'{self.TASK_NAME}_xgboost.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(predictor, f)

        return {
            'mean_test_auroc':    mean_auroc,
            'per_task_metrics':   metrics,
            'model_path':         model_path,
            '_test_predictions':  predictions,   # stored for ensemble, cleaned up later
            '_test_targets':      test_y,
        }

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} task='{self.TASK_NAME}' "
            f"windows={self.WINDOWS} best={self.best_model_name}>"
        )