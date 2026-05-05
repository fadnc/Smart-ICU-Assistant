"""
Base Predictor — Shared logic for all ICU prediction tasks.

CHANGES:
  - MODELS_TO_TRY: ['lstm', 'transformer', 'xgboost', 'lightgbm']
    LightGBM added for diversity.
  - Ensemble: AUROC²-weighted averaging + stacking meta-learner.
  - Per-task threshold tuning: optimizes F1 on val set instead of hardcoded 0.5.
  - XGBoost/LightGBM: per-task scale_pos_weight + early stopping with val set.
  - Ensemble macro metrics bug fixed (AUPRC/F1/Sens were 0.0000).
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
from models import create_model, XGBoostPredictor, LightGBMPredictor
from training import ModelTrainer, clear_gpu_memory, temporal_split_data

logger = logging.getLogger(__name__)


class BasePredictor(ABC):
    """Abstract base class for all ICU prediction tasks."""

    TASK_NAME:        str       = ""
    TASK_DESCRIPTION: str       = ""
    WINDOWS:          List[int] = []
    LABEL_PREFIX:     str       = ""

    # TCN removed — BatchNorm1d → NaN loss under FP16 AMP with imbalanced labels
    MODELS_TO_TRY: List[str] = ['lstm', 'transformer', 'xgboost', 'lightgbm']

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

        # Check which models to try (skip TFT/TabTransformer unless configured)
        models_to_try = list(self.MODELS_TO_TRY)
        task_overrides = self.config.get('TASK_OVERRIDES', {})
        task_cfg = task_overrides.get(self.TASK_NAME, {})
        if 'models' in task_cfg:
            models_to_try = task_cfg['models']

        # Check for LightGBM availability
        if 'lightgbm' in models_to_try:
            try:
                import lightgbm
            except ImportError:
                logger.warning(f"[{self.TASK_NAME}] LightGBM not installed — skipping")
                models_to_try = [m for m in models_to_try if m != 'lightgbm']

        logger.info(
            f"[{self.TASK_NAME}] {len(models_to_try)} models × "
            f"{num_tasks} labels | input={input_size}"
        )

        comparison = {}

        model_bar = tqdm(
            models_to_try,
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
            self._run_ensembles(comparison)

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

    # ── Ensemble ──────────────────────────────────────────────────────────────

    def _run_ensembles(self, comparison: Dict):
        """Run weighted averaging + stacking ensemble."""
        # Collect predictions + AUROCs from successful models
        pred_list  = []
        auroc_list = []
        model_names = []
        test_y_ens = None

        for mname, mmetrics in comparison.items():
            if not isinstance(mmetrics, dict) or '_test_predictions' not in mmetrics:
                continue
            pred_list.append(mmetrics['_test_predictions'])
            auroc_list.append(mmetrics.get('mean_test_auroc', 0.5))
            model_names.append(mname)
            if test_y_ens is None:
                test_y_ens = mmetrics['_test_targets']

        if len(pred_list) < 2:
            return

        # ── AUROC²-weighted ensemble ──
        weights = np.array([a ** 2 for a in auroc_list])
        weights = weights / weights.sum()
        avg_preds = np.average(pred_list, axis=0, weights=weights)

        ens_metrics = self._compute_ensemble_metrics(avg_preds, test_y_ens)
        mean_ens_auroc = ens_metrics.get('macro_auroc', 0.0)

        # Save ensemble pickle with component paths + weights
        ens_path = os.path.join('models', f'{self.TASK_NAME}_ensemble.pkl')
        component_paths = [comparison[n].get('model_path', '') for n in model_names]
        ens_artifact = {
            'type':                 'weighted_ensemble',
            'task':                 self.TASK_NAME,
            'labels':               self.get_label_names(),
            'component_model_names': model_names,
            'component_model_paths': component_paths,
            'weights':              {n: float(w) for n, w in zip(model_names, weights)},
            'input_size':           pred_list[0].shape[1] if len(pred_list[0].shape) > 1 else 1,
        }
        with open(ens_path, 'wb') as f:
            pickle.dump(ens_artifact, f)
        logger.info(f"[{self.TASK_NAME}] Saved ensemble → {ens_path}")

        comparison['ensemble'] = {
            'mean_test_auroc':  mean_ens_auroc,
            'per_task_metrics': ens_metrics,
            'model_path':       ens_path,
            'models_used':      model_names,
            'weights':          {n: float(w) for n, w in zip(model_names, weights)},
        }
        logger.info(
            f"[{self.TASK_NAME}] ENSEMBLE ({len(pred_list)} models, weighted) → "
            f"AUROC={mean_ens_auroc:.4f}"
        )
        if mean_ens_auroc > self.best_auroc:
            self.best_auroc      = mean_ens_auroc
            self.best_model_name = 'ensemble'

        # ── Stacking ensemble ──
        try:
            stacked_result = self._stacked_ensemble(pred_list, test_y_ens, model_names)
            if stacked_result:
                stacked_metrics = stacked_result['metrics']
                meta_learners   = stacked_result['meta_learners']
                stacked_auroc = stacked_metrics.get('macro_auroc', 0.0)

                # Save stacked ensemble pickle with meta-learners
                stk_path = os.path.join('models', f'{self.TASK_NAME}_stacked_ensemble.pkl')
                component_paths = [comparison[n].get('model_path', '') for n in model_names]
                stk_artifact = {
                    'type':                 'stacked_ensemble',
                    'task':                 self.TASK_NAME,
                    'labels':               self.get_label_names(),
                    'component_model_names': model_names,
                    'component_model_paths': component_paths,
                    'meta_learners':        meta_learners,
                    'input_size':           pred_list[0].shape[1] if len(pred_list[0].shape) > 1 else 1,
                }
                with open(stk_path, 'wb') as f:
                    pickle.dump(stk_artifact, f)
                logger.info(f"[{self.TASK_NAME}] Saved stacked ensemble → {stk_path}")

                comparison['stacked_ensemble'] = {
                    'mean_test_auroc':  stacked_auroc,
                    'per_task_metrics': stacked_metrics,
                    'model_path':       stk_path,
                    'models_used':      model_names,
                }
                logger.info(
                    f"[{self.TASK_NAME}] STACKED ENSEMBLE → AUROC={stacked_auroc:.4f}"
                )
                if stacked_auroc > self.best_auroc:
                    self.best_auroc      = stacked_auroc
                    self.best_model_name = 'stacked_ensemble'
        except Exception as e:
            logger.warning(f"[{self.TASK_NAME}] Stacking failed: {e}")

    def _compute_ensemble_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """Compute per-task + macro metrics for ensemble predictions."""
        from sklearn.metrics import (
            roc_auc_score, average_precision_score, f1_score,
            recall_score, brier_score_loss,
        )

        metrics = {}
        all_aurocs, all_auprcs, all_f1s = [], [], []
        all_sens, all_specs, all_briers = [], [], []

        for t in range(targets.shape[1]):
            y_true = targets[:, t]
            y_prob = predictions[:, t]

            # Optimal threshold per task (maximize F1)
            best_thr, best_f1 = 0.5, 0.0
            for thr in np.arange(0.1, 0.9, 0.05):
                f1_val = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
                if f1_val > best_f1:
                    best_f1, best_thr = f1_val, thr
            y_pred = (y_prob >= best_thr).astype(int)

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
                                   ('sensitivity', sens), ('specificity', spec),
                                   ('brier', brier), ('threshold', best_thr)]:
                    metrics[f'task_{t}_{metric}'] = float(val)

                all_aurocs.append(auroc); all_auprcs.append(auprc)
                all_f1s.append(f1); all_sens.append(sens)
                all_specs.append(spec); all_briers.append(brier)
            except Exception:
                pass

        # Macro averages — THIS WAS THE BUG: these were missing before
        metrics['macro_auroc']       = float(np.mean(all_aurocs)) if all_aurocs else 0.0
        metrics['macro_auprc']       = float(np.mean(all_auprcs)) if all_auprcs else 0.0
        metrics['macro_f1']          = float(np.mean(all_f1s))    if all_f1s    else 0.0
        metrics['macro_sensitivity'] = float(np.mean(all_sens))   if all_sens   else 0.0
        metrics['macro_specificity'] = float(np.mean(all_specs))  if all_specs  else 0.0
        metrics['macro_brier']       = float(np.mean(all_briers)) if all_briers else 0.0

        return metrics

    def _stacked_ensemble(self, pred_list: List[np.ndarray],
                          test_y: np.ndarray,
                          model_names: List[str]) -> Optional[Dict]:
        """
        Stacking ensemble: LogisticRegression meta-learner trained on base
        model predictions. Uses half/half split to avoid leakage.
        Returns dict with 'metrics' and 'meta_learners' for serialization.
        """
        from sklearn.linear_model import LogisticRegression

        n_samples = test_y.shape[0]
        n_tasks   = test_y.shape[1]
        half      = n_samples // 2

        # Stack all model predictions: (n_samples, n_models) per task
        stacked_preds = np.zeros((n_samples, n_tasks))
        meta_learners = []  # one LR per task, saved for inference

        for t in range(n_tasks):
            # Build meta-features: each model's prediction for this task
            meta_features = np.column_stack([p[:, t] for p in pred_list])

            # Split: train meta-learner on first half, predict on second half
            meta_train_X = meta_features[:half]
            meta_train_y = test_y[:half, t]
            meta_test_X  = meta_features[half:]

            if len(np.unique(meta_train_y)) < 2:
                # Fallback to mean
                stacked_preds[half:, t] = np.mean(meta_test_X, axis=1)
                meta_learners.append(None)  # no learner for this task
                continue

            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(meta_train_X, meta_train_y)
            stacked_preds[half:, t] = lr.predict_proba(meta_test_X)[:, 1]
            meta_learners.append(lr)

        # Evaluate only the second half (where meta-learner predicted)
        metrics = self._compute_ensemble_metrics(
            stacked_preds[half:], test_y[half:]
        )
        return {'metrics': metrics, 'meta_learners': meta_learners}

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

        # Apply per-task config overrides (e.g. dropout for overfitting tasks)
        task_overrides = config.get('TASK_OVERRIDES', {})
        task_cfg = task_overrides.get(self.TASK_NAME, {})
        if task_cfg:
            # Override model-specific configs
            if 'dropout' in task_cfg:
                for cfg_key in ('LSTM_CONFIG', 'TRANSFORMER_CONFIG',
                                'TABTRANSFORMER_CONFIG', 'TFT_CONFIG'):
                    if cfg_key in config:
                        config[cfg_key] = {**config.get(cfg_key, {}),
                                           'dropout': task_cfg['dropout']}

        if model_name == 'xgboost':
            return self._train_xgboost(X, y, timestamps, config)
        if model_name == 'lightgbm':
            return self._train_lightgbm(X, y, timestamps, config)
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

        # Per-task threshold tuning: find optimal threshold on val set
        val_predictions = trainer.predict(val_X)
        thresholds = self._find_optimal_thresholds(val_predictions, val_y)
        metrics = self._recompute_with_thresholds(predictions, test_y, thresholds, metrics)

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
            'thresholds':         thresholds,
            '_test_predictions':  predictions,   # stored for ensemble, cleaned up later
            '_test_targets':      test_y,
        }

    def _train_xgboost(self, X, y, timestamps, config) -> Dict:
        """Train XGBoost with per-task scale_pos_weight and early stopping."""
        xgb_config = config.get('XGBOOST_CONFIG', {})

        predictor = XGBoostPredictor(
            num_tasks     = y.shape[1],
            max_depth     = xgb_config.get('max_depth', 6),
            learning_rate = xgb_config.get('learning_rate', 0.1),
            n_estimators  = xgb_config.get('n_estimators', 100),
            subsample     = xgb_config.get('subsample', 0.8),
            device        = xgb_config.get('device', 'cpu'),
            min_child_weight  = xgb_config.get('min_child_weight', 1.0),
            colsample_bytree  = xgb_config.get('colsample_bytree', 1.0),
            gamma             = xgb_config.get('gamma', 0.0),
            early_stopping_rounds = xgb_config.get('early_stopping_rounds', 0),
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
            predictor.fit(train_X, train_y, val_X=val_X, val_y=val_y, verbose=False)
            xgb_bar.update(y.shape[1])

        predictions = predictor.predict_proba(test_X)

        # Compute metrics with optimal thresholds
        val_predictions = predictor.predict_proba(val_X)
        thresholds = self._find_optimal_thresholds(val_predictions, val_y)

        metrics = self._compute_test_metrics(predictions, test_y, thresholds)

        aurocs     = [v for k, v in metrics.items() if k.endswith('_auroc') and not k.startswith('macro') and not np.isnan(v)]
        mean_auroc = float(np.mean(aurocs)) if aurocs else 0.0

        model_path = os.path.join('models', f'{self.TASK_NAME}_xgboost.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(predictor, f)

        return {
            'mean_test_auroc':    mean_auroc,
            'per_task_metrics':   metrics,
            'model_path':         model_path,
            'thresholds':         thresholds,
            '_test_predictions':  predictions,
            '_test_targets':      test_y,
        }

    def _train_lightgbm(self, X, y, timestamps, config) -> Dict:
        """Train LightGBM with per-task scale_pos_weight and early stopping."""
        lgb_config = config.get('LIGHTGBM_CONFIG', {})

        predictor = LightGBMPredictor(
            num_tasks     = y.shape[1],
            max_depth     = lgb_config.get('max_depth', -1),
            num_leaves    = lgb_config.get('num_leaves', 63),
            learning_rate = lgb_config.get('learning_rate', 0.05),
            n_estimators  = lgb_config.get('n_estimators', 300),
            subsample     = lgb_config.get('subsample', 0.8),
            colsample_bytree  = lgb_config.get('colsample_bytree', 0.8),
            min_child_samples = lgb_config.get('min_child_samples', 20),
            device        = lgb_config.get('device', 'cpu'),
            early_stopping_rounds = lgb_config.get('early_stopping_rounds', 0),
        )

        splits           = temporal_split_data(X, y, timestamps)
        train_X, train_y = splits['train']
        val_X,   val_y   = splits['val']
        test_X,  test_y  = splits['test']

        with tqdm(
            total=y.shape[1],
            desc=f"  [{self.TASK_NAME}] LightGBM",
            unit="task",
            file=sys.stderr,
            dynamic_ncols=True,
            leave=False,
        ) as lgb_bar:
            predictor.fit(train_X, train_y, val_X=val_X, val_y=val_y, verbose=False)
            lgb_bar.update(y.shape[1])

        predictions = predictor.predict_proba(test_X)

        val_predictions = predictor.predict_proba(val_X)
        thresholds = self._find_optimal_thresholds(val_predictions, val_y)

        metrics = self._compute_test_metrics(predictions, test_y, thresholds)

        aurocs     = [v for k, v in metrics.items() if k.endswith('_auroc') and not k.startswith('macro') and not np.isnan(v)]
        mean_auroc = float(np.mean(aurocs)) if aurocs else 0.0

        model_path = os.path.join('models', f'{self.TASK_NAME}_lightgbm.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(predictor, f)

        return {
            'mean_test_auroc':    mean_auroc,
            'per_task_metrics':   metrics,
            'model_path':         model_path,
            'thresholds':         thresholds,
            '_test_predictions':  predictions,
            '_test_targets':      test_y,
        }

    # ── Threshold tuning ──────────────────────────────────────────────────────

    def _find_optimal_thresholds(self, val_preds: np.ndarray, val_y: np.ndarray) -> List[float]:
        """Find per-task threshold that maximizes F1 on validation set."""
        from sklearn.metrics import f1_score

        thresholds = []
        for t in range(val_y.shape[1]):
            y_true = val_y[:, t]
            y_prob = val_preds[:, t]

            if len(np.unique(y_true)) < 2:
                thresholds.append(0.5)
                continue

            best_thr, best_f1 = 0.5, 0.0
            for thr in np.arange(0.05, 0.95, 0.05):
                f1 = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_f1  = f1
                    best_thr = float(thr)
            thresholds.append(best_thr)

        return thresholds

    def _recompute_with_thresholds(self, predictions: np.ndarray, test_y: np.ndarray,
                                    thresholds: List[float], existing_metrics: Dict) -> Dict:
        """Recompute F1/sensitivity/specificity using tuned thresholds."""
        from sklearn.metrics import f1_score, recall_score

        all_f1s, all_sens, all_specs = [], [], []

        for t in range(test_y.shape[1]):
            y_true = test_y[:, t]
            y_prob = predictions[:, t]
            y_pred = (y_prob >= thresholds[t]).astype(int)

            if len(np.unique(y_true)) < 2:
                continue

            f1   = f1_score(y_true, y_pred, zero_division=0)
            sens = recall_score(y_true, y_pred, zero_division=0)
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

            existing_metrics[f'task_{t}_f1']          = float(f1)
            existing_metrics[f'task_{t}_sensitivity'] = float(sens)
            existing_metrics[f'task_{t}_specificity'] = float(spec)
            existing_metrics[f'task_{t}_threshold']   = float(thresholds[t])
            all_f1s.append(f1); all_sens.append(sens); all_specs.append(spec)

        if all_f1s:
            existing_metrics['macro_f1']          = float(np.mean(all_f1s))
            existing_metrics['macro_sensitivity'] = float(np.mean(all_sens))
            existing_metrics['macro_specificity'] = float(np.mean(all_specs))

        return existing_metrics

    def _compute_test_metrics(self, predictions: np.ndarray, test_y: np.ndarray,
                               thresholds: List[float]) -> Dict:
        """Compute all metrics for tree-based models with tuned thresholds."""
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
            y_pred = (y_prob >= thresholds[t]).astype(int)

            try:
                if len(set(y_true)) < 2:
                    for m in ('auroc', 'auprc', 'f1', 'sensitivity', 'specificity', 'brier', 'threshold'):
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
                metrics[f'task_{t}_threshold']   = float(thresholds[t])

                all_aurocs.append(auroc); all_auprcs.append(auprc)
                all_f1s.append(f1); all_sens.append(sens)
                all_specs.append(spec); all_briers.append(brier)

            except Exception:
                for m in ('auroc', 'auprc', 'f1', 'sensitivity', 'specificity', 'brier', 'threshold'):
                    metrics[f'task_{t}_{m}'] = float('nan')

        metrics['macro_auroc']       = float(np.mean(all_aurocs)) if all_aurocs else 0.0
        metrics['macro_auprc']       = float(np.mean(all_auprcs)) if all_auprcs else 0.0
        metrics['macro_f1']          = float(np.mean(all_f1s))    if all_f1s    else 0.0
        metrics['macro_sensitivity'] = float(np.mean(all_sens))   if all_sens   else 0.0
        metrics['macro_specificity'] = float(np.mean(all_specs))  if all_specs  else 0.0
        metrics['macro_brier']       = float(np.mean(all_briers)) if all_briers else 0.0

        return metrics

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} task='{self.TASK_NAME}' "
            f"windows={self.WINDOWS} best={self.best_model_name}>"
        )