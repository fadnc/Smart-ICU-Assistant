"""
Readmission Predictor — Task 21
Predicts ICU readmission within 30 days using discharge-level tabular features + XGBoost + SHAP.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)


class ReadmissionPredictor:
    """
    ICU Readmission Prediction (XGBoost + SHAP)

    Different from other predictors:
      - Uses one feature vector per ICU stay (not time-series)
      - Only uses XGBoost (tabular data → tree models best)
      - Generates SHAP interpretability plots

    Readmission = same patient has another ICU admission within 30 days of discharge.

    16 discharge features:
      Demographics: age, gender
      Stay: los_hours, los_days
      Complexity: n_diagnoses, n_prescriptions, n_unique_drugs
      Monitoring: n_chart_events, n_last_6h_events, n_lab_tests
      Labs: lab_mean, lab_std
      Output: total_urine_output, n_output_events
      Services: n_service_changes
    """

    TASK_NAME = "readmission"

    def __init__(self, config_path: str = 'config.yaml'):
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        xgb_cfg = self.config.get('XGBOOST_CONFIG', {})
        import torch as _torch
        use_gpu = _torch.cuda.is_available()
        device  = f'cuda:{xgb_cfg.get("gpu_id", 0)}' if use_gpu else 'cpu'

        self.model = xgb.XGBClassifier(
            max_depth=xgb_cfg.get('max_depth', 6),
            learning_rate=xgb_cfg.get('learning_rate', 0.1),
            n_estimators=xgb_cfg.get('n_estimators', 100),
            subsample=xgb_cfg.get('subsample', 0.8),
            eval_metric='logloss',
            tree_method='hist',
            device=device,
            random_state=42,
        )
        self.feature_names: List[str] = []
        self.results: Dict = {}

    def extract_readmission_labels(self, icu_stays: pd.DataFrame,
                                    readmission_window_days: int = 30) -> pd.DataFrame:
        """Detect readmissions: same patient, another ICU stay within N days."""
        stays = icu_stays.sort_values(['subject_id', 'intime']).copy()
        stays['readmission'] = 0

        for subject_id in stays['subject_id'].unique():
            patient = stays[stays['subject_id'] == subject_id].sort_values('intime')
            if len(patient) <= 1:
                continue
            idxs = patient.index.tolist()
            for i in range(len(idxs) - 1):
                outtime = stays.loc[idxs[i], 'outtime']
                next_in = stays.loc[idxs[i + 1], 'intime']
                if pd.notna(outtime) and pd.notna(next_in):
                    gap_days = (next_in - outtime).total_seconds() / 86400
                    if 0 < gap_days <= readmission_window_days:
                        stays.loc[idxs[i], 'readmission'] = 1

        rate = stays['readmission'].mean() * 100
        logger.info(f"Readmission: {stays['readmission'].sum()}/{len(stays)} ({rate:.1f}%)")
        return stays[['icustay_id', 'readmission']]

    def extract_discharge_features(self, merged_data, chartevents, labevents,
                                    diagnoses, prescriptions, services=None,
                                    outputevents=None) -> pd.DataFrame:
        """Extract 16 tabular features per ICU stay at discharge."""
        features = []
        for _, stay in merged_data.iterrows():
            icustay_id = stay['icustay_id']
            hadm_id = stay['hadm_id']
            feat = {'icustay_id': icustay_id}

            # Demographics
            feat['age'] = stay.get('age', 0)
            feat['gender_M'] = 1 if stay.get('gender', '') == 'M' else 0

            # LOS
            if pd.notna(stay.get('intime')) and pd.notna(stay.get('outtime')):
                los_h = (stay['outtime'] - stay['intime']).total_seconds() / 3600
                feat['los_hours'] = los_h
                feat['los_days'] = los_h / 24
            else:
                feat['los_hours'] = feat['los_days'] = 0

            # Diagnoses
            feat['n_diagnoses'] = len(diagnoses[diagnoses['hadm_id'] == hadm_id]) \
                if len(diagnoses) > 0 and 'hadm_id' in diagnoses.columns else 0

            # Prescriptions
            if len(prescriptions) > 0:
                key = 'hadm_id' if 'hadm_id' in prescriptions.columns else 'subject_id'
                val = hadm_id if key == 'hadm_id' else stay.get('subject_id')
                rx = prescriptions[prescriptions[key] == val]
                feat['n_prescriptions'] = len(rx)
                feat['n_unique_drugs'] = rx['drug'].nunique() if 'drug' in rx.columns else 0
            else:
                feat['n_prescriptions'] = feat['n_unique_drugs'] = 0

            # Chart events
            if len(chartevents) > 0 and 'icustay_id' in chartevents.columns:
                sc = chartevents[chartevents['icustay_id'] == icustay_id]
                feat['n_chart_events'] = len(sc)
                if pd.notna(stay.get('outtime')):
                    last6 = sc[sc['charttime'] >= (stay['outtime'] - pd.Timedelta(hours=6))]
                    feat['n_last_6h_events'] = len(last6)
                else:
                    feat['n_last_6h_events'] = 0
            else:
                feat['n_chart_events'] = feat['n_last_6h_events'] = 0

            # Lab stats
            if len(labevents) > 0 and 'icustay_id' in labevents.columns:
                sl = labevents[labevents['icustay_id'] == icustay_id]
                feat['n_lab_tests'] = len(sl)
                if 'valuenum' in sl.columns and len(sl) > 0:
                    feat['lab_mean'] = sl['valuenum'].mean()
                    feat['lab_std'] = sl['valuenum'].std()
                else:
                    feat['lab_mean'] = feat['lab_std'] = 0
            else:
                feat['n_lab_tests'] = 0
                feat['lab_mean'] = feat['lab_std'] = 0

            # Urine output
            if outputevents is not None and len(outputevents) > 0 and 'icustay_id' in outputevents.columns:
                so = outputevents[outputevents['icustay_id'] == icustay_id]
                feat['total_urine_output'] = so['value'].sum() if len(so) > 0 else 0
                feat['n_output_events'] = len(so)
            else:
                feat['total_urine_output'] = feat['n_output_events'] = 0

            # Services
            if services is not None and len(services) > 0 and 'hadm_id' in services.columns:
                feat['n_service_changes'] = max(0, len(services[services['hadm_id'] == hadm_id]) - 1)
            else:
                feat['n_service_changes'] = 0

            features.append(feat)

        df = pd.DataFrame(features)
        self.feature_names = [c for c in df.columns if c != 'icustay_id']
        logger.info(f"Extracted {len(self.feature_names)} discharge features for {len(df)} stays")
        return df

    def train(self, features_df, labels_df, output_dir='output'):
        """Train XGBoost + SHAP on readmission data."""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs('models', exist_ok=True)

        merged = features_df.merge(labels_df, on='icustay_id')
        X = merged[self.feature_names].fillna(0).values
        y_label = merged['readmission'].values

        n_pos = y_label.sum()
        logger.info(f"Training readmission: {len(y_label)} samples, {n_pos} positive ({n_pos/len(y_label)*100:.1f}%)")

        if n_pos < 2:
            logger.warning("Not enough positive readmission labels to train")
            self.results = {'auroc': float('nan'), 'auprc': float('nan')}
            return self.results

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_label, test_size=0.2, random_state=42, stratify=y_label
        )

        self.model.fit(X_train, y_train)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Comprehensive metrics (matching all other predictors)
        from sklearn.metrics import f1_score, recall_score, brier_score_loss
        auroc = roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else float('nan')
        auprc = average_precision_score(y_test, y_prob) if len(set(y_test)) > 1 else float('nan')
        f1    = f1_score(y_test, y_pred, zero_division=0)
        sens  = recall_score(y_test, y_pred, zero_division=0)
        tn = ((y_pred == 0) & (y_test == 0)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        spec  = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        brier = brier_score_loss(y_test, y_prob)

        # SHAP
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_test)
            mean_abs = np.abs(shap_values).mean(axis=0)
            top_idx = np.argsort(mean_abs)[::-1][:5]
            logger.info("Top features by SHAP:")
            for i in top_idx:
                logger.info(f"  {self.feature_names[i]}: {mean_abs[i]:.4f}")

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            shap.summary_plot(shap_values, X_test, feature_names=self.feature_names, show=False)
            plot_path = os.path.join(output_dir, 'shap_readmission.png')
            plt.savefig(plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            logger.info(f"SHAP plot saved to {plot_path}")
        except ImportError:
            logger.info("SHAP not available — using built-in feature importance")

        logger.info(
            f"Readmission | AUROC={auroc:.4f} AUPRC={auprc:.4f} "
            f"F1={f1:.4f} Sens={sens:.4f} Spec={spec:.4f} Brier={brier:.4f}"
        )

        # Save model
        model_path = os.path.join('models', 'readmission_xgboost.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        self.results = {
            'task': 'readmission',
            'best_model': 'xgboost',
            'auroc': auroc,
            'auprc': auprc,
            'f1': f1,
            'sensitivity': sens,
            'specificity': spec,
            'brier': brier,
            'readmission_rate': float(n_pos / len(y_label)),
            'n_samples': len(y_label),
            'model_path': model_path,
        }

        report_path = os.path.join(output_dir, 'readmission_report.json')
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        return self.results
