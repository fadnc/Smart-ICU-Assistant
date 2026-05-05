"""
AKI Predictor — Tasks 7-12
Predicts Acute Kidney Injury at KDIGO stages 1-3 for 24h and 48h horizons.
"""

import pandas as pd
import numpy as np
from typing import Dict
from datetime import timedelta
from predictors.base_predictor import BasePredictor

import logging
logger = logging.getLogger(__name__)


class AKIPredictor(BasePredictor):
    """
    AKI (Acute Kidney Injury) Prediction & Staging

    Clinical definition (KDIGO criteria):
        Baseline = lowest creatinine in past 48 hours
        Future   = max creatinine in next 24 or 48 hours

        Stage 1: cr_increase >= 0.3 mg/dL  OR  future_cr >= 1.5× baseline
        Stage 2: future_cr >= 2.0× baseline
        Stage 3: future_cr >= 3.0× baseline  OR  future_cr > 4.0 mg/dL

    Stages are cumulative: Stage 3 → Stage 2 = 1, Stage 1 = 1
    """

    TASK_NAME = "aki"
    TASK_DESCRIPTION = "AKI prediction & KDIGO staging (stages 1-3 at 24/48h)"
    WINDOWS = [24, 48]
    LABEL_PREFIX = "aki"
    STAGES = [1, 2, 3]

    # KDIGO thresholds (from config or defaults)
    STAGE1_CR_INCREASE = 0.3    # mg/dL absolute increase
    STAGE1_CR_RATIO = 1.5       # × baseline
    STAGE2_CR_RATIO = 2.0
    STAGE3_CR_RATIO = 3.0
    STAGE3_CR_ABSOLUTE = 4.0    # mg/dL

    def __init__(self, config_path: str = 'config.yaml'):
        super().__init__(config_path)
        aki_cfg = self.config.get('AKI_KDIGO_STAGES', {})
        self.STAGE1_CR_INCREASE = aki_cfg.get('stage1_cr_increase', 0.3)
        self.STAGE1_CR_RATIO = aki_cfg.get('stage1_cr_ratio', 1.5)
        self.STAGE2_CR_RATIO = aki_cfg.get('stage2_cr_ratio', 2.0)
        self.STAGE3_CR_RATIO = aki_cfg.get('stage3_cr_ratio', 3.0)

    def get_label_names(self):
        """AKI has compound labels: aki_stage1_24h, aki_stage2_24h, ..."""
        names = []
        for window in self.WINDOWS:
            for stage in self.STAGES:
                names.append(f'aki_stage{stage}_{window}h')
        return names

    def generate_labels(self,
                        stay: pd.Series,
                        vitals: pd.DataFrame,
                        labs: pd.DataFrame,
                        current_time: pd.Timestamp,
                        **extra_data) -> Dict[str, int]:
        labels = {}
        for window in self.WINDOWS:
            stage_labels = self._check_aki(labs, current_time, window)
            for stage in self.STAGES:
                labels[f'aki_stage{stage}_{window}h'] = stage_labels.get(f'aki_stage{stage}', 0)
        return labels

    def _check_aki(self, labs, current_time, window_hours) -> Dict[str, int]:
        result = {'aki_stage1': 0, 'aki_stage2': 0, 'aki_stage3': 0}

        if len(labs) == 0 or 'creatinine' not in labs.columns:
            return result

        # Baseline: lowest creatinine in past 48h
        lookback = current_time - timedelta(hours=48)
        past_cr = labs[
            (labs.index >= lookback) & (labs.index <= current_time)
        ]['creatinine'].dropna()

        if len(past_cr) == 0:
            return result

        baseline = past_cr.min()

        # Future: max creatinine in the window
        window_end = current_time + timedelta(hours=window_hours)
        future_cr = labs[
            (labs.index > current_time) & (labs.index <= window_end)
        ]['creatinine'].dropna()

        if len(future_cr) == 0:
            return result

        max_future = future_cr.max()
        increase = max_future - baseline

        # KDIGO staging
        if increase >= self.STAGE1_CR_INCREASE or max_future >= self.STAGE1_CR_RATIO * baseline:
            result['aki_stage1'] = 1
        if max_future >= self.STAGE2_CR_RATIO * baseline:
            result['aki_stage2'] = 1
        if max_future >= self.STAGE3_CR_RATIO * baseline or max_future > self.STAGE3_CR_ABSOLUTE:
            result['aki_stage3'] = 1

        return result


if __name__ == "__main__":
    p = AKIPredictor()
    print(p)
    print(f"Labels: {p.get_label_names()}")
