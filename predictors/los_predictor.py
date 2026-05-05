"""
Length of Stay Predictor — Regression Task
Predicts remaining ICU length of stay in hours.
"""

import numpy as np
import pandas as pd
from typing import Dict
from predictors.base_predictor import BasePredictor

import logging
logger = logging.getLogger(__name__)


class LOSPredictor(BasePredictor):
    """
    ICU Length of Stay Prediction (Regression)

    Unlike other predictors (binary classification), this predicts a
    continuous value: remaining hours in ICU from current_time.

    Logic: (-- to be implemented...)
        remaining_los = (outtime - current_time) in hours
        Bucketed into categories for classification:
            - Short stay:  < 24h → label = 0
            - Medium stay: 24-72h → label = 1
            - Long stay:   > 72h → label = 2

    For binary classification compatibility, we convert to:
        los_short_24h:  1 if remaining < 24h
        los_long_72h:   1 if remaining > 72h

    """

    TASK_NAME = "los"
    TASK_DESCRIPTION = "ICU length of stay prediction (short <24h / long >72h)"
    WINDOWS = [24, 72]           # thresholds in hours
    LABEL_PREFIX = "los"

    def get_label_names(self):
        return ['los_short_24h', 'los_long_72h']

    def generate_labels(self,
                        stay: pd.Series,
                        vitals: pd.DataFrame,
                        labs: pd.DataFrame,
                        current_time: pd.Timestamp,
                        **extra_data) -> Dict[str, int]:
        labels = {}

        if pd.isna(stay.get('outtime')) or pd.isna(current_time):
            labels['los_short_24h'] = 0
            labels['los_long_72h'] = 0
            return labels

        remaining_hours = (stay['outtime'] - current_time).total_seconds() / 3600

        # Short stay: will be discharged within 24h
        labels['los_short_24h'] = 1 if 0 < remaining_hours <= 24 else 0

        # Long stay: still >72h remaining
        labels['los_long_72h'] = 1 if remaining_hours > 72 else 0

        return labels


if __name__ == "__main__":
    p = LOSPredictor()
    print(p)
    print(f"Labels: {p.get_label_names()}")
