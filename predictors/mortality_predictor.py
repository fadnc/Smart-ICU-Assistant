"""
Mortality Predictor — Tasks 1-3
Predicts ICU mortality at 6, 12, and 24 hour horizons.
"""

import pandas as pd
from typing import Dict
from predictors.base_predictor import BasePredictor

import logging
logger = logging.getLogger(__name__)


class MortalityPredictor(BasePredictor):
    """
    Early ICU Mortality Prediction

    Logic:
        IF patient has a recorded date of death (dod):
            time_to_death = (dod - current_time) in hours
            IF 0 <= time_to_death <= window → label = 1
        ELSE → label = 0
        
    """

    TASK_NAME = "mortality"
    TASK_DESCRIPTION = "Early ICU mortality prediction at 6/12/24h horizons"
    WINDOWS = [6, 12, 24]
    LABEL_PREFIX = "mortality"

    def generate_labels(self,
                        stay: pd.Series,
                        vitals: pd.DataFrame,
                        labs: pd.DataFrame,
                        current_time: pd.Timestamp,
                        **extra_data) -> Dict[str, int]:
        labels = {}
        for window in self.WINDOWS:
            label_name = f'{self.LABEL_PREFIX}_{window}h'

            if pd.isna(stay.get('dod')):
                labels[label_name] = 0
                continue

            time_to_death = (stay['dod'] - current_time).total_seconds() / 3600
            labels[label_name] = 1 if 0 <= time_to_death <= window else 0

        return labels


if __name__ == "__main__":
    p = MortalityPredictor()
    print(p)
    print(f"Labels: {p.get_label_names()}")
