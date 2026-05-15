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
    TASK_DESCRIPTION = "In-ICU mortality prediction"

    def get_label_names(self) -> list:
        return ['mortality_in_icu']

    def generate_labels(self,
                        stay: pd.Series,
                        vitals: pd.DataFrame,
                        labs: pd.DataFrame,
                        current_time: pd.Timestamp,
                        **extra_data) -> Dict[str, int]:
        labels = {}
        label_name = 'mortality_in_icu'

        if not pd.isna(stay.get('deathtime')):
            if stay['deathtime'] <= stay['outtime'] + pd.Timedelta(hours=24):
                labels[label_name] = 1
            else:
                labels[label_name] = 0
        elif not pd.isna(stay.get('dod')):
            if stay['dod'] <= stay['outtime'] + pd.Timedelta(hours=24):
                labels[label_name] = 1
            else:
                labels[label_name] = 0
        else:
            labels[label_name] = 0

        return labels


if __name__ == "__main__":
    p = MortalityPredictor()
    print(p)
    print(f"Labels: {p.get_label_names()}")
