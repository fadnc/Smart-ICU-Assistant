"""
Vasopressor Predictor — Tasks 16-17
Predicts vasopressor requirement at 6 and 12 hour horizons.
"""

import pandas as pd
from typing import Dict
from datetime import timedelta
from predictors.base_predictor import BasePredictor

import logging
logger = logging.getLogger(__name__)


class VasopressorPredictor(BasePredictor):
    """
    Vasopressor Requirement Prediction

    Clinical definition:
        Vasopressors are emergency drugs to raise critically low blood pressure.
        Drugs: norepinephrine, epinephrine, vasopressin, dopamine, dobutamine,
               phenylephrine, milrinone

    Logic:
        Check PRESCRIPTIONS for vasopressor drug names started in the window.
        Also check INPUTEVENTS_MV for IV vasopressor itemids.

    """

    TASK_NAME = "vasopressor"
    TASK_DESCRIPTION = "Vasopressor requirement prediction at 6/12h"
    WINDOWS = [6, 12]
    LABEL_PREFIX = "vasopressor"

    VASOPRESSOR_DRUG_KEYWORDS = [
        'norepinephrine', 'epinephrine', 'vasopressin',
        'dopamine', 'dobutamine', 'phenylephrine', 'milrinone',
    ]

    def __init__(self, config_path: str = 'config.yaml'):
        super().__init__(config_path)
        self.vasopressor_itemids = self.config.get(
            'VASOPRESSOR_ITEMIDS', [221906, 221289, 222315, 221749, 221662]
        )

    def generate_labels(self,
                        stay: pd.Series,
                        vitals: pd.DataFrame,
                        labs: pd.DataFrame,
                        current_time: pd.Timestamp,
                        **extra_data) -> Dict[str, int]:
        prescriptions = extra_data.get('prescriptions', pd.DataFrame())
        inputevents = extra_data.get('inputevents', pd.DataFrame())

        labels = {}
        for window in self.WINDOWS:
            label_name = f'{self.LABEL_PREFIX}_{window}h'
            labels[label_name] = self._check_vasopressor(
                prescriptions, inputevents, current_time, window
            )
        return labels

    def _check_vasopressor(self, prescriptions, inputevents,
                           current_time, window_hours) -> int:
        window_end = current_time + timedelta(hours=window_hours)

        # Check 1: Prescriptions
        if len(prescriptions) > 0 and 'startdate' in prescriptions.columns:
            future_meds = prescriptions[
                (prescriptions['startdate'] > current_time) &
                (prescriptions['startdate'] <= window_end)
            ]
            if len(future_meds) > 0 and 'drug' in future_meds.columns:
                for kw in self.VASOPRESSOR_DRUG_KEYWORDS:
                    if future_meds['drug'].str.contains(kw, case=False, na=False).any():
                        return 1

        # Check 2: INPUTEVENTS_MV (IV administration)
        if len(inputevents) > 0 and 'itemid' in inputevents.columns:
            time_col = 'starttime' if 'starttime' in inputevents.columns else 'charttime'
            if time_col in inputevents.columns:
                future_inputs = inputevents[
                    (inputevents['itemid'].isin(self.vasopressor_itemids)) &
                    (inputevents[time_col] > current_time) &
                    (inputevents[time_col] <= window_end)
                ]
                if len(future_inputs) > 0:
                    return 1

        return 0


if __name__ == "__main__":
    p = VasopressorPredictor()
    print(p)
    print(f"Labels: {p.get_label_names()}")
