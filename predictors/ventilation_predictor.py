"""
Ventilation Predictor — Tasks 18-20
Predicts mechanical ventilation requirement at 6, 12, and 24 hour horizons.
"""

import pandas as pd
from typing import Dict
from predictors.base_predictor import BasePredictor

import logging
logger = logging.getLogger(__name__)


class VentilationPredictor(BasePredictor):
    """
    Mechanical Ventilation Requirement Prediction

    Three-layer detection (any one triggers label = 1):
        1. CHARTEVENTS — ventilation itemids (225792, 225794, 226260)
        2. PROCEDUREEVENTS_MV — same itemids in procedure table
        3. PROCEDURES_ICD — ICD-9 codes (9670-9672, 9604, 9390)

    """

    TASK_NAME = "ventilation"
    TASK_DESCRIPTION = "Mechanical ventilation requirement at 6/12/24h"
    WINDOWS = [6, 12, 24]
    LABEL_PREFIX = "ventilation"

    def __init__(self, config_path: str = 'config.yaml'):
        super().__init__(config_path)
        self.vent_itemids = self.config.get(
            'VENTILATION_ITEMIDS', [225792, 225794, 226260]
        )
        self.vent_icd_codes = self.config.get(
            'VENTILATION_ICD_CODES', ['9670', '9671', '9672', '9604', '9390']
        )

    def generate_labels(self,
                        stay: pd.Series,
                        vitals: pd.DataFrame,
                        labs: pd.DataFrame,
                        current_time: pd.Timestamp,
                        **extra_data) -> Dict[str, int]:
        chartevents = extra_data.get('chartevents', pd.DataFrame())
        procedureevents = extra_data.get('procedureevents', pd.DataFrame())
        procedures_icd = extra_data.get('procedures_icd', pd.DataFrame())

        labels = {}
        for window in self.WINDOWS:
            label_name = f'{self.LABEL_PREFIX}_{window}h'
            labels[label_name] = self._check_ventilation(
                chartevents, procedureevents, procedures_icd,
                current_time, window
            )
        return labels

    def _check_ventilation(self, chartevents, procedureevents, procedures_icd,
                           current_time, window_hours) -> int:
        window_end = current_time + pd.Timedelta(hours=window_hours)

        # Check 1: CHARTEVENTS ventilation itemids
        if len(chartevents) > 0 and 'itemid' in chartevents.columns:
            vent = chartevents[
                (chartevents['itemid'].isin(self.vent_itemids)) &
                (chartevents['charttime'] >= current_time) &
                (chartevents['charttime'] <= window_end)
            ]
            if len(vent) > 0:
                return 1

        # Check 2: PROCEDUREEVENTS_MV
        if len(procedureevents) > 0 and 'itemid' in procedureevents.columns:
            vent = procedureevents[
                (procedureevents['itemid'].isin(self.vent_itemids)) &
                (procedureevents['starttime'] >= current_time) &
                (procedureevents['starttime'] <= window_end)
            ]
            if len(vent) > 0:
                return 1

        # Check 3: PROCEDURES_ICD ventilation codes
        if len(procedures_icd) > 0 and 'icd9_code' in procedures_icd.columns:
            vent = procedures_icd[
                procedures_icd['icd9_code'].astype(str).isin(self.vent_icd_codes)
            ]
            if len(vent) > 0:
                return 1

        return 0


if __name__ == "__main__":
    p = VentilationPredictor()
    print(p)
    print(f"Labels: {p.get_label_names()}")
