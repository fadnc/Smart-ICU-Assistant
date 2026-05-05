"""
Sepsis Predictor — Tasks 4-6
Predicts sepsis onset at 6, 12, and 24 hour horizons using SIRS criteria + infection markers.
"""

import pandas as pd
from typing import Dict
from datetime import timedelta
from predictors.base_predictor import BasePredictor

import logging
logger = logging.getLogger(__name__)


class SepsisPredictor(BasePredictor):
    """
    Sepsis Onset Prediction

    Clinical definition:
        Sepsis = SIRS (≥2 criteria) + Suspected/Confirmed Infection

    SIRS criteria (at any timepoint in the future window):
        1. Temperature > 38.3°C or < 36.0°C
        2. Heart rate > 90 bpm
        3. Respiratory rate > 20 breaths/min
        4. WBC > 12 K/µL or < 4 K/µL

    Infection evidence (either):
        A. Antibiotic prescription started in the window
        B. ICD-9 sepsis diagnosis (038, 995.91, 995.92, 785.52)

    """

    TASK_NAME = "sepsis"
    TASK_DESCRIPTION = "Sepsis onset prediction at 6/12/24h using SIRS + infection markers"
    WINDOWS = [6, 12, 24]
    LABEL_PREFIX = "sepsis"

    ANTIBIOTIC_KEYWORDS = [
        'cillin', 'mycin', 'cycline', 'cephalosporin',
        'quinolone', 'vancomycin', 'meropenem', 'azithromycin',
        'floxacin', 'oxacin', 'penem', 'sulfa', 'metronidazole',
    ]

    SEPSIS_ICD9_CODES = ['038', '995.91', '995.92', '785.52']

    def generate_labels(self,
                        stay: pd.Series,
                        vitals: pd.DataFrame,
                        labs: pd.DataFrame,
                        current_time: pd.Timestamp,
                        **extra_data) -> Dict[str, int]:
        prescriptions = extra_data.get('prescriptions', pd.DataFrame())
        diagnoses = extra_data.get('diagnoses', pd.DataFrame())

        labels = {}
        for window in self.WINDOWS:
            label_name = f'{self.LABEL_PREFIX}_{window}h'
            labels[label_name] = self._check_sepsis(
                vitals, labs, prescriptions, diagnoses, current_time, window
            )
        return labels

    def _check_sepsis(self, vitals, labs, prescriptions, diagnoses,
                      current_time, window_hours) -> int:
        window_end = current_time + timedelta(hours=window_hours)

        # --- SIRS check on future vitals/labs ---
        future_vitals = vitals[
            (vitals.index > current_time) & (vitals.index <= window_end)
        ] if len(vitals) > 0 else pd.DataFrame()

        future_labs = labs[
            (labs.index > current_time) & (labs.index <= window_end)
        ] if len(labs) > 0 else pd.DataFrame()

        if len(future_vitals) == 0:
            return 0

        combined = pd.concat([future_vitals, future_labs], axis=1)
        sirs_count = pd.Series(0, index=combined.index)

        if 'tempc' in combined.columns:
            sirs_count += ((combined['tempc'] > 38.3) | (combined['tempc'] < 36.0)).fillna(False).astype(int)
        if 'heartrate' in combined.columns:
            sirs_count += (combined['heartrate'] > 90).fillna(False).astype(int)
        if 'resprate' in combined.columns:
            sirs_count += (combined['resprate'] > 20).fillna(False).astype(int)
        if 'wbc' in combined.columns:
            sirs_count += ((combined['wbc'] > 12) | (combined['wbc'] < 4)).fillna(False).astype(int)

        if not (sirs_count >= 2).any():
            return 0

        # --- Infection evidence ---
        # 1. Antibiotic prescriptions
        if len(prescriptions) > 0 and 'startdate' in prescriptions.columns:
            future_meds = prescriptions[
                (prescriptions['startdate'] > current_time) &
                (prescriptions['startdate'] <= window_end)
            ]
            if len(future_meds) > 0 and 'drug' in future_meds.columns:
                for kw in self.ANTIBIOTIC_KEYWORDS:
                    if future_meds['drug'].str.contains(kw, case=False, na=False).any():
                        return 1

        # 2. ICD-9 sepsis codes
        if len(diagnoses) > 0 and 'icd9_code' in diagnoses.columns:
            for code in self.SEPSIS_ICD9_CODES:
                if diagnoses['icd9_code'].str.startswith(code, na=False).any():
                    return 1

        return 0


if __name__ == "__main__":
    p = SepsisPredictor()
    print(p)
    print(f"Labels: {p.get_label_names()}")
