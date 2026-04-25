"""
Smart ICU Assistant — Clinical Logic Unit Tests
=================================================
Validates the correctness of clinical algorithms:
  • SIRS scoring (feature_engineering.py)
  • KDIGO AKI staging (aki_predictor.py)
  • Mortality label generation (mortality_predictor.py)
  • Sepsis label generation (sepsis_predictor.py)
  • LOS label generation (los_predictor.py)

Run:  pytest tests/test_clinical_logic.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from datetime import timedelta


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SIRS SCORE  (feature_engineering.py → compute_sirs_score)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSIRSScore:
    """Validate SIRS criteria: temp, HR, RR, WBC → score 0-4."""

    @pytest.fixture
    def extractor(self):
        from feature_engineering import FeatureEngineer
        return FeatureEngineer.__new__(FeatureEngineer)   # skip __init__ (no config needed)

    def _make_vitals(self, tempc=37.0, heartrate=75, resprate=16):
        idx = pd.DatetimeIndex([pd.Timestamp("2150-01-01 12:00")])
        return pd.DataFrame({"tempc": [tempc], "heartrate": [heartrate], "resprate": [resprate]}, index=idx)

    def _make_labs(self, wbc=8.0):
        idx = pd.DatetimeIndex([pd.Timestamp("2150-01-01 12:00")])
        return pd.DataFrame({"wbc": [wbc]}, index=idx)

    def test_sirs_all_normal_returns_zero(self, extractor):
        """All vitals within normal range → SIRS = 0."""
        score = extractor.compute_sirs_score(
            self._make_vitals(tempc=37.0, heartrate=75, resprate=16),
            self._make_labs(wbc=8.0),
        )
        assert score.iloc[0] == 0

    def test_sirs_all_criteria_met_returns_four(self, extractor):
        """Temp >38.3, HR >90, RR >20, WBC >12 → SIRS = 4."""
        score = extractor.compute_sirs_score(
            self._make_vitals(tempc=39.5, heartrate=110, resprate=25),
            self._make_labs(wbc=15.0),
        )
        assert score.iloc[0] == 4

    def test_sirs_hypothermia_counts(self, extractor):
        """Temp <36°C triggers the temperature criterion → SIRS ≥ 1."""
        score = extractor.compute_sirs_score(
            self._make_vitals(tempc=35.0, heartrate=75, resprate=16),
            self._make_labs(wbc=8.0),
        )
        assert score.iloc[0] == 1

    def test_sirs_leukopenia_counts(self, extractor):
        """WBC <4 K/µL triggers the WBC criterion → SIRS ≥ 1."""
        score = extractor.compute_sirs_score(
            self._make_vitals(tempc=37.0, heartrate=75, resprate=16),
            self._make_labs(wbc=3.0),
        )
        assert score.iloc[0] == 1

    def test_sirs_partial_two_criteria(self, extractor):
        """Only HR >90 and RR >20 → SIRS = 2 (sepsis threshold)."""
        score = extractor.compute_sirs_score(
            self._make_vitals(tempc=37.0, heartrate=95, resprate=22),
            self._make_labs(wbc=8.0),
        )
        assert score.iloc[0] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  KDIGO AKI STAGING  (aki_predictor.py → _check_aki)
# ═══════════════════════════════════════════════════════════════════════════════

class TestKDIGOStaging:
    """Validate KDIGO creatinine-based AKI staging."""

    @pytest.fixture
    def predictor(self):
        from predictors.aki_predictor import AKIPredictor
        p = AKIPredictor.__new__(AKIPredictor)
        # Set default thresholds manually (skip config loading)
        p.STAGE1_CR_INCREASE = 0.3
        p.STAGE1_CR_RATIO = 1.5
        p.STAGE2_CR_RATIO = 2.0
        p.STAGE3_CR_RATIO = 3.0
        p.STAGE3_CR_ABSOLUTE = 4.0
        return p

    def _make_labs(self, past_values, future_values, current_time):
        """Create lab DataFrame with past + future creatinine values."""
        past_times = [current_time - timedelta(hours=i+1) for i in range(len(past_values))]
        future_times = [current_time + timedelta(hours=i+1) for i in range(len(future_values))]
        all_times = past_times + future_times
        all_values = past_values + future_values
        return pd.DataFrame({"creatinine": all_values}, index=pd.DatetimeIndex(all_times))

    def test_no_aki_normal_creatinine(self, predictor):
        """Stable creatinine 1.0 → no AKI at any stage."""
        t = pd.Timestamp("2150-01-01 12:00")
        labs = self._make_labs([1.0, 1.0], [1.0, 1.1], t)
        result = predictor._check_aki(labs, t, window_hours=24)
        assert result == {"aki_stage1": 0, "aki_stage2": 0, "aki_stage3": 0}

    def test_stage1_absolute_increase(self, predictor):
        """Baseline 1.0 → future 1.3 (increase ≥0.3) → Stage 1."""
        t = pd.Timestamp("2150-01-01 12:00")
        labs = self._make_labs([1.0, 1.0], [1.3], t)
        result = predictor._check_aki(labs, t, window_hours=24)
        assert result["aki_stage1"] == 1
        assert result["aki_stage2"] == 0

    def test_stage1_ratio_trigger(self, predictor):
        """Baseline 1.0 → future 1.5 (1.5× baseline) → Stage 1."""
        t = pd.Timestamp("2150-01-01 12:00")
        labs = self._make_labs([1.0, 1.0], [1.5], t)
        result = predictor._check_aki(labs, t, window_hours=24)
        assert result["aki_stage1"] == 1

    def test_stage2_doubles_baseline(self, predictor):
        """Baseline 1.0 → future 2.0 (2× baseline) → Stage 1 + Stage 2."""
        t = pd.Timestamp("2150-01-01 12:00")
        labs = self._make_labs([1.0, 1.0], [2.0], t)
        result = predictor._check_aki(labs, t, window_hours=24)
        assert result["aki_stage1"] == 1
        assert result["aki_stage2"] == 1
        assert result["aki_stage3"] == 0

    def test_stage3_triples_baseline(self, predictor):
        """Baseline 1.0 → future 3.0 (3× baseline) → all stages positive."""
        t = pd.Timestamp("2150-01-01 12:00")
        labs = self._make_labs([1.0, 1.0], [3.0], t)
        result = predictor._check_aki(labs, t, window_hours=24)
        assert result["aki_stage1"] == 1
        assert result["aki_stage2"] == 1
        assert result["aki_stage3"] == 1

    def test_stage3_absolute_threshold(self, predictor):
        """Creatinine >4.0 mg/dL → Stage 3 regardless of baseline."""
        t = pd.Timestamp("2150-01-01 12:00")
        labs = self._make_labs([3.5, 3.5], [4.1], t)
        result = predictor._check_aki(labs, t, window_hours=24)
        assert result["aki_stage3"] == 1

    def test_no_future_labs_returns_zero(self, predictor):
        """No future creatinine values → no AKI."""
        t = pd.Timestamp("2150-01-01 12:00")
        labs = self._make_labs([1.0, 1.0], [], t)
        result = predictor._check_aki(labs, t, window_hours=24)
        assert result == {"aki_stage1": 0, "aki_stage2": 0, "aki_stage3": 0}


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  MORTALITY LABELS  (mortality_predictor.py → generate_labels)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMortalityLabels:
    """Validate time-to-death mortality label generation."""

    @pytest.fixture
    def predictor(self):
        from predictors.mortality_predictor import MortalityPredictor
        p = MortalityPredictor.__new__(MortalityPredictor)
        p.WINDOWS = [6, 12, 24]
        p.LABEL_PREFIX = "mortality"
        return p

    def test_death_within_6h(self, predictor):
        """Patient dies 5h from now → all 3 horizons positive."""
        t = pd.Timestamp("2150-01-01 12:00")
        stay = pd.Series({"dod": t + timedelta(hours=5)})
        labels = predictor.generate_labels(stay, pd.DataFrame(), pd.DataFrame(), t)
        assert labels["mortality_6h"] == 1
        assert labels["mortality_12h"] == 1
        assert labels["mortality_24h"] == 1

    def test_death_at_18h(self, predictor):
        """Patient dies 18h from now → 6h=0, 12h=0, 24h=1."""
        t = pd.Timestamp("2150-01-01 12:00")
        stay = pd.Series({"dod": t + timedelta(hours=18)})
        labels = predictor.generate_labels(stay, pd.DataFrame(), pd.DataFrame(), t)
        assert labels["mortality_6h"] == 0
        assert labels["mortality_12h"] == 0
        assert labels["mortality_24h"] == 1

    def test_no_death(self, predictor):
        """Patient has no recorded death → all labels = 0."""
        t = pd.Timestamp("2150-01-01 12:00")
        stay = pd.Series({"dod": pd.NaT})
        labels = predictor.generate_labels(stay, pd.DataFrame(), pd.DataFrame(), t)
        assert all(v == 0 for v in labels.values())

    def test_death_in_past(self, predictor):
        """Death time before current_time → all labels = 0."""
        t = pd.Timestamp("2150-01-01 12:00")
        stay = pd.Series({"dod": t - timedelta(hours=2)})
        labels = predictor.generate_labels(stay, pd.DataFrame(), pd.DataFrame(), t)
        assert all(v == 0 for v in labels.values())

    def test_death_beyond_24h(self, predictor):
        """Patient dies 30h from now → all labels = 0."""
        t = pd.Timestamp("2150-01-01 12:00")
        stay = pd.Series({"dod": t + timedelta(hours=30)})
        labels = predictor.generate_labels(stay, pd.DataFrame(), pd.DataFrame(), t)
        assert all(v == 0 for v in labels.values())


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  SEPSIS LABELS — requires SIRS ≥ 2 AND infection evidence
# ═══════════════════════════════════════════════════════════════════════════════

class TestSepsisLabels:
    """Validate sepsis = SIRS ≥ 2 + infection evidence."""

    @pytest.fixture
    def predictor(self):
        from predictors.sepsis_predictor import SepsisPredictor
        p = SepsisPredictor.__new__(SepsisPredictor)
        p.WINDOWS = [6, 12, 24]
        p.LABEL_PREFIX = "sepsis"
        p.ANTIBIOTIC_KEYWORDS = ['cillin', 'mycin', 'vancomycin', 'meropenem']
        p.SEPSIS_ICD9_CODES = ['038', '995.91', '995.92', '785.52']
        return p

    def test_sirs_alone_not_sepsis(self, predictor):
        """SIRS ≥ 2 but no infection evidence → sepsis = 0."""
        t = pd.Timestamp("2150-01-01 12:00")
        vitals = pd.DataFrame({
            "tempc": [39.0], "heartrate": [110], "resprate": [22],
        }, index=pd.DatetimeIndex([t + timedelta(hours=1)]))
        labs = pd.DataFrame({"wbc": [15.0]}, index=pd.DatetimeIndex([t + timedelta(hours=1)]))
        result = predictor._check_sepsis(
            vitals, labs, pd.DataFrame(), pd.DataFrame(), t, window_hours=6
        )
        assert result == 0

    def test_sirs_plus_antibiotic_is_sepsis(self, predictor):
        """SIRS ≥ 2 + antibiotic prescription → sepsis = 1."""
        t = pd.Timestamp("2150-01-01 12:00")
        vitals = pd.DataFrame({
            "tempc": [39.0], "heartrate": [110], "resprate": [22],
        }, index=pd.DatetimeIndex([t + timedelta(hours=1)]))
        labs = pd.DataFrame({"wbc": [15.0]}, index=pd.DatetimeIndex([t + timedelta(hours=1)]))
        prescriptions = pd.DataFrame({
            "startdate": [t + timedelta(hours=2)],
            "drug": ["Vancomycin 1g IV"],
        })
        result = predictor._check_sepsis(
            vitals, labs, prescriptions, pd.DataFrame(), t, window_hours=6
        )
        assert result == 1

    def test_sirs_plus_icd9_is_sepsis(self, predictor):
        """SIRS ≥ 2 + ICD-9 sepsis code → sepsis = 1."""
        t = pd.Timestamp("2150-01-01 12:00")
        vitals = pd.DataFrame({
            "tempc": [39.0], "heartrate": [110], "resprate": [22],
        }, index=pd.DatetimeIndex([t + timedelta(hours=1)]))
        labs = pd.DataFrame({"wbc": [15.0]}, index=pd.DatetimeIndex([t + timedelta(hours=1)]))
        diagnoses = pd.DataFrame({"icd9_code": ["995.91"]})
        result = predictor._check_sepsis(
            vitals, labs, pd.DataFrame(), diagnoses, t, window_hours=6
        )
        assert result == 1

    def test_low_sirs_with_infection_not_sepsis(self, predictor):
        """SIRS = 1 + infection evidence → sepsis = 0 (requires ≥ 2)."""
        t = pd.Timestamp("2150-01-01 12:00")
        vitals = pd.DataFrame({
            "tempc": [37.0], "heartrate": [95], "resprate": [16],
        }, index=pd.DatetimeIndex([t + timedelta(hours=1)]))
        labs = pd.DataFrame({"wbc": [8.0]}, index=pd.DatetimeIndex([t + timedelta(hours=1)]))
        prescriptions = pd.DataFrame({
            "startdate": [t + timedelta(hours=2)],
            "drug": ["Vancomycin 1g IV"],
        })
        result = predictor._check_sepsis(
            vitals, labs, prescriptions, pd.DataFrame(), t, window_hours=6
        )
        assert result == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  LENGTH-OF-STAY LABELS
# ═══════════════════════════════════════════════════════════════════════════════

class TestLOSLabels:
    """Validate LOS binary classification (short <24h / long >72h)."""

    @pytest.fixture
    def predictor(self):
        from predictors.los_predictor import LOSPredictor
        p = LOSPredictor.__new__(LOSPredictor)
        p.WINDOWS = [24, 72]
        p.LABEL_PREFIX = "los"
        return p

    def test_short_stay(self, predictor):
        """Remaining LOS = 12h → short=1, long=0."""
        t = pd.Timestamp("2150-01-01 12:00")
        stay = pd.Series({"outtime": t + timedelta(hours=12)})
        labels = predictor.generate_labels(stay, pd.DataFrame(), pd.DataFrame(), t)
        assert labels["los_short_24h"] == 1
        assert labels["los_long_72h"] == 0

    def test_long_stay(self, predictor):
        """Remaining LOS = 100h → short=0, long=1."""
        t = pd.Timestamp("2150-01-01 12:00")
        stay = pd.Series({"outtime": t + timedelta(hours=100)})
        labels = predictor.generate_labels(stay, pd.DataFrame(), pd.DataFrame(), t)
        assert labels["los_short_24h"] == 0
        assert labels["los_long_72h"] == 1

    def test_medium_stay(self, predictor):
        """Remaining LOS = 48h → short=0, long=0 (medium gap)."""
        t = pd.Timestamp("2150-01-01 12:00")
        stay = pd.Series({"outtime": t + timedelta(hours=48)})
        labels = predictor.generate_labels(stay, pd.DataFrame(), pd.DataFrame(), t)
        assert labels["los_short_24h"] == 0
        assert labels["los_long_72h"] == 0
