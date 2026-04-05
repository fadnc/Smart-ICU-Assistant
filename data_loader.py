"""
MIMIC-III Data Loader
Loads and merges MIMIC-III demo tables for ICU patient analysis.

CHANGES:
  - tqdm progress bars on every load method (stderr, so they don't clash with logging)
  - Chunked CHARTEVENTS bar shows MB read vs estimated total
  - Chunked LABEVENTS bar shows rows processed
  - merge_data() shows an overall step bar
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import yaml
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MIMICDataLoader:
    """Load and process MIMIC-III clinical database."""

    def __init__(self, data_dir: str, config_path: str):
        self.data_dir = data_dir
        self.config   = self._load_config(config_path)

        # Core tables
        self.patients      = None
        self.admissions    = None
        self.icu_stays     = None
        self.chartevents   = None
        self.labevents     = None
        self.diagnoses     = None
        self.prescriptions = None
        self.d_items       = None
        self.d_labitems    = None

        # Additional tables
        self.inputevents_mv    = None
        self.outputevents      = None
        self.procedureevents   = None
        self.procedures_icd    = None
        self.microbiologyevents= None
        self.transfers         = None
        self.callout           = None
        self.services          = None
        self.d_icd_diagnoses   = None
        self.d_icd_procedures  = None

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase all column names for MIMIC-III full/demo compatibility."""
        df.columns = [c.lower() for c in df.columns]
        return df

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _pbar(desc: str, total=None, unit='rows') -> tqdm:
        """Create a consistent tqdm bar writing to stderr."""
        return tqdm(
            total=total,
            desc=f"  {desc}",
            unit=unit,
            unit_scale=True,
            file=sys.stderr,
            dynamic_ncols=True,
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

    # ── Core table loaders ────────────────────────────────────────────────────

    def load_patients(self) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, 'PATIENTS.csv')
        with self._pbar("Loading PATIENTS") as pbar:
            self.patients = self._normalize_columns(pd.read_csv(filepath))
            pbar.update(len(self.patients))
        self.patients['dob'] = pd.to_datetime(self.patients['dob'])
        self.patients['dod'] = pd.to_datetime(self.patients['dod'])
        logger.info(f"PATIENTS: {len(self.patients):,} rows")
        return self.patients

    def load_admissions(self) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, 'ADMISSIONS.csv')
        with self._pbar("Loading ADMISSIONS") as pbar:
            self.admissions = self._normalize_columns(pd.read_csv(filepath))
            pbar.update(len(self.admissions))
        for col in ['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime']:
            if col in self.admissions.columns:
                self.admissions[col] = pd.to_datetime(
                    self.admissions[col], format='%Y-%m-%d %H:%M:%S', errors='coerce'
                )
        logger.info(f"ADMISSIONS: {len(self.admissions):,} rows")
        return self.admissions

    def load_icu_stays(self) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, 'ICUSTAYS.csv')
        with self._pbar("Loading ICUSTAYS") as pbar:
            self.icu_stays = self._normalize_columns(pd.read_csv(filepath))
            pbar.update(len(self.icu_stays))
        self.icu_stays['intime']  = pd.to_datetime(self.icu_stays['intime'])
        self.icu_stays['outtime'] = pd.to_datetime(self.icu_stays['outtime'])
        logger.info(f"ICUSTAYS: {len(self.icu_stays):,} rows")
        return self.icu_stays

    def load_chartevents(self, sample_n: Optional[int] = None) -> pd.DataFrame:
        """
        Load vital signs and chart events.
        Small files (<500 MB): loaded directly with a row-count bar.
        Large files (≥500 MB): chunked loading with an MB-based bar.
        """
        filepath      = os.path.join(self.data_dir, 'CHARTEVENTS.csv')
        file_size_mb  = os.path.getsize(filepath) / (1024 * 1024)
        file_size_gb  = file_size_mb / 1024

        if file_size_mb < 500 or sample_n:
            desc = "Loading CHARTEVENTS" + (f" (sample={sample_n:,})" if sample_n else "")
            with self._pbar(desc) as pbar:
                if sample_n:
                    self.chartevents = self._normalize_columns(
                        pd.read_csv(filepath, nrows=sample_n, low_memory=False)
                    )
                else:
                    self.chartevents = self._normalize_columns(
                        pd.read_csv(filepath, low_memory=False)
                    )
                pbar.update(len(self.chartevents))
        else:
            # Large file — chunked with MB progress bar
            logger.info(
                f"CHARTEVENTS is {file_size_gb:.1f} GB — starting chunked load "
                f"(filtering to vital/vent itemids only)"
            )
            relevant_ids = self._get_relevant_chartevents_itemids()
            logger.info(f"  Filtering to {len(relevant_ids)} relevant itemids")

            CHARTEVENTS_DTYPES = {
                'ROW_ID':    'int32',
                'SUBJECT_ID':'int32',
                'HADM_ID':   'float32',
                'ICUSTAY_ID':'float32',
                'ITEMID':    'int32',
                'VALUENUM':  'float32',
                'VALUEUOM':  'str',
                'ERROR':     'float32',
            }

            chunk_size  = 2_000_000
            chunks      = []
            total_read  = 0
            bytes_read  = 0
            file_bytes  = os.path.getsize(filepath)

            bar = tqdm(
                total=file_bytes,
                desc="  CHARTEVENTS (chunked)",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                file=sys.stderr,
                dynamic_ncols=True,
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] kept={postfix}",
            )

            reader = pd.read_csv(
                filepath,
                chunksize=chunk_size,
                dtype=CHARTEVENTS_DTYPES,
                usecols=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID',
                         'ITEMID', 'CHARTTIME', 'VALUENUM'],
                low_memory=False,
            )

            for chunk in reader:
                chunk_bytes = chunk.memory_usage(deep=True).sum()
                chunk = self._normalize_columns(chunk)
                if 'itemid' in chunk.columns:
                    filtered = chunk[chunk['itemid'].isin(relevant_ids)]
                    if len(filtered) > 0:
                        chunks.append(filtered)
                total_read += len(chunk)
                # Approximate bytes by row ratio
                bytes_read += int(file_bytes * len(chunk) / 330_000_000)
                bytes_read  = min(bytes_read, file_bytes)
                kept        = sum(len(c) for c in chunks)
                bar.n       = bytes_read
                bar.set_postfix_str(f"{kept:,} rows")
                bar.refresh()

            bar.n = file_bytes
            bar.refresh()
            bar.close()

            self.chartevents = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            logger.info(
                f"CHARTEVENTS: kept {len(self.chartevents):,} rows "
                f"(from ~{total_read:,} total)"
            )

        if len(self.chartevents) > 0:
            if 'charttime' in self.chartevents.columns:
                self.chartevents['charttime'] = pd.to_datetime(self.chartevents['charttime'])
            if 'valuenum' in self.chartevents.columns:
                self.chartevents['valuenum'] = pd.to_numeric(
                    self.chartevents['valuenum'], errors='coerce'
                )
        return self.chartevents

    def _get_relevant_chartevents_itemids(self) -> set:
        relevant = set()
        vent_ids = self.config.get('VENTILATION_ITEMIDS', [225792, 225794, 226260])
        relevant.update(vent_ids)
        vital_itemids = {
            220045, 211,
            220050, 220179, 51, 455,
            220051, 220180, 8368, 8441,
            220052, 220181, 52, 456,
            220210, 224690, 618, 615,
            223761, 223762, 676, 678,
            220277, 646,
            220621, 226537, 807, 811, 1529,
        }
        relevant.update(vital_itemids)
        return relevant

    def load_labevents(self) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, 'LABEVENTS.csv')
        relevant_subjects = set(self.icu_stays['subject_id'].unique())
        file_bytes = os.path.getsize(filepath)
        chunk_size = 1_000_000
        chunks     = []
        bytes_est  = 0

        bar = tqdm(
            total=file_bytes,
            desc="  LABEVENTS (chunked)",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            file=sys.stderr,
            dynamic_ncols=True,
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        reader = pd.read_csv(
            filepath,
            chunksize=chunk_size,
            dtype={'SUBJECT_ID': 'int32', 'ITEMID': 'int32', 'VALUENUM': 'float32'},
            usecols=['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM'],
        )

        total_rows = 0
        for chunk in reader:
            chunk = self._normalize_columns(chunk)
            chunk = chunk[chunk['subject_id'].isin(relevant_subjects)]
            if len(chunk) > 0:
                chunks.append(chunk)
            total_rows += chunk_size
            # Approximate progress
            bytes_est = min(int(file_bytes * total_rows / 27_000_000), file_bytes)
            bar.n = bytes_est
            bar.refresh()

        bar.n = file_bytes
        bar.refresh()
        bar.close()

        self.labevents = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        self.labevents['charttime'] = pd.to_datetime(self.labevents['charttime'])
        self.labevents['valuenum']  = pd.to_numeric(self.labevents['valuenum'], errors='coerce')

        # Assign icustay_id via time-overlap join
        if self.icu_stays is not None:
            logger.info("Assigning icustay_id to lab events via time overlap...")
            with self._pbar("  Merging LABEVENTS → ICUSTAYS", unit="stays") as pbar:
                merged = self.labevents.merge(
                    self.icu_stays[['subject_id', 'hadm_id', 'icustay_id', 'intime', 'outtime']],
                    on=['subject_id', 'hadm_id'],
                    how='left',
                )
                mask = (
                    merged['icustay_id'].notna() &
                    (merged['charttime'] >= merged['intime']) &
                    (merged['charttime'] <= merged['outtime'])
                )
                self.labevents = merged[mask].drop(columns=['intime', 'outtime']).copy()
                pbar.update(len(self.labevents))
        logger.info(f"LABEVENTS: {len(self.labevents):,} rows (in-stay)")
        return self.labevents

    def load_diagnoses(self) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, 'DIAGNOSES_ICD.csv')
        if not os.path.exists(filepath):
            logger.warning("DIAGNOSES_ICD.csv not found, skipping")
            self.diagnoses = pd.DataFrame()
            return self.diagnoses
        with self._pbar("Loading DIAGNOSES_ICD") as pbar:
            self.diagnoses = self._normalize_columns(pd.read_csv(filepath))
            pbar.update(len(self.diagnoses))
        logger.info(f"DIAGNOSES_ICD: {len(self.diagnoses):,} rows")
        return self.diagnoses

    def load_prescriptions(self) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, 'PRESCRIPTIONS.csv')
        with self._pbar("Loading PRESCRIPTIONS") as pbar:
            self.prescriptions = self._normalize_columns(
                pd.read_csv(filepath, low_memory=False)
            )
            pbar.update(len(self.prescriptions))
        for col in ['startdate', 'enddate']:
            if col in self.prescriptions.columns:
                self.prescriptions[col] = pd.to_datetime(self.prescriptions[col])
        logger.info(f"PRESCRIPTIONS: {len(self.prescriptions):,} rows")
        return self.prescriptions

    def load_inputevents(self) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, 'INPUTEVENTS_MV.csv')
        if not os.path.exists(filepath):
            logger.warning("INPUTEVENTS_MV.csv not found, skipping")
            self.inputevents_mv = pd.DataFrame()
            return self.inputevents_mv
        with self._pbar("Loading INPUTEVENTS_MV") as pbar:
            self.inputevents_mv = self._normalize_columns(pd.read_csv(filepath))
            pbar.update(len(self.inputevents_mv))
        for col in ['starttime', 'endtime']:
            if col in self.inputevents_mv.columns:
                self.inputevents_mv[col] = pd.to_datetime(self.inputevents_mv[col])
        logger.info(f"INPUTEVENTS_MV: {len(self.inputevents_mv):,} rows")
        return self.inputevents_mv

    def load_outputevents(self) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, 'OUTPUTEVENTS.csv')
        if not os.path.exists(filepath):
            logger.warning("OUTPUTEVENTS.csv not found, skipping")
            self.outputevents = pd.DataFrame()
            return self.outputevents
        with self._pbar("Loading OUTPUTEVENTS") as pbar:
            self.outputevents = self._normalize_columns(pd.read_csv(filepath))
            pbar.update(len(self.outputevents))
        if 'charttime' in self.outputevents.columns:
            self.outputevents['charttime'] = pd.to_datetime(self.outputevents['charttime'])
        self.outputevents['value'] = pd.to_numeric(self.outputevents['value'], errors='coerce')
        logger.info(f"OUTPUTEVENTS: {len(self.outputevents):,} rows")
        return self.outputevents

    def load_procedureevents(self) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, 'PROCEDUREEVENTS_MV.csv')
        if not os.path.exists(filepath):
            logger.warning("PROCEDUREEVENTS_MV.csv not found, skipping")
            self.procedureevents = pd.DataFrame()
            return self.procedureevents
        with self._pbar("Loading PROCEDUREEVENTS_MV") as pbar:
            self.procedureevents = self._normalize_columns(pd.read_csv(filepath))
            pbar.update(len(self.procedureevents))
        for col in ['starttime', 'endtime']:
            if col in self.procedureevents.columns:
                self.procedureevents[col] = pd.to_datetime(self.procedureevents[col])
        logger.info(f"PROCEDUREEVENTS_MV: {len(self.procedureevents):,} rows")
        return self.procedureevents

    def load_procedures_icd(self) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, 'PROCEDURES_ICD.csv')
        if not os.path.exists(filepath):
            logger.warning("PROCEDURES_ICD.csv not found, skipping")
            self.procedures_icd = pd.DataFrame()
            return self.procedures_icd
        with self._pbar("Loading PROCEDURES_ICD") as pbar:
            self.procedures_icd = self._normalize_columns(pd.read_csv(filepath))
            pbar.update(len(self.procedures_icd))
        logger.info(f"PROCEDURES_ICD: {len(self.procedures_icd):,} rows")
        return self.procedures_icd

    def load_microbiologyevents(self) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, 'MICROBIOLOGYEVENTS.csv')
        if not os.path.exists(filepath):
            logger.warning("MICROBIOLOGYEVENTS.csv not found, skipping")
            self.microbiologyevents = pd.DataFrame()
            return self.microbiologyevents
        with self._pbar("Loading MICROBIOLOGYEVENTS") as pbar:
            self.microbiologyevents = self._normalize_columns(pd.read_csv(filepath))
            pbar.update(len(self.microbiologyevents))
        for col in ['chartdate', 'charttime']:
            if col in self.microbiologyevents.columns:
                self.microbiologyevents[col] = pd.to_datetime(
                    self.microbiologyevents[col], errors='coerce'
                )
        logger.info(f"MICROBIOLOGYEVENTS: {len(self.microbiologyevents):,} rows")
        return self.microbiologyevents

    def load_transfers(self) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, 'TRANSFERS.csv')
        if not os.path.exists(filepath):
            logger.warning("TRANSFERS.csv not found, skipping")
            self.transfers = pd.DataFrame()
            return self.transfers
        with self._pbar("Loading TRANSFERS") as pbar:
            self.transfers = self._normalize_columns(pd.read_csv(filepath))
            pbar.update(len(self.transfers))
        for col in ['intime', 'outtime']:
            if col in self.transfers.columns:
                self.transfers[col] = pd.to_datetime(self.transfers[col])
        logger.info(f"TRANSFERS: {len(self.transfers):,} rows")
        return self.transfers

    def load_callout(self) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, 'CALLOUT.csv')
        if not os.path.exists(filepath):
            logger.warning("CALLOUT.csv not found, skipping")
            self.callout = pd.DataFrame()
            return self.callout
        with self._pbar("Loading CALLOUT") as pbar:
            self.callout = self._normalize_columns(pd.read_csv(filepath))
            pbar.update(len(self.callout))
        for col in ['createtime', 'updatetime', 'acknowledgetime', 'outcometime']:
            if col in self.callout.columns:
                self.callout[col] = pd.to_datetime(
                    self.callout[col], format='%Y-%m-%d %H:%M:%S', errors='coerce'
                )
        logger.info(f"CALLOUT: {len(self.callout):,} rows")
        return self.callout

    def load_services(self) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, 'SERVICES.csv')
        if not os.path.exists(filepath):
            logger.warning("SERVICES.csv not found, skipping")
            self.services = pd.DataFrame()
            return self.services
        with self._pbar("Loading SERVICES") as pbar:
            self.services = self._normalize_columns(pd.read_csv(filepath))
            pbar.update(len(self.services))
        if 'transfertime' in self.services.columns:
            self.services['transfertime'] = pd.to_datetime(self.services['transfertime'])
        logger.info(f"SERVICES: {len(self.services):,} rows")
        return self.services

    def load_dictionaries(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        with self._pbar("Loading D_ITEMS + D_LABITEMS") as pbar:
            self.d_items = self._normalize_columns(
                pd.read_csv(os.path.join(self.data_dir, 'D_ITEMS.csv'))
            )
            self.d_labitems = self._normalize_columns(
                pd.read_csv(os.path.join(self.data_dir, 'D_LABITEMS.csv'))
            )
            pbar.update(len(self.d_items) + len(self.d_labitems))
        logger.info(
            f"D_ITEMS: {len(self.d_items):,} | D_LABITEMS: {len(self.d_labitems):,}"
        )
        return self.d_items, self.d_labitems

    # ── Master merge ──────────────────────────────────────────────────────────

    def merge_data(self, load_chart_sample: Optional[int] = None) -> pd.DataFrame:
        """Load all 14 tables, merge, and compute derived columns."""

        ALL_STEPS = [
            "PATIENTS", "ADMISSIONS", "ICUSTAYS", "DICTIONARIES",
            "CHARTEVENTS", "LABEVENTS", "DIAGNOSES_ICD", "PRESCRIPTIONS",
            "INPUTEVENTS_MV", "OUTPUTEVENTS", "PROCEDUREEVENTS_MV",
            "PROCEDURES_ICD", "MICROBIOLOGYEVENTS", "TRANSFERS",
            "CALLOUT", "SERVICES", "MERGE",
        ]

        step_bar = tqdm(
            ALL_STEPS,
            desc="  Data loading",
            unit="step",
            file=sys.stderr,
            dynamic_ncols=True,
            leave=True,
        )

        def advance(label):
            step_bar.set_description(f"  {label:<35}")
            step_bar.update(1)

        logger.info("=" * 60)
        logger.info("DATA LOADING PIPELINE")
        logger.info("=" * 60)

        self.load_patients();      advance("PATIENTS ✓")
        self.load_admissions();    advance("ADMISSIONS ✓")
        self.load_icu_stays();     advance("ICUSTAYS ✓")
        self.load_dictionaries();  advance("DICTIONARIES ✓")
        self.load_chartevents(sample_n=load_chart_sample); advance("CHARTEVENTS ✓")
        self.load_labevents();     advance("LABEVENTS ✓")
        self.load_diagnoses();     advance("DIAGNOSES_ICD ✓")
        self.load_prescriptions(); advance("PRESCRIPTIONS ✓")
        self.load_inputevents();   advance("INPUTEVENTS_MV ✓")
        self.load_outputevents();  advance("OUTPUTEVENTS ✓")
        self.load_procedureevents(); advance("PROCEDUREEVENTS_MV ✓")
        self.load_procedures_icd(); advance("PROCEDURES_ICD ✓")
        self.load_microbiologyevents(); advance("MICROBIOLOGYEVENTS ✓")
        self.load_transfers();     advance("TRANSFERS ✓")
        self.load_callout();       advance("CALLOUT ✓")
        self.load_services();      advance("SERVICES ✓")

        logger.info("Merging tables...")
        merged = self.icu_stays.merge(
            self.patients[['subject_id', 'gender', 'dob', 'dod', 'expire_flag']],
            on='subject_id', how='left',
        )
        merged = merged.merge(
            self.admissions[[
                'subject_id', 'hadm_id', 'admittime', 'dischtime',
                'deathtime', 'admission_type', 'diagnosis',
            ]],
            on=['subject_id', 'hadm_id'], how='left',
        )

        # Age calculation (MIMIC-III shifted dates; >89 → clamp to 91.4)
        try:
            merged['age'] = (
                (merged['intime'] - merged['dob']).dt.total_seconds()
                / (365.25 * 24 * 3600)
            )
        except OverflowError:
            def safe_age(row):
                try:
                    return (row['intime'] - row['dob']).total_seconds() / (365.25 * 24 * 3600)
                except Exception:
                    return np.nan
            merged['age'] = merged.apply(safe_age, axis=1)

        n_elderly = (merged['age'] > 89).sum()
        if n_elderly > 0:
            merged.loc[merged['age'] > 89, 'age'] = 91.4
            logger.info(f"  Clamped {n_elderly} ages >89 → 91.4 (MIMIC-III de-id)")

        n_neg = (merged['age'] < 0).sum()
        if n_neg > 0:
            merged.loc[merged['age'] < 0, 'age'] = np.nan
            logger.info(f"  Set {n_neg} negative ages → NaN")

        try:
            merged['hours_to_death'] = (
                (merged['dod'] - merged['intime']).dt.total_seconds() / 3600
            )
        except Exception:
            def safe_death(row):
                try:
                    return (
                        np.nan if pd.isna(row['dod'])
                        else (row['dod'] - row['intime']).total_seconds() / 3600
                    )
                except Exception:
                    return np.nan
            merged['hours_to_death'] = merged.apply(safe_death, axis=1)

        advance("MERGE ✓")
        step_bar.close()

        logger.info(f"Merged dataset: {len(merged):,} ICU stays")
        logger.info(f"  Unique patients : {merged['subject_id'].nunique():,}")
        logger.info(f"  Deceased        : {int(merged['expire_flag'].sum()):,}")
        logger.info(f"  Mean age        : {merged['age'].mean():.1f} years")
        return merged

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_vital_sign_itemids(self) -> Dict[str, List[int]]:
        if self.d_items is None:
            self.load_dictionaries()
        vital_mapping = {
            'heartrate': ['heart rate', 'hr'],
            'sysbp':     ['systolic', 'nbp systolic', 'arterial bp systolic'],
            'diasbp':    ['diastolic', 'nbp diastolic', 'arterial bp diastolic'],
            'meanbp':    ['mean', 'nbp mean', 'arterial bp mean'],
            'resprate':  ['respiratory rate', 'resp rate', 'rr'],
            'tempc':     ['temperature c', 'temperature celsius'],
            'spo2':      ['spo2', 'o2 saturation'],
            'glucose':   ['glucose', 'fingerstick glucose'],
        }
        itemid_map = {}
        for name, keywords in vital_mapping.items():
            mask = self.d_items['label'].str.lower().str.contains(
                '|'.join(keywords), na=False, case=False
            )
            itemid_map[name] = self.d_items[mask]['itemid'].unique().tolist()
        return itemid_map

    def get_lab_itemids(self) -> Dict[str, List[int]]:
        if self.d_labitems is None:
            self.load_dictionaries()
        itemid_map = {}
        for lab_name in self.config.get('LAB_TESTS', []):
            mask = self.d_labitems['label'].str.lower().str.contains(
                lab_name, na=False, case=False
            )
            itemid_map[lab_name] = self.d_labitems[mask]['itemid'].unique().tolist()
        return itemid_map

    def get_patient_timeseries(self, icustay_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        charts = self.chartevents[
            self.chartevents['icustay_id'] == icustay_id
        ].sort_values('charttime').copy()
        labs = self.labevents[
            self.labevents['icustay_id'] == icustay_id
        ].sort_values('charttime').copy()
        return charts, labs


if __name__ == "__main__":
    loader = MIMICDataLoader('data', 'config.yaml')
    merged = loader.merge_data()
    print(f"\nLoaded {len(merged):,} ICU stays")
    print(f"Columns: {list(merged.columns)}")