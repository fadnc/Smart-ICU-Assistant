"""
MIMIC-III Data Loader
Loads and merges MIMIC-III demo tables for ICU patient analysis
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MIMICDataLoader:
    """Load and process MIMIC-III clinical database"""
    
    def __init__(self, data_dir: str, config_path: str):
        """
        Initialize data loader
        
        Args:
            data_dir: Path to MIMIC-III data directory (e.g., 'demo/')
            config_path: Path to config.yaml file
        """
        self.data_dir = data_dir
        self.config = self._load_config(config_path)
        
        # Data containers
        self.patients = None
        self.admissions = None
        self.icu_stays = None
        self.chartevents = None
        self.labevents = None
        self.diagnoses = None
        self.prescriptions = None
        self.d_items = None
        self.d_labitems = None
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_patients(self) -> pd.DataFrame:
        """Load patient demographics"""
        logger.info("Loading PATIENTS...")
        filepath = os.path.join(self.data_dir, 'PATIENTS.csv')
        self.patients = pd.read_csv(filepath)
        
        # Convert dates
        self.patients['dob'] = pd.to_datetime(self.patients['dob'])
        self.patients['dod'] = pd.to_datetime(self.patients['dod'])
        
        logger.info(f"Loaded {len(self.patients)} patients")
        return self.patients
    
    def load_admissions(self) -> pd.DataFrame:
        """Load hospital admissions"""
        logger.info("Loading ADMISSIONS...")
        filepath = os.path.join(self.data_dir, 'ADMISSIONS.csv')
        self.admissions = pd.read_csv(filepath)
        
        # Convert timestamps
        time_cols = ['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime']
        for col in time_cols:
            if col in self.admissions.columns:
                self.admissions[col] = pd.to_datetime(self.admissions[col])
        
        logger.info(f"Loaded {len(self.admissions)} admissions")
        return self.admissions
    
    def load_icu_stays(self) -> pd.DataFrame:
        """Load ICU stays"""
        logger.info("Loading ICUSTAYS...")
        filepath = os.path.join(self.data_dir, 'ICUSTAYS.csv')
        self.icu_stays = pd.read_csv(filepath)
        
        # Convert timestamps
        self.icu_stays['intime'] = pd.to_datetime(self.icu_stays['intime'])
        self.icu_stays['outtime'] = pd.to_datetime(self.icu_stays['outtime'])
        
        logger.info(f"Loaded {len(self.icu_stays)} ICU stays")
        return self.icu_stays
    
    def load_chartevents(self, sample_n: Optional[int] = None) -> pd.DataFrame:
        """
        Load vital signs and chart events
        
        Args:
            sample_n: If provided, load only first N rows (for large files)
        """
        logger.info("Loading CHARTEVENTS...")
        filepath = os.path.join(self.data_dir, 'CHARTEVENTS.csv')
        
        # Load with optional sampling for large files
        if sample_n:
            self.chartevents = pd.read_csv(filepath, nrows=sample_n, low_memory=False)
            logger.info(f"Loaded {sample_n} chart events (sampled)")
        else:
            self.chartevents = pd.read_csv(filepath, low_memory=False)
            logger.info(f"Loaded {len(self.chartevents)} chart events")
        
        # Convert timestamps
        self.chartevents['charttime'] = pd.to_datetime(self.chartevents['charttime'])
        
        # Convert numeric values
        self.chartevents['valuenum'] = pd.to_numeric(self.chartevents['valuenum'], errors='coerce')
        
        return self.chartevents
    
    def load_labevents(self) -> pd.DataFrame:
        """Load laboratory test results and assign icustay_id via time-based join"""
        logger.info("Loading LABEVENTS...")
        filepath = os.path.join(self.data_dir, 'LABEVENTS.csv')
        self.labevents = pd.read_csv(filepath)
        
        # Convert timestamps
        self.labevents['charttime'] = pd.to_datetime(self.labevents['charttime'])
        
        # Convert numeric values
        self.labevents['valuenum'] = pd.to_numeric(self.labevents['valuenum'], errors='coerce')
        
        logger.info(f"Loaded {len(self.labevents)} lab events")
        
        # LABEVENTS.csv has no icustay_id column — assign via ICUSTAYS join
        if self.icu_stays is not None:
            logger.info("Assigning icustay_id to lab events via hadm_id + time overlap...")
            labevents_merged = self.labevents.merge(
                self.icu_stays[['subject_id', 'hadm_id', 'icustay_id', 'intime', 'outtime']],
                on=['subject_id', 'hadm_id'],
                how='left'
            )
            # Keep only lab events that fall within an ICU stay timeframe
            mask = (
                labevents_merged['icustay_id'].notna() &
                (labevents_merged['charttime'] >= labevents_merged['intime']) &
                (labevents_merged['charttime'] <= labevents_merged['outtime'])
            )
            self.labevents = labevents_merged[mask].drop(columns=['intime', 'outtime']).copy()
            logger.info(f"Assigned icustay_id to {len(self.labevents)} lab events within ICU stays")
        else:
            logger.warning("ICU stays not loaded yet — labevents will lack icustay_id")
        
        return self.labevents
    
    def load_diagnoses(self) -> pd.DataFrame:
        """Load ICD-9 diagnosis codes"""
        logger.info("Loading DIAGNOSES_ICD...")
        filepath = os.path.join(self.data_dir, 'DIAGNOSES_ICD.csv')
        self.diagnoses = pd.read_csv(filepath)
        
        logger.info(f"Loaded {len(self.diagnoses)} diagnoses")
        return self.diagnoses
    
    def load_prescriptions(self) -> pd.DataFrame:
        """Load medication prescriptions"""
        logger.info("Loading PRESCRIPTIONS...")
        filepath = os.path.join(self.data_dir, 'PRESCRIPTIONS.csv')
        self.prescriptions = pd.read_csv(filepath)
        
        # Convert timestamps
        time_cols = ['startdate', 'enddate']
        for col in time_cols:
            if col in self.prescriptions.columns:
                self.prescriptions[col] = pd.to_datetime(self.prescriptions[col])
        
        logger.info(f"Loaded {len(self.prescriptions)} prescriptions")
        return self.prescriptions
    
    def load_dictionaries(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data dictionaries for items and lab tests"""
        logger.info("Loading data dictionaries...")
        
        # D_ITEMS - chart event definitions
        filepath = os.path.join(self.data_dir, 'D_ITEMS.csv')
        self.d_items = pd.read_csv(filepath)
        
        # D_LABITEMS - lab test definitions
        filepath = os.path.join(self.data_dir, 'D_LABITEMS.csv')
        self.d_labitems = pd.read_csv(filepath)
        
        logger.info(f"Loaded {len(self.d_items)} item definitions, {len(self.d_labitems)} lab definitions")
        return self.d_items, self.d_labitems
    
    def get_vital_sign_itemids(self) -> Dict[str, List[int]]:
        """
        Map vital sign names to itemids using D_ITEMS dictionary
        
        Returns:
            Dictionary mapping vital names to list of itemids
        """
        if self.d_items is None:
            self.load_dictionaries()
        
        # Mapping of vital signs to search keywords
        vital_mapping = {
            'heartrate': ['heart rate', 'hr'],
            'sysbp': ['systolic', 'nbp systolic', 'arterial bp systolic'],
            'diasbp': ['diastolic', 'nbp diastolic', 'arterial bp diastolic'],
            'meanbp': ['mean', 'nbp mean', 'arterial bp mean'],
            'resprate': ['respiratory rate', 'resp rate', 'rr'],
            'tempc': ['temperature c', 'temperature celsius'],
            'spo2': ['spo2', 'o2 saturation'],
            'glucose': ['glucose', 'fingerstick glucose']
        }
        
        itemid_map = {}
        for vital_name, keywords in vital_mapping.items():
            # Search for items matching keywords
            mask = self.d_items['label'].str.lower().str.contains('|'.join(keywords), na=False, case=False)
            itemids = self.d_items[mask]['itemid'].unique().tolist()
            itemid_map[vital_name] = itemids
        
        return itemid_map
    
    def get_lab_itemids(self) -> Dict[str, List[int]]:
        """
        Map lab test names to itemids using D_LABITEMS dictionary
        
        Returns:
            Dictionary mapping lab names to list of itemids
        """
        if self.d_labitems is None:
            self.load_dictionaries()
        
        # Lab tests from config
        lab_tests = self.config.get('LAB_TESTS', [])
        
        itemid_map = {}
        for lab_name in lab_tests:
            # Search for items matching lab name
            mask = self.d_labitems['label'].str.lower().str.contains(lab_name, na=False, case=False)
            itemids = self.d_labitems[mask]['itemid'].unique().tolist()
            itemid_map[lab_name] = itemids
        
        return itemid_map
    
    def merge_data(self, load_chart_sample: Optional[int] = None) -> pd.DataFrame:
        """
        Load and merge all relevant tables
        
        Args:
            load_chart_sample: Optional number of chart events to sample
            
        Returns:
            Merged dataframe with patients, ICU stays, and demographics
        """
        logger.info("Starting data merge pipeline...")
        
        # Load all tables (ICU stays must be loaded before labevents for icustay_id join)
        self.load_patients()
        self.load_admissions()
        self.load_icu_stays()
        self.load_chartevents(sample_n=load_chart_sample)
        self.load_labevents()  # Requires icu_stays to be loaded first
        self.load_diagnoses()
        self.load_prescriptions()
        self.load_dictionaries()
        
        # Merge patients with ICU stays
        logger.info("Merging patients with ICU stays...")
        merged = self.icu_stays.merge(
            self.patients[['subject_id', 'gender', 'dob', 'dod', 'expire_flag']],
            on='subject_id',
            how='left'
        )
        
        # Merge with admissions
        logger.info("Merging with admissions...")
        merged = merged.merge(
            self.admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime', 
                           'deathtime', 'admission_type', 'diagnosis']],
            on=['subject_id', 'hadm_id'],
            how='left'
        )
        
        # Calculate age at ICU admission (handle overflow for invalid dates)
        try:
            merged['age'] = (merged['intime'] - merged['dob']).dt.total_seconds() / (365.25 * 24 * 3600)
        except OverflowError:
            # Calculate age row by row for problematic dates
            def safe_age_calc(row):
                try:
                    return (row['intime'] - row['dob']).total_seconds() / (365.25 * 24 * 3600)
                except:
                    return np.nan
            merged['age'] = merged.apply(safe_age_calc, axis=1)
        
        # Calculate time to death (if applicable)
        try:
            merged['hours_to_death'] = (merged['dod'] - merged['intime']).dt.total_seconds() / 3600
        except:
            def safe_death_calc(row):
                try:
                    if pd.isna(row['dod']):
                        return np.nan
                    return (row['dod'] - row['intime']).total_seconds() / 3600
                except:
                    return np.nan
            merged['hours_to_death'] = merged.apply(safe_death_calc, axis=1)
        
        logger.info(f"Merged dataset: {len(merged)} ICU stays")
        logger.info(f"  - Unique patients: {merged['subject_id'].nunique()}")
        logger.info(f"  - Deceased patients: {merged['expire_flag'].sum()}")
        logger.info(f"  - Mean age: {merged['age'].mean():.1f} years")
        
        return merged
    
    def get_patient_timeseries(self, icustay_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get time-series data for a specific ICU stay
        
        Args:
            icustay_id: ICU stay identifier
            
        Returns:
            Tuple of (chartevents_df, labevents_df) for this stay
        """
        charts = self.chartevents[self.chartevents['icustay_id'] == icustay_id].copy()
        labs = self.labevents[self.labevents['icustay_id'] == icustay_id].copy()
        
        # Sort by time
        charts = charts.sort_values('charttime')
        labs = labs.sort_values('charttime')
        
        return charts, labs


if __name__ == "__main__":
    # Test the data loader
    loader = MIMICDataLoader('demo', 'config.yaml')
    
    # Load and merge data
    merged_data = loader.merge_data()
    
    print("\n=== Data Loader Test ===")
    print(f"Loaded {len(merged_data)} ICU stays")
    print(f"\nColumns: {list(merged_data.columns)}")
    print(f"\nFirst 3 rows:\n{merged_data.head(3)}")
    
    # Test vital sign mapping
    vital_itemids = loader.get_vital_sign_itemids()
    print(f"\n=== Vital Sign Itemids ===")
    for vital, itemids in vital_itemids.items():
        print(f"{vital}: {len(itemids)} items")
    
    # Test lab mapping
    lab_itemids = loader.get_lab_itemids()
    print(f"\n=== Lab Test Itemids ===")
    for lab, itemids in lab_itemids.items():
        print(f"{lab}: {len(itemids)} items")
