"""
Feature Engineering for MIMIC-III ICU Data
Extracts and processes vital signs, lab tests, and creates temporal features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import timedelta
import yaml
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Extract and engineer features from MIMIC-III time-series data"""
    
    def __init__(self, config_path: str):
        """
        Initialize feature engineer
        
        Args:
            config_path: Path to config.yaml
        """
        self.config = self._load_config(config_path)
        self.scaler = StandardScaler()
        
        # Feature names
        self.vital_features = self.config.get('VITAL_SIGNS', [])
        self.lab_features = self.config.get('LAB_TESTS', [])
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def extract_vital_signs(self, 
                           chartevents: pd.DataFrame, 
                           d_items: pd.DataFrame,
                           icustay_id: int,
                           start_time: pd.Timestamp,
                           end_time: pd.Timestamp) -> pd.DataFrame:
        """
        Extract vital signs for a specific ICU stay and time window
        
        Args:
            chartevents: Chart events dataframe
            d_items: Items dictionary
            icustay_id: ICU stay identifier
            start_time: Window start time
            end_time: Window end time
            
        Returns:
            DataFrame with vital signs indexed by charttime
        """
        # Filter for this ICU stay and time window
        mask = (
            (chartevents['icustay_id'] == icustay_id) &
            (chartevents['charttime'] >= start_time) &
            (chartevents['charttime'] <= end_time)
        )
        stay_charts = chartevents[mask].copy()
        
        if len(stay_charts) == 0:
            return pd.DataFrame()
        
        # Map itemids to vital names
        vital_mapping = self._create_vital_itemid_mapping(d_items)
        
        # Extract vitals
        vitals_list = []
        for vital_name, itemids in vital_mapping.items():
            vital_data = stay_charts[stay_charts['itemid'].isin(itemids)].copy()
            if len(vital_data) > 0:
                vital_data['vital_name'] = vital_name
                vitals_list.append(vital_data[['charttime', 'vital_name', 'valuenum']])
        
        if not vitals_list:
            return pd.DataFrame()
        
        # Combine all vitals
        vitals = pd.concat(vitals_list, ignore_index=True)
        
        # Pivot to wide format
        vitals_wide = vitals.pivot_table(
            index='charttime',
            columns='vital_name',
            values='valuenum',
            aggfunc='mean'  # Average if multiple readings at same time
        )
        
        return vitals_wide
    
    def extract_lab_tests(self,
                         labevents: pd.DataFrame,
                         d_labitems: pd.DataFrame,
                         icustay_id: int,
                         start_time: pd.Timestamp,
                         end_time: pd.Timestamp) -> pd.DataFrame:
        """
        Extract lab test results for a specific ICU stay and time window
        
        Args:
            labevents: Lab events dataframe
            d_labitems: Lab items dictionary
            icustay_id: ICU stay identifier
            start_time: Window start time
            end_time: Window end time
            
        Returns:
            DataFrame with lab values indexed by charttime
        """
        # Filter for this ICU stay and time window
        mask = (
            (labevents['icustay_id'] == icustay_id) &
            (labevents['charttime'] >= start_time) &
            (labevents['charttime'] <= end_time)
        )
        stay_labs = labevents[mask].copy()
        
        if len(stay_labs) == 0:
            return pd.DataFrame()
        
        # Map itemids to lab names
        lab_mapping = self._create_lab_itemid_mapping(d_labitems)
        
        # Extract labs
        labs_list = []
        for lab_name, itemids in lab_mapping.items():
            lab_data = stay_labs[stay_labs['itemid'].isin(itemids)].copy()
            if len(lab_data) > 0:
                lab_data['lab_name'] = lab_name
                labs_list.append(lab_data[['charttime', 'lab_name', 'valuenum']])
        
        if not labs_list:
            return pd.DataFrame()
        
        # Combine all labs
        labs = pd.concat(labs_list, ignore_index=True)
        
        # Pivot to wide format
        labs_wide = labs.pivot_table(
            index='charttime',
            columns='lab_name',
            values='valuenum',
            aggfunc='mean'
        )
        
        return labs_wide
    
    def _create_vital_itemid_mapping(self, d_items: pd.DataFrame) -> Dict[str, List[int]]:
        """Create mapping of vital names to itemids"""
        vital_keywords = {
            'heartrate': ['heart rate', 'hr'],
            'sysbp': ['systolic', 'nbp systolic', 'arterial bp systolic'],
            'diasbp': ['diastolic', 'nbp diastolic', 'arterial bp diastolic'],
            'meanbp': ['mean', 'nbp mean', 'arterial bp mean'],
            'resprate': ['respiratory rate', 'resp rate', 'rr'],
            'tempc': ['temperature c', 'temperature celsius', 'temp c'],
            'spo2': ['spo2', 'o2 saturation'],
            'glucose': ['glucose', 'fingerstick glucose']
        }
        
        mapping = {}
        for vital_name, keywords in vital_keywords.items():
            pattern = '|'.join(keywords)
            mask = d_items['label'].str.lower().str.contains(pattern, na=False, case=False)
            itemids = d_items[mask]['itemid'].unique().tolist()
            if itemids:
                mapping[vital_name] = itemids
        
        return mapping
    
    def _create_lab_itemid_mapping(self, d_labitems: pd.DataFrame) -> Dict[str, List[int]]:
        """Create mapping of lab names to itemids"""
        lab_tests = self.lab_features
        
        mapping = {}
        for lab_name in lab_tests:
            mask = d_labitems['label'].str.lower().str.contains(lab_name, na=False, case=False)
            itemids = d_labitems[mask]['itemid'].unique().tolist()
            if itemids:
                mapping[lab_name] = itemids
        
        return mapping
    
    def create_time_windows(self, 
                           timeseries: pd.DataFrame,
                           window_hours: int) -> pd.DataFrame:
        """
        Create aggregated features over time windows
        
        Args:
            timeseries: DataFrame with time-indexed features
            window_hours: Window size in hours
            
        Returns:
            DataFrame with aggregated statistics (mean, std, min, max, trend)
        """
        if len(timeseries) == 0:
            return pd.DataFrame()
        
        # Resample to hourly intervals and forward fill
        hourly = timeseries.resample('1h').mean()
        hourly = hourly.ffill(limit=24)  # Forward fill up to 24 hours
        
        # Rolling window aggregations
        window_size = window_hours
        
        features = {}
        for col in hourly.columns:
            # Mean
            features[f'{col}_mean_{window_hours}h'] = hourly[col].rolling(
                window=window_size, min_periods=1).mean()
            
            # Standard deviation
            features[f'{col}_std_{window_hours}h'] = hourly[col].rolling(
                window=window_size, min_periods=1).std()
            
            # Min
            features[f'{col}_min_{window_hours}h'] = hourly[col].rolling(
                window=window_size, min_periods=1).min()
            
            # Max
            features[f'{col}_max_{window_hours}h'] = hourly[col].rolling(
                window=window_size, min_periods=1).max()
            
            # Trend (slope of linear regression over window)
            def calc_trend(x):
                if len(x) < 2:
                    return 0
                x_clean = x.dropna()
                if len(x_clean) < 2:
                    return 0
                return np.polyfit(range(len(x_clean)), x_clean, 1)[0]
            
            features[f'{col}_trend_{window_hours}h'] = hourly[col].rolling(
                window=window_size, min_periods=2).apply(calc_trend, raw=False)
        
        result = pd.DataFrame(features, index=hourly.index)
        return result
    
    def compute_derived_features(self, vitals: pd.DataFrame) -> pd.DataFrame:
        """
        Compute clinical derived features
        
        Args:
            vitals: DataFrame with vital signs
            
        Returns:
            DataFrame with derived features added
        """
        derived = vitals.copy()
        
        # Mean Arterial Pressure (if not already present)
        if 'meanbp' not in derived.columns and 'sysbp' in derived.columns and 'diasbp' in derived.columns:
            derived['meanbp'] = (derived['sysbp'] + 2 * derived['diasbp']) / 3
        
        # Shock Index (HR / SBP)
        if 'heartrate' in derived.columns and 'sysbp' in derived.columns:
            derived['shock_index'] = derived['heartrate'] / derived['sysbp'].replace(0, np.nan)
        
        # Pulse Pressure
        if 'sysbp' in derived.columns and 'diasbp' in derived.columns:
            derived['pulse_pressure'] = derived['sysbp'] - derived['diasbp']
        
        return derived
    
    def compute_sirs_score(self, vitals: pd.DataFrame, labs: pd.DataFrame) -> pd.Series:
        """
        Compute SIRS (Systemic Inflammatory Response Syndrome) score
        
        SIRS criteria:
        - Temperature > 38°C or < 36°C
        - Heart rate > 90 bpm
        - Respiratory rate > 20
        - WBC > 12,000 or < 4,000
        
        Args:
            vitals: Vital signs DataFrame
            labs: Lab tests DataFrame
            
        Returns:
            Series with SIRS score (0-4)
        """
        # Align vitals and labs on time index
        combined = pd.concat([vitals, labs], axis=1)
        
        sirs_score = pd.Series(0, index=combined.index)
        
        # Temperature criterion
        if 'tempc' in combined.columns:
            temp_abnormal = (combined['tempc'] > 38.3) | (combined['tempc'] < 36.0)
            sirs_score += temp_abnormal.fillna(False).astype(int)
        
        # Heart rate criterion
        if 'heartrate' in combined.columns:
            hr_abnormal = combined['heartrate'] > 90
            sirs_score += hr_abnormal.fillna(False).astype(int)
        
        # Respiratory rate criterion
        if 'resprate' in combined.columns:
            rr_abnormal = combined['resprate'] > 20
            sirs_score += rr_abnormal.fillna(False).astype(int)
        
        # WBC criterion
        if 'wbc' in combined.columns:
            wbc_abnormal = (combined['wbc'] > 12) | (combined['wbc'] < 4)
            sirs_score += wbc_abnormal.fillna(False).astype(int)
        
        return sirs_score
    
    def create_sequences(self,
                        timeseries: pd.DataFrame,
                        sequence_length: int = 24,
                        step_size: int = 1) -> Tuple[np.ndarray, List[pd.Timestamp]]:
        """
        Create fixed-length sequences for LSTM/TCN models
        
        Args:
            timeseries: Time-indexed features DataFrame
            sequence_length: Number of timesteps per sequence
            step_size: Step size for sliding window (in hours)
            
        Returns:
            Tuple of (sequences_array, timestamps_list)
            - sequences_array: shape [n_sequences, sequence_length, n_features]
            - timestamps_list: List of end timestamps for each sequence
        """
        if len(timeseries) == 0:
            return np.array([]), []
        
        # Resample to hourly
        hourly = timeseries.resample('1h').mean()
        hourly = hourly.ffill(limit=24)
        
        # Fill remaining NaNs with 0
        hourly = hourly.fillna(0)
        
        # Create sequences using sliding window
        sequences = []
        timestamps = []
        
        for i in range(0, len(hourly) - sequence_length + 1, step_size):
            seq = hourly.iloc[i:i+sequence_length].values
            sequences.append(seq)
            timestamps.append(hourly.index[i + sequence_length - 1])
        
        if not sequences:
            return np.array([]), []
        
        sequences_array = np.array(sequences)  # Shape: [n_sequences, seq_len, n_features]
        
        return sequences_array, timestamps
    
    def normalize_features(self, features: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize features using StandardScaler
        
        Args:
            features: Feature array [n_samples, n_features] or [n_samples, seq_len, n_features]
            fit: Whether to fit scaler (True for train, False for test)
            
        Returns:
            Normalized features
        """
        original_shape = features.shape
        
        # Reshape to 2D if needed
        if len(original_shape) == 3:
            n_samples, seq_len, n_features = original_shape
            features_2d = features.reshape(-1, n_features)
        else:
            features_2d = features
        
        # Normalize
        if fit:
            normalized = self.scaler.fit_transform(features_2d)
        else:
            normalized = self.scaler.transform(features_2d)
        
        # Reshape back to original
        if len(original_shape) == 3:
            normalized = normalized.reshape(original_shape)
        
        return normalized
    
    def extract_features_for_stay(self,
                                  icustay_id: int,
                                  icu_intime: pd.Timestamp,
                                  icu_outtime: pd.Timestamp,
                                  chartevents: pd.DataFrame,
                                  labevents: pd.DataFrame,
                                  d_items: pd.DataFrame,
                                  d_labitems: pd.DataFrame,
                                  window_hours: int = 24) -> pd.DataFrame:
        """
        Extract all features for a single ICU stay
        
        Args:
            icustay_id: ICU stay identifier
            icu_intime: ICU admission time
            icu_outtime: ICU discharge time
            chartevents: All chart events
            labevents: All lab events
            d_items: Items dictionary
            d_labitems: Lab items dictionary
            window_hours: Time window for aggregations
            
        Returns:
            DataFrame with all engineered features
        """
        # Extract vitals and labs
        vitals = self.extract_vital_signs(
            chartevents, d_items, icustay_id, icu_intime, icu_outtime
        )
        
        labs = self.extract_lab_tests(
            labevents, d_labitems, icustay_id, icu_intime, icu_outtime
        )
        
        if len(vitals) == 0 and len(labs) == 0:
            return pd.DataFrame()
        
        # Compute derived features
        if len(vitals) > 0:
            vitals = self.compute_derived_features(vitals)
        
        # Combine vitals and labs
        combined = pd.concat([vitals, labs], axis=1)
        
        # Create time-windowed features
        windowed = self.create_time_windows(combined, window_hours)
        
        # Add SIRS score
        if len(vitals) > 0 and len(labs) > 0:
            sirs = self.compute_sirs_score(vitals, labs)
            windowed['sirs_score'] = sirs
        
        return windowed


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import MIMICDataLoader
    
    logger.info("Testing Feature Engineering...")
    
    # Load data
    loader = MIMICDataLoader('demo', 'config.yaml')
    merged = loader.merge_data()
    
    # Initialize feature engineer
    fe = FeatureEngineer('config.yaml')
    
    # Test on first ICU stay
    first_stay = merged.iloc[0]
    icustay_id = first_stay['icustay_id']
    
    logger.info(f"\nExtracting features for ICU stay {icustay_id}...")
    
    features = fe.extract_features_for_stay(
        icustay_id=icustay_id,
        icu_intime=first_stay['intime'],
        icu_outtime=first_stay['outtime'],
        chartevents=loader.chartevents,
        labevents=loader.labevents,
        d_items=loader.d_items,
        d_labitems=loader.d_labitems,
        window_hours=6
    )
    
    print(f"\n=== Feature Engineering Test ===")
    print(f"Features shape: {features.shape}")
    print(f"\nFeature columns ({len(features.columns)}):")
    print(list(features.columns)[:20])
    print(f"\nFirst 3 rows:\n{features.head(3)}")
    
    # Test sequence creation
    if len(features) > 0:
        sequences, timestamps = fe.create_sequences(features, sequence_length=24, step_size=6)
        print(f"\n=== Sequence Creation ===")
        print(f"Sequences shape: {sequences.shape}")
        print(f"Number of timestamps: {len(timestamps)}")
