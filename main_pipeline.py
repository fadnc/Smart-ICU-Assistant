"""
Main Pipeline for Smart ICU Assistant
End-to-end orchestration of data loading, feature engineering, label generation, and model training
"""

import os
import sys
import yaml
import logging
import argparse
import pickle
import json
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd

# Import pipeline components
from data_loader import MIMICDataLoader
from feature_engineering import FeatureEngineer
from label_generation import LabelGenerator
from models import LSTMModel, TCNModel, XGBoostPredictor, create_model
from training import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmartICUPipeline:
    """End-to-end pipeline for Smart ICU Assistant"""
    
    def __init__(self, config_path: str = 'config.yaml', data_dir: str = 'demo'):
        """
        Initialize pipeline
        
        Args:
            config_path: Path to configuration file
            data_dir: Path to MIMIC-III data directory
        """
        self.config_path = config_path
        self.data_dir = data_dir
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create output directories
        self.model_dir = self.config.get('MODEL_DIR', 'models')
        self.output_dir = self.config.get('OUTPUT_DIR', 'output')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.data_loader = None
        self.feature_engineer = None
        self.label_generator = None
        
        logger.info("Smart ICU Pipeline initialized")
    
    def load_data(self):
        """Step 1: Load MIMIC-III data"""
        logger.info("=" * 60)
        logger.info("STEP 1: Loading MIMIC-III Data")
        logger.info("=" * 60)
        
        self.data_loader = MIMICDataLoader(self.data_dir, self.config_path)
        self.merged_data = self.data_loader.merge_data()
        
        logger.info(f"✓ Loaded {len(self.merged_data)} ICU stays")
        logger.info(f"  - {self.merged_data['subject_id'].nunique()} unique patients")
        logger.info(f"  - {self.merged_data['expire_flag'].sum()} deceased patients")
        
        return self.merged_data
    
    def extract_features_and_labels(self, sample_size: int = None):
        """Step 2 & 3: Extract features and generate labels"""
        logger.info("=" * 60)
        logger.info("STEP 2 & 3: Feature Engineering & Label Generation")
        logger.info("=" * 60)
        
        self.feature_engineer = FeatureEngineer(self.config_path)
        self.label_generator = LabelGenerator(self.config_path)
        
        # Process each ICU stay
        all_sequences = []
        all_labels = []
        all_timestamps = []
        
        stays_to_process = self.merged_data if sample_size is None else self.merged_data.head(sample_size)
        
        logger.info(f"Processing {len(stays_to_process)} ICU stays...")
        
        for idx, stay in stays_to_process.iterrows():
            try:
                icustay_id = stay['icustay_id']
                
                # Extract features for this stay
                features = self.feature_engineer.extract_features_for_stay(
                    icustay_id=icustay_id,
                    icu_intime=stay['intime'],
                    icu_outtime=stay['outtime'],
                    chartevents=self.data_loader.chartevents,
                    labevents=self.data_loader.labevents,
                    d_items=self.data_loader.d_items,
                    d_labitems=self.data_loader.d_labitems,
                    window_hours=6
                )
                
                if len(features) == 0:
                    continue
                
                # Create sequences
                sequences, timestamps = self.feature_engineer.create_sequences(
                    features, sequence_length=24, step_size=6
                )
                
                if len(sequences) == 0:
                    continue
                
                # Generate labels for each sequence
                for seq_idx, timestamp in enumerate(timestamps):
                    # Filter features for this timestamp
                    vital_cols = [c for c in features.columns if any(v in c for v in ['heartrate', 'bp', 'temp', 'spo2', 'resp'])]
                    lab_cols = [c for c in features.columns if any(l in c for l in ['lactate', 'creatinine', 'wbc'])]
                    
                    vitals_df = features[vital_cols] if vital_cols else pd.DataFrame()
                    labs_df = features[lab_cols] if lab_cols else pd.DataFrame()
                    
                    # Get prescriptions and diagnoses
                    stay_prescriptions = self.data_loader.prescriptions[
                        self.data_loader.prescriptions['subject_id'] == stay['subject_id']
                    ] if 'subject_id' in self.data_loader.prescriptions.columns else pd.DataFrame()
                    
                    stay_diagnoses = self.data_loader.diagnoses[
                        self.data_loader.diagnoses['hadm_id'] == stay['hadm_id']
                    ]
                    
                    # Generate labels
                    try:
                        labels = self.label_generator.generate_all_labels(
                            icu_stay_data=stay,
                            vitals=vitals_df,
                            labs=labs_df,
                            prescriptions=stay_prescriptions,
                            diagnoses=stay_diagnoses,
                            current_time=timestamp
                        )
                    except Exception as label_err:
                        logger.debug(f"Label generation error for stay {icustay_id} at {timestamp}: {label_err}")
                        # Use all-zeros as fallback labels
                        label_names = [f'mortality_{w}h' for w in [6,12,24]] + \
                                      [f'sepsis_{w}h' for w in [6,12,24]] + \
                                      [f'aki_stage{s}_{w}h' for w in [24,48] for s in [1,2,3]] + \
                                      [f'hypotension_{w}h' for w in [1,3,6]] + \
                                      [f'vasopressor_{w}h' for w in [6,12]] + \
                                      [f'ventilation_{w}h' for w in [6,12,24]]
                        labels = {name: 0 for name in label_names}
                    
                    # Store
                    all_sequences.append(sequences[seq_idx])
                    all_labels.append(list(labels.values()))
                    all_timestamps.append(timestamp)
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"  Processed {idx + 1}/{len(stays_to_process)} stays, {len(all_sequences)} sequences collected")
            
            except Exception as e:
                logger.debug(f"Error processing stay {stay.get('icustay_id', 'unknown')}: {str(e)}")
                continue
        
        # Convert to arrays — pad sequences to uniform feature count
        if len(all_sequences) == 0:
            logger.warning("No sequences collected!")
            return np.array([]), np.array([]), []
        
        # Find max feature dimension across all sequences
        max_features = max(seq.shape[1] for seq in all_sequences)
        
        # Pad sequences with zeros to match max_features
        padded_sequences = []
        for seq in all_sequences:
            if seq.shape[1] < max_features:
                padding = np.zeros((seq.shape[0], max_features - seq.shape[1]))
                seq = np.hstack([seq, padding])
            padded_sequences.append(seq)
        
        X = np.array(padded_sequences)
        y = np.array(all_labels)
        
        logger.info(f"✓ Feature extraction complete")
        logger.info(f"  - Sequences shape: {X.shape}")
        logger.info(f"  - Labels shape: {y.shape}")
        logger.info(f"  - Label count per sample: {y.shape[1]}")
        
        return X, y, all_timestamps
    
    def train_models(self, X: np.ndarray, y: np.ndarray, timestamps: List):
        """Step 4: Train all models"""
        logger.info("=" * 60)
        logger.info("STEP 4: Model Training")
        logger.info("=" * 60)
        
        results = {}
        
        # Get feature size and number of tasks
        input_size = X.shape[2]
        num_tasks = y.shape[1]
        
        # Update config with dynamic values
        self.config['input_size'] = input_size
        self.config['num_tasks'] = num_tasks
        
        # Train LSTM
        logger.info("\n--- Training LSTM Model ---")
        lstm_model = create_model('lstm', self.config)
        lstm_trainer = ModelTrainer(lstm_model, self.config)
        
        (train_X, train_y), (val_X, val_y), (test_X, test_y) = lstm_trainer.temporal_split(
            X, y, timestamps
        )
        
        lstm_history = lstm_trainer.train(train_X, train_y, val_X, val_y, verbose=True)
        
        # Evaluate on test set
        test_pred_lstm = lstm_trainer.predict(test_X)
        test_metrics_lstm = lstm_trainer.compute_metrics(test_pred_lstm, test_y)
        
        results['lstm'] = {
            'model': lstm_model,
            'trainer': lstm_trainer,
            'history': lstm_history,
            'test_metrics': test_metrics_lstm,
            'test_predictions': test_pred_lstm
        }
        
        logger.info(f"✓ LSTM - Test AUROC: {test_metrics_lstm['mean_auroc']:.4f}")
        
        # Train TCN
        logger.info("\n--- Training TCN Model ---")
        tcn_model = create_model('tcn', self.config)
        tcn_trainer = ModelTrainer(tcn_model, self.config)
        
        tcn_history = tcn_trainer.train(train_X, train_y, val_X, val_y, verbose=True)
        
        test_pred_tcn = tcn_trainer.predict(test_X)
        test_metrics_tcn = tcn_trainer.compute_metrics(test_pred_tcn, test_y)
        
        results['tcn'] = {
            'model': tcn_model,
            'trainer': tcn_trainer,
            'history': tcn_history,
            'test_metrics': test_metrics_tcn,
            'test_predictions': test_pred_tcn
        }
        
        logger.info(f"✓ TCN - Test AUROC: {test_metrics_tcn['mean_auroc']:.4f}")
        
        # Train XGBoost
        logger.info("\n--- Training XGBoost Model ---")
        xgb_model = XGBoostPredictor(
            num_tasks=num_tasks,
            **self.config.get('XGBOOST_CONFIG', {})
        )
        
        xgb_model.fit(train_X, train_y, verbose=True)
        test_pred_xgb = xgb_model.predict_proba(test_X)
        
        dummy_trainer = ModelTrainer(LSTMModel(1, 1, 1), self.config)
        test_metrics_xgb = dummy_trainer.compute_metrics(test_pred_xgb, test_y)
        
        results['xgboost'] = {
            'model': xgb_model,
            'test_metrics': test_metrics_xgb,
            'test_predictions': test_pred_xgb
        }
        
        logger.info(f"✓ XGBoost - Test AUROC: {test_metrics_xgb['mean_auroc']:.4f}")
        
        # Store test data
        results['test_data'] = {
            'X': test_X,
            'y': test_y
        }
        
        return results
    
    def save_results(self, results: Dict):
        """Step 5: Save models and results"""
        logger.info("=" * 60)
        logger.info("STEP 5: Saving Results")
        logger.info("=" * 60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        for model_name in ['lstm', 'tcn']:
            if model_name in results:
                model_path = os.path.join(self.model_dir, f'{model_name}_model_{timestamp}.pth')
                results[model_name]['trainer'].save_checkpoint(model_path)
                logger.info(f"✓ Saved {model_name.upper()} model to {model_path}")
        
        # Save XGBoost
        if 'xgboost' in results:
            xgb_path = os.path.join(self.model_dir, f'xgboost_model_{timestamp}.pkl')
            with open(xgb_path, 'wb') as f:
                pickle.dump(results['xgboost']['model'], f)
            logger.info(f"✓ Saved XGBoost model to {xgb_path}")
        
        # Save metrics report
        metrics_report = {
            'timestamp': timestamp,
            'config': self.config,
            'models': {}
        }
        
        for model_name in ['lstm', 'tcn', 'xgboost']:
            if model_name in results:
                metrics_report['models'][model_name] = {
                    'test_auroc': results[model_name]['test_metrics']['auroc'],
                    'test_auprc': results[model_name]['test_metrics']['auprc'],
                    'test_brier': results[model_name]['test_metrics']['brier'],
                    'mean_auroc': results[model_name]['test_metrics']['mean_auroc'],
                    'mean_auprc': results[model_name]['test_metrics']['mean_auprc'],
                    'mean_brier': results[model_name]['test_metrics']['mean_brier']
                }
        
        report_path = os.path.join(self.output_dir, f'metrics_report_{timestamp}.json')
        with open(report_path, 'w') as f:
            json.dump(metrics_report, f, indent=2, default=str)
        
        logger.info(f"✓ Saved metrics report to {report_path}")
        
        return report_path
    
    def run(self, sample_size: int = None):
        """Run complete pipeline"""
        logger.info("\n" + "=" * 60)
        logger.info("SMART ICU ASSISTANT - TRAINING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Start time: {datetime.now()}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Config file: {self.config_path}")
        if sample_size:
            logger.info(f"Sample size: {sample_size} ICU stays")
        logger.info("=" * 60 + "\n")
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2-3: Extract features and labels
            X, y, timestamps = self.extract_features_and_labels(sample_size=sample_size)
            
            # Check if we have enough data
            if len(X) < 10:
                logger.error("Insufficient data for training (< 10 sequences)")
                return None
            
            # Step 4: Train models
            results = self.train_models(X, y, timestamps)
            
            # Step 5: Save results
            report_path = self.save_results(results)
            
            logger.info("\n" + "=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"End time: {datetime.now()}")
            logger.info(f"Results saved to: {report_path}")
            logger.info("=" * 60 + "\n")
            
            # Print summary
            self.print_summary(results)
            
            return results
        
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
            raise
    
    def print_summary(self, results: Dict):
        """Print results summary"""
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 60)
        
        for model_name in ['lstm', 'tcn', 'xgboost']:
            if model_name in results:
                metrics = results[model_name]['test_metrics']
                logger.info(f"\n{model_name.upper()} Model:")
                logger.info(f"  Mean AUROC: {metrics['mean_auroc']:.4f}")
                logger.info(f"  Mean AUPRC: {metrics['mean_auprc']:.4f}")
                logger.info(f"  Mean Brier: {metrics['mean_brier']:.4f}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Smart ICU Assistant Training Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='demo', help='Path to MIMIC-III data')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of ICU stays to process (for testing)')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = SmartICUPipeline(
        config_path=args.config,
        data_dir=args.data_dir
    )
    
    results = pipeline.run(sample_size=args.sample_size)
    
    return results


if __name__ == "__main__":
    main()
