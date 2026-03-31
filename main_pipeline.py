"""
Main Pipeline for Smart ICU Assistant
Orchestrates all prediction tasks using modular predictors.

FIXES:
  - set_label_indices() now called BEFORE train_all_predictors(), not after.
    Previously DL models received the full 19-column label matrix instead of
    their task-specific columns, causing shape mismatches and AUROC = 0.
  - Progress logging fixed: reports every 10% correctly including 10%, 60-100%.
  - XGBoost deprecated params (gpu_id, tree_method=gpu_hist, use_label_encoder)
    updated to XGBoost 2.x API (device='cuda', tree_method='hist').
"""

import os
import sys
import yaml
import logging
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime
from tqdm import tqdm

from data_loader import MIMICDataLoader
from feature_engineering import FeatureEngineer
from predictors import (
    MortalityPredictor,
    SepsisPredictor,
    AKIPredictor,
    VasopressorPredictor,
    VentilationPredictor,
    ReadmissionPredictor,
    LOSPredictor,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmartICUPipeline:
    """End-to-end pipeline for Smart ICU Assistant using modular predictors."""

    CACHE_DIR = 'output'
    CACHE_FILE_X = 'feature_cache_X.npy'
    CACHE_FILE_Y = 'feature_cache_y.npy'
    CACHE_FILE_META = 'feature_cache_meta.pkl'

    def __init__(self, config_path: str = 'config.yaml', data_dir: str = 'data'):
        self.config_path = config_path
        self.data_dir = data_dir

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_loader = MIMICDataLoader(data_dir, config_path)
        self.feature_engineer = FeatureEngineer(config_path)

        self.predictors = {
            'mortality':   MortalityPredictor(config_path),
            'sepsis':      SepsisPredictor(config_path),
            'aki':         AKIPredictor(config_path),
            'vasopressor': VasopressorPredictor(config_path),
            'ventilation': VentilationPredictor(config_path),
            'los':         LOSPredictor(config_path),
        }
        self.readmission_predictor = ReadmissionPredictor(config_path)

        self.merged_data = None
        logger.info("Smart ICU Pipeline initialized")

    # ── Step 1: Load Data ──────────────────────────────────────────────────

    def load_data(self):
        logger.info("=" * 60)
        logger.info("STEP 1: Loading MIMIC-III Data")
        logger.info("=" * 60)

        self.merged_data = self.data_loader.merge_data()

        n_stays = len(self.merged_data)
        n_patients = self.merged_data['subject_id'].nunique()
        logger.info(f"✓ Loaded {n_stays} ICU stays ({n_patients} patients)")
        return self.merged_data

    # ── Step 2-3: Features & Labels ────────────────────────────────────────

    def extract_features_and_labels(self, sample_size: int = None):
        """
        Extract feature sequences and generate labels from all predictors.

        Returns:
            Tuple of (X, y, timestamps, label_names)
        """
        logger.info("=" * 60)
        logger.info("STEP 2-3: Feature Engineering & Label Generation")
        logger.info("=" * 60)

        # Build ordered list of all label names
        all_label_names = []
        for predictor in self.predictors.values():
            all_label_names.extend(predictor.get_label_names())

        # FIX: Set label indices NOW, before any training happens.
        # Previously this was called after train_all_predictors(), meaning DL models
        # received the full 19-column label matrix regardless of task.
        for predictor in self.predictors.values():
            predictor.set_label_indices(all_label_names)
        logger.info(f"Label indices set for {len(self.predictors)} predictors "
                    f"({len(all_label_names)} total labels: {all_label_names})")

        all_sequences = []
        all_labels = []
        all_timestamps = []

        stays = self.merged_data if sample_size is None else self.merged_data.head(sample_size)
        n_stays = len(stays)
        logger.info(f"Processing {n_stays} ICU stays...")

        # Pre-index by icustay_id for O(1) lookup
        logger.info("Pre-indexing chartevents and labevents by icustay_id...")
        charts_grouped = dict(tuple(
            self.data_loader.chartevents.groupby('icustay_id')
        )) if self.data_loader.chartevents is not None else {}

        labs_grouped = dict(tuple(
            self.data_loader.labevents.groupby('icustay_id')
        )) if self.data_loader.labevents is not None else {}
        logger.info("Pre-indexing complete.")

        # FIX: Correct progress reporting — every 10% of stays.
        # Previously used idx (DataFrame index, not sequential 0..n), so many
        # milestones never printed. Now uses enumerate() counter.
        report_every = max(1, n_stays // 10)

        for counter, (idx, stay) in enumerate(stays.iterrows()):
            try:
                icustay_id = stay['icustay_id']
                stay_charts = charts_grouped.get(icustay_id, pd.DataFrame())
                stay_labs = labs_grouped.get(icustay_id, pd.DataFrame())

                features = self.feature_engineer.extract_features_for_stay(
                    icustay_id=icustay_id,
                    icu_intime=stay['intime'],
                    icu_outtime=stay['outtime'],
                    chartevents=stay_charts,
                    labevents=stay_labs,
                    d_items=self.data_loader.d_items,
                    d_labitems=self.data_loader.d_labitems,
                    window_hours=self.config.get('LSTM_CONFIG', {}).get('sequence_length', 24)
                )

                if features is None or len(features) == 0:
                    continue

                sequences, timestamps = self.feature_engineer.create_sequences(
                    features, sequence_length=24, step_size=6
                )

                if len(sequences) == 0:
                    continue

                raw_vitals = self.feature_engineer.extract_vital_signs(
                    stay_charts, self.data_loader.d_items,
                    icustay_id, stay['intime'], stay['outtime']
                )
                raw_labs = self.feature_engineer.extract_lab_tests(
                    stay_labs, self.data_loader.d_labitems,
                    icustay_id, stay['intime'], stay['outtime']
                )

                extra_data = self._prepare_extra_data(stay, icustay_id, stay_charts=stay_charts)

                for seq_idx, timestamp in enumerate(timestamps):
                    try:
                        labels = {}
                        for predictor in self.predictors.values():
                            task_labels = predictor.generate_labels(
                                stay, raw_vitals, raw_labs, timestamp, **extra_data
                            )
                            labels.update(task_labels)

                        label_array = [labels.get(name, 0) for name in all_label_names]
                        all_labels.append(label_array)
                        all_sequences.append(sequences[seq_idx].astype(np.float32))
                        all_timestamps.append(timestamp)
                    except Exception as e:
                        logger.debug(f"Label error stay {icustay_id}: {e}")
                        all_labels.append([0] * len(all_label_names))
                        all_sequences.append(sequences[seq_idx].astype(np.float32))
                        all_timestamps.append(timestamp)

            except Exception as e:
                logger.debug(f"Feature extraction error stay {stay.get('icustay_id', '?')}: {e}")
                continue

            # FIX: Use counter (0..n) not idx (DataFrame index) for reliable % reporting
            if (counter + 1) % report_every == 0 or (counter + 1) == n_stays:
                logger.info(
                    f"  Processed {counter + 1}/{n_stays} stays, "
                    f"{len(all_sequences)} sequences collected"
                )

        # Free grouped data before building arrays
        del charts_grouped, labs_grouped
        import gc
        gc.collect()

        logger.info(f"  Building arrays from {len(all_sequences)} sequences...")

        if all_sequences:
            feature_sizes = [seq.shape[1] for seq in all_sequences]
            target_features = max(feature_sizes)
            n_seqs = len(all_sequences)
            seq_len = all_sequences[0].shape[0]

            X = np.zeros((n_seqs, seq_len, target_features), dtype=np.float32)
            for i, seq in enumerate(all_sequences):
                X[i, :, :seq.shape[1]] = seq
            del all_sequences
            gc.collect()
        else:
            X = np.array([], dtype=np.float32)

        y = np.array(all_labels, dtype=np.float32)
        del all_labels
        gc.collect()

        input_size = X.shape[2] if len(X) > 0 else 0

        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        if nan_count > 0 or inf_count > 0:
            logger.info(f"  Cleaning {nan_count:,} NaN and {inf_count:,} inf values → 0")
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        nan_labels = np.isnan(y).sum()
        if nan_labels > 0:
            logger.info(f"  Cleaning {nan_labels:,} NaN labels → 0")
            np.nan_to_num(y, copy=False, nan=0.0)

        mem_gb = X.nbytes / (1024 ** 3)
        logger.info(f"  Padded all sequences to feature size: {input_size}")
        logger.info(f"  Array memory: {mem_gb:.1f} GB (float32)")
        logger.info(f"✓ Feature extraction complete")
        logger.info(f"  Sequences: {X.shape}, Labels: {y.shape}")

        return X, y, all_timestamps, all_label_names

    def _prepare_extra_data(self, stay, icustay_id, stay_charts=None) -> Dict:
        extra = {}

        extra['chartevents'] = stay_charts if stay_charts is not None else pd.DataFrame()

        if hasattr(self.data_loader, 'prescriptions') and self.data_loader.prescriptions is not None:
            pkey = 'subject_id' if 'subject_id' in self.data_loader.prescriptions.columns else None
            if pkey:
                extra['prescriptions'] = self.data_loader.prescriptions[
                    self.data_loader.prescriptions[pkey] == stay.get('subject_id')
                ]
            else:
                extra['prescriptions'] = pd.DataFrame()
        else:
            extra['prescriptions'] = pd.DataFrame()

        if hasattr(self.data_loader, 'diagnoses') and self.data_loader.diagnoses is not None:
            extra['diagnoses'] = self.data_loader.diagnoses[
                self.data_loader.diagnoses['hadm_id'] == stay.get('hadm_id')
            ]
        else:
            extra['diagnoses'] = pd.DataFrame()

        if hasattr(self.data_loader, 'procedureevents') and self.data_loader.procedureevents is not None:
            if len(self.data_loader.procedureevents) > 0 and 'icustay_id' in self.data_loader.procedureevents.columns:
                extra['procedureevents'] = self.data_loader.procedureevents[
                    self.data_loader.procedureevents['icustay_id'] == icustay_id
                ]
            else:
                extra['procedureevents'] = pd.DataFrame()
        else:
            extra['procedureevents'] = pd.DataFrame()

        if hasattr(self.data_loader, 'procedures_icd') and self.data_loader.procedures_icd is not None:
            if len(self.data_loader.procedures_icd) > 0 and 'hadm_id' in self.data_loader.procedures_icd.columns:
                extra['procedures_icd'] = self.data_loader.procedures_icd[
                    self.data_loader.procedures_icd['hadm_id'] == stay.get('hadm_id')
                ]
            else:
                extra['procedures_icd'] = pd.DataFrame()
        else:
            extra['procedures_icd'] = pd.DataFrame()

        if hasattr(self.data_loader, 'inputevents_mv') and self.data_loader.inputevents_mv is not None:
            if len(self.data_loader.inputevents_mv) > 0 and 'icustay_id' in self.data_loader.inputevents_mv.columns:
                extra['inputevents'] = self.data_loader.inputevents_mv[
                    self.data_loader.inputevents_mv['icustay_id'] == icustay_id
                ]
            else:
                extra['inputevents'] = pd.DataFrame()
        else:
            extra['inputevents'] = pd.DataFrame()

        return extra

    # ── Step 4: Train All Predictors ───────────────────────────────────────

    def train_all_predictors(self, X, y, timestamps, label_names, output_dir='output'):
        """
        Train each predictor independently.
        Label indices are already set (done in extract_features_and_labels).
        """
        logger.info("=" * 60)
        logger.info("STEP 4: Training All Predictors (Best Model Selection)")
        logger.info("=" * 60)

        results = {}
        predictors_list = list(self.predictors.items())
        
        # Create progress bar for model training
        with tqdm(total=len(predictors_list), desc="Model Training Progress", unit="model") as pbar:
            for name, predictor in predictors_list:
                logger.info(f"\n{'─' * 50}")
                logger.info(f"Training: {predictor.TASK_DESCRIPTION}")
                logger.info(f"{'─' * 50}")
                task_result = predictor.train_all_models(X, y, timestamps, output_dir)
                results[name] = task_result
                
                # Update progress bar with model name
                pbar.set_postfix({'current': name})
                pbar.update(1)

        return results

    # ── Step 5: Readmission ────────────────────────────────────────────────

    def train_readmission(self, output_dir='output', use_cache=True):
        logger.info("=" * 60)
        logger.info("STEP 5: ICU Readmission Prediction")
        logger.info("=" * 60)

        readmission_cache = self._cache_path('readmission_cache.pkl')

        if use_cache and os.path.exists(readmission_cache):
            logger.info("Loading readmission features from cache...")
            with open(readmission_cache, 'rb') as f:
                cached = pickle.load(f)
            features_df = cached['features_df']
            labels_df = cached['labels_df']
            self.readmission_predictor.feature_names = cached['feature_names']
            logger.info(f"✓ Readmission cache loaded: {len(features_df)} stays")
        else:
            if not hasattr(self, 'merged_data') or self.merged_data is None:
                self.load_data()

            labels_df = self.readmission_predictor.extract_readmission_labels(
                self.data_loader.icu_stays
            )
            features_df = self.readmission_predictor.extract_discharge_features(
                self.merged_data,
                self.data_loader.chartevents,
                self.data_loader.labevents,
                self.data_loader.diagnoses,
                self.data_loader.prescriptions,
                services=getattr(self.data_loader, 'services', None),
                outputevents=getattr(self.data_loader, 'outputevents', None),
            )
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            with open(readmission_cache, 'wb') as f:
                pickle.dump({
                    'features_df': features_df,
                    'labels_df': labels_df,
                    'feature_names': self.readmission_predictor.feature_names,
                }, f)
            logger.info("✓ Readmission features cached")

        result = self.readmission_predictor.train(features_df, labels_df, output_dir)
        return result

    # ── Step 6: Save & Report ──────────────────────────────────────────────

    def save_summary(self, results, readmission_result, output_dir='output'):
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        report = {
            'timestamp': ts,
            'data_dir': self.data_dir,
            'predictor_results': {},
        }

        for name, result in results.items():
            report['predictor_results'][name] = {
                'best_model': result.get('best_model', 'N/A'),
                'best_auroc': result.get('best_auroc', 0),
                'comparison': {
                    m: {'mean_test_auroc': v.get('mean_test_auroc', 0)}
                    for m, v in result.get('comparison', {}).items()
                    if isinstance(v, dict) and 'mean_test_auroc' in v
                },
            }

        report['readmission'] = readmission_result

        report_path = os.path.join(output_dir, f'metrics_report_{ts}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"✓ Saved metrics report to {report_path}")
        return report_path

    # ── Cache helpers ──────────────────────────────────────────────────────

    def _cache_path(self, filename):
        return os.path.join(self.CACHE_DIR, filename)

    def save_feature_cache(self, X, y, timestamps, label_names):
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        logger.info("Saving feature cache to disk...")
        np.save(self._cache_path(self.CACHE_FILE_X), X)
        np.save(self._cache_path(self.CACHE_FILE_Y), y)
        with open(self._cache_path(self.CACHE_FILE_META), 'wb') as f:
            pickle.dump({'timestamps': timestamps, 'label_names': label_names}, f)
        size_gb = os.path.getsize(self._cache_path(self.CACHE_FILE_X)) / (1024 ** 3)
        logger.info(f"✓ Feature cache saved ({size_gb:.1f} GB on disk)")

    def load_feature_cache(self):
        """Load cached features. Returns (X, y, timestamps, label_names) or None."""
        x_path = self._cache_path(self.CACHE_FILE_X)
        y_path = self._cache_path(self.CACHE_FILE_Y)
        meta_path = self._cache_path(self.CACHE_FILE_META)

        if not all(os.path.exists(p) for p in [x_path, y_path, meta_path]):
            return None

        logger.info("Loading feature cache from disk...")
        X = np.array(np.load(x_path, mmap_mode='r'))
        y = np.load(y_path)

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        nan_count = np.isnan(X).sum() + np.isinf(X).sum()
        if nan_count > 0:
            logger.info(f"  Cleaning {nan_count:,} NaN/inf in cached data → 0")
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(y, copy=False, nan=0.0)

        logger.info(
            f"✓ Cache loaded: X={X.shape}, y={y.shape}, "
            f"{len(meta['label_names'])} labels"
        )
        return X, y, meta['timestamps'], meta['label_names']

    # ── Run ────────────────────────────────────────────────────────────────

    def run(self, sample_size: int = None, use_cache: bool = True):
        start = datetime.now()
        logger.info("=" * 60)
        logger.info("SMART ICU ASSISTANT — TRAINING PIPELINE")
        logger.info(f"Start: {start}")
        logger.info(f"Data: {self.data_dir}")
        logger.info("=" * 60)

        cached = None
        if use_cache:
            cached = self.load_feature_cache()

        if cached is not None:
            X, y, timestamps, label_names = cached
            logger.info("⚡ Skipped Steps 1-3 (loaded from cache)")

            # FIX: When loading from cache, label indices must still be set
            # before training. Previously this only happened inside
            # extract_features_and_labels(), so cache runs skipped it entirely.
            for predictor in self.predictors.values():
                predictor.set_label_indices(label_names)
            logger.info(f"Label indices set from cache metadata ({len(label_names)} labels)")
        else:
            self.load_data()
            # label indices are set inside extract_features_and_labels
            X, y, timestamps, label_names = self.extract_features_and_labels(sample_size)
            if len(X) == 0:
                logger.error("No sequences extracted — aborting")
                return
            self.save_feature_cache(X, y, timestamps, label_names)

        if len(X) == 0:
            logger.error("No sequences — aborting")
            return

        results = self.train_all_predictors(X, y, timestamps, label_names)

        readmission_result = self.train_readmission(use_cache=use_cache)
        results['readmission'] = readmission_result

        report_path = self.save_summary(results, readmission_result)
        self.print_summary(results)

        end = datetime.now()
        logger.info(f"\n{'=' * 60}")
        logger.info(f"PIPELINE COMPLETED in {end - start}")
        logger.info(f"Report: {report_path}")
        logger.info(f"{'=' * 60}")

    def print_summary(self, results):
        logger.info(f"\n{'=' * 60}")
        logger.info("RESULTS SUMMARY — BEST MODEL PER TASK")
        logger.info(f"{'=' * 60}")

        for name, result in results.items():
            if isinstance(result, dict) and 'best_model' in result:
                best = result.get('best_model', 'N/A')
                auroc = result.get('best_auroc', result.get('auroc', 0))
                logger.info(f"  {name:20s} → {str(best):20s} AUROC: {auroc:.4f}")

                comparison = result.get('comparison', {})
                if comparison:
                    for model_name, metrics in comparison.items():
                        if isinstance(metrics, dict) and 'mean_test_auroc' in metrics:
                            marker = " ★" if model_name == best else "  "
                            logger.info(
                                f"    {marker} {model_name:18s} "
                                f"{metrics['mean_test_auroc']:.4f}"
                            )


def main():
    parser = argparse.ArgumentParser(description='Smart ICU Assistant Pipeline')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to MIMIC-III data directory')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of ICU stays to process (None = all)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force re-processing (ignore saved feature cache)')
    args = parser.parse_args()

    pipeline = SmartICUPipeline(
        config_path=args.config,
        data_dir=args.data_dir,
    )
    pipeline.run(sample_size=args.sample_size, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()