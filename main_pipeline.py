"""
Main Pipeline for Smart ICU Assistant
Orchestrates all prediction tasks using modular predictors.

FIXES / CHANGES:
  - Input normalization (StandardScaler) applied to X before caching — fixes NaN loss
  - Feature extraction loop has a tqdm progress bar (stays processed, sequences collected)
  - Overall pipeline step bar
  - set_label_indices() called BEFORE train_all_predictors()
  - XGBoost deprecated params updated to 2.x API
  - Cache normalization helper: normalize_existing_cache.py instructions printed
    if cache is loaded without normalization flag
"""

import os
import sys
import gc
import yaml
import logging
import argparse
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

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

# ── Suppress known harmless warnings ─────────────────────────────────────────
# PyTorch TransformerEncoder emits this when norm_first=True (which we want)
warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')
# Triton-related warnings on Windows
warnings.filterwarnings('ignore', message='.*triton.*')
# XGBoost ctypes callback (known Windows+GPU issue, non-fatal)
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')


# ── Logging that works with tqdm ─────────────────────────────────────────────
class TqdmLoggingHandler(logging.Handler):
    """Route log messages through tqdm.write() so they don't split progress bars."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
        except Exception:
            self.handleError(record)


def setup_logging():
    """Configure logging once for the entire pipeline."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Remove any existing handlers (prevents duplicate output)
    root.handlers.clear()

    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(
        '%(levelname)s:%(name)s:%(message)s'
    ))
    root.addHandler(handler)


setup_logging()
logger = logging.getLogger(__name__)


class SmartICUPipeline:
    """End-to-end pipeline for Smart ICU Assistant."""

    CACHE_DIR       = 'output'
    CACHE_FILE_X    = 'feature_cache_X.npy'
    CACHE_FILE_Y    = 'feature_cache_y.npy'
    CACHE_FILE_META = 'feature_cache_meta.pkl'
    # Flag file written after normalization so we know the cache is normalized
    CACHE_NORM_FLAG = 'feature_cache_normalized.flag'

    def __init__(self, config_path: str = 'config.yaml', data_dir: str = 'data'):
        self.config_path = config_path
        self.data_dir    = data_dir

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_loader       = MIMICDataLoader(data_dir, config_path)
        self.feature_engineer  = FeatureEngineer(config_path)

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

    # ── Step 1 ────────────────────────────────────────────────────────────────

    def load_data(self):
        logger.info("=" * 60)
        logger.info("STEP 1: Loading MIMIC-III Data")
        logger.info("=" * 60)
        self.merged_data = self.data_loader.merge_data()
        logger.info(
            f"✓ Loaded {len(self.merged_data):,} ICU stays "
            f"({self.merged_data['subject_id'].nunique():,} patients)"
        )
        return self.merged_data

    # ── Steps 2-3 ─────────────────────────────────────────────────────────────

    def extract_features_and_labels(self, sample_size: int = None):
        """Extract feature sequences and generate labels."""
        logger.info("=" * 60)
        logger.info("STEP 2-3: Feature Engineering & Label Generation")
        logger.info("=" * 60)

        # Build label list and set indices
        all_label_names = []
        for predictor in self.predictors.values():
            all_label_names.extend(predictor.get_label_names())

        for predictor in self.predictors.values():
            predictor.set_label_indices(all_label_names)

        logger.info(
            f"Labels: {len(all_label_names)} total → {all_label_names}"
        )

        stays    = self.merged_data if sample_size is None else self.merged_data.head(sample_size)
        n_stays  = len(stays)

        # Pre-index events by icustay_id for O(1) lookup
        logger.info("Pre-indexing chartevents / labevents by icustay_id...")
        charts_grouped = (
            dict(tuple(self.data_loader.chartevents.groupby('icustay_id')))
            if self.data_loader.chartevents is not None else {}
        )
        labs_grouped = (
            dict(tuple(self.data_loader.labevents.groupby('icustay_id')))
            if self.data_loader.labevents is not None else {}
        )
        logger.info("Pre-indexing complete.")

        all_sequences  = []
        all_labels     = []
        all_timestamps = []

        # ── Stay-level progress bar ───────────────────────────────────────────
        stay_bar = tqdm(
            stays.iterrows(),
            total=n_stays,
            desc="  Feature extraction",
            unit="stay",
            file=sys.stderr,
            dynamic_ncols=True,
            leave=True,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} stays "
                "[{elapsed}<{remaining}] seqs={postfix}"
            ),
        )

        for counter, (idx, stay) in enumerate(stay_bar):
            try:
                icustay_id  = stay['icustay_id']
                stay_charts = charts_grouped.get(icustay_id, pd.DataFrame())
                stay_labs   = labs_grouped.get(icustay_id, pd.DataFrame())

                features = self.feature_engineer.extract_features_for_stay(
                    icustay_id=icustay_id,
                    icu_intime=stay['intime'],
                    icu_outtime=stay['outtime'],
                    chartevents=stay_charts,
                    labevents=stay_labs,
                    d_items=self.data_loader.d_items,
                    d_labitems=self.data_loader.d_labitems,
                    window_hours=self.config.get('LSTM_CONFIG', {}).get('sequence_length', 24),
                )
                if features is None or len(features) == 0:
                    continue

                sequences, timestamps = self.feature_engineer.create_sequences(
                    features, sequence_length=24, step_size=12
                )
                if len(sequences) == 0:
                    continue

                raw_vitals = self.feature_engineer.extract_vital_signs(
                    stay_charts, self.data_loader.d_items,
                    icustay_id, stay['intime'], stay['outtime'],
                )
                raw_labs = self.feature_engineer.extract_lab_tests(
                    stay_labs, self.data_loader.d_labitems,
                    icustay_id, stay['intime'], stay['outtime'],
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
                        label_array = [labels.get(n, 0) for n in all_label_names]
                        all_labels.append(label_array)
                        all_sequences.append(sequences[seq_idx].astype(np.float32))
                        all_timestamps.append(timestamp)
                    except Exception as e:
                        logger.debug(f"Label error stay {icustay_id}: {e}")
                        all_labels.append([0] * len(all_label_names))
                        all_sequences.append(sequences[seq_idx].astype(np.float32))
                        all_timestamps.append(timestamp)

            except Exception as e:
                logger.debug(f"Feature extraction error stay {stay.get('icustay_id','?')}: {e}")
                continue

            stay_bar.set_postfix_str(f"{len(all_sequences):,}")

        stay_bar.close()

        del charts_grouped, labs_grouped
        gc.collect()

        logger.info(f"Building arrays from {len(all_sequences):,} sequences...")

        if all_sequences:
            target_features = max(s.shape[1] for s in all_sequences)
            n_seqs  = len(all_sequences)
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

        # Clean NaN / inf
        nan_x = np.isnan(X).sum() + np.isinf(X).sum()
        if nan_x > 0:
            logger.info(f"  Cleaning {nan_x:,} NaN/inf in X → 0")
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        nan_y = np.isnan(y).sum()
        if nan_y > 0:
            logger.info(f"  Cleaning {nan_y:,} NaN in y → 0")
            np.nan_to_num(y, copy=False, nan=0.0)

        # ── NORMALIZE (critical: prevents NaN loss in DL models) ──────────────
        X = self._normalize_X(X)

        input_size = X.shape[2] if len(X) > 0 else 0
        mem_gb     = X.nbytes / (1024 ** 3)
        logger.info(f"✓ Extraction complete | X={X.shape} | y={y.shape} | {mem_gb:.1f} GB")
        return X, y, all_timestamps, all_label_names

    @staticmethod
    def _normalize_X(X: np.ndarray) -> np.ndarray:
        """
        Normalize features per-column with StandardScaler.
        ICU features span wildly different ranges (HR≈80, WBC≈10,000)
        which causes gradient explosion and NaN loss without normalization.
        """
        if len(X) == 0:
            return X
        n_seqs, seq_len, n_feat = X.shape
        logger.info(f"  Normalizing X ({n_seqs:,} × {seq_len} × {n_feat}) with StandardScaler...")
        X_2d    = X.reshape(-1, n_feat)
        scaler  = StandardScaler()
        X_norm  = scaler.fit_transform(X_2d).astype(np.float32)
        np.nan_to_num(X_norm, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        logger.info("  Normalization complete ✓")
        return X_norm.reshape(n_seqs, seq_len, n_feat)

    def _prepare_extra_data(self, stay, icustay_id, stay_charts=None) -> Dict:
        extra = {}
        extra['chartevents'] = stay_charts if stay_charts is not None else pd.DataFrame()

        dl = self.data_loader
        if getattr(dl, 'prescriptions', None) is not None:
            pkey = 'subject_id' if 'subject_id' in dl.prescriptions.columns else None
            extra['prescriptions'] = (
                dl.prescriptions[dl.prescriptions[pkey] == stay.get('subject_id')]
                if pkey else pd.DataFrame()
            )
        else:
            extra['prescriptions'] = pd.DataFrame()

        extra['diagnoses'] = (
            dl.diagnoses[dl.diagnoses['hadm_id'] == stay.get('hadm_id')]
            if getattr(dl, 'diagnoses', None) is not None else pd.DataFrame()
        )

        if (getattr(dl, 'procedureevents', None) is not None
                and len(dl.procedureevents) > 0
                and 'icustay_id' in dl.procedureevents.columns):
            extra['procedureevents'] = dl.procedureevents[
                dl.procedureevents['icustay_id'] == icustay_id
            ]
        else:
            extra['procedureevents'] = pd.DataFrame()

        if (getattr(dl, 'procedures_icd', None) is not None
                and len(dl.procedures_icd) > 0
                and 'hadm_id' in dl.procedures_icd.columns):
            extra['procedures_icd'] = dl.procedures_icd[
                dl.procedures_icd['hadm_id'] == stay.get('hadm_id')
            ]
        else:
            extra['procedures_icd'] = pd.DataFrame()

        if (getattr(dl, 'inputevents_mv', None) is not None
                and len(dl.inputevents_mv) > 0
                and 'icustay_id' in dl.inputevents_mv.columns):
            extra['inputevents'] = dl.inputevents_mv[
                dl.inputevents_mv['icustay_id'] == icustay_id
            ]
        else:
            extra['inputevents'] = pd.DataFrame()

        return extra

    # ── Step 4 ────────────────────────────────────────────────────────────────

    def train_all_predictors(self, X, y, timestamps, label_names, output_dir='output'):
        logger.info("=" * 60)
        logger.info("STEP 4: Training All Predictors")
        logger.info("=" * 60)

        results         = {}
        predictors_list = list(self.predictors.items())

        # ── Overall task progress bar ─────────────────────────────────────────
        task_bar = tqdm(
            predictors_list,
            desc="  Tasks",
            unit="task",
            file=sys.stderr,
            dynamic_ncols=True,
            leave=True,
            bar_format=(
                "{l_bar}{bar:25}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}] {postfix}"
            ),
        )

        for name, predictor in task_bar:
            task_bar.set_postfix_str(f"▶ {name}")
            logger.info(f"\n── {name}: {predictor.TASK_DESCRIPTION}")

            task_result  = predictor.train_all_models(X, y, timestamps, output_dir)
            results[name] = task_result

            best = task_result.get('best_model', '?')
            auroc= task_result.get('best_auroc', 0)
            task_bar.set_postfix_str(f"{name} ✓ {best} ({auroc:.4f})")

        task_bar.close()
        return results

    # ── Step 5 ────────────────────────────────────────────────────────────────

    def train_readmission(self, output_dir='output', use_cache=True):
        logger.info("=" * 60)
        logger.info("STEP 5: ICU Readmission Prediction")
        logger.info("=" * 60)

        readmission_cache = self._cache_path('readmission_cache.pkl')

        if use_cache and os.path.exists(readmission_cache):
            logger.info("Loading readmission features from cache...")
            with open(readmission_cache, 'rb') as f:
                cached = pickle.load(f)
            features_df  = cached['features_df']
            labels_df    = cached['labels_df']
            self.readmission_predictor.feature_names = cached['feature_names']
            logger.info(f"✓ Readmission cache loaded: {len(features_df):,} stays")
        else:
            if self.merged_data is None:
                self.load_data()

            labels_df   = self.readmission_predictor.extract_readmission_labels(
                self.data_loader.icu_stays
            )
            features_df = self.readmission_predictor.extract_discharge_features(
                self.merged_data,
                self.data_loader.chartevents,
                self.data_loader.labevents,
                self.data_loader.diagnoses,
                self.data_loader.prescriptions,
                services    =getattr(self.data_loader, 'services', None),
                outputevents=getattr(self.data_loader, 'outputevents', None),
            )
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            with open(readmission_cache, 'wb') as f:
                pickle.dump({
                    'features_df':  features_df,
                    'labels_df':    labels_df,
                    'feature_names':self.readmission_predictor.feature_names,
                }, f)
            logger.info("✓ Readmission features cached")

        return self.readmission_predictor.train(features_df, labels_df, output_dir)

    # ── Step 6 ────────────────────────────────────────────────────────────────

    def save_summary(self, results, readmission_result, output_dir='output'):
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        report = {
            'timestamp':         ts,
            'data_dir':          self.data_dir,
            'predictor_results': {},
        }
        for name, result in results.items():
            report['predictor_results'][name] = {
                'best_model':  result.get('best_model', 'N/A'),
                'best_auroc':  result.get('best_auroc', 0),
                'comparison':  {
                    m: {'mean_test_auroc': v.get('mean_test_auroc', 0)}
                    for m, v in result.get('comparison', {}).items()
                    if isinstance(v, dict) and 'mean_test_auroc' in v
                },
            }
        report['readmission'] = readmission_result

        path = os.path.join(output_dir, f'metrics_report_{ts}.json')
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"✓ Saved metrics report → {path}")
        return path

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_path(self, filename):
        return os.path.join(self.CACHE_DIR, filename)

    def save_feature_cache(self, X, y, timestamps, label_names):
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        logger.info("Saving feature cache…")
        np.save(self._cache_path(self.CACHE_FILE_X), X)
        np.save(self._cache_path(self.CACHE_FILE_Y), y)
        with open(self._cache_path(self.CACHE_FILE_META), 'wb') as f:
            pickle.dump({'timestamps': timestamps, 'label_names': label_names}, f)
        # Write normalization flag so future loads know this cache is normalized
        open(self._cache_path(self.CACHE_NORM_FLAG), 'w').close()
        size_gb = os.path.getsize(self._cache_path(self.CACHE_FILE_X)) / (1024 ** 3)
        logger.info(f"✓ Cache saved ({size_gb:.1f} GB) — normalization flag written")

    def load_feature_cache(self):
        x_path    = self._cache_path(self.CACHE_FILE_X)
        y_path    = self._cache_path(self.CACHE_FILE_Y)
        meta_path = self._cache_path(self.CACHE_FILE_META)
        norm_flag = self._cache_path(self.CACHE_NORM_FLAG)

        if not all(os.path.exists(p) for p in [x_path, y_path, meta_path]):
            return None

        logger.info("Loading feature cache from disk…")
        X = np.array(np.load(x_path, mmap_mode='r'))
        y = np.load(y_path)

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        nan_count = np.isnan(X).sum() + np.isinf(X).sum()
        if nan_count > 0:
            logger.info(f"  Cleaning {nan_count:,} NaN/inf → 0")
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(y, copy=False, nan=0.0)

        # Warn if normalization flag is absent (old cache)
        if not os.path.exists(norm_flag):
            logger.warning(
                "⚠️  Cache was created WITHOUT normalization (old cache). "
                "DL models may produce NaN loss. Run with --no-cache to rebuild, "
                "or run: python normalize_cache.py"
            )
        else:
            logger.info("  ✓ Cache is normalized (flag present)")

        logger.info(
            f"✓ Cache loaded: X={X.shape} | y={y.shape} | "
            f"{len(meta['label_names'])} labels"
        )
        return X, y, meta['timestamps'], meta['label_names']

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, sample_size: int = None, use_cache: bool = True):
        start = datetime.now()
        logger.info("=" * 60)
        logger.info("SMART ICU ASSISTANT — TRAINING PIPELINE")
        logger.info(f"Start : {start}")
        logger.info(f"Data  : {self.data_dir}")
        logger.info("=" * 60)

        # ── Pipeline-level step bar ───────────────────────────────────────────
        PIPELINE_STEPS = [
            "Load / Cache data",
            "Train predictors",
            "Train readmission",
            "Save report",
        ]
        pipe_bar = tqdm(
            PIPELINE_STEPS,
            desc="  Pipeline",
            unit="step",
            file=sys.stderr,
            dynamic_ncols=True,
            leave=True,
        )

        def next_step(label):
            pipe_bar.set_description(f"  {label:<35}")
            pipe_bar.update(1)

        # Step 1-3: data + features
        cached = self.load_feature_cache() if use_cache else None

        if cached is not None:
            X, y, timestamps, label_names = cached
            logger.info("⚡ Steps 1-3 skipped (cache loaded)")
            for predictor in self.predictors.values():
                predictor.set_label_indices(label_names)
            logger.info(f"Label indices set from cache ({len(label_names)} labels)")
        else:
            self.load_data()
            X, y, timestamps, label_names = self.extract_features_and_labels(sample_size)
            if len(X) == 0:
                logger.error("No sequences extracted — aborting")
                return
            self.save_feature_cache(X, y, timestamps, label_names)

        if len(X) == 0:
            logger.error("No sequences — aborting")
            return

        next_step("Data loaded ✓")

        # Step 4: train
        results = self.train_all_predictors(X, y, timestamps, label_names)
        next_step("Predictors trained ✓")

        # Step 5: readmission
        readmission_result = self.train_readmission(use_cache=use_cache)
        results['readmission'] = readmission_result
        next_step("Readmission trained ✓")

        # Step 6: report
        report_path = self.save_summary(results, readmission_result)
        self.print_summary(results)
        next_step("Report saved ✓")

        pipe_bar.close()

        end = datetime.now()
        logger.info(f"\n{'='*60}")
        logger.info(f"PIPELINE COMPLETED in {end - start}")
        logger.info(f"Report: {report_path}")
        logger.info(f"{'='*60}")

    def print_summary(self, results):
        logger.info(f"\n{'='*70}")
        logger.info("RESULTS SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"  {'Task':<15s} {'Model':<14s} {'AUROC':>7s} {'AUPRC':>7s} {'F1':>7s} {'Sens':>7s}")
        logger.info(f"  {'─'*13}   {'─'*12}   {'─'*5}   {'─'*5}   {'─'*5}   {'─'*5}")
        for name, result in results.items():
            if isinstance(result, dict) and 'best_model' in result:
                best = result.get('best_model', 'N/A')
                
                # Standard tasks (with comparison block)
                if 'comparison' in result:
                    for mname, mmetrics in result.get('comparison', {}).items():
                        if isinstance(mmetrics, dict) and 'mean_test_auroc' in mmetrics:
                            pm = mmetrics.get('per_task_metrics', {})
                            auroc = mmetrics.get('mean_test_auroc', 0)
                            auprc = pm.get('macro_auprc', 0)
                            f1    = pm.get('macro_f1', 0)
                            sens  = pm.get('macro_sensitivity', 0)
                            star  = " ★" if mname == best else "  "
                            logger.info(
                                f"  {name:<15s}{star}{mname:<12s} "
                                f"{auroc:>7.4f} {auprc:>7.4f} {f1:>7.4f} {sens:>7.4f}"
                            )
                
                # Readmission task (flat metrics)
                elif 'auroc' in result:
                    auroc = result.get('auroc', 0)
                    auprc = result.get('auprc', 0)
                    f1    = result.get('f1', 0)
                    sens  = result.get('sensitivity', 0)
                    logger.info(
                        f"  {name:<15s} ★{best:<12s} "
                        f"{auroc:>7.4f} {auprc:>7.4f} {f1:>7.4f} {sens:>7.4f}"
                    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Smart ICU Assistant Pipeline')
    parser.add_argument('--data_dir',    type=str, default='data')
    parser.add_argument('--config',      type=str, default='config.yaml')
    parser.add_argument('--sample_size', type=int, default=None)
    parser.add_argument('--no-cache',    action='store_true',
                        help='Force re-processing (ignore saved feature cache)')
    args = parser.parse_args()

    pipeline = SmartICUPipeline(config_path=args.config, data_dir=args.data_dir)
    pipeline.run(sample_size=args.sample_size, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()