# Smart ICU Assistant — Complete Project Documentation

> **Combined from**: `System_Arch.md`, `new-arch.md`, `new_log.md`, `why.md`
> This is the single-source-of-truth reference for architecture, design decisions, and development history.

---

## Table of Contents

1. [Data Loading](#1-data-loading)
2. [Feature Engineering](#2-feature-engineering)
3. [Clinical Time Windows](#3-clinical-time-windows)
4. [Label Generation](#4-label-generation)
5. [Model Architecture](#5-model-architecture)
6. [Training Pipeline](#6-training-pipeline)
7. [API Backend](#7-api-backend)
8. [Dashboard Frontend](#8-dashboard-frontend)
9. [File Architecture](#9-file-architecture)
10. [MIMIC-III Dataset Notes](#10-mimic-iii-dataset-notes)
11. [Development Changelog](#11-development-changelog)

---

## 1. Data Loading

**Source file**: `data_loader.py`

### About MIMIC-III

MIMIC-III is a publicly available clinical database of **~46,000 patients** admitted to the ICU at Beth Israel Deaconess Medical Center (Boston) between 2001–2012. All dates are **shifted to the 2100–2200 range** for de-identification, but are internally consistent per patient (relative time differences are preserved). Patients >89 years old have DOB shifted ~300 years before admission (clamped to 91.4 years).

### The 14 CSV Files We Load

The data loader reads tables in a **specific order** because some tables depend on others:

| Order | File | Rows (Full) | What it contains | Why we need it |
|-------|------|-------------|-----------------|----------------|
| 1 | `PATIENTS.csv` | 46K | Gender, DOB, DOD, expire_flag | Base patient demographics; DOD is critical for mortality labels |
| 2 | `ADMISSIONS.csv` | 59K | Admit/discharge times, death time, diagnosis | Admission context; `deathtime` = in-hospital death |
| 3 | `ICUSTAYS.csv` | 61K | ICU in/out times, care unit, LOS | **Core unit of analysis** — one row = one ICU stay |
| 4 | `D_ITEMS.csv` | 12K | Dictionary: itemid → label (e.g., 220045 → "Heart Rate") | **Must load before CHARTEVENTS** — tells us which itemids are vitals |
| 4 | `D_LABITEMS.csv` | 753 | Dictionary: itemid → lab name (e.g., 50912 → "Creatinine") | Maps lab item IDs to readable names |
| 5 | **`CHARTEVENTS.csv`** | **330M** | Timestamped bedside measurements (HR, BP, SpO₂, temp…) | **Main vitals source** — 33.6 GB, requires chunked loading |
| 6 | `LABEVENTS.csv` | 27M | Lab results (creatinine, lactate, WBC, etc.) | Lab values for AKI, sepsis detection |
| 7 | `DIAGNOSES_ICD.csv` | 650K | ICD-9 diagnosis codes per admission | Sepsis ICD codes (038, 995.91) |
| 8 | `PRESCRIPTIONS.csv` | 4M | Medication orders (drug name, start/end dates) | Antibiotic detection (sepsis), vasopressor drugs |
| 9 | `INPUTEVENTS_MV.csv` | 3M | IV drip administration (drug, rate, amount) | **Vasopressor detection** — norepinephrine, dopamine, etc. |
| 10 | `OUTPUTEVENTS.csv` | 4M | Fluid outputs (urine, drains) | **AKI detection** — urine output criteria |
| 11 | `PROCEDUREEVENTS_MV.csv` | 258K | ICU procedures with timestamps | **Ventilation detection** — mechanical vent itemids |
| 12 | `PROCEDURES_ICD.csv` | 240K | ICD-9 procedure codes | Ventilation ICD codes (9670-9672) |
| 13 | `MICROBIOLOGYEVENTS.csv` | 631K | Culture results (organism, antibiotic sensitivity) | Infection evidence for sepsis |
| 14 | `TRANSFERS.csv` | 261K | Patient transfers between units | Unit transfer tracking |

**Tables intentionally skipped** (6 files):
- `NOTEEVENTS` — requires NLP, different model type entirely
- `DATETIMEEVENTS` — non-numeric events, low predictive value
- `CPTEVENTS` / `DRGCODES` / `D_CPT` — billing codes, no clinical value
- `CAREGIVERS` — caregiver ID doesn't predict patient outcomes

### CHARTEVENTS Chunked Loading

CHARTEVENTS is **33.6 GB** / **330 million rows**. It tracks everything the nurse charts: heart rate, blood pressure, ventilator settings, Glasgow Coma Scale, etc.

**Strategy** (can't fit 330M rows in RAM):

1. Get the 32 relevant itemids (vitals + ventilation) via `_get_relevant_chartevents_itemids()` — e.g., 220045 (HR), 220050/220051 (SysBP/DiaBP), 220210 (RR)
2. Read CSV in chunks of 2,000,000 rows using `pd.read_csv(filepath, chunksize=2_000_000)`. Only loads 6 columns: `SUBJECT_ID, HADM_ID, ICUSTAY_ID, ITEMID, CHARTTIME, VALUENUM`
3. Filter each chunk: keep only rows where `itemid ∈ {32 relevant IDs}` — drops ~85% of rows
4. Concatenate kept chunks → ~50M rows (from 330M), fits in ~4-6 GB RAM

### Merging

After loading all 14 tables, `merge_data()` joins them:

```
ICU_STAYS ← merge → PATIENTS (on subject_id)            → adds gender, DOB, DOD
          ← merge → ADMISSIONS (on subject_id + hadm_id) → adds admit/disch time, diagnosis

Computes:
  age = (ICU intime - DOB) / 365.25 days
  hours_to_death = (DOD - ICU intime) in hours  ← KEY for mortality labels
```

Result: **61,532 rows** — one per ICU stay, with demographics, times, and death info.

---

## 2. Feature Engineering

**Source file**: `feature_engineering.py`

### Vital Signs (8 features from CHARTEVENTS)

| Vital | Clinical Significance |
|-------|----------------------|
| Heart Rate (bpm) | Tachycardia > 90 = SIRS criterion |
| Systolic BP (mmHg) | Hypotension indicator |
| Diastolic BP (mmHg) | Used to compute MAP |
| Mean Arterial Pressure (mmHg) | MAP < 65 = hypotension = organ damage |
| Respiratory Rate (/min) | RR > 20 = SIRS criterion |
| Temperature (°C) | Fever > 38.3°C or hypothermia < 36°C = SIRS |
| SpO₂ (%) | Oxygen saturation; < 94% = respiratory failure |
| Glucose (mg/dL) | Stress hyperglycemia in critical illness |

### Lab Values (7 features from LABEVENTS)

| Lab | Clinical Significance |
|-----|----------------------|
| Creatinine (mg/dL) | **AKI detection** — rising creatinine = kidney failure |
| Lactate (mmol/L) | Tissue hypoxia marker; > 2 = sepsis, > 4 = severe |
| WBC (/µL) | > 12K or < 4K = SIRS criterion (infection) |
| Hemoglobin (g/dL) | Anemia, bleeding |
| Platelets (×10³) | Low = DIC, sepsis-related coagulopathy |
| Bicarbonate (mEq/L) | Metabolic acidosis (low) = tissue hypoperfusion |
| Chloride (mEq/L) | Electrolyte balance |

### Derived Features

```python
shock_index    = heartrate / sysbp          # > 1.0 = hemodynamic instability → shock
pulse_pressure = sysbp - diasbp             # < 25 = low cardiac output
meanbp         = (sysbp + 2 * diasbp) / 3   # < 65 mmHg = insufficient organ perfusion
```

### SIRS Score (0–4)

Computed at each timepoint from vitals + labs:

```
+1 if temperature > 38.3°C or < 36.0°C
+1 if heart rate > 90 bpm
+1 if respiratory rate > 20 /min
+1 if WBC > 12,000 or < 4,000 /µL

Score ≥ 2 = systemic inflammatory response
Score ≥ 2 + infection evidence = SEPSIS
```

### Time Window Aggregations

For each feature, rolling window statistics are computed:

- **Window sizes**: 6h, 12h, 24h
- **Per window, per feature → 5 statistics**: mean, std, min, max, trend (slope)
- Example: `heartrate_6h_mean`, `heartrate_6h_std`, etc.

This turns ~15 raw features into **~90+ engineered features** per timepoint.

### Sequence Creation

Deep learning models (LSTM, Transformer) need fixed-length sequences. A sliding window is used:

```
Parameters: sequence_length = 24, step_size = 6

For a 72-hour stay:
  Sequence 1: hours [0–23]   → predict what happens after hour 23
  Sequence 2: hours [6–29]   → predict what happens after hour 29
  ...
  Sequence 9: hours [48–71]  → predict what happens after hour 71

Output shape: [n_sequences, 24, n_features]
```

---

## 3. Clinical Time Windows

Each prediction task uses different time horizons because **clinical interventions have different lead times**:

| Task | Windows | Rationale |
|------|---------|-----------|
| **Mortality** | 6, 12, 24h | 6h = code blue/palliative; 12h = family notification; 24h = standard planning horizon |
| **Sepsis** | 6, 12, 24h | 6h = "golden hour" (early antibiotics reduce mortality 7%/hr); 12h = targeted antibiotic switch; 24h = full workup |
| **AKI** | 24, 48h | Kidneys are slow — creatinine rises 12-24h **after** injury. KDIGO requires 0.3 mg/dL rise within 48h |
| **Hypotension** | 1, 3, 6h | MAP < 65 = organs not getting blood NOW. 3h = fluid resuscitation bundle. Not 12/24h — too many confounders |
| **Vasopressor** | 6, 12h | 6h = urgent hemodynamic support; 12h = OR scheduling. Not 24h — depends on treatment response |
| **Ventilation** | 6, 12, 24h | 6h = prepare intubation team; 12h = ICU bed planning; 24h = resource allocation (ventilators are limited) |

---

## 4. Label Generation

**Source files**: `predictors/*.py`

A **label** is the ground truth the model learns: "Did this bad thing actually happen in the next X hours?" → **0 or 1**.

### Label Definitions by Task

**Mortality**: `label = 1` if `0 ≤ (date_of_death - T) ≤ window_hours`

**Sepsis (SIRS + Infection)**: `label = 1` if SIRS ≥ 2 (fever, tachycardia, tachypnea, WBC abnormality) AND infection evidence (antibiotic prescription or sepsis ICD-9 codes 038, 995.91, 995.92)

**AKI (KDIGO)**: Based on creatinine kinetics from 48h baseline:
- Stage 1: increase ≥ 0.3 mg/dL OR ≥ 1.5× baseline
- Stage 2: ≥ 2.0× baseline
- Stage 3: ≥ 3.0× baseline OR > 4.0 mg/dL absolute
- Stages are cumulative (Stage 3 → also Stage 2 = 1, Stage 1 = 1)

**Hypotension**: `label = 1` if any MAP reading < 65 mmHg in window

**Vasopressor**: `label = 1` if vasopressor started in window. Checks PRESCRIPTIONS (drug keywords) and INPUTEVENTS_MV (itemids 221906, 221289, 222315, 221749, 221662)

**Ventilation**: Three-layer detection (any triggers label = 1): CHARTEVENTS itemids (225792, 225794, 226260), PROCEDUREEVENTS_MV, ICD-9 codes (9670-9672, 9604, 9390)

**Length of Stay**: `los_short_24h = 1` if remaining ≤ 24h; `los_long_72h = 1` if remaining > 72h

### Full Label Vector (17 binary labels per sequence)

```
[ mortality_6h, mortality_12h, mortality_24h,          # 1-3
  sepsis_6h, sepsis_12h, sepsis_24h,                   # 4-6
  aki_stage1_24h, aki_stage2_24h, aki_stage3_24h,      # 7-9
  aki_stage1_48h, aki_stage2_48h, aki_stage3_48h,      # 10-12
  vasopressor_6h, vasopressor_12h,                     # 13-14
  ventilation_6h, ventilation_12h, ventilation_24h,    # 15-17
  los_short_24h, los_long_72h ]                        # 16-17
```

---

## 5. Model Architecture

**Source file**: `models.py`

### Current Models (3 per task)

| Model | Type | Strengths | Best for |
|-------|------|-----------|----------|
| **LSTM** | Recurrent Neural Net | Long-term memory, captures deterioration trends | Mortality, Ventilation, Sepsis |
| **Transformer** | Attention-based | Cross-feature attention, subtle interactions | Sepsis (temp × HR × WBC) |
| **XGBoost** | Gradient-boosted trees | Handles class imbalance, interpretable | Vasopressor, LOS, Mortality |

### LSTM (Bidirectional, 2-layer)

```
Input [batch, 24, features]
  → Bidirectional LSTM (hidden=128, layers=2, dropout=0.3)
  → Attention pooling over time steps
  → Dropout (0.3) → Linear → Sigmoid → [batch, num_tasks]
```

### Transformer Encoder

```
Input [batch, 24, features]
  → Positional Encoding
  → TransformerEncoder (d_model=128, heads=8, layers=3, norm_first=True)
  → Mean pooling → Linear → Sigmoid → [batch, num_tasks]
```

### XGBoost (one model per label)

```
Input [n_samples, 24, features] → flatten → [n_samples, 24 × features]
  → 17 independent XGBClassifier models
  → max_depth=8, 300 trees, early stopping (20 rounds)
  → tree_method='hist', device='cuda'


### Best Model Selection

```
For each task (e.g., "sepsis"):
  1. Train LSTM        → test AUROC
  2. Train Transformer → test AUROC
  3. Train XGBoost     → test AUROC

  → Save the model with highest AUROC
  → Save report to output/<task>_report.json
```

**AUROC interpretation**: 0.5 = random guessing, 0.7 = acceptable, 0.8 = good, 0.9+ = excellent.

---

## 6. Training Pipeline

**Source file**: `main_pipeline.py`

### Execution

```bash
python main_pipeline.py --data_dir data
```

**Steps**: Load Data → Pre-index (group by icustay_id) → Feature Engineering → Label Generation → Train 3 Models Per Task → Save Reports

### Temporal Split (no data leakage)

Data is split **chronologically**, not randomly, to prevent leaking future information into training:

```
Sort all sequences by (shifted) timestamp — relative order preserved

|←————— 70% Train ————→|← 15% Val →|← 15% Test →|
   earliest shifted dates .................. latest
```

### GPU Optimizations (RTX 3050, 4GB VRAM)

| Optimization | Effect |
|-------------|--------|
| Mixed Precision (FP16) | Halves memory → model fits in 4GB |
| Gradient Accumulation | batch 32 × 2 steps = effective batch 64 |
| Pin Memory + non_blocking | Faster CPU→GPU data transfer |
| VRAM Cleanup | `torch.cuda.empty_cache()` between models |
| XGBoost `device=cuda` | GPU-accelerated tree boosting |
| `BCEWithLogitsLoss` | Numerically stable under AMP (fused sigmoid + BCE) |
| Gradient Clipping | `clip_grad_norm_(max_norm=1.0)` prevents exploding gradients |

### Feature Cache

After the first run, features are cached to disk:
- `feature_cache_X.npy` — Feature matrix (memory-mapped for large datasets)
- `feature_cache_y.npy` — Label matrix
- `feature_cache_meta.pkl` — Metadata (label names, timestamps, scaler)
- `feature_cache_normalized.flag` — Normalization indicator

Subsequent runs skip feature extraction entirely and load from cache.

### Resource Estimates

| Metric | Demo (100 patients) | Full MIMIC-III |
|--------|-------------------|----------------|
| Patients | 100 | ~46,000 |
| ICU Stays | 136 | ~61,000 |
| Sequences | 1,931 | ~500,000+ |
| CHARTEVENTS | 758K rows | ~330M rows |
| Training time | ~10 min | ~2-8 hours (GPU) |
| RAM needed | ~2 GB | ~16-32 GB (with chunking) |

---

## 7. API Backend

**Source file**: `app.py`

### FastAPI Endpoints

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/api/stats` | GET | Dataset statistics (total stays, mean age, mortality rate, care unit distribution) |
| `/api/patients` | GET | Paginated patient list with search, risk filtering |
| `/api/patients/{id}/predictions` | GET | All 17 prediction scores for an existing patient |
| `/api/patients/{id}/vitals` | GET | Time-series vital signs for charting |
| `/api/patients/{id}/labs` | GET | Time-series lab values for charting |
| `/api/alerts` | GET | All HIGH/MEDIUM risk patients with triggered labels |
| **`/api/predict`** | **POST** | **New patient assessment** — accepts vitals/labs/meds → returns risk scores |
| `/api/health` | GET | Server health check |

### `/api/predict` — New Patient Endpoint

**Mode 1: Clinical Rules** (always available, no models needed):
- SIRS score (0-4) from temp, HR, RR, WBC
- KDIGO staging from creatinine
- MAP threshold check (< 65 = hypotension)
- Shock index (HR / SBP > 1.0 = shock)
- Vasopressor/vent check from medications toggles

**Mode 2: Trained Models** (after pipeline completes):
- Build 24-step feature vector → run through trained models → per-task probability scores → merge with clinical rules → generate alerts

---

## 8. Dashboard Frontend

**Source file**: `smart_icu_dashboard.html` + `templates/`

### Pages

| Page | Purpose |
|------|---------|
| **Overview** | KPI cards (total stays, age, LOS, mortality) + risk distribution + care units + critical alerts |
| **Patient Detail** | Click patient → vitals charts, lab trends, all 17 prediction scores, SHAP features |
| **Alerts** | All high-risk patients sorted by composite score, with triggered labels |
| **New Assessment** | Input form for new patient → real-time risk prediction |
| **Validation** | Model performance metrics and training reports |

### Composite Risk Score

```
composite = mortality_24h   × 0.30
          + sepsis_24h      × 0.25
          + aki_stage1_24h  × 0.15
          + vasopressor_12h × 0.15
          + ventilation_24h × 0.15
```

Patients are classified as **HIGH** (>0.6), **MEDIUM** (>0.3), or **LOW** risk.

---

## 9. File Architecture

```
SEM-4-PROJECT/
├── data/                          # MIMIC-III CSV files (33+ GB)
├── config.yaml                    # All thresholds, model params, GPU config
├── data_loader.py                 # Loads + merges 14 CSV files
├── feature_engineering.py         # Extracts vitals/labs, creates sequences
├── training.py                    # GPU training loop, AMP, grad accumulation
├── models.py                      # LSTM, Transformer, XGBoost architectures
├── main_pipeline.py               # Orchestrates everything end-to-end
├── rebuild_ensembles.py           # Rebuild ensemble pickles from checkpoints
├── show_predictions.py            # CLI display of training results
│
├── predictors/                    # One file per prediction task
│   ├── base_predictor.py          # MODELS_TO_TRY = ['lstm','transformer','xgboost']
│   ├── mortality_predictor.py     # 6/12/24h mortality
│   ├── sepsis_predictor.py        # SIRS + infection detection
│   ├── aki_predictor.py           # KDIGO staging
│   ├── vasopressor_predictor.py   # Drug requirement detection
│   ├── ventilation_predictor.py   # 3-layer vent detection
│   └── los_predictor.py           # Short/long stay
│
├── templates/                     # Jinja2 HTML templates
│   ├── base.html                  # Base layout with theme toggle
│   ├── index.html                 # Dashboard landing
│   ├── patients.html              # Patient list
│   ├── patient_detail.html        # Individual patient view
│   └── validation.html            # Model validation results
│
├── app.py                         # FastAPI backend
├── smart_icu_dashboard.html       # Standalone React dashboard
├── models/                        # Saved model checkpoints (.pth, .pkl)
└── output/                        # Training reports, feature cache, SHAP plots
```

---

## 10. MIMIC-III Dataset Notes

### Date Shifting

All dates are shifted to 2100-2200 range for patient privacy:

- **Consistent per patient**: `intime - dob` gives correct age, `dod - intime` gives correct hours_to_death
- **Random per patient**: Two patients on "2150-01-01" were NOT in the ICU at the same time
- **Preserved**: Time of day, day of week, seasonality
- **Removed**: Absolute year, day of month, inter-patient timing
- **Special case**: Patients >89 years → DOB shifted ~300 years before first admission → ages show as 300+, clamped to 91.4
- Actual data collected 2001-2012 at Beth Israel Deaconess Medical Center (Boston)

---

## 11. Development Changelog

### Memory & Data Pipeline Fixes

- **float64 → float32**: Halved memory from 14.2 GB → 7.1 GB for the final numpy array creation
- **Pre-allocate instead of list→array**: Avoided the 2× memory spike during conversion
- **Feature caching**: Added `_vital_itemid_cache` and `_lab_itemid_cache` so itemid-to-name mapping runs once instead of 492,000 times
- **Rolling trend optimization**: `calc_trend_fast` uses `raw=True` (numpy/Cython path) instead of `raw=False` (interpreted) — 2-4× faster per stay

### Training Bug Fixes

- **BCELoss → BCEWithLogitsLoss**: BCELoss is unsafe under AMP. Models output raw logits; `BCEWithLogitsLoss` fuses sigmoid + loss internally and is numerically stable under FP16
- **Label indices bug**: `set_label_indices()` was called **after** `train_all_models()` instead of before — root cause of AUROC = 0 for all DL models (they received all 19 labels instead of task-specific columns)
- **NaN loss handling**: Added `not np.isnan(val_loss)` check for checkpointing; NaN epochs are skipped without incrementing patience counter
- **best_model.pth conflicts**: All models saved to same path → crash when checkpoint missing. Fixed with unique temp paths (`models/_best_<uuid>.pth`)
- **"Mean of empty slice" warnings**: All mortality labels in validation batch were 0 → AUROC = NaN. Now filters NaN values before averaging
- **Gradient clipping**: Added `clip_grad_norm_(max_norm=1.0)` to prevent exploding gradients producing inf/NaN loss
- **Early stopping tuning**: Patience raised 10 → 20, `min_delta=1e-4` added

### Windows Compatibility

- **DataLoader `num_workers`**: `num_workers=2` crashes on Windows without `if __name__ == '__main__'` guard. Added `_safe_num_workers()` forcing `num_workers=0` on Windows
- **`torch.compile` suppressed**: Triton not supported on Windows

### A100 GPU Optimizations (for server deployment)

- BF16 replaces FP16 (A100 has dedicated BF16 tensor cores, no GradScaler needed)
- Batch size 32 → 2048 (~60× fewer kernel launches per epoch)
- `torch.compile(mode='reduce-overhead')` for CUDA graph fusion (20-40% speedup)
- DataLoader workers 2 → 16 with `prefetch_factor=4` and `persistent_workers=True`
- Fused AdamW (`fused=True, foreach=True`)
- LSTM hidden 128 → 256, layers 2 → 3, Transformer d_model 128 → 256
- Expected speedup: ~10-20× (mortality/LSTM from ~60 min/epoch to ~3-6 min/epoch)

### RTX 3050 GPU Optimizations (for local development)

- Batch size 32 → 128 (uses ~1.8 GB of 4 GB VRAM at FP16, was using only 5%)
- Gradient accumulation steps 2 → 1 (no longer needed at batch=128)
- `prefetch_factor=2` + `persistent_workers=True` on Linux/macOS
- `weights_only=True` on `torch.load` calls (removes PyTorch 2.x FutureWarning)

---

*Last updated: April 2026*
