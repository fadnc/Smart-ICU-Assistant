# Smart ICU Assistant — Complete Project Documentation

> **Single-source-of-truth** reference for architecture, design decisions, and development history.
> **Last updated**: April 2026

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

MIMIC-III is a publicly available clinical database of **~46,000 patients** admitted to the ICU at Beth Israel Deaconess Medical Center (Boston) between 2001–2012. Total size: **~43 GB**. All dates are **shifted to the 2100–2200 range** for de-identification, but are internally consistent per patient (relative time differences are preserved). Patients >89 years old have DOB shifted ~300 years before admission (clamped to 91.4 years).

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

CHARTEVENTS is **33.6 GB** / **330 million rows**.

**Strategy** (can't fit 330M rows in RAM):

1. Get the 32 relevant itemids (vitals + ventilation) via `_get_relevant_chartevents_itemids()`
2. Read CSV in chunks of 2,000,000 rows. Only loads 6 columns: `SUBJECT_ID, HADM_ID, ICUSTAY_ID, ITEMID, CHARTTIME, VALUENUM`
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

This turns ~15 raw features into **81 engineered features** per timepoint.

### Sequence Creation

Deep learning models (LSTM, Transformer) need fixed-length sequences:

```
Parameters: sequence_length = 24, step_size = 12

For a 72-hour stay:
  Sequence 1: hours [0–23]   → predict what happens after hour 23
  Sequence 2: hours [12–35]  → predict what happens after hour 35
  ...

Output shape: [n_sequences, 24, 81]
```

---

## 3. Clinical Time Windows

Each prediction task uses different time horizons because **clinical interventions have different lead times**:

| Task | Windows | Rationale |
|------|---------|-----------| 
| **Mortality** | 6, 12, 24h | 6h = code blue/palliative; 12h = family notification; 24h = standard planning horizon |
| **Sepsis** | 6, 12, 24h | 6h = "golden hour" (early antibiotics reduce mortality 7%/hr); 12h = targeted antibiotic switch; 24h = full workup |
| **AKI** | 24, 48h | Kidneys are slow — creatinine rises 12-24h **after** injury. KDIGO requires 0.3 mg/dL rise within 48h |
| **Vasopressor** | 6, 12h | 6h = urgent hemodynamic support; 12h = OR scheduling. Not 24h — depends on treatment response |
| **Ventilation** | 6, 12, 24h | 6h = prepare intubation team; 12h = ICU bed planning; 24h = resource allocation (ventilators are limited) |
| **Length of Stay** | 24h, 72h | 24h = imminent discharge planning; 72h = extended care resource allocation |

**Removed Tasks**: Hypotension (1/3/6h) and ICU Readmission (48-72h) from the original PDF design were dropped during implementation — Hypotension overlaps with vasopressor prediction, and Readmission relies on static features with minimal sequential dependency, making it unsuitable for the time-series pipeline.

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
  los_short_24h, los_long_72h ]                        # 18-19
```

> **Note**: The label vector contains 19 entries during feature extraction, but training uses **17 task-specific labels** (LOS labels are indexed separately). Each predictor extracts only its own labels via `set_label_indices()`.

---

## 5. Model Architecture

**Source file**: `models.py`

### Current Models (4 per task + 2 ensembles)

| Model | Type | Strengths | Best for |
|-------|------|-----------|----------|
| **LSTM** | Bidirectional RNN with Attention | Long-term memory, captures deterioration trends | Mortality, Ventilation |
| **Transformer** | Pre-LayerNorm Attention Encoder | Cross-feature attention, subtle interactions | Sepsis (temp × HR × WBC) |
| **XGBoost** | Gradient-boosted trees | Handles class imbalance, interpretable via SHAP | Vasopressor, AKI, LOS |
| **LightGBM** | Leaf-wise gradient-boosted trees | Different growth strategy, adds ensemble diversity | AKI, LOS |
| **Weighted Ensemble** | AUROC²-weighted average of all 4 | Combines strengths, reduces variance | Mortality, Vasopressor |
| **Stacked Ensemble** | LogisticRegression meta-learner | Learns optimal model combination per task | Sepsis, AKI, Ventilation, LOS |

**Removed**: TCN (Temporal Convolutional Network) — `nn.BatchNorm1d` collapses under FP16 AMP with imbalanced ICU labels, producing NaN loss from epoch 2-7.

### LSTM (Bidirectional, 2-layer, with Temporal Attention)

```
Input [batch, 24, 81]
  → Bidirectional LSTM (hidden=128, layers=2, dropout=0.3)
  → Temporal Attention pooling (learns which timesteps matter most)
  → Per-task heads: Linear(256→64) → ReLU → Dropout → Linear(64→1)
  → Concatenate → [batch, num_tasks]
```

### Transformer Encoder (Pre-LayerNorm)

```
Input [batch, 24, 81]
  → Linear projection (81 → 128)
  → Positional Encoding
  → TransformerEncoder (d_model=128, heads=8, layers=3, norm_first=True)
  → FP32 autocast for attention (prevents FP16 overflow)
  → Mean pooling → Linear(128→64) → ReLU → Linear(64→num_tasks)
```

### XGBoost (one model per label)

```
Input [n_samples, 24, 81] → flatten → [n_samples, 1944]
  → Per-task XGBClassifier with auto scale_pos_weight
  → max_depth=8, 300 trees, early stopping (20 rounds)
  → tree_method='hist', device='cuda'
```

### LightGBM (one model per label)

```
Input [n_samples, 24, 81] → flatten → [n_samples, 1944]
  → Per-task LGBMClassifier with auto scale_pos_weight
  → num_leaves=63, 300 trees, early stopping (20 rounds)
  → Leaf-wise growth (different from XGBoost's level-wise)
```

### Ensemble Methods

**Weighted Ensemble**: Each model's contribution is weighted by AUROC², so better-performing models have more influence.

**Stacked Ensemble**: A LogisticRegression meta-learner is trained on base model predictions (half/half split to avoid leakage), learning the optimal combination per task.

### Best Model Selection

```
For each task (e.g., "sepsis"):
  1. Train LSTM        → test AUROC
  2. Train Transformer → test AUROC
  3. Train XGBoost     → test AUROC
  4. Train LightGBM    → test AUROC
  5. Weighted Ensemble → test AUROC
  6. Stacked Ensemble  → test AUROC

  → Best model = highest AUROC across all 6
  → Save report to output/<task>_report.json
```

### Achieved Results (Full MIMIC-III)

| Task | Best Model | AUROC | Labels |
|------|-----------|-------|--------|
| Mortality | Ensemble | 0.7906 | 3 |
| Sepsis | Stacked Ensemble | 0.8015 | 3 |
| AKI | Stacked Ensemble | 0.8249 | 6 |
| Vasopressor | Ensemble | 0.8032 | 2 |
| Ventilation | Stacked Ensemble | 0.8531 | 3 |
| LOS | Stacked Ensemble | 0.7940 | 2 |

---

## 6. Training Pipeline

**Source file**: `main_pipeline.py`

### Execution

```bash
python main_pipeline.py --data_dir data
```

**Steps**: Load Data → Pre-index (group by icustay_id) → Feature Engineering → Label Generation → Normalize (StandardScaler) → Train 4 Models + Ensembles Per Task → Save Reports

### Temporal Split (no data leakage)

```
Sort all sequences by (shifted) timestamp — relative order preserved

|←————— 70% Train ————→|← 15% Val →|← 15% Test →|
   earliest shifted dates .................. latest
```

### GPU Optimizations (RTX 3050, 4GB VRAM)

| Optimization | Effect |
|-------------|--------|
| Mixed Precision (FP16) | Halves memory → model fits in 4GB |
| Gradient Accumulation | batch 64 × 1 step = effective batch 64 |
| Pin Memory + non_blocking | Faster CPU→GPU data transfer |
| VRAM Cleanup | `torch.cuda.empty_cache()` between models |
| XGBoost `device=cuda` | GPU-accelerated tree boosting |
| `BCEWithLogitsLoss` | Numerically stable under AMP (fused sigmoid + BCE) |
| Gradient Clipping | `clip_grad_norm_(max_norm=1.0)` prevents exploding gradients |
| StandardScaler normalization | Prevents NaN loss from raw feature scale differences |

### Feature Cache

After the first run, features are cached to disk:
- `feature_cache_X.npy` — Feature matrix (81 features, normalized)
- `feature_cache_y.npy` — Label matrix (17-19 labels)
- `feature_cache_meta.pkl` — Metadata (label names, timestamps)
- `feature_cache_normalized.flag` — Normalization indicator

Subsequent runs skip feature extraction entirely and load from cache.

---

## 7. API Backend

**Source file**: `app.py`

### FastAPI Endpoints

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/api/stats` | GET | Dataset statistics (total stays, mean age, mortality rate, care unit distribution) |
| `/api/patients` | GET | Paginated patient list with search, risk filtering |
| `/api/patients/{id}/predictions` | GET | All prediction scores for an existing patient |
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
| **Patient Detail** | Click patient → vitals charts, lab trends, prediction scores, SHAP features |
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
├── data/                          # MIMIC-III CSV files (~43 GB)
├── config.yaml                    # All thresholds, model params, GPU config
├── data_loader.py                 # Loads + merges 14 CSV files
├── feature_engineering.py         # Extracts vitals/labs, creates sequences
├── training.py                    # GPU training loop, AMP, grad accumulation
├── models.py                      # LSTM, Transformer, XGBoost, LightGBM
├── main_pipeline.py               # Orchestrates everything end-to-end
├── rebuild_ensembles.py           # Rebuild ensemble pickles from checkpoints
├── show_predictions.py            # CLI display of training results
│
├── predictors/                    # One file per prediction task
│   ├── base_predictor.py          # MODELS_TO_TRY = ['lstm','transformer','xgboost','lightgbm']
│   ├── mortality_predictor.py     # 6/12/24h mortality
│   ├── sepsis_predictor.py        # SIRS + infection detection
│   ├── aki_predictor.py           # KDIGO staging (stages 1-3, 24/48h)
│   ├── vasopressor_predictor.py   # Drug requirement detection
│   ├── ventilation_predictor.py   # 3-layer vent detection
│   └── los_predictor.py           # Short/long stay classification
│
├── templates/                     # Jinja2 HTML templates
│   ├── base.html                  # Base layout with theme toggle
│   ├── index.html                 # Dashboard landing
│   ├── patients.html              # Patient list
│   ├── patient_detail.html        # Individual patient view
│   └── validation.html            # Model validation results
│
├── app.py                         # FastAPI backend
├── smart_icu_dashboard.html       # Standalone dashboard
├── models/                        # Saved model checkpoints (.pth, .pkl) — 36 files
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
- **StandardScaler normalization**: Added input normalization before caching — fixes NaN loss in DL models caused by raw feature scale differences (HR≈80, WBC≈10000)

### Training Bug Fixes

- **BCELoss → BCEWithLogitsLoss**: BCELoss is unsafe under AMP. Models output raw logits; `BCEWithLogitsLoss` fuses sigmoid + loss internally and is numerically stable under FP16
- **Label indices bug**: `set_label_indices()` was called **after** `train_all_models()` instead of before — root cause of AUROC = 0 for all DL models (they received all labels instead of task-specific columns)
- **NaN loss handling**: Added `not np.isnan(val_loss)` check for checkpointing; NaN epochs are skipped without incrementing patience counter
- **best_model.pth conflicts**: All models saved to same path → crash when checkpoint missing. Fixed with unique temp paths (`models/_best_<uuid>.pth`)
- **"Mean of empty slice" warnings**: All mortality labels in validation batch were 0 → AUROC = NaN. Now filters NaN values before averaging
- **Gradient clipping**: Added `clip_grad_norm_(max_norm=1.0)` to prevent exploding gradients producing inf/NaN loss
- **Early stopping tuning**: Patience raised 10 → 20, `min_delta=1e-4` added
- **Ensemble macro metrics bug**: AUPRC/F1/Sensitivity were 0.0000 — fixed by computing per-task metrics then averaging

### Model Changes

- **TCN removed**: `nn.BatchNorm1d` in TCNBlock collapses under FP16 AMP when mini-batches contain all-zero labels (common with rare ICU events). Running mean/variance → 0, normalization divides near-zero by near-zero, producing NaN loss from epoch 2-7 on every task
- **LightGBM added**: Leaf-wise growth provides different perspective from XGBoost's level-wise approach, adding model diversity to the ensemble
- **Ensemble methods added**: AUROC²-weighted averaging + stacking meta-learner with LogisticRegression
- **Per-task threshold tuning**: Optimizes F1 on validation set instead of hardcoded 0.5
- **Hypotension predictor removed**: Overlaps with vasopressor prediction; MAP threshold already captured in clinical rules
- **Readmission predictor removed**: Relies on static/summary features with minimal sequential dependency; not suitable for the time-series pipeline

### Windows Compatibility

- **DataLoader `num_workers`**: `num_workers=2` crashes on Windows without `if __name__ == '__main__'` guard. Added `_safe_num_workers()` forcing `num_workers=0` on Windows
- **`torch.compile` suppressed**: Triton not supported on Windows

---

*Last updated: April 2026*
