# Smart ICU Assistant — Complete System Architecture

A deep technical explanation of every layer, from raw CSVs to clinical predictions.

---

## 1. Data Loading (`data_loader.py`)

### What is MIMIC-III?

MIMIC-III is a publicly available clinical database of **~46,000 patients** who were admitted to the ICU at Beth Israel Deaconess Medical Center (Boston) between 2001–2012. All dates are **shifted to the 2100–2200 range** for de-identification, but are internally consistent per patient (relative time differences are preserved). Patients >89 years old have DOB shifted ~300 years before admission (we clamp these to 91.4 years).

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
| 14 | `TRANSFERS.csv` | 261K | Patient transfers between units | ICU readmission detection |

### CHARTEVENTS Chunked Loading (the hard part)

CHARTEVENTS is **33.6 GB** and **330 million rows**. It tracks **everything** the nurse charts: heart rate, blood pressure, ventilator settings, Glasgow Coma Scale, etc.

We can't load 330M rows into RAM. The solution:

```
1. Get the 32 relevant itemids we need (vitals + ventilation)
   → from _get_relevant_chartevents_itemids()
   → e.g., 220045 (HR), 220050/220051 (SysBP/DiaBP), 220210 (RR), etc.

2. Read CSV in chunks of 2,000,000 rows at a time
   → Uses pd.read_csv(filepath, chunksize=2_000_000)
   → Only loads 6 columns: SUBJECT_ID, HADM_ID, ICUSTAY_ID, ITEMID, CHARTTIME, VALUENUM

3. Filter each chunk: keep only rows where itemid ∈ {32 relevant IDs}
   → This drops ~85% of rows (ventilator settings, GCS, I/O, nursing notes)

4. Concatenate kept chunks → ~50M rows (from 330M)
   → Fits in ~4-6 GB RAM
```

### Merging

After loading all 14 tables, `merge_data()` joins them:

```
ICU_STAYS ← merge → PATIENTS (on subject_id)       → adds gender, DOB, DOD
          ← merge → ADMISSIONS (on subject_id + hadm_id) → adds admit/disch time, diagnosis

Then computes:
  age = (ICU intime - DOB) / 365.25 days
  hours_to_death = (DOD - ICU intime) in hours  ← KEY for mortality labels
```

Result: **61,532 rows** — one per ICU stay, with demographics, times, and death info.

---

## 2. Feature Engineering (`feature_engineering.py`)

### What features get extracted?

For each ICU stay, we extract two types of raw time series:

**8 Vital Signs** (from CHARTEVENTS):
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

**7 Lab Values** (from LABEVENTS):
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
shock_index   = heartrate / sysbp           # > 1.0 = hemodynamic instability
pulse_pressure = sysbp - diasbp             # < 25 = low cardiac output
meanbp        = (sysbp + 2 * diasbp) / 3   # < 65 mmHg = organ hypoperfusion
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

For each feature, we compute **rolling window statistics** (mean, std, min, max, trend) over 6h / 12h / 24h windows, turning ~15 raw features into **~90+ engineered features** per timepoint.

### Sequence Creation

```
Sliding window: sequence_length=24, step_size=6
Output: [n_sequences, 24, n_features]
```

---

## 3. Why Those Time Windows?

Each prediction task uses different horizons because clinical interventions have different lead times. Mortality/Sepsis/Ventilation use 6/12/24h; AKI uses 24/48h (kidneys are slow — creatinine lags injury by 12-24h); Hypotension uses 1/3/6h (organ damage is immediate); Vasopressor uses 6/12h.

---

## 4. Label Generation (Predictors)

### Full label vector: **19 binary labels per sequence**

```
[ mortality_6h, mortality_12h, mortality_24h,          # 1-3
  sepsis_6h, sepsis_12h, sepsis_24h,                   # 4-6
  aki_stage1_24h, aki_stage2_24h, aki_stage3_24h,      # 7-9
  aki_stage1_48h, aki_stage2_48h, aki_stage3_48h,      # 10-12
  vasopressor_6h, vasopressor_12h,                     # 13-14
  ventilation_6h, ventilation_12h, ventilation_24h,    # 15-17
  los_short_24h, los_long_72h ]                        # 18-19
```

---

## 5. Model Selection

### Why three models per task?

TCN was removed from the codebase (see below). We now train **3 models** per task:

| Model | Type | Strengths | Best for |
|-------|------|-----------|----------|
| **LSTM** | Recurrent Neural Net | Long-term memory, captures deterioration trends over hours | Mortality, Ventilation, Sepsis |
| **Transformer** | Attention-based | Cross-feature attention, finds subtle interactions | Sepsis (temp × HR × WBC interaction) |
| **XGBoost** | Gradient-boosted trees | Handles class imbalance, tabular data, interpretable | Vasopressor, LOS, Readmission, Mortality |

### Why TCN was removed

TCN (`TCNBlock`) used `nn.BatchNorm1d`. Under FP16 AMP with severely imbalanced labels (rare events: vasopressor requirement is <8% of sequences, AKI Stage 3 <3%), entire mini-batches frequently contain all-zero labels. BatchNorm's running mean/variance collapses toward zero, the normalization then divides near-zero by near-zero, producing NaN from epoch 2–7 onward on every task. This caused TCN to waste ~3 hours per task with AUROC stuck at 0.0000, while LSTM and XGBoost matched or exceeded its best-case performance.

The fix would be replacing `nn.BatchNorm1d` with `nn.GroupNorm(1, out_channels)`, but since LSTM + XGBoost already achieve state-of-the-art results on all confirmed tasks (mortality LSTM 0.82, XGBoost 0.87; sepsis LSTM 0.87), the complexity is not justified.

### How the best model is selected

```
For each task (e.g., "sepsis"):
  1. Train LSTM        → test AUROC = 0.871  ← BEST
  2. Train Transformer → test AUROC = ~0.80
  3. Train XGBoost     → test AUROC = ~0.85

  → Save LSTM as best model for sepsis
  → Save report to output/sepsis_report.json
```

---

## 6. Training Pipeline (`main_pipeline.py`)

### Step-by-step execution flow

```
python main_pipeline.py --data_dir data
```

Steps: Load Data → Pre-index → Feature Engineering → Label Generation → Train 3 Models Per Task → Train Readmission → Save Reports

### Temporal Split (no data leakage)

```
Sort by (shifted) timestamp — relative order preserved

|←————— 70% Train ————→|← 15% Val →|← 15% Test →|
   earliest shifted dates .................. latest
```

### GPU Optimizations (RTX 3050, 4GB VRAM)

```
Mixed Precision (FP16):   Halves memory → model fits in 4GB
Gradient Accumulation:    batch 32 × 2 steps = effective batch 64
Pin Memory:               Faster CPU→GPU data transfer
VRAM Cleanup:             torch.cuda.empty_cache() between models
XGBoost device=cuda:      GPU-accelerated tree boosting
```

---

## 7. File Architecture

```
fadnc-project-h/
├── data/                          # MIMIC-III CSV files (33+ GB)
├── config.yaml                    # All thresholds, model params, GPU config
│                                  # (TCN_CONFIG removed)
├── data_loader.py                 # Loads + merges 14 CSV files
├── feature_engineering.py         # Extracts vitals/labs, creates sequences
├── training.py                    # GPU training loop, AMP, grad accumulation
├── models.py                      # LSTM, Transformer, XGBoost architectures
│                                  # (TCNBlock, TCNModel removed)
├── main_pipeline.py               # Orchestrates everything end-to-end
│
├── predictors/                    # One file per prediction task
│   ├── base_predictor.py          # MODELS_TO_TRY = ['lstm','transformer','xgboost']
│   │                              # (tcn removed)
│   ├── mortality_predictor.py
│   ├── sepsis_predictor.py
│   ├── aki_predictor.py
│   ├── vasopressor_predictor.py
│   ├── ventilation_predictor.py
│   ├── los_predictor.py
│   └── readmission_predictor.py
│
├── models/                        # Saved checkpoints (.pth, .pkl)
│                                  # mortality_tcn.pth will no longer be created
└── output/                        # Training reports, SHAP plots
```