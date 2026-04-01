# Smart ICU Assistant — Complete System Architecture

A deep technical explanation of every layer, from raw CSVs to clinical predictions.

---

## 1. Data Loading (`data_loader.py`)

### What is MIMIC-III?

MIMIC-III is a publicly available clinical database of **~46,000 patients** who were admitted to the ICU at Beth Israel Deaconess Medical Center (Boston) between 2001–2012. All dates are **shifted to the 2100–2200 range** for de-identification, but are internally consistent per patient (relative time differences are preserved). Patients >89 years old have DOB shifted ~300 years before admission (we clamp these to 91.4 years).

### The 14 CSV Files We Load

| Order | File | Rows (Full) | What it contains | Why we need it |
|-------|------|-------------|-----------------|----------------|
| 1 | `PATIENTS.csv` | 46K | Gender, DOB, DOD, expire_flag | Base patient demographics |
| 2 | `ADMISSIONS.csv` | 59K | Admit/discharge times, death time, diagnosis | Admission context |
| 3 | `ICUSTAYS.csv` | 61K | ICU in/out times, care unit, LOS | Core unit of analysis |
| 4 | `D_ITEMS.csv` | 12K | itemid → label dictionary | Must load before CHARTEVENTS |
| 4 | `D_LABITEMS.csv` | 753 | itemid → lab name dictionary | Maps lab item IDs |
| 5 | **`CHARTEVENTS.csv`** | **330M** | Timestamped bedside measurements | Main vitals source — 33.6 GB |
| 6 | `LABEVENTS.csv` | 27M | Lab results | Lab values for AKI, sepsis |
| 7 | `DIAGNOSES_ICD.csv` | 650K | ICD-9 diagnosis codes | Sepsis ICD codes |
| 8 | `PRESCRIPTIONS.csv` | 4M | Medication orders | Antibiotic detection |
| 9 | `INPUTEVENTS_MV.csv` | 3M | IV drip administration | Vasopressor detection |
| 10 | `OUTPUTEVENTS.csv` | 4M | Fluid outputs | AKI detection — urine output |
| 11 | `PROCEDUREEVENTS_MV.csv` | 258K | ICU procedures with timestamps | Ventilation detection |
| 12 | `PROCEDURES_ICD.csv` | 240K | ICD-9 procedure codes | Ventilation ICD codes |
| 13 | `MICROBIOLOGYEVENTS.csv` | 631K | Culture results | Infection evidence for sepsis |
| 14 | `TRANSFERS.csv` | 261K | Patient transfers | ICU readmission detection |

---

## 2. Feature Engineering (`feature_engineering.py`)

### Vital Signs (from CHARTEVENTS)

| Vital | Clinical Significance |
|-------|----------------------|
| Heart Rate (bpm) | Tachycardia > 90 = SIRS criterion |
| Systolic BP (mmHg) | Hypotension indicator |
| Diastolic BP (mmHg) | Used to compute MAP |
| Mean Arterial Pressure (mmHg) | MAP < 65 = hypotension |
| Respiratory Rate (/min) | RR > 20 = SIRS criterion |
| Temperature (°C) | Fever > 38.3°C or < 36°C = SIRS |
| SpO₂ (%) | < 94% = respiratory failure |
| Glucose (mg/dL) | Stress hyperglycemia |

### Lab Values (from LABEVENTS)

| Lab | Clinical Significance |
|-----|----------------------|
| Creatinine (mg/dL) | AKI detection |
| Lactate (mmol/L) | Tissue hypoxia marker |
| WBC (/µL) | > 12K or < 4K = SIRS |
| Hemoglobin (g/dL) | Anemia, bleeding |
| Platelets (×10³) | Low = DIC, sepsis coagulopathy |
| Bicarbonate (mEq/L) | Metabolic acidosis |
| Chloride (mEq/L) | Electrolyte balance |

### Derived Features

```python
shock_index   = heartrate / sysbp          # > 1.0 = hemodynamic instability
pulse_pressure = sysbp - diasbp            # Narrow < 25 = low cardiac output
meanbp        = (sysbp + 2 * diasbp) / 3  # < 65 mmHg = insufficient perfusion
```

### SIRS Score (0–4)

```
+1 temperature > 38.3°C or < 36.0°C
+1 heart rate > 90 bpm
+1 respiratory rate > 20 /min
+1 WBC > 12,000 or < 4,000 /µL

Score ≥ 2 = systemic inflammatory response
Score ≥ 2 + infection evidence = SEPSIS
```

### Sequence Creation

```
sequence_length = 24   (look at 24 hours at a time)
step_size       = 6    (slide forward 6 hours)

Output shape: [n_sequences, 24, n_features]
```

---

## 3. Why Those Time Windows?

| Task | Windows | Reason |
|------|---------|--------|
| Mortality | 6/12/24h | 6h = code blue; 12h = family notification; 24h = clinical planning |
| Sepsis | 6/12/24h | Golden hour — each hour of delayed antibiotics increases mortality ~7% |
| AKI | 24/48h | Creatinine is a slow marker — rises 12-24h after renal injury |
| Hypotension | 1/3/6h | MAP < 65 is an emergency requiring immediate response |
| Vasopressor | 6/12h | Hemodynamic support scheduling window |
| Ventilation | 6/12/24h | Respiratory therapy preparation + ICU bed planning |

---

## 4. Model Selection

### Why TCN was removed

`TCNBlock` uses `nn.BatchNorm1d`. Under FP16 AMP with severely imbalanced ICU labels (rare events like vasopressor, AKI Stage 3), entire mini-batches often have all-zero labels. BatchNorm's running mean/variance collapses to zero, normalization divides near-zero by near-zero, and every epoch from ~epoch 3-7 onward produces NaN loss. This was confirmed across all 6 tasks in the current training run. The fix (replacing `BatchNorm1d` with `GroupNorm`) would be a 1-line change but the model's AUROC gains over LSTM were marginal (0.8199 vs 0.8239 on mortality). TCN was wasting ~3 hours per task for no meaningful benefit.

### Active Models

| Model | Type | Strengths | Best for |
|-------|------|-----------|----------|
| **LSTM** | Recurrent | Long-term memory, captures deterioration trends | Mortality, Ventilation, Sepsis |
| **Transformer** | Attention | Cross-feature interactions (temp × HR × WBC) | Sepsis, AKI |
| **XGBoost** | Gradient-boosted trees | Class imbalance, tabular features, fast | All tasks (frequent winner) |

### LSTM Architecture

```
Input [batch, 24, features]
  → Bidirectional LSTM (hidden=128, layers=2, dropout=0.3)
  → Last hidden state
  → Dropout (0.3)
  → Per-task linear heads → Sigmoid → [batch, num_tasks]
```

### XGBoost Architecture

```
Input [n_samples, 24, features] → flatten → [n_samples, 24 × features]
  → One XGBClassifier per label (independent)
  → Each: max_depth=6, 100 trees, device=cuda
```

### Model Selection

```
For each task:
  1. Train LSTM       → Mean AUROC on test set
  2. Train Transformer → Mean AUROC on test set
  3. Train XGBoost    → Mean AUROC on test set
  → Save best model, write report to output/{task}_report.json
```

---

## 5. Label Generation

### Full label vector: 22 binary labels per sequence

```
[ mortality_6h,  mortality_12h,  mortality_24h,           # 1-3
  sepsis_6h,     sepsis_12h,     sepsis_24h,              # 4-6
  aki_stage1_24h, aki_stage2_24h, aki_stage3_24h,         # 7-9
  aki_stage1_48h, aki_stage2_48h, aki_stage3_48h,         # 10-12
  hypotension_1h, hypotension_3h, hypotension_6h,         # 13-15
  vasopressor_6h, vasopressor_12h,                        # 16-17
  ventilation_6h, ventilation_12h, ventilation_24h,       # 18-20
  los_short_24h,  los_long_72h ]                          # 21-22
```

---

## 6. Training Pipeline (`main_pipeline.py`)

```
python main_pipeline.py --data_dir data
```

```
Step 1: Load 14 CSV files → 61,532 ICU stays
Step 2: Pre-index chartevents/labevents by icustay_id (O(1) lookup)
Step 3: Feature extraction → sliding-window sequences × 90 features
Step 4: StandardScaler normalization (prevents NaN loss)
Step 5: Train 3 models per task, pick best AUROC
Step 6: Train readmission (XGBoost + SHAP)
Step 7: Save reports
```

### Temporal Split (no data leakage)

```
Sort all sequences by (shifted) timestamp

|←——— 70% Train ———→|← 15% Val →|← 15% Test →|
  earliest dates .............. latest dates
```

### GPU Optimizations

```
Mixed Precision (FP16)   — halves VRAM usage
Gradient Accumulation    — batch 32 × 2 = effective batch 64
Pin Memory               — faster CPU→GPU transfer
XGBoost device=cuda      — GPU-accelerated tree boosting
```

---

## 7. File Architecture

```
fadnc-project-h/
├── config.yaml                    # All thresholds, model params (TCN block removed)
├── data_loader.py                 # Loads + merges 14 CSV files
├── feature_engineering.py         # Extracts vitals/labs, creates sequences
├── training.py                    # GPU training loop, AMP, grad accumulation
├── models.py                      # LSTM, Transformer, XGBoost (TCN removed)
├── main_pipeline.py               # Orchestrates everything end-to-end
├── normalize_cache.py             # One-shot cache normalization utility
├── show_predictions.py            # Prints per-task results table
│
├── predictors/
│   ├── base_predictor.py          # MODELS_TO_TRY = [lstm, transformer, xgboost]
│   ├── mortality_predictor.py
│   ├── sepsis_predictor.py
│   ├── aki_predictor.py
│   ├── vasopressor_predictor.py
│   ├── ventilation_predictor.py
│   ├── los_predictor.py
│   └── readmission_predictor.py
│
├── data/                          # MIMIC-III CSV files
├── models/                        # Saved checkpoints (.pth, .pkl)
└── output/                        # Training reports, SHAP plots
```