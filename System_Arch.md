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

Progress log:
  Read 10,000,000 rows, kept 4,698,698
  Read 20,000,000 rows, kept 9,466,518
  ...
  Read 330,000,000 rows, kept 49,212,073
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

### Derived Features (computed, not measured)

```python
# 1. Shock Index = Heart Rate / Systolic BP
#    Normal: 0.5–0.7   |   > 1.0 = hemodynamic instability → shock
shock_index = heartrate / sysbp

# 2. Pulse Pressure = Systolic - Diastolic
#    Normal: 40 mmHg   |   Narrow (<25) = low cardiac output
pulse_pressure = sysbp - diasbp

# 3. MAP (if not directly measured)
#    MAP = (SBP + 2×DBP) / 3
#    < 65 mmHg = insufficient organ perfusion
meanbp = (sysbp + 2 * diasbp) / 3
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

For each feature, we compute **rolling window statistics**:

```
Window sizes: 6h, 12h, 24h
Per window, per feature → 5 statistics:
  - mean    (average value in window)
  - std     (variability — high std = instability)
  - min     (worst low value)
  - max     (worst high value)
  - trend   (slope — rising or falling)

Example: heartrate over 6h → heartrate_6h_mean, heartrate_6h_std, etc.
```

This turns each timepoint from ~15 raw features into **~90+ engineered features**.

### Sequence Creation (the key step)

Deep learning models (LSTM, TCN) need **fixed-length sequences**. We use a **sliding window**:

```
Raw: hourly samples from ICU admission → discharge (e.g., 72 hours of data)

Sliding window parameters:
  sequence_length = 24 (look at 24 hours at a time)
  step_size = 6 (slide forward 6 hours between sequences)

Example for a 72-hour stay:
  Sequence 1: hours [0–23]   → predict what happens after hour 23
  Sequence 2: hours [6–29]   → predict what happens after hour 29
  Sequence 3: hours [12–35]  → predict what happens after hour 35
  ...
  Sequence 9: hours [48–71]  → predict what happens after hour 71

Output shape: [n_sequences, 24, n_features]
                ↑              ↑      ↑
           how many     24-hour   ~90 features
           windows      lookback  per timestep
```

---

## 3. Why Those Time Windows? (6h / 12h / 24h / 48h)

Each prediction task uses different time horizons because **clinical interventions have different lead times**:

### Mortality: `[6, 12, 24]` hours
- **6h**: Last-resort interventions (code blue, palliative care decision)
- **12h**: Time for family notification, escalation to attending physician
- **24h**: Standard clinical planning horizon (enough time for treatment adjustment)

### Sepsis: `[6, 12, 24]` hours
- **6h**: "Golden hour" concept — early antibiotics reduce mortality by 7% per hour
- **12h**: Time for blood cultures + targeted antibiotic switch
- **24h**: Full sepsis workup completion window

### AKI: `[24, 48]` hours
- **24h**: KDIGO AKI Stage 1 definition requires 0.3 mg/dL creatinine rise within 48h
- **48h**: KDIGO baseline comparison window — creatinine changes are slow (4-6h lab turnaround)
- Not shorter because: kidneys are slow organs — creatinine rises 12-24h **after** injury

### Hypotension: `[1, 3, 6]` hours
- **1h**: Emergency — MAP < 65 means organs are not getting blood RIGHT NOW
- **3h**: Fluid resuscitation window (3-hour sepsis bundle)
- **6h**: Vasopressor titration window
- Not 12h/24h because: hypotension that far out has too many confounders

### Vasopressor: `[6, 12]` hours
- **6h**: Urgent hemodynamic support needed soon
- **12h**: Operating room / procedure scheduling window
- Not 24h because: vasopressor need beyond 12h depends on treatment response

### Ventilation: `[6, 12, 24]` hours
- **6h**: Prepare respiratory therapy, intubation team
- **12h**: ICU bed planning (ventilated patients need different beds)
- **24h**: Resource allocation (ventilators are limited)

---

## 4. Label Generation (Predictors)

### What is a "label" and why do we need it?

A **label** is the ground truth answer the model needs to learn:
- Input: 24 hours of patient data (vitals + labs)
- Label: "Did this bad thing actually happen in the next X hours?" → **0 or 1**

Without labels, there's nothing to train on. Labels are generated **from the data itself** — we look at what actually happened to each patient and retroactively tag it.

### How each predictor generates labels:

#### Mortality Predictor
```
At prediction_time T:
  label = 1  IF  0 ≤ (date_of_death - T) ≤ window_hours
  label = 0  OTHERWISE

Example: Patient died at hour 50 of ICU stay
  At T=44:  mortality_6h = 1  (dies in 6h)
  At T=26:  mortality_24h = 1 (dies in 24h)
  At T=20:  mortality_6h = 0  (doesn't die in next 6h, dies in 30h)
```

#### Sepsis Predictor (SIRS + Infection)
```
At prediction_time T, looking forward window_hours:
  Step 1: Check SIRS criteria in future vitals/labs
          Need ≥ 2 of: fever, tachycardia, tachypnea, WBC abnormality
  Step 2: Check infection evidence:
          A. Antibiotic prescription started in the window (prescriptions table)
          B. Sepsis ICD-9 code on this admission (038, 995.91, 995.92)
  
  label = 1  IF SIRS ≥ 2 AND infection evidence found
  label = 0  OTHERWISE
```

#### AKI Predictor (KDIGO Criteria)
```
At prediction_time T:
  baseline_creatinine = lowest creatinine in past 48h
  future_creatinine   = highest creatinine in next 24h or 48h
  
  Stage 1: increase ≥ 0.3 mg/dL  OR  future ≥ 1.5× baseline
  Stage 2: future ≥ 2.0× baseline
  Stage 3: future ≥ 3.0× baseline  OR  future > 4.0 mg/dL

  Stages are cumulative: Stage 3 patient also gets Stage 2 = 1, Stage 1 = 1
```

#### Hypotension Predictor
```
At prediction_time T:
  label = 1  IF any MAP reading < 65 mmHg in next [1/3/6] hours
  label = 0  otherwise

  Simple but clinically critical — MAP < 65 = organ damage
```

#### Vasopressor Predictor
```
Two sources checked:
  1. PRESCRIPTIONS table: drug name contains vasopressor keywords
     (norepinephrine, epinephrine, vasopressin, dopamine, phenylephrine)
  2. INPUTEVENTS_MV table: itemid ∈ {221906, 221289, 222315, 221749, 221662}

  label = 1  IF any vasopressor started in future window
```

#### Ventilation Predictor
```
Three-layer detection (any one triggers label = 1):
  1. CHARTEVENTS: itemid ∈ {225792, 225794, 226260}  (vent-related charting)
  2. PROCEDUREEVENTS_MV: same itemids in procedure table
  3. PROCEDURES_ICD: ICD-9 codes 9670-9672, 9604, 9390

  label = 1  IF mechanical ventilation detected in future window
```

#### Length of Stay Predictor
```
remaining_hours = outtime - current_time

  los_short_24h = 1  IF remaining ≤ 24h  (will be discharged soon)
  los_long_72h  = 1  IF remaining > 72h  (still has 3+ days to go)
```

#### Readmission Predictor (different — tabular, not time-series)
```
For each ICU stay:
  label = 1  IF same patient has another ICU admission within 30 days of discharge
  label = 0  OTHERWISE

Uses 16 discharge-level features (age, LOS, # diagnoses, # drugs, etc.)
Not time-series — uses XGBoost, not LSTM
```

### Full label vector: **22 binary labels per sequence**

```
[ mortality_6h, mortality_12h, mortality_24h,          # 1-3
  sepsis_6h, sepsis_12h, sepsis_24h,                   # 4-6
  aki_stage1_24h, aki_stage2_24h, aki_stage3_24h,      # 7-9
  aki_stage1_48h, aki_stage2_48h, aki_stage3_48h,      # 10-12
  hypotension_1h, hypotension_3h, hypotension_6h,      # 13-15
  vasopressor_6h, vasopressor_12h,                     # 16-17
  ventilation_6h, ventilation_12h, ventilation_24h,    # 18-20
  los_short_24h, los_long_72h ]                        # 21-22
```

---

## 5. Model Selection

### Why multiple models per task?

Each task has different data characteristics. We train **4 models** per task and auto-select the best:

| Model | Type | Strengths | Best for |
|-------|------|-----------|----------|
| **LSTM** | Recurrent Neural Net | Long-term memory, captures deterioration trends over hours | Mortality, Ventilation (slow multi-organ decline) |
| **TCN** | Convolutional (temporal) | Fast local pattern detection, parallel computation | Hypotension (sudden MAP drops) |
| **Transformer** | Attention-based | Cross-feature attention, finds subtle interactions | Sepsis (temp × HR × WBC interaction) |
| **XGBoost** | Gradient-boosted trees | Handles class imbalance, tabular data, interpretable | Vasopressor, LOS, Readmission (imbalanced labels) |

### Model Architectures

**LSTM** (Bidirectional, 2-layer):
```
Input [batch, 24, features] 
  → Bidirectional LSTM (hidden=128, layers=2, dropout=0.3)
  → Attention pooling over time steps
  → Dropout (0.3)
  → Linear → Sigmoid → [batch, num_tasks]
```

**TCN** (Dilated Causal Convolutions):
```
Input [batch, 24, features] → transpose → [batch, features, 24]
  → TCNBlock (dilations: 1, 2, 4, 8) with residual connections
  → Channels: 64 → 128 → 256
  → Global Average Pooling
  → Linear → Sigmoid → [batch, num_tasks]
```

**XGBoost** (one model per label):
```
Input [n_samples, 24, features] → flatten → [n_samples, 24 × features]
  → 22 independent XGBClassifier models (one per label)
  → Each: max_depth=6, 100 trees, gpu_hist acceleration
```

### How the best model is selected

```
For each prediction task (e.g., "sepsis"):
  1. Train LSTM      → evaluate on test set → Mean AUROC = 0.72
  2. Train TCN       → evaluate on test set → Mean AUROC = 0.68
  3. Train Transformer → evaluate on test set → Mean AUROC = 0.75  ← BEST
  4. Train XGBoost   → evaluate on test set → Mean AUROC = 0.71

  → Save Transformer as best model for sepsis
  → Save report to output/sepsis_report.json
```

**AUROC** (Area Under ROC Curve) measures how well the model separates positive from negative cases:
- `0.5` = random guessing (useless)
- `0.7` = acceptable
- `0.8` = good
- `0.9+` = excellent

---

## 6. Training Pipeline (`main_pipeline.py`)

### Step-by-step execution flow:

```
python main_pipeline.py --data_dir data
```

```mermaid
graph TD
    A[Step 1: Load Data] --> B[Step 2: Pre-index]
    B --> C[Step 3: Feature Engineering]
    C --> D[Step 4: Label Generation]
    D --> E[Step 5: Train Predictors]
    E --> F[Step 6: Train Readmission]
    F --> G[Step 7: Save Reports]
    
    A -->|14 CSV files| A1[61,532 ICU stays merged]
    B -->|Group by icustay_id| B1[O(1) lookup dict]
    C -->|Per stay: vitals + labs| C1[24-step sequences × 90 features]
    D -->|Per sequence: 22 binary labels| D1[Label matrix: n × 22]
    E -->|Per task: 4 models, pick best| E1[Best model saved to models/]
```

### Temporal Split (no data leakage)

**Critical**: We can't randomly split ICU data — we'd leak future information into training.

```
Sort all sequences by (shifted) timestamp — relative order is preserved

|←————— 70% Train ————→|← 15% Val →|← 15% Test →|
   earliest shifted dates .................. latest shifted dates

- Training: only sees older data
- Validation: tunes hyperparameters on more recent data
- Test: final evaluation on the most recent data (never seen)

Note: MIMIC-III dates are shifted to 2100-2200, but intra-patient
ordering is preserved, so temporal splitting works correctly.
```

### GPU Optimizations (RTX 3050, 4GB VRAM)

```
Mixed Precision (FP16):   Halves memory → model fits in 4GB
Gradient Accumulation:    batch 32 × 2 steps = effective batch 64
Pin Memory:               Faster CPU→GPU data transfer
VRAM Cleanup:             torch.cuda.empty_cache() between models
XGBoost gpu_hist:         GPU-accelerated tree boosting
```

---

## 7. API Backend (`app.py`)

### FastAPI endpoints:

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/api/stats` | GET | Dataset statistics (total stays, mean age, mortality rate, care unit distribution) |
| `/api/patients` | GET | Paginated patient list with search, risk filtering |
| `/api/patients/{id}/predictions` | GET | All 22 prediction scores for an existing patient |
| `/api/patients/{id}/vitals` | GET | Time-series vital signs for charting |
| `/api/patients/{id}/labs` | GET | Time-series lab values for charting |
| `/api/alerts` | GET | All HIGH/MEDIUM risk patients with triggered labels |
| **`/api/predict`** | **POST** | **New patient assessment** — accepts vitals/labs/meds → returns risk scores |
| `/api/health` | GET | Server health check |

### `/api/predict` — The New Patient Endpoint

This is the **clinical decision support** endpoint. It works in two modes:

**Mode 1: Clinical Rules** (always available, no models needed)
```
Input: patient vitals + labs
  → SIRS score (0-4) from temp, HR, RR, WBC
  → KDIGO staging from creatinine
  → MAP threshold check (< 65 = hypotension)
  → Shock index (HR / SBP > 1.0 = shock)
  → Vasopressor/vent check from medications toggles
  → Composite score = weighted average
```

**Mode 2: Trained Models** (after pipeline completes)
```
Input: patient vitals + labs
  → Build 24-step feature vector (pad/replicate single measurement)
  → Run through each trained model (LSTM/TCN/XGBoost)
  → Get per-task probability scores
  → Merge with clinical rules (model scores take priority)
  → Generate alerts for high-risk categories
```

---

## 8. Dashboard Frontend (`smart_icu_dashboard.html`)

### 4 Tabs:

| Tab | Purpose |
|-----|---------|
| **Overview** | KPI cards (total stays, age, LOS, mortality) + risk distribution + care units + critical alerts |
| **Patient Detail** | Click patient → vitals charts, lab trends, all 22 prediction scores, SHAP features |
| **Alerts** | All high-risk patients sorted by composite score, with triggered labels |
| **⚡ New Assessment** | Input form for new patient → real-time risk prediction |

### Tech Stack:
- **React** (via Babel in-browser transpilation)
- **Recharts** for line charts (vitals, labs)
- **Custom CSS** with dark theme, glassmorphism, glow effects
- **No build step** — single HTML file, loads JS from `/static/js/`

---

## 9. File Architecture

```
SEM-4-PROJECT/
├── data/                          # MIMIC-III CSV files (33+ GB)
│   ├── PATIENTS.csv
│   ├── ADMISSIONS.csv
│   ├── ICUSTAYS.csv
│   ├── CHARTEVENTS.csv            # 33.6 GB — chunked loading
│   ├── LABEVENTS.csv
│   └── ...13 more tables
│
├── config.yaml                    # All thresholds, model params, GPU config
├── data_loader.py                 # Loads + merges 14 CSV files
├── feature_engineering.py         # Extracts vitals/labs, creates sequences
├── training.py                    # GPU training loop, AMP, grad accumulation
├── models.py                      # LSTM, TCN, Transformer, XGBoost architectures
├── main_pipeline.py               # Orchestrates everything end-to-end
│
├── predictors/                    # One file per prediction task
│   ├── base_predictor.py          # Abstract base: train/eval/save logic
│   ├── mortality_predictor.py     # 6/12/24h mortality
│   ├── sepsis_predictor.py        # SIRS + infection detection
│   ├── aki_predictor.py           # KDIGO staging
│   ├── hypotension_predictor.py   # MAP < 65 mmHg
│   ├── vasopressor_predictor.py   # Drug requirement detection
│   ├── ventilation_predictor.py   # 3-layer vent detection
│   ├── los_predictor.py           # Short/long stay
│   ├── readmission_predictor.py   # 30-day readmission (XGBoost + SHAP)
│   └── composite_predictor.py     # Unified deterioration score
│
├── app.py                         # FastAPI backend
├── smart_icu_dashboard.html       # React frontend
├── models/                        # Saved model checkpoints (.pth, .pkl)
└── output/                        # Training reports, SHAP plots
```
