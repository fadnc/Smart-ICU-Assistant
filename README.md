# Smart ICU Assistant

A machine learning pipeline for real-time ICU patient monitoring, clinical risk prediction, and alerting. The system processes MIMIC-III clinical data to predict seven critical outcomes across 19 binary classification labels, using an ensemble of deep learning and gradient-boosted tree models. A FastAPI-powered dashboard provides real-time visualization of patient vitals, lab trends, risk scores, and SHAP-based feature importance.

---

## Table of Contents

- [Overview](#overview)
- [Clinical Prediction Tasks](#clinical-prediction-tasks)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Model Architectures](#model-architectures)
- [Training Pipeline](#training-pipeline)
- [Ensemble Strategy](#ensemble-strategy)
- [Web Dashboard](#web-dashboard)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Hardware Requirements](#hardware-requirements)
- [License](#license)

---

## Overview

The Smart ICU Assistant addresses the need for early warning systems in intensive care units. It ingests patient vitals (heart rate, blood pressure, SpO2, respiratory rate, temperature, glucose) and laboratory values (creatinine, lactate, WBC, hemoglobin, platelets, bicarbonate, chloride) to generate risk predictions across seven clinical categories. Each predictor is trained on MIMIC-III data using temporal cross-validation and evaluated with clinically relevant metrics (AUROC, AUPRC, sensitivity, specificity, Brier score).

### Key Features

- **19 prediction labels** across 7 clinical tasks with configurable time horizons
- **4 model types** per task: LSTM, Transformer, XGBoost, LightGBM
- **Automated ensemble**: AUROC-squared weighted averaging and stacking meta-learner
- **Per-task threshold tuning**: optimizes F1 score on validation set instead of hardcoded 0.5
- **Mixed-precision training** (FP16 AMP) with NaN-safe loss computation
- **Feature caching** with normalization flags to skip redundant preprocessing
- **SHAP interpretability** for the readmission predictor
- **Real-time dashboard** with vitals charts, lab trends, risk scoring, and clinical alerts

---

## Clinical Prediction Tasks

The system implements seven prediction tasks producing 19 binary classification labels:

| No. | Task | Labels | Time Windows | Clinical Definition |
|-----|------|--------|-------------|---------------------|
| 1 | **Mortality** | `mortality_6h`, `mortality_12h`, `mortality_24h` | 6, 12, 24 hours | Patient has a recorded date of death within the prediction window |
| 2 | **Sepsis** | `sepsis_6h`, `sepsis_12h`, `sepsis_24h` | 6, 12, 24 hours | SIRS criteria (>=2 of: temp >38.3 or <36.0, HR >90, RR >20, WBC >12 or <4 K/uL) plus infection evidence (antibiotic prescription or ICD-9 sepsis codes 038, 995.91, 995.92, 785.52) |
| 3 | **Acute Kidney Injury** | `aki_stage1_24h`, `aki_stage2_24h`, `aki_stage3_24h`, `aki_stage1_48h`, `aki_stage2_48h`, `aki_stage3_48h` | 24, 48 hours | KDIGO staging based on creatinine kinetics. Stage 1: increase >=0.3 mg/dL or >=1.5x baseline. Stage 2: >=2.0x baseline. Stage 3: >=3.0x baseline or >4.0 mg/dL absolute |
| 4 | **Vasopressor** | `vasopressor_6h`, `vasopressor_12h` | 6, 12 hours | Vasopressor drug (norepinephrine, epinephrine, vasopressin, dopamine, dobutamine, phenylephrine, milrinone) prescribed or administered via IV within the window |
| 5 | **Ventilation** | `ventilation_6h`, `ventilation_12h`, `ventilation_24h` | 6, 12, 24 hours | Three-layer detection: CHARTEVENTS ventilation itemids (225792, 225794, 226260), PROCEDUREEVENTS_MV, or ICD-9 procedure codes (9670-9672, 9604, 9390) |
| 6 | **Length of Stay** | `los_short_24h`, `los_long_72h` | 24, 72 hours | Binary classification of remaining ICU stay. Short: discharged within 24 hours. Long: >72 hours remaining |
| 7 | **Readmission** | `readmission` | 30 days | Same patient has another ICU admission within 30 days of discharge. Uses tabular discharge features with XGBoost and SHAP explanations |

---

## System Architecture

```
                    MIMIC-III CSV Files
                          |
                    [Data Loader]
                    17 clinical tables
                          |
                 [Feature Engineering]
                 Vital signs + lab values
                 Rolling aggregations
                 Time-window sequences
                          |
              [Label Generation per Task]
              7 predictors x configurable windows
                          |
           [Model Training (GPU / CPU)]
           LSTM | Transformer | XGBoost | LightGBM
           AMP (FP16) | Early stopping | LR scheduling
                          |
             [Ensemble Construction]
             AUROC2-weighted averaging
             Stacking meta-learner (LogisticRegression)
                          |
               [Model Checkpoints]
               .pth (DL) | .pkl (tree-based)
                          |
                [FastAPI Dashboard]
                Real-time predictions
                Vitals + Lab charts
                Risk alerts + SHAP
```

---

## Project Structure

```
SEM-4-PROJECT/
|-- app.py                        # FastAPI backend (dashboard + API endpoints)
|-- main_pipeline.py              # Training pipeline orchestrator
|-- models.py                     # Model architectures (LSTM, Transformer, XGBoost, LightGBM, TabTransformer, TFT)
|-- training.py                   # GPU training loop with AMP, early stopping, scheduling
|-- feature_engineering.py        # Feature extraction, rolling windows, sequence creation
|-- data_loader.py                # MIMIC-III CSV loading with chunked I/O
|-- config.yaml                   # All hyperparameters, clinical thresholds, feature lists
|-- requirements.txt              # Python dependencies
|-- rebuild_ensembles.py          # Rebuild ensemble pickles from existing model checkpoints
|-- show_predictions.py           # CLI display of training results per task
|-- smart_icu_dashboard.html      # Standalone React dashboard (single-file)
|
|-- predictors/
|   |-- __init__.py               # Package exports
|   |-- base_predictor.py         # Abstract base class with ensemble, threshold tuning, metrics
|   |-- mortality_predictor.py    # Tasks 1-3: ICU mortality at 6/12/24h
|   |-- sepsis_predictor.py       # Tasks 4-6: Sepsis onset (SIRS + infection)
|   |-- aki_predictor.py          # Tasks 7-12: AKI KDIGO stages 1-3 at 24/48h
|   |-- vasopressor_predictor.py  # Tasks 13-14: Vasopressor requirement at 6/12h
|   |-- ventilation_predictor.py  # Tasks 15-17: Mechanical ventilation at 6/12/24h
|   |-- los_predictor.py          # Tasks 18-19: Length of stay (short/long)
|   |-- readmission_predictor.py  # Task 20: 30-day ICU readmission (XGBoost + SHAP)
|
|-- templates/
|   |-- base.html                 # Jinja2 base layout with theme toggle
|   |-- index.html                # Dashboard landing page
|   |-- patients.html             # Patient list view
|   |-- patient_detail.html       # Individual patient detail view
|   |-- validation.html           # Model validation results page
|
|-- data/                         # MIMIC-III CSV files (not included in repo)
|-- models/                       # Trained model checkpoints (.pth, .pkl)
|-- output/                       # Training reports, feature cache, SHAP plots
```

---

## Model Architectures

### LSTM with Attention

Bidirectional LSTM with temporal attention pooling. Two stacked layers with dropout, followed by an attention mechanism that learns which timesteps are most informative for prediction. Outputs are raw logits passed through a shared linear head.

- Hidden size: 128
- Layers: 2 (bidirectional)
- Dropout: 0.3
- Learning rate: 0.001

### Transformer Encoder

Multi-head self-attention encoder with Pre-LayerNorm (norm_first=True) for FP16 stability. Positional encoding is added to the input sequence. The final representation is obtained by mean-pooling the encoder output across the temporal dimension.

- d_model: 128
- Attention heads: 8
- Encoder layers: 3
- Feed-forward dimension: 256
- Learning rate: 0.0001 (10x lower than LSTM to prevent attention overflow under AMP)

### XGBoost

Gradient-boosted decision trees using the XGBoost 2.x API. One binary classifier per label with per-task `scale_pos_weight` computed from class imbalance ratios. The input is flattened from (batch, timesteps, features) to (batch, timesteps * features).

- Max depth: 8
- Estimators: 300 with early stopping (20 rounds)
- Subsample: 0.8
- Tree method: hist (GPU-accelerated via `device: cuda`)

### LightGBM

Leaf-wise gradient boosting providing a different inductive bias from XGBoost. Adds model diversity to the ensemble. Per-task `scale_pos_weight` for class imbalance handling.

- Num leaves: 63
- Estimators: 300 with early stopping (20 rounds)
- Learning rate: 0.05
- Device: CPU (GPU requires separate LightGBM build)

### Additional Architectures (configurable)

- **MultitaskLSTM**: Shared LSTM backbone with task-group-specific heads. Produces individual predictions plus a composite deterioration score.
- **TabTransformer**: Column-wise attention over features with learned embeddings per feature, then temporal pooling.
- **Temporal Fusion Transformer (TFT)**: Variable selection networks with interpretable temporal attention for multivariate time-series.

---

## Training Pipeline

### Data Flow

1. **Data Loading** (`data_loader.py`): Loads 17 MIMIC-III tables (PATIENTS, ADMISSIONS, ICUSTAYS, CHARTEVENTS, LABEVENTS, PRESCRIPTIONS, DIAGNOSES_ICD, PROCEDURES_ICD, INPUTEVENTS_MV, OUTPUTEVENTS, SERVICES, D_ITEMS, D_LABITEMS, etc.) with chunked reading for large files (CHARTEVENTS can exceed 30 GB).

2. **Feature Engineering** (`feature_engineering.py`): Extracts 81 features per timestep from vital signs and lab values. Applies rolling aggregations (mean, std, min, max, trend) over configurable windows. Creates fixed-length sequences (default: 24 timesteps with step size 6).

3. **Label Generation** (`predictors/*.py`): Each predictor implements `generate_labels()` using clinically validated definitions. Labels are generated at each observation timepoint by looking forward into the prediction window.

4. **Normalization**: StandardScaler is fit on the training set and applied to all splits. A `feature_cache_normalized.flag` file tracks whether the cache contains normalized data. Unnormalized features will trigger a warning.

5. **Training** (`training.py`): GPU training loop with:
   - Mixed-precision training (FP16 AMP) with `GradScaler`
   - `BCEWithLogitsLoss` with per-task `pos_weight` for class imbalance
   - Label smoothing (configurable, default: 0.05)
   - Linear warmup followed by ReduceLROnPlateau or CosineAnnealing scheduler
   - Early stopping with configurable patience (default: 20 epochs)
   - NaN loss detection with automatic abort after N consecutive NaN epochs
   - Gradient accumulation (configurable steps)

6. **Temporal Split**: Data is split chronologically (not randomly) into train/val/test sets to prevent temporal leakage.

### Feature Cache

After the first run, features are cached to disk as NumPy arrays:
- `feature_cache_X.npy`: Feature matrix (memory-mapped for large datasets)
- `feature_cache_y.npy`: Label matrix
- `feature_cache_meta.pkl`: Metadata (label names, timestamps, scaler)
- `feature_cache_normalized.flag`: Normalization indicator

Subsequent runs skip feature extraction entirely and load from cache.

---

## Ensemble Strategy

Two ensemble methods are applied when enabled in config:

### AUROC-Squared Weighted Averaging

Each model's test AUROC is squared to compute weights, amplifying the contribution of better-performing models. The weighted average of predicted probabilities produces the ensemble output.

```
weight_i = auroc_i^2 / sum(auroc_j^2)
ensemble_pred = sum(weight_i * pred_i)
```

### Stacking Meta-Learner

A LogisticRegression meta-learner is trained per label on the base models' predictions. To avoid data leakage, the test set is split in half: the first half trains the meta-learner, the second half is used for evaluation.

Both ensemble types are serialized as `.pkl` files containing component model references, weights, and meta-learners for inference.

---

## Web Dashboard

The dashboard is served by FastAPI and provides:

### Pages

- **Overview**: KPI tiles (total stays, mean age, mean LOS, mortality rate), risk distribution bar chart, care unit breakdown, critical alerts
- **Patient List**: Searchable, filterable patient list with composite risk scores and pagination
- **Patient Detail**: Demographics, current vitals with normal range indicators, HR/MAP and SpO2/RR time-series charts, creatinine/lactate and WBC/hemoglobin lab trend charts, all 19 prediction scores grouped by task category, SHAP feature importance for readmission
- **New Assessment**: Manual data entry form for clinical parameters (demographics, vitals, labs, medications, ICU context) with instant risk prediction
- **Validation**: Model performance metrics and training reports
- **Alerts**: Active critical and warning alerts sorted by severity and score

### Composite Risk Score

A weighted aggregation of task-specific predictions used for patient triage:

```
composite = mortality_24h * 0.30
          + sepsis_24h    * 0.25
          + aki_stage1_24h * 0.15
          + vasopressor_12h * 0.15
          + ventilation_24h * 0.15
```

Patients are classified as HIGH (>0.6), MEDIUM (>0.3), or LOW risk.

### Clinical Alerts

Rule-based alerts are generated from vital signs and lab values:
- Mortality risk >50%
- SIRS score >=3/4 or SIRS >=2 with antibiotics
- Creatinine >2.0 mg/dL (critical) or >1.3 mg/dL (warning)
- MAP <65 mmHg (critical) or <70 mmHg (warning)
- Active vasopressor administration
- SpO2 <90% (critical) or <94% with RR >24 (warning)
- Shock index >1.0
- Lactate >4.0 mmol/L (critical) or >2.0 mmol/L (warning)

---

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended, not required)
- MIMIC-III dataset (requires PhysioNet credentialed access)

### Setup

```bash
# Clone the repository
git clone https://github.com/fadnc/SEM-4-PROJECT.git
cd SEM-4-PROJECT

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### MIMIC-III Data

Place the following CSV files in the `data/` directory:

| Required Files | Description |
|---------------|-------------|
| PATIENTS.csv | Patient demographics |
| ADMISSIONS.csv | Hospital admissions |
| ICUSTAYS.csv | ICU stay records |
| CHARTEVENTS.csv | Charted vital signs |
| LABEVENTS.csv | Laboratory test results |
| D_ITEMS.csv | Item dictionary (chartevents) |
| D_LABITEMS.csv | Item dictionary (labevents) |
| PRESCRIPTIONS.csv | Medication prescriptions |
| DIAGNOSES_ICD.csv | ICD-9 diagnosis codes |
| PROCEDURES_ICD.csv | ICD-9 procedure codes |
| INPUTEVENTS_MV.csv | IV fluid/drug inputs (MetaVision) |
| OUTPUTEVENTS.csv | Patient outputs (urine, drains) |
| SERVICES.csv | Service transfers |

---

## Configuration

All parameters are centralized in `config.yaml`:

```yaml
# Time windows per task (hours)
TIME_WINDOWS:
  mortality: [6, 12, 24]
  sepsis: [6, 12, 24]
  aki: [24, 48]
  vasopressor: [6, 12]
  ventilation: [6, 12, 24]

# GPU training settings
GPU_CONFIG:
  batch_size: 64
  use_amp: true
  num_workers: 0          # Required on Windows

# Ensemble
ENSEMBLE:
  enabled: true

# Per-task dropout overrides
TASK_OVERRIDES:
  vasopressor:
    dropout: 0.5          # Higher dropout for tasks with severe overfitting
  sepsis:
    dropout: 0.5
```

See `config.yaml` for the full list of configurable parameters including model hyperparameters, clinical thresholds (KDIGO stages, SIRS criteria, ventilation itemids, vasopressor itemids), early stopping settings, and learning rate scheduler options.

---

## Usage

### Training the Pipeline

```bash
# Full training run (all 7 tasks)
python main_pipeline.py --data_dir data/

# The pipeline will:
# 1. Load MIMIC-III data (17 tables)
# 2. Extract features and generate labels
# 3. Cache features to output/ for future runs
# 4. Train 4 models per task (LSTM, Transformer, XGBoost, LightGBM)
# 5. Build ensembles (weighted + stacking)
# 6. Train readmission predictor separately (discharge features + SHAP)
# 7. Save checkpoints to models/ and reports to output/
```

### Viewing Training Results

```bash
python show_predictions.py
```

### Rebuilding Ensembles (No GPU Needed)

```bash
python rebuild_ensembles.py
```

### Running the Dashboard

```bash
# Start the FastAPI server
python app.py

# Or with uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Access the dashboard at `http://localhost:8000`. API documentation is available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc` (ReDoc).

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard HTML page |
| GET | `/api/health` | Health check |
| GET | `/api/stats` | Aggregate ICU statistics (total stays, mortality rate, care units) |
| GET | `/api/patients` | Patient list with search, risk filter, and pagination |
| GET | `/api/patients/{icustay_id}` | Individual patient details |
| GET | `/api/patients/{icustay_id}/vitals` | Hourly vital sign time-series |
| GET | `/api/patients/{icustay_id}/labs` | 6-hourly laboratory values |
| GET | `/api/patients/{icustay_id}/predictions` | All 19 prediction scores with composite risk |
| GET | `/api/alerts` | Active critical and warning alerts |
| POST | `/api/predict` | Manual clinical assessment with custom input data |
| GET | `/docs` | Swagger UI (auto-generated) |
| GET | `/redoc` | ReDoc documentation (auto-generated) |

### Query Parameters for `/api/patients`

| Parameter | Type | Description |
|-----------|------|-------------|
| `q` | string | Search by ICU stay ID, diagnosis, or care unit |
| `risk` | string | Filter by risk level: HIGH, MEDIUM, or LOW |
| `page` | int | Page number (default: 1) |
| `per_page` | int | Results per page (default: 18) |

---

## Hardware Requirements

### Minimum

- CPU: 4 cores
- RAM: 16 GB
- Storage: 50 GB (MIMIC-III is approximately 40 GB uncompressed)
- GPU: Not required (XGBoost and LightGBM run on CPU)

### Recommended (for full training)

- CPU: 8+ cores
- RAM: 32 GB
- GPU: NVIDIA RTX 3050 or higher (4 GB VRAM minimum)
- CUDA: 11.8 or higher
- Storage: 100 GB SSD

The pipeline is optimized for constrained GPU environments. Key optimizations include:
- Batch size 64 with gradient accumulation step 1
- Mixed-precision training (FP16) to halve VRAM usage
- `num_workers=0` for Windows compatibility
- `torch.compile` suppression on Windows (Triton not supported)
- TF32 enabled on Ampere+ GPUs for faster matmul

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

### Data License

MIMIC-III data is subject to the PhysioNet Credentialed Health Data License. Access requires completion of the CITI training course and a signed data use agreement. The data files are not included in this repository.

---

## Acknowledgments

- MIMIC-III Clinical Database: Johnson, A., Pollard, T., Shen, L. et al. MIMIC-III, a freely accessible critical care database. Scientific Data 3, 160035 (2016).
- KDIGO Acute Kidney Injury Guidelines
- Surviving Sepsis Campaign: International Guidelines for Management of Sepsis and Septic Shock
