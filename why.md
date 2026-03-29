## MIMIC-III Date Shifting
All dates are shifted to 2100-2200 range for patient privacy. The shift is:
- **Consistent per patient**: intime - dob gives correct age, dod - intime gives correct hours_to_death
- **Random per patient**: two patients on "2150-01-01" were NOT in the ICU at the same time
- **Preserves**: time of day, day of week, seasonality
- **Removes**: absolute year, day of month, inter-patient timing
- **Special case**: patients >89 years old have DOB shifted ~300 years before first admission → ages show as 300+. We clamp to 91.4.
- Actual data was collected between 2001-2012 at Beth Israel Deaconess Medical Center (Boston).

CHARTEVENTS is 33.6 GB. Cannot load into RAM all at once. Must use chunked processing or Parquet conversion.

Tables not worth adding (6 skip)
NOTEEVENTS — requires NLP, different model type entirely
DATETIMEEVENTS — non-numeric events, low predictive value
CPTEVENTS / DRGCODES / D_CPT — billing codes, no clinical value
CAREGIVERS — caregiver ID doesn't predict patient outcomes

The refactoring is strongly recommended. The modularity benefits for a 2-person team project far outweigh the small overhead of more files. The base class approach eliminates code duplication, and the independent validation capability will save significant debugging time.

Mortality	LSTM or XGBoost	Long memory for multi-organ failure
Sepsis	Transformer or XGBoost	Cross-feature attention for subtle interactions
AKI	XGBoost or LSTM	Handles creatinine kinetics well
Hypotension	TCN or XGBoost	Short-term temporal patterns
Vasopressor	XGBoost	Best with class imbalance
Ventilation	LSTM or MultitaskLSTM	Sequential respiratory patterns
Readmission	XGBoost + SHAP	Tabular features, interpretability
Composite	MultitaskLSTM	Shared encoder by design

Metric	Demo (Current)	Full MIMIC-III (Expected)
Patients	100	~46,000
ICU Stays	136	~61,000
Sequences	1,931	~500,000+
CHARTEVENTS	758K rows	~330 million rows
Training time	~10 min	~2-8 hours (GPU recommended)
RAM needed	~2 GB	~16-32 GB (with chunking)


This will take ~30-45 minutes to load the full 33.6 GB CHARTEVENTS file via chunked reading. After that, feature extraction and model training will take another ~1-2 hours on CPU.

 CHARTEVENTS loading is at 10M rows. This will take 30-45 minutes for the 330M row file.


to run : python main_pipeline.py --data_dir data 2>&1

training.py — Core GPU optimizations:

Mixed Precision (FP16) via torch.amp.autocast — halves VRAM usage (~2× memory savings)
Gradient Accumulation — batch 32 × 2 steps = effective batch 64 (fits in 4GB)
Pin Memory + non_blocking=True — faster CPU→GPU data transfers
get_device() — auto-detects your RTX 3050, logs name + VRAM
clear_gpu_memory() — frees VRAM cache between model trainings
config.yaml — New GPU_CONFIG section:

yaml
GPU_CONFIG:
  batch_size: 32           # Safe for 4GB VRAM
  grad_accum_steps: 2      # Effective batch = 64
  use_amp: true            # FP16 mixed precision
models.py — XGBoost GPU:

tree_method: gpu_hist — GPU-accelerated tree boosting
Auto-falls back to CPU if no CUDA
base_predictor.py — VRAM cleanup:

Calls clear_gpu_memory() before each model training to prevent OOM

The pipeline processed all 61K stays successfully but crashed creating the final numpy array — 14.2 GB needed, only ~1.4 GB RAM free. Two quick fixes will solve this:

float64 → float32: Halves memory from 14.2 GB → 7.1 GB (sufficient for ML, no precision loss)
Pre-allocate instead of list→array: Avoids the 2× memory spike during conversion

We identified three technical issues during training. First, a PyTorch AMP compatibility issue with BCELoss prevented our deep learning models from training — this is a known issue with a one-line fix. Second, a shape mismatch in the composite model configuration. Third, a Windows shared memory limitation with large datasets. Despite these, XGBoost successfully trained on all tasks and achieved strong results — which is consistent with published research showing XGBoost is highly competitive on structured EHR data, often matching or outperforming deep learning on clinical prediction tasks."

All dates in the database have been shifted to protect patient confidentiality. Dates will be internally consistent for the same patient, but randomly distributed in the future. This means that if measurement A is made at 2150-01-01 14:00:00, and measurement B is made at 2150-01-01 15:00:00, then measurement B was made 1 hour after measurement A.

The date shifting preserved the following:

Time of day - a measurement made at 15:00:00 was actually made at 15:00:00 local standard time.
Day of the week - a measurement made on a Sunday will appear on a Sunday in the future.
Seasonality - a measurement made during the winter months will appear during a winter month.
The date shifting removed the following:

Year - The year is randomly distributed between 2100 - 2200.
Day of the month - The absolute day of the month is not preserved.
Inter-patient information - Two patients in the ICU on 2150-01-01 were not in the ICU at the same time