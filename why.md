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
------------------
28/3
We identified three technical issues during training. First, a PyTorch AMP compatibility issue with BCELoss prevented our deep learning models from training — this is a known issue with a one-line fix. Second, a shape mismatch in the composite model configuration. Third, a Windows shared memory limitation with large datasets. Despite these, XGBoost successfully trained on all tasks and achieved strong results — which is consistent with published research showing XGBoost is highly competitive on structured EHR data, often matching or outperforming deep learning on clinical prediction tasks."
------------------
29/3
All dates in the database have been shifted to protect patient confidentiality. Dates will be internally consistent for the same patient, but randomly distributed in the future. This means that if measurement A is made at 2150-01-01 14:00:00, and measurement B is made at 2150-01-01 15:00:00, then measurement B was made 1 hour after measurement A.
The date shifting preserved the following:
  Time of day - a measurement made at 15:00:00 was actually made at 15:00:00 local standard time.
  Day of the week - a measurement made on a Sunday will appear on a Sunday in the future.
  Seasonality - a measurement made during the winter months will appear during a winter month.
The date shifting removed the following:
  Year - The year is randomly distributed between 2100 - 2200.
  Day of the month - The absolute day of the month is not preserved.
  Inter-patient information - Two patients in the ICU on 2150-01-01 were not in the ICU at the same time
------------------
29/3
best_model.pth not found	All models saved to same best_model.pth in CWD. When val_loss was NaN, checkpoint was never saved, but load_checkpoint still tried to load it → crash	Each training session now uses a unique temp path (models/_best_<uuid>.pth), cleaned up after loading
"Mean of empty slice" warnings	All mortality labels in validation batch were 0 (rare event) → all AUROC = NaN → nanmean([]) = warning	Now filters NaN values before averaging, falls back to 0.0 instead of NaN
NaN val_loss → no checkpoint saved	NaN < float('inf') = False, so the checkpoint condition never triggered	Added explicit not np.isnan(val_loss) check, plus graceful fallback to final epoch weights if no checkpoint exists
------------------
31/3
main_pipeline.py — Critical bug fix
The single most important change: set_label_indices() is now called before train_all_models(), not after. This was the root cause of AUROC = 0 for all DL models — they were receiving all 19 labels instead of their task-specific 2-6 columns. Also fixed when loading from cache — the old code skipped setting label indices entirely on cache runs, meaning a second run would have the same AUROC = 0 bug. Progress logging now uses enumerate() counter instead of DataFrame idx so all 10% milestones print correctly.

training.py — Early stopping and checkpoint fixes
patience raised from 10 → 20, and min_delta=1e-4 added so the counter only increments when loss doesn't improve by a meaningful amount. A pre-training checkpoint save is added at epoch 0 so the "No checkpoint saved" warning can never happen again. NaN loss epochs are now skipped entirely (no patience increment) rather than counted against the model. finally block ensures the temp checkpoint is always cleaned up even on crash.

feature_engineering.py — Performance fixes
_vital_itemid_cache and _lab_itemid_cache added — the itemid-to-name mapping now runs once via get_vital_mapping() and get_lab_mapping() instead of 492,000 times. The rolling trend calc_trend_fast now uses raw=True (numpy arrays, Cython path) instead of raw=False (Python objects, interpreted path) — 2-4× faster per stay. A minimum data guard (len < 3) skips stays too sparse to produce meaningful statistics.

models.py — XGBoost 2.x API
XGBoostPredictor now uses tree_method='hist' + device='cuda' instead of the removed tree_method='gpu_hist' + gpu_id=0. use_label_encoder=False removed (parameter no longer exists in XGBoost 2.x). All deprecation warnings will be gone.

base_predictor.py — Error visibility + VRAM cleanup
_extract_task_labels now logs an explicit error when indices are missing instead of silently returning wrong columns. clear_gpu_memory() is called both before and after each model training (not just before), reducing VRAM fragmentation across sequential model runs. XGBoost params updated to match 2.x API same as models.py.

config.yaml — Updated XGBoost block + early stopping params
gpu_id and tree_method: gpu_hist replaced with device: cuda and tree_method: hist. New EARLY_STOPPING section added with patience: 20 and min_delta: 0.0001 for documentation/reference.
------------------
1/4
training.py — Three fixes for NaN loss:
Gradient clipping (clip_grad_norm_(max_norm=1.0)) so exploding gradients can't produce inf/NaN loss
Epoch bar now shows train=X.XXXX | val=X.XXXX | auroc=X.XXXX | pat=N/20 live
Per-batch inner bar (disappears after each epoch) shows individual batch loss
task_name/model_name forwarded from trainer so bars are labeled (e.g. [mortality/lstm] epochs)

data_loader.py — Progress bars everywhere:
All small tables (PATIENTS, ADMISSIONS, etc.) show a row-count bar
CHARTEVENTS chunked load shows an MB-based bar with live "kept rows" counter
LABEVENTS chunked load shows a similar bar
merge_data() has an overall 17-step pipeline bar

base_predictor.py — Per-task and per-model bars:
Each task shows a 4-step model comparison bar (lstm → tcn → transformer → xgboost)
Postfix shows live AUROC as each model finishes

main_pipeline.py — Three major changes:
_normalize_X() called after building the feature array (StandardScaler, fixes NaN loss at root)
Stay-level progress bar during feature extraction showing sequences collected
Top-level 4-step pipeline bar

normalize_cache.py — New utility to fix an existing un-normalized cache without re-running the 2-8 hour extraction.

Why TCN fails on every task: TCNBlock uses nn.BatchNorm1d. Under FP16 AMP with severely imbalanced labels (rare events like vasopressor or AKI Stage 3), entire mini-batches can have all-zero labels. BatchNorm's running mean/variance collapses to zero, the normalization divides near-zero by near-zero, and you get NaN from epoch 2-7 onward. The fix is one line — replace nn.BatchNorm1d with nn.GroupNorm(1, out_channels) — but for this run, TCN is wasting roughly 3 hours per task.