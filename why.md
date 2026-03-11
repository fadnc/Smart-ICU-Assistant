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