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

 CHARTEVENTS loading is at 10M rows. This will take 30-45 minutes for the 330M row file. Let me keep monitoring


to run : python main_pipeline.py --data_dir data 2>&1
