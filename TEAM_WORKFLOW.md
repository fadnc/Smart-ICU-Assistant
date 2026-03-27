# Team Workflow — Smart ICU Assistant

## Branch Structure
```
demo_ds ← SHARED BASE (both start from here)
├── feat/molyy  ← Her branch: mortality, AKI, readmission
└── feat/fadhi  ← Your branch: sepsis, vasopressor, ventilation, hypotension, LOS, composite
```

---

## Task Assignment

### Raha (3 Tasks)
| Task | File | Model | What to do |
|------|------|-------|-----------|
| **Mortality** (6/12/24h) | `predictors/mortality_predictor.py` | LSTM | Tune label logic, try XGBoost vs LSTM |
| **AKI** (KDIGO stages 1-3) | `predictors/aki_predictor.py` | LSTM multiclass | Improve creatinine features, add urine output |
| **ICU Readmission** | `predictors/readmission_predictor.py` | XGBoost + SHAP | Add more discharge features, interpretability |

### Fadhil (5 Tasks)
| Task | File | Model | What to do |
|------|------|-------|-----------|
| **Sepsis** (6/12/24h) | `predictors/sepsis_predictor.py` | Transformer/LSTM | SIRS + infection logic, cross-feature attention |
| **Vasopressor** (6/12h) | `predictors/vasopressor_predictor.py` | XGBoost | Drug requirement detection from prescriptions + IV |
| **Ventilation** (6/12/24h) | `predictors/ventilation_predictor.py` | LSTM | 3-layer detection (chartevents, procedures, ICD) |
| **Hypotension** (1/3/6h) | `predictors/hypotension_predictor.py` | TCN | MAP < 65 mmHg, short-term patterns |
| **Length of Stay** | `predictors/los_predictor.py` | XGBoost/LSTM | Short (<24h) vs Long (>72h) stay prediction |

###  Shared (don't edit without coordinating)
- `base_predictor.py` — shared training logic
- `data_loader.py` — data loading
- `feature_engineering.py` — feature extraction
- `models.py` — model architectures
- `main_pipeline.py` — final integration
- `config.yaml`

### Integration (done together)
- `composite_predictor.py` — combines ALL tasks via MultitaskLSTM

---

## Git Commands

### Step 1: Push shared base
```bash
git checkout demo_ds
git push origin demo_ds
```

### Step 2: She creates her branch
```bash
git checkout demo_ds
git checkout -b feat/molyy
# Work on her 3 predictors...
git add predictors/mortality_predictor.py predictors/aki_predictor.py predictors/readmission_predictor.py
git commit -m "feat: improve mortality/AKI/readmission predictors"
git push origin feat/molyy
```

### Step 3: You create your branch
```bash
git checkout demo_ds
git checkout -b feat/fadhi
# Work on your 5 predictors...
git add predictors/sepsis_predictor.py predictors/vasopressor_predictor.py predictors/ventilation_predictor.py predictors/hypotension_predictor.py predictors/los_predictor.py
git commit -m "feat: improve sepsis/vasopressor/ventilation/hypotension/LOS predictors"
git push origin feat/fadhi
```

### Step 4: Merge both into demo_ds
```bash
git checkout demo_ds
git merge feat/molyy    # No conflicts (different files)
git merge feat/fadhi     # No conflicts (different files)
git push origin demo_ds
```

---

## Running Independently

### Test your predictor only
```bash
python -c "from predictors.sepsis_predictor import SepsisPredictor; p = SepsisPredictor(); print(p, p.get_label_names())"
```

### Run full pipeline
```bash
# Full MIMIC-III (slower, ~1-2h)
python main_pipeline.py --data_dir data

# Quick test with 10 stays
python main_pipeline.py --data_dir data --sample_size 10

# View results
python show_predictions.py
```

---

## File Map
```
SEM-4-PROJECT/
├── predictors/
│   ├── base_predictor.py          ← SHARED: don't edit alone
│   ├── mortality_predictor.py     ← MOLYY
│   ├── aki_predictor.py           ← MOLYY
│   ├── readmission_predictor.py   ← MOLYY
│   ├── sepsis_predictor.py        ← FADHI
│   ├── vasopressor_predictor.py   ← FADHI
│   ├── ventilation_predictor.py   ← FADHI
│   ├── hypotension_predictor.py   ← FADHI
│   ├── los_predictor.py           ← FADHI
│   └── composite_predictor.py     ← TOGETHER (final step)
├── data_loader.py                 ← SHARED
├── feature_engineering.py         ← SHARED
├── models.py                      ← SHARED
├── training.py                    ← SHARED
├── main_pipeline.py               ← SHARED (integration)
├── show_predictions.py            ← SHARED (results)
└── config.yaml                    ← SHARED
```
