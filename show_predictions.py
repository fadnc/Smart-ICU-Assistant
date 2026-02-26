"""
Show all 20 prediction tasks and their performance metrics
"""
import json, os

# Load latest metrics report
reports = sorted([f for f in os.listdir('output') if f.startswith('metrics_report')])
with open(f'output/{reports[-1]}') as f:
    content = f.read().replace('NaN', 'null')
    metrics = json.loads(content)

# The 20 prediction labels in order
labels = [
    'mortality_6h',   'mortality_12h',  'mortality_24h',
    'sepsis_6h',      'sepsis_12h',     'sepsis_24h',
    'aki_stage1_24h', 'aki_stage2_24h', 'aki_stage3_24h',
    'aki_stage1_48h', 'aki_stage2_48h', 'aki_stage3_48h',
    'hypotension_1h', 'hypotension_3h', 'hypotension_6h',
    'vasopressor_6h', 'vasopressor_12h',
    'ventilation_6h', 'ventilation_12h','ventilation_24h'
]

print('=' * 90)
print('SMART ICU ASSISTANT — ALL 20 PREDICTION TASKS')
print('=' * 90)
print(f'{"#":>3} | {"Prediction Task":<22} | {"LSTM AUROC":>12} | {"TCN AUROC":>12} | {"XGB AUROC":>12} | Status')
print('-' * 90)

for i, label in enumerate(labels):
    lstm_val = metrics['models']['lstm']['test_auroc'][i]
    tcn_val  = metrics['models']['tcn']['test_auroc'][i]
    xgb_val  = metrics['models']['xgboost']['test_auroc'][i]

    lstm_s = f'{lstm_val:.3f}' if lstm_val else '  NaN'
    tcn_s  = f'{tcn_val:.3f}' if tcn_val else '  NaN'
    xgb_s  = f'{xgb_val:.3f}' if xgb_val else '  NaN'
    status = 'OK' if lstm_val else 'No +ve labels'

    print(f'{i+1:>3} | {label:<22} | {lstm_s:>12} | {tcn_s:>12} | {xgb_s:>12} | {status}')

print('=' * 90)
print()
print('PREDICTION CATEGORIES:')
print()
print('  [1-3]   MORTALITY    — Will the patient die within the next 6/12/24 hours?')
print('                         Uses: date of death vs current prediction time')
print()
print('  [4-6]   SEPSIS       — Will the patient develop sepsis within 6/12/24 hours?')
print('                         Uses: SIRS criteria (temp, HR, RR, WBC) + antibiotic/ICD codes')
print()
print('  [7-12]  AKI          — Will the patient develop Acute Kidney Injury (stages 1-3)?')
print('                         Uses: KDIGO criteria (creatinine increase vs baseline)')
print()
print('  [13-15] HYPOTENSION  — Will blood pressure drop critically in 1/3/6 hours?')
print('                         Uses: Mean Arterial Pressure < 65 mmHg')
print()
print('  [16-17] VASOPRESSOR  — Will the patient need vasopressor drugs in 6/12 hours?')
print('                         Uses: norepinephrine, epinephrine, vasopressin prescriptions')
print()
print('  [18-20] VENTILATION  — Will the patient need mechanical ventilation in 6/12/24 hours?')
print('                         Uses: placeholder (returns 0) — needs ventilator itemids')
print()
print('NOTE: "No +ve labels" means all test-set patients had label=0 for that task.')
print('      AUROC requires both positive and negative examples. This is expected on')
print('      the small demo dataset (100 patients). Full MIMIC-III will fix this.')
