"""
Show Predictions — Display results from all Smart ICU predictors.
Reads per-task reports and displays best model per task + full comparison.
"""

import os
import json
import glob


def load_latest_report():
    """Load the most recent metrics report."""
    reports = sorted(glob.glob('output/metrics_report_*.json'))
    if not reports:
        return None
    with open(reports[-1], 'r') as f:
        return json.load(f)


def load_task_reports():
    """Load individual per-task reports."""
    tasks = {}
    for path in glob.glob('output/*_report.json'):
        name = os.path.basename(path).replace('_report.json', '')
        with open(path, 'r') as f:
            tasks[name] = json.load(f)
    return tasks


def main():
    print("=" * 90)
    print("SMART ICU ASSISTANT — PREDICTION RESULTS")
    print("=" * 90)

    task_reports = load_task_reports()
    main_report  = load_latest_report()

    if not task_reports and not main_report:
        print("\n  No results found. Run the pipeline first:")
        print("    python main_pipeline.py --data_dir data/")
        return

    task_order = [
        'mortality', 'sepsis', 'aki', 'hypotension',
        'vasopressor', 'ventilation', 'los', 'readmission', 'composite',
    ]

    print(f"\n{'#':>3} | {'Task':<25} | {'Best Model':<18} | {'Best AUROC':>10} | Models Tried")
    print("-" * 90)

    for task_num, task_name in enumerate(task_order, start=1):
        report = task_reports.get(task_name, {})
        if not report and main_report and 'predictor_results' in main_report:
            report = main_report['predictor_results'].get(task_name, {})

        if not report:
            print(f"{task_num:>3} | {task_name:<25} | {'—':<18} | {'—':>10} | Not trained")
            continue

        best_model = report.get('best_model', 'N/A')
        best_auroc = report.get('best_auroc', report.get('auroc', 0))
        comparison = report.get('comparison', {})

        auroc_str = f"{best_auroc:.4f}" if isinstance(best_auroc, (int, float)) and best_auroc > 0 else "N/A"

        tried = []
        for m, v in comparison.items():
            if isinstance(v, dict):
                m_auroc = v.get('mean_test_auroc', 0)
                if isinstance(m_auroc, (int, float)) and m_auroc > 0:
                    marker = "★" if m == best_model else " "
                    tried.append(f"{marker}{m}({m_auroc:.3f})")
        tried_str = ", ".join(tried) if tried else best_model or "—"

        print(f"{task_num:>3} | {task_name:<25} | {str(best_model):<18} | {auroc_str:>10} | {tried_str}")

    print("=" * 90)
    print(f"\nPREDICTION CATEGORIES:")

    # TCN removed from model descriptions
    categories = [
        ("[1]   MORTALITY",     "LSTM / XGBoost — 6/12/24h death prediction"),
        ("[2]   SEPSIS",        "Transformer / LSTM — SIRS + infection at 6/12/24h"),
        ("[3]   AKI",           "XGBoost / LSTM — KDIGO stages 1-3 at 24/48h"),
        ("[4]   HYPOTENSION",   "XGBoost / LSTM — MAP < 65 mmHg at 1/3/6h"),
        ("[5]   VASOPRESSOR",   "XGBoost primary — drug requirement at 6/12h"),
        ("[6]   VENTILATION",   "LSTM / XGBoost — mechanical vent at 6/12/24h"),
        ("[7]   LENGTH OF STAY","XGBoost primary — short (<24h) / long (>72h) stay"),
        ("[8]   READMISSION",   "XGBoost + SHAP — 30-day ICU readmission"),
        ("[9]   COMPOSITE",     "MultitaskLSTM — unified deterioration score"),
    ]
    for cat, desc in categories:
        print(f"  {cat:<20s} — {desc}")

    latest = sorted(glob.glob('output/metrics_report_*.json'))
    if latest:
        print(f"\nLatest report: {latest[-1]}")

    task_report_files = sorted(glob.glob('output/*_report.json'))
    if task_report_files:
        print(f"Task reports:  {len(task_report_files)} files in output/")

    print(f"\nNOTE: TCN removed — BatchNorm1d NaN under FP16 AMP. Active models: LSTM, Transformer, XGBoost.")


if __name__ == "__main__":
    main()