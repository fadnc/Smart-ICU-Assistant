"""
Show Predictions — Display results from all Smart ICU predictors.
Reads per-task reports and displays best model per task + full comparison.
Now includes per-subtask breakdown for AKI (stages 1-3) and LOS (24h/72h),
plus F1 and Sensitivity metrics.
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


def fmt(val, width=8):
    """Format a metric value for display."""
    if isinstance(val, (int, float)) and val > 0:
        return f"{val:.4f}".rjust(width)
    return "N/A".rjust(width)


def get_best_per_task_metrics(report):
    """Get per-task metrics from the best model in a report."""
    best_model = report.get('best_model', '')
    comparison = report.get('comparison', {})
    model_data = comparison.get(best_model, {})
    return model_data.get('per_task_metrics', {}), best_model


def print_subtask_rows(task_num, report, label_map):
    """Print individual subtask rows with AUROC, F1, Sensitivity."""
    comparison = report.get('comparison', {})
    best_model = report.get('best_model', '')

    if not comparison:
        for sub_name, _ in label_map:
            print(f"  {'':<3} {'  └─ ' + sub_name:<30} {'—':<16} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
        return

    for sub_name, task_idx in label_map:
        print(f"  {'':<3} {'  +- ' + sub_name:<30}")
        for m, v in comparison.items():
            if isinstance(v, dict):
                metrics = v.get('per_task_metrics', {})
                auroc = metrics.get(f'task_{task_idx}_auroc', 0)
                f1    = metrics.get(f'task_{task_idx}_f1', 0)
                sens  = metrics.get(f'task_{task_idx}_sensitivity', 0)
                marker = "*" if m == best_model else ""
                print(f"  {'':<3} {'':<30} {marker + m:<16} {fmt(auroc)} {fmt(f1)} {fmt(sens)}")


def print_task_summary(task_num, task_name, report):
    """Print the summary row for a task with macro metrics."""
    best_model = report.get('best_model', 'N/A')
    best_auroc = report.get('best_auroc', 0)
    metrics, _ = get_best_per_task_metrics(report)
    macro_f1   = metrics.get('macro_f1', 0)
    macro_sens = metrics.get('macro_sensitivity', 0)

    # Model comparison string
    comparison = report.get('comparison', {})
    tried = []
    for m, v in comparison.items():
        if isinstance(v, dict):
            m_auroc = v.get('mean_test_auroc', 0)
            if isinstance(m_auroc, (int, float)) and m_auroc > 0:
                marker = "*" if m == best_model else " "
                tried.append(f"{marker}{m}({m_auroc:.3f})")
    tried_str = ", ".join(tried) if tried else str(best_model)

    print(f"  {task_num:>3} {task_name:<30} {str(best_model):<16} {fmt(best_auroc)} {fmt(macro_f1)} {fmt(macro_sens)}  {tried_str}")


def main():
    SEP = "=" * 120
    print(SEP)
    print("SMART ICU ASSISTANT - PREDICTION RESULTS (Detailed)")
    print(SEP)

    task_reports = load_task_reports()
    main_report  = load_latest_report()

    if not task_reports and not main_report:
        print("\n  No results found. Run the pipeline first:")
        print("    python main_pipeline.py --data_dir data/")
        return

    # Header
    print(f"\n  {'#':>3} {'Task':<30} {'Best Model':<16} {'AUROC':>8} {'F1':>8} {'Sens.':>8}  Models Tried")
    print("  " + "-" * 116)

    # --- Define task order and subtask mappings ---
    task_order = [
        'mortality', 'sepsis', 'aki',
        'vasopressor', 'ventilation', 'los',
    ]

    # Subtask label index mappings (matching labels list order in reports)
    subtask_map = {
        'mortality': [
            ('Mortality 6h',  0),
            ('Mortality 12h', 1),
            ('Mortality 24h', 2),
        ],
        'sepsis': [
            ('Sepsis 6h',  0),
            ('Sepsis 12h', 1),
            ('Sepsis 24h', 2),
        ],
        'aki': [
            ('AKI Stage 1 - 24h', 0),
            ('AKI Stage 2 - 24h', 1),
            ('AKI Stage 3 - 24h', 2),
            ('AKI Stage 1 - 48h', 3),
            ('AKI Stage 2 - 48h', 4),
            ('AKI Stage 3 - 48h', 5),
        ],
        'vasopressor': [
            ('Vasopressor 6h',  0),
            ('Vasopressor 12h', 1),
        ],
        'ventilation': [
            ('Ventilation 6h',  0),
            ('Ventilation 12h', 1),
            ('Ventilation 24h', 2),
        ],
        'los': [
            ('LOS Short (<24h)', 0),
            ('LOS Long (>72h)',  1),
        ],
    }

    for task_num, task_name in enumerate(task_order, start=1):
        report = task_reports.get(task_name, {})
        if not report and main_report and 'predictor_results' in main_report:
            report = main_report['predictor_results'].get(task_name, {})

        if not report:
            print(f"  {task_num:>3} {task_name:<30} {'—':<16} {'N/A':>8} {'N/A':>8} {'N/A':>8}  Not trained")
            continue

        # Print task summary row
        print_task_summary(task_num, task_name.upper(), report)

        # Print subtask breakdown
        if task_name in subtask_map:
            print_subtask_rows(task_num, report, subtask_map[task_name])

        print()  # blank line between tasks

    print(SEP)

    # --- Category summary ---
    print(f"\nPREDICTION CATEGORIES:")
    categories = [
        ("[1]   MORTALITY",     "LSTM / XGBoost — 6/12/24h death prediction"),
        ("[2]   SEPSIS",        "Transformer / LSTM — SIRS + infection at 6/12/24h"),
        ("[3]   AKI",           "XGBoost / LSTM — KDIGO stages 1-3 at 24/48h"),
        ("[4]   VASOPRESSOR",   "XGBoost primary — drug requirement at 6/12h"),
        ("[5]   VENTILATION",   "LSTM / XGBoost — mechanical vent at 6/12/24h"),
        ("[6]   LENGTH OF STAY","XGBoost primary — short (<24h) / long (>72h) stay")
    ]
    for cat, desc in categories:
        print(f"  {cat:<20s} — {desc}")

    # --- File info ---
    latest = sorted(glob.glob('output/metrics_report_*.json'))
    if latest:
        print(f"\nLatest report: {latest[-1]}")

    task_report_files = sorted(glob.glob('output/*_report.json'))
    if task_report_files:
        print(f"Task reports:  {len(task_report_files)} files in output/")


if __name__ == "__main__":
    main()