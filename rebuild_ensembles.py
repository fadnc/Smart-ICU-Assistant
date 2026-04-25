"""
Rebuild ensemble pickles from existing trained models.

This script re-runs ONLY the ensemble step (weighted averaging + stacking)
using the already-saved individual model checkpoints. No GPU training needed.
Takes ~2-5 minutes total.

Usage:
    python rebuild_ensembles.py
"""
import os, sys, json, pickle, glob
import numpy as np
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")


def load_config():
    import yaml
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {}


def load_feature_cache():
    """Load feature cache and return X, y, meta."""
    X = np.load(os.path.join(OUTPUT_DIR, "feature_cache_X.npy"), mmap_mode='r')
    y = np.load(os.path.join(OUTPUT_DIR, "feature_cache_y.npy"))
    with open(os.path.join(OUTPUT_DIR, "feature_cache_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return X, y, meta


def temporal_split(X, y, test_frac=0.2, val_frac=0.1):
    """Simple temporal split: train / val / test."""
    n = len(X)
    n_test  = int(n * test_frac)
    n_val   = int(n * val_frac)
    n_train = n - n_val - n_test
    return {
        "train": (X[:n_train], y[:n_train]),
        "val":   (X[n_train:n_train+n_val], y[n_train:n_train+n_val]),
        "test":  (X[n_train+n_val:], y[n_train+n_val:]),
    }


def predict_with_model(model_path, X_seq, X_flat, n_labels):
    """Run a saved model and return predictions array (n_samples, n_labels)."""
    if model_path.endswith(".pkl"):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        preds = np.zeros((X_flat.shape[0], n_labels))
        for i, m in enumerate(model.models):
            if m is None or i >= n_labels:
                continue
            expected = m.n_features_in_ if hasattr(m, 'n_features_in_') else X_flat.shape[1]
            x_in = X_flat
            if X_flat.shape[1] < expected:
                x_in = np.pad(X_flat, ((0, 0), (0, expected - X_flat.shape[1])))
            elif X_flat.shape[1] > expected:
                x_in = X_flat[:, :expected]
            preds[:, i] = m.predict_proba(x_in)[:, 1]
        return preds

    elif model_path.endswith(".pth"):
        import torch, yaml
        from models import create_model

        config = load_config()
        config["input_size"] = X_seq.shape[2]
        config["num_tasks"]  = n_labels

        # Infer model type from filename
        basename = os.path.basename(model_path)
        if "lstm" in basename:
            model_type = "lstm"
        elif "transformer" in basename:
            model_type = "transformer"
        else:
            print(f"  [WARN] Cannot infer model type from {basename}, skipping")
            return None

        model = create_model(model_type, config)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            out = model(torch.FloatTensor(X_seq)).numpy()
        return out[:, :n_labels]

    return None


def rebuild_for_task(task_name, report, X_test, y_test):
    """Rebuild ensemble + stacked for one task."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    labels = report.get("labels", [])
    comparison = report.get("comparison", {})
    n_labels = len(labels)

    # Find all label indices in y
    all_label_names = pickle.load(
        open(os.path.join(OUTPUT_DIR, "feature_cache_meta.pkl"), "rb")
    ).get("label_names", [])
    label_indices = [all_label_names.index(l) for l in labels if l in all_label_names]
    if not label_indices:
        print(f"  [SKIP] No matching labels for {task_name}")
        return

    task_y = y_test[:, label_indices]
    n_labels = len(label_indices)

    # Collect predictions from each saved model
    model_names = []
    pred_list   = []
    auroc_list  = []

    X_flat = X_test.reshape(X_test.shape[0], -1).astype(np.float32) if len(X_test.shape) == 3 else X_test.astype(np.float32)
    X_seq  = X_test.astype(np.float32) if len(X_test.shape) == 3 else X_test.reshape(X_test.shape[0], 1, -1).astype(np.float32)

    for mname in ["lstm", "transformer", "xgboost", "lightgbm"]:
        if mname not in comparison:
            continue
        mpath = os.path.join(MODELS_DIR, f"{task_name}_{mname}.pth" if mname in ("lstm", "transformer") else f"{task_name}_{mname}.pkl")
        if not os.path.exists(mpath):
            continue

        print(f"  Running {mname}...")
        preds = predict_with_model(mpath, X_seq, X_flat, n_labels)
        if preds is not None:
            model_names.append(mname)
            pred_list.append(preds)
            auroc = comparison[mname].get("mean_test_auroc", 0.5)
            auroc_list.append(auroc)

    if len(pred_list) < 2:
        print(f"  [SKIP] Only {len(pred_list)} models — need >= 2 for ensemble")
        return

    print(f"  {len(pred_list)} models loaded: {model_names}")

    # ── Weighted ensemble ──
    weights = np.array([a ** 2 for a in auroc_list])
    weights = weights / weights.sum()
    avg_preds = np.average(pred_list, axis=0, weights=weights)

    # Compute AUROC
    aurocs = []
    for t in range(n_labels):
        y_true = task_y[:, t]
        if len(np.unique(y_true)) >= 2:
            aurocs.append(roc_auc_score(y_true, avg_preds[:, t]))
    ens_auroc = float(np.mean(aurocs)) if aurocs else 0.0

    ens_path = os.path.join(MODELS_DIR, f"{task_name}_ensemble.pkl")
    component_paths = [os.path.join(MODELS_DIR, f"{task_name}_{n}.pth" if n in ("lstm", "transformer") else f"{task_name}_{n}.pkl") for n in model_names]
    ens_artifact = {
        "type": "weighted_ensemble",
        "task": task_name,
        "labels": labels,
        "component_model_names": model_names,
        "component_model_paths": component_paths,
        "weights": {n: float(w) for n, w in zip(model_names, weights)},
        "input_size": n_labels,
    }
    with open(ens_path, "wb") as f:
        pickle.dump(ens_artifact, f)
    print(f"  ★ Weighted ensemble AUROC={ens_auroc:.4f} → {ens_path}")

    # ── Stacked ensemble ──
    half = len(task_y) // 2
    meta_learners = []
    stacked_preds = np.zeros_like(avg_preds)

    for t in range(n_labels):
        meta_features = np.column_stack([p[:, t] for p in pred_list])
        meta_train_X = meta_features[:half]
        meta_train_y = task_y[:half, t]
        meta_test_X  = meta_features[half:]

        if len(np.unique(meta_train_y)) < 2:
            stacked_preds[half:, t] = np.mean(meta_test_X, axis=1)
            meta_learners.append(None)
            continue

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(meta_train_X, meta_train_y)
        stacked_preds[half:, t] = lr.predict_proba(meta_test_X)[:, 1]
        meta_learners.append(lr)

    # AUROC on second half
    stk_aurocs = []
    for t in range(n_labels):
        y_true = task_y[half:, t]
        if len(np.unique(y_true)) >= 2:
            stk_aurocs.append(roc_auc_score(y_true, stacked_preds[half:, t]))
    stk_auroc = float(np.mean(stk_aurocs)) if stk_aurocs else 0.0

    stk_path = os.path.join(MODELS_DIR, f"{task_name}_stacked_ensemble.pkl")
    stk_artifact = {
        "type": "stacked_ensemble",
        "task": task_name,
        "labels": labels,
        "component_model_names": model_names,
        "component_model_paths": component_paths,
        "meta_learners": meta_learners,
        "input_size": n_labels,
    }
    with open(stk_path, "wb") as f:
        pickle.dump(stk_artifact, f)
    print(f"  ★ Stacked ensemble AUROC={stk_auroc:.4f} → {stk_path}")

    # Update report JSON (only if keys exist)
    if "ensemble" in report.get("comparison", {}):
        report["comparison"]["ensemble"]["model_path"] = ens_path
    if "stacked_ensemble" in report.get("comparison", {}):
        report["comparison"]["stacked_ensemble"]["model_path"] = stk_path


def main():
    print("=" * 60)
    print("  REBUILD ENSEMBLE PICKLES")
    print("=" * 60)

    print("\n[1/2] Loading feature cache...")
    X, y, meta = load_feature_cache()
    splits = temporal_split(X, y)
    X_test, y_test = splits["test"]
    print(f"  Test set: {len(X_test)} samples")

    report_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_report.json")))
    if not report_files:
        print("[ERROR] No *_report.json files found in output/")
        return

    print(f"\n[2/2] Rebuilding ensembles for {len(report_files)} tasks...\n")

    for rpath in report_files:
        with open(rpath) as f:
            report = json.load(f)
        task = report.get("task", "?")
        best = report.get("best_model", "?")

        if "best_auroc" not in report:
            print(f"── {task}: skipped (no best_auroc, different format)")
            continue

        print(f"── {task} (current best: {best}, AUROC {report.get('best_auroc', 0):.4f})")

        if "ensemble" not in report.get("comparison", {}) and "stacked_ensemble" not in report.get("comparison", {}):
            print(f"  [SKIP] No ensemble data in report")
            continue

        rebuild_for_task(task, report, np.array(X_test), y_test)

        # Save updated report
        with open(rpath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Updated {os.path.basename(rpath)}")
        print()

    print("=" * 60)
    print("  DONE — Ensemble pickles saved to models/")
    print("  Run: python app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
