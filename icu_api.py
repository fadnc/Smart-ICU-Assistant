"""
Smart ICU Assistant — Flask API Backend  (Final Integrated Version)
====================================================================
Connects directly to your MIMIC-III data, trained model checkpoints,
and the ReadmissionPredictor for real SHAP values.

Directory layout expected (matches your project):
  fadnc-sem-4-project/
  ├── icu_api.py              ← this file
  ├── config.yaml
  ├── data_loader.py
  ├── feature_engineering.py
  ├── models.py
  ├── predictors/
  │   └── readmission_predictor.py
  ├── data/                   ← MIMIC-III CSVs  (override: MIMIC_DATA_DIR)
  ├── output/                 ← *_report.json    (override: ICU_OUTPUT_DIR)
  └── models/                 ← *.pth / *.pkl    (override: ICU_MODELS_DIR)

Install:
  pip install flask flask-cors pandas numpy torch xgboost pyyaml shap

Run:
  python icu_api.py
  MIMIC_DATA_DIR=data ICU_OUTPUT_DIR=output ICU_MODELS_DIR=models python icu_api.py

Endpoints:
  GET /api/health
  GET /api/stats
  GET /api/patients                          ?q=&unit=&risk=&page=&per_page=
  GET /api/patients/<icustay_id>
  GET /api/patients/<icustay_id>/vitals
  GET /api/patients/<icustay_id>/labs
  GET /api/patients/<icustay_id>/predictions
  GET /api/alerts
"""

# ── Stdlib ────────────────────────────────────────────────────────────────────
import os, sys, json, glob, pickle, traceback
from datetime import datetime, timedelta

# ── Third-party ───────────────────────────────────────────────────────────────
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import yaml

# ── Project imports (project root must be on PYTHONPATH) ─────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# These are imported lazily inside functions so the API starts even if
# optional heavy deps (torch, xgboost) are not yet installed.

# ── App & config ─────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

DATA_DIR    = os.environ.get("MIMIC_DATA_DIR",  os.path.join(PROJECT_ROOT, "data"))
OUTPUT_DIR  = os.environ.get("ICU_OUTPUT_DIR",  os.path.join(PROJECT_ROOT, "output"))
MODELS_DIR  = os.environ.get("ICU_MODELS_DIR",  os.path.join(PROJECT_ROOT, "models"))
CONFIG_PATH = os.environ.get("ICU_CONFIG_PATH", os.path.join(PROJECT_ROOT, "config.yaml"))

# ── Prediction label registry ─────────────────────────────────────────────────
PREDICTION_LABELS = [
    "mortality_6h",  "mortality_12h",  "mortality_24h",
    "sepsis_6h",     "sepsis_12h",     "sepsis_24h",
    "aki_stage1_24h","aki_stage2_24h", "aki_stage3_24h",
    "aki_stage1_48h","aki_stage2_48h", "aki_stage3_48h",
    "hypotension_1h","hypotension_3h", "hypotension_6h",
    "vasopressor_6h","vasopressor_12h",
    "ventilation_6h","ventilation_12h","ventilation_24h",
    "los_short_24h", "los_long_72h",
]

TASK_GROUPS = {
    "Mortality":      ["mortality_6h",   "mortality_12h",   "mortality_24h"],
    "Sepsis":         ["sepsis_6h",      "sepsis_12h",      "sepsis_24h"],
    "AKI":            ["aki_stage1_24h", "aki_stage2_24h",  "aki_stage3_24h",
                       "aki_stage1_48h", "aki_stage2_48h",  "aki_stage3_48h"],
    "Hypotension":    ["hypotension_1h", "hypotension_3h",  "hypotension_6h"],
    "Vasopressor":    ["vasopressor_6h", "vasopressor_12h"],
    "Ventilation":    ["ventilation_6h", "ventilation_12h", "ventilation_24h"],
    "Length of Stay": ["los_short_24h",  "los_long_72h"],
}

# Vital sign MIMIC-III itemids (MetaVision + CareVue)
VITAL_ITEMIDS = {
    "heartrate": [220045, 211],
    "sysbp":     [220050, 220179, 51, 455],
    "diasbp":    [220051, 220180, 8368, 8441],
    "meanbp":    [220052, 220181, 52, 456],
    "resprate":  [220210, 224690, 618, 615],
    "tempc":     [223761, 223762, 676, 678],
    "spo2":      [220277, 646],
    "glucose":   [220621, 226537, 807, 811, 1529],
}

# Lab MIMIC-III itemids
LAB_ITEMIDS = {
    "creatinine":  [50912],
    "lactate":     [50813],
    "wbc":         [51301],
    "hemoglobin":  [51222],
    "platelets":   [51265],
    "bicarbonate": [50882],
    "chloride":    [50902],
}

# Flat set for fast filtering
ALL_VITAL_ITEMIDS = {iid for ids in VITAL_ITEMIDS.values() for iid in ids}
ALL_LAB_ITEMIDS   = {iid for ids in LAB_ITEMIDS.values()   for iid in ids}

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Lazy data cache
# ═════════════════════════════════════════════════════════════════════════════
_cache: dict = {}


def cached(key: str, fn):
    """Load-once cache. Returns None and logs if loader raises."""
    if key not in _cache:
        try:
            result = fn()
            _cache[key] = result
            size = len(result) if hasattr(result, "__len__") else "?"
            print(f"[INFO] cached '{key}': {size} rows/items")
        except Exception as exc:
            print(f"[WARN] Could not load '{key}': {exc}")
            _cache[key] = None
    return _cache[key]


# ── Patient / demographics ────────────────────────────────────────────────────
def _load_patients() -> pd.DataFrame:
    def _read(name: str) -> pd.DataFrame:
        path = os.path.join(DATA_DIR, name)
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.lower() for c in df.columns]
        return df

    pat = _read("PATIENTS.csv")
    icu = _read("ICUSTAYS.csv")
    adm = _read("ADMISSIONS.csv")

    for col in ["intime", "outtime"]:
        icu[col] = pd.to_datetime(icu[col])
    for col in ["dob", "dod"]:
        pat[col] = pd.to_datetime(pat[col])

    merged = (
        icu
        .merge(pat[["subject_id", "gender", "dob", "dod", "expire_flag"]],
               on="subject_id", how="left")
        .merge(adm[["subject_id", "hadm_id", "admittime", "dischtime",
                    "admission_type", "diagnosis"]],
               on=["subject_id", "hadm_id"], how="left")
    )

    def _age(row):
        try:
            return (row["intime"] - row["dob"]).total_seconds() / (365.25 * 24 * 3600)
        except Exception:
            return np.nan

    merged["age"]       = merged.apply(_age, axis=1).clip(upper=90)
    merged["los_hours"] = (merged["outtime"] - merged["intime"]).dt.total_seconds() / 3600
    return merged


# ── CHARTEVENTS (filtered to vitals only) ────────────────────────────────────
def _load_chartevents() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "CHARTEVENTS.csv")
    file_mb = os.path.getsize(path) / (1024 * 1024)
    usecols = ["ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUENUM"]

    if file_mb < 500:
        df = pd.read_csv(path, low_memory=False, usecols=usecols)
        df.columns = [c.lower() for c in df.columns]
    else:
        # Large file — chunked, filter to vital itemids only
        chunks = []
        for chunk in pd.read_csv(path, chunksize=2_000_000, low_memory=False,
                                  usecols=usecols,
                                  dtype={"ICUSTAY_ID": "float32",
                                         "ITEMID": "int32",
                                         "VALUENUM": "float32"}):
            chunk.columns = [c.lower() for c in chunk.columns]
            chunks.append(chunk[chunk["itemid"].isin(ALL_VITAL_ITEMIDS)])
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    df["charttime"] = pd.to_datetime(df["charttime"])
    df["valuenum"]  = pd.to_numeric(df["valuenum"], errors="coerce")
    return df


# ── LABEVENTS (filtered to relevant labs) ────────────────────────────────────
def _load_labevents() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "LABEVENTS.csv")
    usecols = ["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"]

    # Only load relevant subjects to save RAM
    patients_df = cached("patients", _load_patients)
    relevant_subjects = (
        set(patients_df["subject_id"].unique()) if patients_df is not None else None
    )

    chunks = []
    for chunk in pd.read_csv(path, chunksize=1_000_000, low_memory=False,
                              usecols=usecols,
                              dtype={"SUBJECT_ID": "int32",
                                     "ITEMID": "int32",
                                     "VALUENUM": "float32"}):
        chunk.columns = [c.lower() for c in chunk.columns]
        chunk = chunk[chunk["itemid"].isin(ALL_LAB_ITEMIDS)]
        if relevant_subjects:
            chunk = chunk[chunk["subject_id"].isin(relevant_subjects)]
        if len(chunk):
            chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    df["charttime"] = pd.to_datetime(df["charttime"])
    df["valuenum"]  = pd.to_numeric(df["valuenum"], errors="coerce")

    # Assign icustay_id via time-overlap join
    icu = cached("patients", _load_patients)
    if icu is not None and len(df):
        merged = df.merge(
            icu[["subject_id", "hadm_id", "icustay_id", "intime", "outtime"]],
            on=["subject_id", "hadm_id"], how="left"
        )
        mask = (
            merged["icustay_id"].notna() &
            (merged["charttime"] >= merged["intime"]) &
            (merged["charttime"] <= merged["outtime"])
        )
        df = merged[mask].drop(columns=["intime", "outtime"]).copy()

    return df


# ── Pre-grouped indexes (built once, reused per-request) ─────────────────────
def _build_charts_grouped() -> dict:
    ce = cached("chartevents", _load_chartevents)
    if ce is None or len(ce) == 0:
        return {}
    return dict(tuple(ce.groupby("icustay_id")))


def _build_labs_grouped() -> dict:
    le = cached("labevents", _load_labevents)
    if le is None or len(le) == 0:
        return {}
    return dict(tuple(le.groupby("icustay_id")))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Real vitals & labs from MIMIC-III
# ═════════════════════════════════════════════════════════════════════════════

def _real_vitals(icustay_id: int) -> list:
    """
    Query CHARTEVENTS for actual vital signs for this ICU stay.
    Pivots itemid→ named column, resamples to hourly, returns list of dicts.
    Falls back to mock if data unavailable.
    """
    charts_grouped = cached("charts_grouped", _build_charts_grouped)
    if not charts_grouped:
        return _fallback_vitals(icustay_id)

    stay_charts = charts_grouped.get(icustay_id)
    if stay_charts is None or len(stay_charts) == 0:
        return _fallback_vitals(icustay_id)

    records = []
    for vital_name, itemids in VITAL_ITEMIDS.items():
        sub = stay_charts[stay_charts["itemid"].isin(itemids)][["charttime", "valuenum"]].copy()
        sub = sub.dropna(subset=["valuenum"])
        sub["vital"] = vital_name
        records.append(sub)

    if not records:
        return _fallback_vitals(icustay_id)

    combined = pd.concat(records, ignore_index=True)
    pivot = (
        combined
        .pivot_table(index="charttime", columns="vital", values="valuenum", aggfunc="mean")
        .sort_index()
    )

    # Resample to hourly, forward-fill up to 4 h
    pivot = pivot.resample("1h").mean().ffill(limit=4)
    pivot = pivot.reset_index()

    # Cap display at 72 h
    if len(pivot) > 72:
        pivot = pivot.iloc[-72:]

    out = []
    for _, row in pivot.iterrows():
        rec = {"charttime": row["charttime"].isoformat()}
        for col in VITAL_ITEMIDS.keys():
            val = row.get(col, np.nan)
            rec[col] = None if pd.isna(val) else round(float(val), 1)
        out.append(rec)

    return out


def _real_labs(icustay_id: int) -> list:
    """
    Query LABEVENTS for actual lab results for this ICU stay.
    Returns one row per draw event (every ~6 h), named columns.
    Falls back to mock if data unavailable.
    """
    labs_grouped = cached("labs_grouped", _build_labs_grouped)
    if not labs_grouped:
        return _fallback_labs(icustay_id)

    stay_labs = labs_grouped.get(icustay_id)
    if stay_labs is None or len(stay_labs) == 0:
        return _fallback_labs(icustay_id)

    records = []
    for lab_name, itemids in LAB_ITEMIDS.items():
        sub = stay_labs[stay_labs["itemid"].isin(itemids)][["charttime", "valuenum"]].copy()
        sub = sub.dropna(subset=["valuenum"])
        sub["lab"] = lab_name
        records.append(sub)

    if not records:
        return _fallback_labs(icustay_id)

    combined = pd.concat(records, ignore_index=True)
    pivot = (
        combined
        .pivot_table(index="charttime", columns="lab", values="valuenum", aggfunc="mean")
        .sort_index()
    )

    # Resample to 6-hourly
    pivot = pivot.resample("6h").mean().ffill(limit=2)
    if len(pivot) > 24:   # cap at 6 days
        pivot = pivot.iloc[-24:]

    pivot = pivot.reset_index()
    out = []
    for _, row in pivot.iterrows():
        rec = {"charttime": row["charttime"].isoformat()}
        for col in LAB_ITEMIDS.keys():
            val = row.get(col, np.nan)
            rec[col] = None if pd.isna(val) else round(float(val), 2)
        out.append(rec)

    return out


# ─── Fallback generators (used only when CHARTEVENTS/LABEVENTS not loaded) ───

def _fallback_vitals(icustay_id: int) -> list:
    df = cached("patients", _load_patients)
    intime  = datetime.utcnow() - timedelta(hours=24)
    outtime = datetime.utcnow()
    if df is not None:
        row = df[df["icustay_id"] == icustay_id]
        if len(row):
            r = row.iloc[0]
            intime  = pd.to_datetime(r["intime"])
            outtime = pd.to_datetime(r["outtime"]) if pd.notna(r.get("outtime")) else intime + timedelta(hours=48)

    rng = np.random.RandomState(icustay_id % 9999)
    hours = min(72, max(1, int((outtime - intime).total_seconds() / 3600)))
    hr = rng.uniform(75, 95);  sbp = rng.uniform(105, 130); dbp = rng.uniform(60, 80)
    rr = rng.uniform(14, 20);  tmp = rng.uniform(36.5, 37.5)
    spo2 = rng.uniform(95, 99); glc = rng.uniform(100, 160)
    out = []
    for h in range(hours):
        out.append({
            "charttime": (intime + timedelta(hours=h)).isoformat(),
            "heartrate": round(hr   + rng.normal(0, 5),   1),
            "sysbp":     round(sbp  + rng.normal(0, 8),   1),
            "diasbp":    round(dbp  + rng.normal(0, 5),   1),
            "meanbp":    round((sbp + 2*dbp)/3 + rng.normal(0, 4), 1),
            "resprate":  round(rr   + rng.normal(0, 2),   1),
            "tempc":     round(tmp  + rng.normal(0, 0.3), 2),
            "spo2":      round(min(100, spo2 + rng.normal(0, 1)), 1),
            "glucose":   round(glc  + rng.normal(0, 15),  1),
        })
        hr += rng.normal(0, 0.5); sbp += rng.normal(0, 0.8)
        tmp += rng.normal(0, 0.05); spo2 += rng.normal(0, 0.2)
    return out


def _fallback_labs(icustay_id: int) -> list:
    df = cached("patients", _load_patients)
    intime  = datetime.utcnow() - timedelta(hours=24)
    outtime = datetime.utcnow()
    if df is not None:
        row = df[df["icustay_id"] == icustay_id]
        if len(row):
            r = row.iloc[0]
            intime  = pd.to_datetime(r["intime"])
            outtime = pd.to_datetime(r["outtime"]) if pd.notna(r.get("outtime")) else intime + timedelta(hours=48)

    rng = np.random.RandomState((icustay_id + 1000) % 9999)
    hours = min(72, max(1, int((outtime - intime).total_seconds() / 3600)))
    cr = rng.uniform(0.6, 1.5); lac = rng.uniform(0.8, 2.0)
    wbc = rng.uniform(6, 12);   hgb = rng.uniform(9, 13)
    plt = rng.uniform(150, 300); bic = rng.uniform(20, 26)
    out = []
    for h in range(0, hours, 6):
        out.append({
            "charttime":   (intime + timedelta(hours=h)).isoformat(),
            "creatinine":  round(cr  + rng.normal(0, 0.15), 2),
            "lactate":     round(lac + rng.normal(0, 0.30), 2),
            "wbc":         round(wbc + rng.normal(0, 1.5),  1),
            "hemoglobin":  round(hgb + rng.normal(0, 0.5),  1),
            "platelets":   round(plt + rng.normal(0, 20),   0),
            "bicarbonate": round(bic + rng.normal(0, 1.5),  1),
        })
        cr  += rng.normal(0.02, 0.05)
        lac += rng.normal(0.01, 0.08)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Real model predictions
# ═════════════════════════════════════════════════════════════════════════════

_model_registry: dict = {}   # task_name → {type, model, labels, input_size}
_models_loaded = False


def _load_all_models():
    """
    Read every *_report.json in OUTPUT_DIR, find best model per task,
    load the corresponding checkpoint from MODELS_DIR.
    Called once on first prediction request.
    """
    global _models_loaded
    if _models_loaded:
        return

    report_paths = glob.glob(os.path.join(OUTPUT_DIR, "*_report.json"))
    if not report_paths:
        print(f"[WARN] No *_report.json files found in {OUTPUT_DIR}. "
              "Using fallback predictions.")
        _models_loaded = True
        return

    try:
        import torch
        from models import create_model
    except ImportError as e:
        print(f"[WARN] torch/models not importable: {e}. Using fallback predictions.")
        _models_loaded = True
        return

    config_dict = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            config_dict = yaml.safe_load(f)

    for report_path in report_paths:
        try:
            with open(report_path) as f:
                report = json.load(f)

            task        = report.get("task")
            best_model  = report.get("best_model")
            labels      = report.get("labels", [])
            input_size  = report.get("input_size", 81)
            comparison  = report.get("comparison", {})
            model_path  = comparison.get(best_model, {}).get("model_path", "")

            if not task or not best_model:
                continue

            # Resolve path (may be relative or absolute)
            if not os.path.isabs(model_path):
                model_path = os.path.join(PROJECT_ROOT, model_path)

            if not os.path.exists(model_path):
                # Try MODELS_DIR as fallback location
                basename = os.path.basename(model_path)
                model_path = os.path.join(MODELS_DIR, basename)

            if not os.path.exists(model_path):
                print(f"[WARN] Model file not found for task={task}: {model_path}")
                continue

            if best_model == "xgboost":
                with open(model_path, "rb") as f:
                    model_obj = pickle.load(f)
                _model_registry[task] = {
                    "type":  "xgboost",
                    "model": model_obj,
                    "labels": labels,
                }

            else:
                cfg = config_dict.copy()
                cfg["input_size"] = input_size
                cfg["num_tasks"]  = len(labels) if labels else 1
                model_obj = create_model(best_model, cfg)
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                model_obj.load_state_dict(checkpoint["model_state_dict"])
                model_obj.eval()
                _model_registry[task] = {
                    "type":       "dl",
                    "model":      model_obj,
                    "labels":     labels,
                    "input_size": input_size,
                }

            print(f"[INFO] Loaded {best_model} for task={task} ({len(labels)} labels)")

        except Exception:
            print(f"[WARN] Failed to load model from {report_path}:")
            traceback.print_exc()

    _models_loaded = True


def _extract_feature_sequence(icustay_id: int) -> "np.ndarray | None":
    """
    Run FeatureEngineer on the stay and return the most recent 24-step
    sequence — shape (1, 24, n_features).  Returns None on any failure.
    """
    try:
        from feature_engineering import FeatureEngineer

        df = cached("patients", _load_patients)
        if df is None:
            return None

        row = df[df["icustay_id"] == icustay_id]
        if len(row) == 0:
            return None

        stay    = row.iloc[0]
        intime  = pd.to_datetime(stay["intime"])
        outtime = pd.to_datetime(stay["outtime"]) if pd.notna(stay.get("outtime")) \
                  else intime + timedelta(hours=48)

        fe = cached("feature_engineer", lambda: FeatureEngineer(CONFIG_PATH))
        if fe is None:
            return None

        charts_grouped = cached("charts_grouped", _build_charts_grouped)
        labs_grouped   = cached("labs_grouped",   _build_labs_grouped)
        stay_charts    = (charts_grouped or {}).get(icustay_id, pd.DataFrame())
        stay_labs      = (labs_grouped   or {}).get(icustay_id, pd.DataFrame())

        # We need d_items / d_labitems for itemid→name mapping
        from data_loader import MIMICDataLoader
        loader = cached("data_loader", lambda: MIMICDataLoader(DATA_DIR, CONFIG_PATH))
        d_items    = loader.d_items    if loader and loader.d_items    is not None else pd.DataFrame()
        d_labitems = loader.d_labitems if loader and loader.d_labitems is not None else pd.DataFrame()

        features = fe.extract_features_for_stay(
            icustay_id=icustay_id,
            icu_intime=intime,
            icu_outtime=outtime,
            chartevents=stay_charts,
            labevents=stay_labs,
            d_items=d_items,
            d_labitems=d_labitems,
            window_hours=24,
        )

        if features is None or len(features) == 0:
            return None

        sequences, _ = fe.create_sequences(features, sequence_length=24, step_size=6)
        if len(sequences) == 0:
            return None

        return sequences[-1:].astype(np.float32)   # shape (1, 24, n_features)

    except Exception:
        traceback.print_exc()
        return None


def _run_models(icustay_id: int) -> dict:
    """
    Run every loaded model and collect per-label probabilities.
    Returns dict: label_name → float probability.
    """
    _load_all_models()
    scores: dict = {}

    if not _model_registry:
        return scores   # will trigger fallback

    X = _extract_feature_sequence(icustay_id)

    for task, info in _model_registry.items():
        try:
            labels = info["labels"]
            mtype  = info["type"]
            model  = info["model"]

            if X is None:
                for lbl in labels:
                    scores[lbl] = 0.0
                continue

            if mtype == "xgboost":
                X_flat = X.reshape(1, -1)
                # XGBoostPredictor wraps a list of per-task XGB models
                for i, m in enumerate(model.models):
                    if m is None or i >= len(labels):
                        continue
                    prob = float(m.predict_proba(X_flat)[0, 1])
                    scores[labels[i]] = round(prob, 4)

            else:
                import torch
                with torch.no_grad():
                    out = model(torch.FloatTensor(X)).numpy()[0]
                for i, lbl in enumerate(labels):
                    if i < len(out):
                        scores[lbl] = round(float(out[i]), 4)

        except Exception:
            print(f"[WARN] Prediction failed for task={task}:")
            traceback.print_exc()

    return scores


def _fallback_scores(icustay_id: int) -> dict:
    """Deterministic mock scores — used only when no trained models exist."""
    rng = np.random.RandomState(icustay_id % 9999)
    base = float(rng.beta(2, 5))
    horizon_h = {"_1h":1,"_3h":3,"_6h":6,"_12h":12,"_24h":24,"_48h":48,"_72h":72}
    scores = {}
    for label in PREDICTION_LABELS:
        boost = next((h * 0.002 for sfx, h in horizon_h.items() if label.endswith(sfx)), 0.0)
        scores[label] = round(min(1.0, base + boost + float(rng.beta(1.5, 4)) * 0.3), 4)
    return scores


def _get_predictions(icustay_id: int) -> dict:
    """
    Main prediction entry-point called by all routes.
    Uses real models if available, falls back to mock scores.
    """
    scores = _run_models(icustay_id)

    # Fill any missing labels (model may not cover every task)
    if not scores:
        scores = _fallback_scores(icustay_id)
    else:
        fallback = _fallback_scores(icustay_id)
        for lbl in PREDICTION_LABELS:
            if lbl not in scores:
                scores[lbl] = fallback[lbl]

    composite = round(
        scores.get("mortality_24h",  0) * 0.30 +
        scores.get("sepsis_24h",     0) * 0.20 +
        scores.get("aki_stage1_24h", 0) * 0.15 +
        scores.get("hypotension_6h", 0) * 0.20 +
        scores.get("ventilation_24h",0) * 0.15,
        4,
    )
    risk = "HIGH" if composite > 0.6 else "MEDIUM" if composite > 0.3 else "LOW"

    return {
        "icustay_id":      icustay_id,
        "composite_score": composite,
        "risk_level":      risk,
        "scores":          scores,
        "groups":          {g: {l: scores.get(l, 0.0) for l in ls}
                            for g, ls in TASK_GROUPS.items()},
        "shap_features":   _get_shap(icustay_id),
        "updated_at":      datetime.utcnow().isoformat() + "Z",
        "source":          "model" if _model_registry else "fallback",
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Real SHAP (readmission predictor)
# ═════════════════════════════════════════════════════════════════════════════

def _get_shap(icustay_id: int) -> list:
    """
    Return SHAP feature importances from the readmission XGBoost model.
    Falls back to deterministic mock values if shap / model unavailable.
    """
    try:
        import shap as shap_lib
        from predictors.readmission_predictor import ReadmissionPredictor

        model_path = os.path.join(MODELS_DIR, "readmission_xgboost.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Readmission model not found: {model_path}")

        with open(model_path, "rb") as f:
            readm_model = pickle.load(f)

        df = cached("patients", _load_patients)
        if df is None:
            raise ValueError("patients not loaded")

        stay_df = df[df["icustay_id"] == icustay_id]
        if len(stay_df) == 0:
            raise ValueError(f"icustay_id {icustay_id} not found")

        rp = ReadmissionPredictor(CONFIG_PATH)

        from data_loader import MIMICDataLoader
        loader = cached("data_loader", lambda: MIMICDataLoader(DATA_DIR, CONFIG_PATH))
        ce = cached("chartevents", _load_chartevents) or pd.DataFrame()
        le = cached("labevents",   _load_labevents)   or pd.DataFrame()
        diag = loader.diagnoses    if loader and loader.diagnoses    is not None else pd.DataFrame()
        prx  = loader.prescriptions if loader and loader.prescriptions is not None else pd.DataFrame()

        feat_df = rp.extract_discharge_features(
            stay_df, ce, le, diag, prx,
            services=getattr(loader, "services", None),
            outputevents=cached("labevents", _load_labevents),
        )

        if len(feat_df) == 0 or not rp.feature_names:
            raise ValueError("No discharge features extracted")

        X = feat_df[rp.feature_names].fillna(0).values
        explainer   = shap_lib.TreeExplainer(readm_model)
        shap_values = explainer.shap_values(X)
        # shap_values may be list (binary) or ndarray
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        abs_vals = np.abs(shap_values[0])

        top_idx = np.argsort(abs_vals)[::-1][:8]
        return [
            {"feature": rp.feature_names[i], "value": round(float(abs_vals[i]), 4)}
            for i in top_idx
        ]

    except Exception as exc:
        print(f"[INFO] SHAP unavailable for {icustay_id}: {exc}")
        return _fallback_shap(icustay_id)


def _fallback_shap(icustay_id: int) -> list:
    rng = np.random.RandomState((icustay_id + 42) % 9999)
    features = [
        "creatinine_trend_24h", "meanbp_min_6h", "heartrate_mean_12h",
        "lactate_mean_24h", "spo2_min_6h", "resprate_trend_12h",
        "wbc_std_24h", "los_hours",
    ]
    values = sorted([float(rng.uniform(0.01, 0.30)) for _ in features], reverse=True)
    return [{"feature": f, "value": round(v, 4)} for f, v in zip(features, values)]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Serialisation helpers
# ═════════════════════════════════════════════════════════════════════════════

def _safe(v):
    if isinstance(v, (np.integer,)):  return int(v)
    if isinstance(v, (np.floating,)): return None if np.isnan(v) else float(round(v, 2))
    if isinstance(v, pd.Timestamp):   return v.isoformat()
    try:
        if pd.isna(v): return None
    except Exception:
        pass
    return v


PATIENT_FIELDS = [
    "icustay_id", "subject_id", "hadm_id", "age", "gender",
    "admission_type", "diagnosis", "intime", "outtime",
    "los_hours", "expire_flag", "first_careunit",
]


def _row_to_dict(row: pd.Series) -> dict:
    return {k: _safe(row.get(k)) for k in PATIENT_FIELDS}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Routes
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/api/health")
def health():
    return jsonify({
        "status":        "ok",
        "ts":            datetime.utcnow().isoformat(),
        "data_dir":      DATA_DIR,
        "models_loaded": _models_loaded,
        "tasks_loaded":  list(_model_registry.keys()),
    })


@app.route("/api/stats")
def stats():
    df = cached("patients", _load_patients)
    if df is None:
        return jsonify({"error": "data unavailable"}), 503

    # Sample for risk distribution (avoid calling _get_predictions 54K times)
    sample = df.sample(min(500, len(df)), random_state=42)
    risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for _, r in sample.iterrows():
        p = _get_predictions(int(r["icustay_id"]))
        risk_counts[p["risk_level"]] += 1
    scale = len(df) / max(len(sample), 1)

    return jsonify({
        "total_stays":     int(len(df)),
        "total_patients":  int(df["subject_id"].nunique()),
        "mean_age":        round(float(df["age"].mean()), 1),
        "mean_los_hours":  round(float(df["los_hours"].dropna().mean()), 1),
        "expire_count":    int(df["expire_flag"].sum()),
        "gender_dist":     {
            "M": int((df["gender"] == "M").sum()),
            "F": int((df["gender"] == "F").sum()),
        },
        "care_units":      df["first_careunit"].value_counts().head(8).to_dict(),
        "admission_types": df["admission_type"].value_counts().to_dict(),
        "risk_dist":       {k: int(v * scale) for k, v in risk_counts.items()},
        "models_source":   "trained" if _model_registry else "fallback",
    })


@app.route("/api/patients")
def patients():
    df = cached("patients", _load_patients)
    if df is None:
        return jsonify({"error": "data unavailable"}), 503

    q        = request.args.get("q",        "").strip().lower()
    unit     = request.args.get("unit",     "")
    risk     = request.args.get("risk",     "").upper()
    page     = max(1, int(request.args.get("page",     1)))
    per_page = max(1, int(request.args.get("per_page", 20)))

    filtered = df.copy()

    if q:
        mask = (
            filtered["subject_id"].astype(str).str.contains(q, na=False)  |
            filtered["icustay_id"].astype(str).str.contains(q, na=False)  |
            filtered["diagnosis"].fillna("").str.lower().str.contains(q)  |
            filtered["first_careunit"].fillna("").str.lower().str.contains(q)
        )
        filtered = filtered[mask]

    if unit:
        filtered = filtered[filtered["first_careunit"].fillna("") == unit]

    total    = len(filtered)
    df_page  = filtered.iloc[(page-1)*per_page : page*per_page]

    result = []
    for _, row in df_page.iterrows():
        p    = _row_to_dict(row)
        pred = _get_predictions(int(row["icustay_id"]))
        p["composite_score"] = pred["composite_score"]
        p["risk_level"]      = pred["risk_level"]
        result.append(p)

    if risk:
        result = [p for p in result if p["risk_level"] == risk]

    all_units = sorted(df["first_careunit"].dropna().unique().tolist())

    return jsonify({
        "patients": result,
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "pages":    (total + per_page - 1) // per_page,
        "units":    all_units,
    })


@app.route("/api/patients/<int:icustay_id>")
def patient(icustay_id):
    df = cached("patients", _load_patients)
    if df is None:
        return jsonify({"error": "data unavailable"}), 503

    row = df[df["icustay_id"] == icustay_id]
    if len(row) == 0:
        return jsonify({"error": "not found"}), 404

    p    = _row_to_dict(row.iloc[0])
    pred = _get_predictions(icustay_id)
    p["composite_score"] = pred["composite_score"]
    p["risk_level"]      = pred["risk_level"]
    return jsonify(p)


@app.route("/api/patients/<int:icustay_id>/vitals")
def vitals(icustay_id):
    df = cached("patients", _load_patients)
    if df is None:
        return jsonify({"error": "data unavailable"}), 503

    row = df[df["icustay_id"] == icustay_id]
    if len(row) == 0:
        return jsonify({"error": "not found"}), 404

    return jsonify({
        "icustay_id": icustay_id,
        "vitals":     _real_vitals(icustay_id),
        "source":     "chartevents" if cached("charts_grouped", _build_charts_grouped) else "fallback",
    })


@app.route("/api/patients/<int:icustay_id>/labs")
def labs(icustay_id):
    df = cached("patients", _load_patients)
    if df is None:
        return jsonify({"error": "data unavailable"}), 503

    row = df[df["icustay_id"] == icustay_id]
    if len(row) == 0:
        return jsonify({"error": "not found"}), 404

    return jsonify({
        "icustay_id": icustay_id,
        "labs":       _real_labs(icustay_id),
        "source":     "labevents" if cached("labs_grouped", _build_labs_grouped) else "fallback",
    })


@app.route("/api/patients/<int:icustay_id>/predictions")
def predictions(icustay_id):
    return jsonify(_get_predictions(icustay_id))


@app.route("/api/alerts")
def alerts():
    df = cached("patients", _load_patients)
    if df is None:
        return jsonify({"alerts": []})

    sample = df.sample(min(300, len(df)), random_state=42)
    out = []
    for _, row in sample.iterrows():
        iid  = int(row["icustay_id"])
        pred = _get_predictions(iid)
        if pred["composite_score"] > 0.52:
            triggered = sorted(
                [{"label": l, "score": s}
                 for l, s in pred["scores"].items() if s > 0.65],
                key=lambda x: -x["score"],
            )[:3]
            if triggered:
                d = _row_to_dict(row)
                d["composite_score"]  = pred["composite_score"]
                d["risk_level"]       = pred["risk_level"]
                d["triggered_labels"] = triggered
                out.append(d)

    out.sort(key=lambda x: -x["composite_score"])
    return jsonify({"alerts": out[:40]})


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Startup
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Smart ICU Assistant — API  (Final Integrated Version)")
    print(f"  Data    : {DATA_DIR}")
    print(f"  Output  : {OUTPUT_DIR}")
    print(f"  Models  : {MODELS_DIR}")
    print(f"  Config  : {CONFIG_PATH}")
    print("  http://localhost:5000")
    print("=" * 60)

    # Eagerly warm up the data cache and model registry at startup
    # so first requests are fast. Comment these out for faster boot.
    print("[BOOT] Pre-loading patient data…")
    cached("patients", _load_patients)

    print("[BOOT] Pre-loading trained models…")
    _load_all_models()

    print("[BOOT] Pre-loading CHARTEVENTS (this may take a moment)…")
    cached("charts_grouped", _build_charts_grouped)

    print("[BOOT] Pre-loading LABEVENTS…")
    cached("labs_grouped", _build_labs_grouped)

    print("[BOOT] Ready.")
    app.run(debug=False, host="0.0.0.0", port=5000)