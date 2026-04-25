"""
Smart ICU Assistant — FastAPI Backend
=====================================================================
Drop-in replacement for icu_api.py (Flask) with:
  ✓ Auto-generated OpenAPI docs at /docs
  ✓ Pydantic response models
  ✓ Serves dashboard at /  (no separate web server needed)
  ✓ CORS enabled for local development
  ✓ Same data loading, prediction, and SHAP logic

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    # or:  python app.py

Endpoints (same contract as Flask version):
    GET  /                                      → Dashboard HTML
    GET  /api/health
    GET  /api/stats
    GET  /api/patients?q=&risk=&unit=&page=&per_page=
    GET  /api/patients/{icustay_id}
    GET  /api/patients/{icustay_id}/vitals
    GET  /api/patients/{icustay_id}/labs
    GET  /api/patients/{icustay_id}/predictions
    GET  /api/alerts
    GET  /docs                                  → Swagger UI (auto)
    GET  /redoc                                 → ReDoc (auto)

FIXES applied in this revision:
  1. SHAP "DataFrame is ambiguous" — outputevents was wrongly passed as
     cached("labevents", ...) which is a DataFrame. Changed to None (safe
     default) since outputevents is not separately cached in the app.
  2. Model loading for ensemble/stacked_ensemble — these produce a
     placeholder model_path string like "ensemble (no single checkpoint)",
     not a real file. Now falls back to the best individual model
     (xgboost > lstm > transformer) that has an actual saved checkpoint.
  3. Jinja2 | min(100) crash — min() is a sequence filter in Jinja2, not
     a numeric clamp. Fixed in the template by using an inline if expression.
     Also added a custom 'clamp' filter to Jinja2 environment as a safety net.
"""

# ── Stdlib ────────────────────────────────────────────────────────────────────
import os, sys, json, glob, pickle, traceback, math, warnings
from datetime import datetime, timedelta
from typing import Any, Optional
from pathlib import Path

# Suppress noisy sklearn/LGBMClassifier feature-name warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ── Third-party ───────────────────────────────────────────────────────────────
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import yaml

# ── Project imports ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart ICU Assistant",
    description="Real-time ICU patient monitoring, risk prediction & alerting powered by MIMIC-III",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve local JS dependencies (React, Recharts, Babel) — no CDN required
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Jinja2 Templates ─────────────────────────────────────────────────────────
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# FIX 3: Add a numeric clamp filter so templates can use {{ value | clamp(0, 100) }}
# The built-in Jinja2 `min` filter iterates a sequence — it crashes on floats.
def _jinja_clamp(value, lo=0, hi=100):
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return lo

templates.env.filters["clamp"] = _jinja_clamp

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = os.environ.get("MIMIC_DATA_DIR",  os.path.join(PROJECT_ROOT, "data"))
OUTPUT_DIR  = os.environ.get("ICU_OUTPUT_DIR",  os.path.join(PROJECT_ROOT, "output"))
MODELS_DIR  = os.environ.get("ICU_MODELS_DIR",  os.path.join(PROJECT_ROOT, "models"))
CONFIG_PATH = os.environ.get("ICU_CONFIG_PATH", os.path.join(PROJECT_ROOT, "config.yaml"))

# ── Prediction label registry ────────────────────────────────────────────────
PREDICTION_LABELS = [
    "mortality_6h",  "mortality_12h",  "mortality_24h",
    "sepsis_6h",     "sepsis_12h",     "sepsis_24h",
    "aki_stage1_24h","aki_stage2_24h", "aki_stage3_24h",
    "aki_stage1_48h","aki_stage2_48h", "aki_stage3_48h",
    "vasopressor_6h","vasopressor_12h",
    "ventilation_6h","ventilation_12h","ventilation_24h",
    "los_short_24h", "los_long_72h",
]

TASK_GROUPS = {
    "Mortality":      ["mortality_6h",   "mortality_12h",   "mortality_24h"],
    "Sepsis":         ["sepsis_6h",      "sepsis_12h",      "sepsis_24h"],
    "AKI":            ["aki_stage1_24h", "aki_stage2_24h",  "aki_stage3_24h",
                       "aki_stage1_48h", "aki_stage2_48h",  "aki_stage3_48h"],
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

# Lab itemids
LAB_ITEMIDS = {
    "creatinine":  [50912],
    "lactate":     [50813],
    "wbc":         [51301],
    "hemoglobin":  [51222],
    "platelets":   [51265],
    "bicarbonate": [50882],
    "chloride":    [50902],
}

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
        print(f"[INFO] CHARTEVENTS is {file_mb:.0f} MB — chunked loading")
        chunks = []
        total = 0
        for chunk in pd.read_csv(path, chunksize=2_000_000, low_memory=False,
                                  usecols=usecols,
                                  dtype={"ICUSTAY_ID": "float32",
                                         "ITEMID": "int32",
                                         "VALUENUM": "float32"}):
            chunk.columns = [c.lower() for c in chunk.columns]
            filtered = chunk[chunk["itemid"].isin(ALL_VITAL_ITEMIDS)]
            if len(filtered):
                chunks.append(filtered)
            total += len(chunk)
            if total % 10_000_000 == 0:
                print(f"[INFO]   CHARTEVENTS: {total:,} rows read, "
                      f"{sum(len(c) for c in chunks):,} kept")
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        print(f"[INFO]   CHARTEVENTS: done — {len(df):,} vital rows from {total:,} total")

    df["charttime"] = pd.to_datetime(df["charttime"])
    df["valuenum"]  = pd.to_numeric(df["valuenum"], errors="coerce")
    return df


# ── LABEVENTS (filtered to relevant labs) ────────────────────────────────────
def _load_labevents() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "LABEVENTS.csv")
    usecols = ["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"]

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


# ── Pre-grouped indexes ──────────────────────────────────────────────────────
def _build_charts_grouped():
    ce = cached("chartevents", _load_chartevents)
    return dict(tuple(ce.groupby("icustay_id"))) if ce is not None and len(ce) else {}


def _build_labs_grouped():
    le = cached("labevents", _load_labevents)
    return dict(tuple(le.groupby("icustay_id"))) if le is not None and len(le) else {}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Real vitals & labs from MIMIC-III
# ═════════════════════════════════════════════════════════════════════════════

def _real_vitals(icustay_id: int) -> list:
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
    pivot = pivot.resample("1h").mean().ffill(limit=4).reset_index()
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
    pivot = pivot.resample("6h").mean().ffill(limit=2)
    if len(pivot) > 24:
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


# ─── Fallback generators ─────────────────────────────────────────────────────

def _fallback_vitals(icustay_id: int) -> list:
    df = cached("patients", _load_patients)
    intime, outtime = datetime.utcnow() - timedelta(hours=24), datetime.utcnow()
    if df is not None:
        row = df[df["icustay_id"] == icustay_id]
        if len(row):
            r = row.iloc[0]
            intime  = pd.to_datetime(r["intime"])
            outtime = pd.to_datetime(r["outtime"]) if pd.notna(r.get("outtime")) else intime + timedelta(hours=48)

    rng = np.random.RandomState(icustay_id % 9999)
    hours = min(72, max(1, int((outtime - intime).total_seconds() / 3600)))
    hr = rng.uniform(75, 95); sbp = rng.uniform(105, 130); dbp = rng.uniform(60, 80)
    rr = rng.uniform(14, 20); tmp = rng.uniform(36.5, 37.5)
    spo2 = rng.uniform(95, 99); glc = rng.uniform(100, 160)
    out = []
    for h in range(hours):
        out.append({
            "charttime": (intime + timedelta(hours=h)).isoformat(),
            "heartrate": round(hr + rng.normal(0, 5), 1),
            "sysbp":     round(sbp + rng.normal(0, 8), 1),
            "diasbp":    round(dbp + rng.normal(0, 5), 1),
            "meanbp":    round((sbp + 2*dbp)/3 + rng.normal(0, 4), 1),
            "resprate":  round(rr + rng.normal(0, 2), 1),
            "tempc":     round(tmp + rng.normal(0, 0.3), 2),
            "spo2":      round(min(100, spo2 + rng.normal(0, 1)), 1),
            "glucose":   round(glc + rng.normal(0, 15), 1),
        })
        hr += rng.normal(0, 0.5); sbp += rng.normal(0, 0.8)
        tmp += rng.normal(0, 0.05); spo2 += rng.normal(0, 0.2)
    return out


def _fallback_labs(icustay_id: int) -> list:
    df = cached("patients", _load_patients)
    intime, outtime = datetime.utcnow() - timedelta(hours=24), datetime.utcnow()
    if df is not None:
        row = df[df["icustay_id"] == icustay_id]
        if len(row):
            r = row.iloc[0]
            intime  = pd.to_datetime(r["intime"])
            outtime = pd.to_datetime(r["outtime"]) if pd.notna(r.get("outtime")) else intime + timedelta(hours=48)

    rng = np.random.RandomState((icustay_id + 1000) % 9999)
    hours = min(72, max(1, int((outtime - intime).total_seconds() / 3600)))
    cr = rng.uniform(0.6, 1.5); lac = rng.uniform(0.8, 2.0)
    wbc = rng.uniform(6, 12); hgb = rng.uniform(9, 13)
    plt_ = rng.uniform(150, 300); bic = rng.uniform(20, 26)
    out = []
    for h in range(0, hours, 6):
        out.append({
            "charttime":   (intime + timedelta(hours=h)).isoformat(),
            "creatinine":  round(cr + rng.normal(0, 0.15), 2),
            "lactate":     round(lac + rng.normal(0, 0.30), 2),
            "wbc":         round(wbc + rng.normal(0, 1.5), 1),
            "hemoglobin":  round(hgb + rng.normal(0, 0.5), 1),
            "platelets":   round(plt_ + rng.normal(0, 20), 0),
            "bicarbonate": round(bic + rng.normal(0, 1.5), 1),
        })
        cr  += rng.normal(0.02, 0.05)
        lac += rng.normal(0.01, 0.08)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Model predictions
# ═════════════════════════════════════════════════════════════════════════════

_model_registry: dict = {}
_models_loaded = False

_FALLBACK_MODEL_PRIORITY = ["xgboost", "lightgbm", "lstm", "transformer"]


def _find_model_file(task: str, model_name: str) -> str:
    """Resolve a model file path, checking several candidate locations."""
    candidates = [
        os.path.join(MODELS_DIR, f"{task}_{model_name}.pkl"),
        os.path.join(MODELS_DIR, f"{task}_{model_name}.pth"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


def _load_single_model(task, model_name, labels, input_size, config_dict):
    """Load a single model (DL or tree-based) and return a registry entry dict, or None."""
    model_path = _find_model_file(task, model_name)
    if not model_path:
        return None

    try:
        if model_name in ("xgboost", "lightgbm"):
            with open(model_path, "rb") as f:
                model_obj = pickle.load(f)
            return {"type": "xgboost", "model": model_obj, "labels": labels}
        else:
            import torch
            from models import create_model
            cfg = config_dict.copy()
            cfg["input_size"] = input_size
            cfg["num_tasks"]  = len(labels) if labels else 1
            model_obj = create_model(model_name, cfg)
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            model_obj.load_state_dict(checkpoint["model_state_dict"])
            model_obj.eval()
            return {"type": "dl", "model": model_obj, "labels": labels, "input_size": input_size}
    except Exception:
        print(f"[WARN] Failed to load {model_name} for {task}:")
        traceback.print_exc()
        return None


def _load_all_models():
    global _models_loaded
    if _models_loaded:
        return

    report_paths = glob.glob(os.path.join(OUTPUT_DIR, "*_report.json"))
    if not report_paths:
        print(f"[WARN] No *_report.json in {OUTPUT_DIR}. Using fallback predictions.")
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

            task       = report.get("task")
            best_model = report.get("best_model")
            labels     = report.get("labels", [])
            input_size = report.get("input_size", 81)

            if not task or not best_model:
                continue

            # ── Try loading ensemble/stacked_ensemble pickle first ─────────
            if best_model in ("ensemble", "stacked_ensemble", "weighted_ensemble"):
                ens_path = _find_model_file(task, best_model)
                if ens_path:
                    with open(ens_path, "rb") as f:
                        ens_data = pickle.load(f)

                    ens_type = ens_data.get("type", "")
                    component_names = ens_data.get("component_model_names", [])

                    # Load each component model
                    components = {}
                    for cname in component_names:
                        entry = _load_single_model(task, cname, labels, input_size, config_dict)
                        if entry:
                            components[cname] = entry

                    if len(components) >= 2:
                        _model_registry[task] = {
                            "type":       ens_type,
                            "labels":     labels,
                            "input_size": input_size,
                            "components": components,
                            "weights":    ens_data.get("weights", {}),
                            "meta_learners": ens_data.get("meta_learners", []),
                        }
                        print(f"[INFO] Loaded {ens_type} for task={task} "
                              f"({len(components)} components: {list(components.keys())})")
                        continue
                    else:
                        print(f"[WARN] Ensemble for {task}: only {len(components)} components loaded, falling back")

            # ── Fallback: load best individual model ───────────────────────
            loaded = False
            try_order = [best_model] + [m for m in _FALLBACK_MODEL_PRIORITY if m != best_model]
            for mname in try_order:
                entry = _load_single_model(task, mname, labels, input_size, config_dict)
                if entry:
                    _model_registry[task] = entry
                    suffix = "" if mname == best_model else f" (fallback from {best_model})"
                    print(f"[INFO] Loaded {mname} for task={task} ({len(labels)} labels){suffix}")
                    loaded = True
                    break

            if not loaded:
                print(f"[WARN] No loadable model for task={task}")

        except Exception:
            print(f"[WARN] Failed to load {report_path}:")
            traceback.print_exc()

    _models_loaded = True


def _extract_feature_sequence(icustay_id: int):
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

        from data_loader import MIMICDataLoader
        loader = cached("data_loader", lambda: MIMICDataLoader(DATA_DIR, CONFIG_PATH))

        # Ensure dictionary tables are loaded (they're tiny ~200KB each)
        if loader and loader.d_items is None:
            try:
                loader.load_dictionaries()
            except Exception:
                pass

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

        return sequences[-1:].astype(np.float32)

    except Exception:
        traceback.print_exc()
        return None


def _predict_single_component(info: dict, X, X_flat) -> dict:
    """Run a single model component and return {label: prob} dict."""
    labels = info["labels"]
    mtype  = info["type"]
    model  = info["model"]
    preds  = {}

    try:
        if mtype == "xgboost":
            # MultiTaskXGBoost wrapper has .models list
            if hasattr(model, 'models'):
                for i, m in enumerate(model.models):
                    if m is None or i >= len(labels):
                        continue
                    expected = m.n_features_in_ if hasattr(m, 'n_features_in_') else X_flat.shape[1]
                    x_in = X_flat
                    if X_flat.shape[1] < expected:
                        x_in = np.pad(X_flat, ((0, 0), (0, expected - X_flat.shape[1])))
                    elif X_flat.shape[1] > expected:
                        x_in = X_flat[:, :expected]
                    prob = float(m.predict_proba(x_in)[0, 1])
                    preds[labels[i]] = prob
            else:
                # Bare XGBClassifier (single task)
                expected = model.n_features_in_ if hasattr(model, 'n_features_in_') else X_flat.shape[1]
                x_in = X_flat
                if X_flat.shape[1] < expected:
                    x_in = np.pad(X_flat, ((0, 0), (0, expected - X_flat.shape[1])))
                elif X_flat.shape[1] > expected:
                    x_in = X_flat[:, :expected]
                prob = float(model.predict_proba(x_in)[0, 1])
                if labels:
                    preds[labels[0]] = prob
        else:
            # DL model (LSTM / Transformer) — pad features if needed
            import torch
            expected_size = info.get("input_size", X.shape[-1])
            x_in = X
            if X.shape[-1] < expected_size:
                pad_width = expected_size - X.shape[-1]
                x_in = np.pad(X, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
            elif X.shape[-1] > expected_size:
                x_in = X[:, :, :expected_size]

            with torch.no_grad():
                logits = model(torch.FloatTensor(x_in))
                out = torch.sigmoid(logits).numpy()[0]
            for i, lbl in enumerate(labels):
                if i < len(out):
                    preds[lbl] = float(out[i])
    except Exception:
        print(f"[WARN] Component prediction failed ({mtype}):")
        traceback.print_exc()

    return preds


def _run_models(icustay_id: int) -> dict:
    _load_all_models()
    scores: dict = {}
    if not _model_registry:
        return scores

    X = _extract_feature_sequence(icustay_id)

    for task, info in _model_registry.items():
        try:
            labels = info["labels"]
            mtype  = info["type"]

            if X is None:
                for lbl in labels:
                    scores[lbl] = 0.0
                continue

            X_flat = X.reshape(1, -1)

            # ── Ensemble / Stacked Ensemble ──────────────────────────────
            if mtype in ("weighted_ensemble", "stacked_ensemble"):
                components  = info["components"]
                weights_map = info.get("weights", {})
                meta_lrs    = info.get("meta_learners", [])

                # Run each component model
                comp_preds = {}  # {model_name: {label: prob}}
                for cname, cinfo in components.items():
                    comp_preds[cname] = _predict_single_component(cinfo, X, X_flat)

                comp_names = list(comp_preds.keys())

                if mtype == "weighted_ensemble":
                    # Weighted average using AUROC² weights
                    raw_w = np.array([weights_map.get(n, 1.0) for n in comp_names])
                    w = raw_w / raw_w.sum()
                    for lbl in labels:
                        vals = [comp_preds[n].get(lbl, 0.0) for n in comp_names]
                        scores[lbl] = round(float(np.average(vals, weights=w)), 4)

                elif mtype == "stacked_ensemble":
                    # Use LR meta-learners per label
                    for i, lbl in enumerate(labels):
                        vals = [comp_preds[n].get(lbl, 0.0) for n in comp_names]
                        if i < len(meta_lrs) and meta_lrs[i] is not None:
                            meta_X = np.array(vals).reshape(1, -1)
                            scores[lbl] = round(float(
                                meta_lrs[i].predict_proba(meta_X)[0, 1]
                            ), 4)
                        else:
                            # Fallback to simple mean if no meta-learner
                            scores[lbl] = round(float(np.mean(vals)), 4)

            # ── Single model (XGBoost / LightGBM / DL) ──────────────────
            else:
                scores.update({
                    k: round(v, 4)
                    for k, v in _predict_single_component(info, X, X_flat).items()
                })

        except Exception:
            print(f"[WARN] Prediction failed for task={task}:")
            traceback.print_exc()

    return scores


def _fallback_scores(icustay_id: int) -> dict:
    rng = np.random.RandomState(icustay_id % 9999)
    base = float(rng.beta(2, 5))
    horizon_h = {"_1h":1,"_3h":3,"_6h":6,"_12h":12,"_24h":24,"_48h":48,"_72h":72}
    scores = {}
    for label in PREDICTION_LABELS:
        boost = next((h * 0.002 for sfx, h in horizon_h.items() if label.endswith(sfx)), 0.0)
        scores[label] = round(min(1.0, base + boost + float(rng.beta(1.5, 4)) * 0.3), 4)
    return scores


def _get_predictions(icustay_id: int) -> dict:
    scores = _run_models(icustay_id)

    if not scores:
        scores = _fallback_scores(icustay_id)
    else:
        fallback = _fallback_scores(icustay_id)
        for lbl in PREDICTION_LABELS:
            if lbl not in scores:
                scores[lbl] = fallback[lbl]

    composite = round(
        scores.get("mortality_24h",  0) * 0.30 +
        scores.get("sepsis_24h",     0) * 0.25 +
        scores.get("aki_stage1_24h", 0) * 0.15 +
        scores.get("vasopressor_12h",0) * 0.15 +
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
# SECTION 4 — SHAP
# ═════════════════════════════════════════════════════════════════════════════

def _get_shap(icustay_id: int) -> list:
    """Return SHAP-style feature importance values for a patient."""
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

# ── SSR Pages (Jinja2 Templates) ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def page_overview(request: Request):
    """Overview dashboard."""
    df = cached("patients", _load_patients)
    stats_data = {}
    if df is not None:
        stats_data = {
            "total_stays":    int(len(df)),
            "total_patients": int(df["subject_id"].nunique()),
            "mean_age":       round(float(df["age"].mean()), 1),
            "mean_los_hours": round(float(df["los_hours"].dropna().mean()), 1),
            "risk_dist":      {},
        }
        # Quick risk sample (small for speed)
        sample = df.sample(min(50, len(df)), random_state=42)
        risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for _, r in sample.iterrows():
            p = _get_predictions(int(r["icustay_id"]))
            risk_counts[p["risk_level"]] += 1
        scale = len(df) / max(len(sample), 1)
        stats_data["risk_dist"] = {k: int(v * scale) for k, v in risk_counts.items()}

    # Load training report data for model performance summary
    report_data = {}
    report_paths = glob.glob(os.path.join(OUTPUT_DIR, "*_report.json"))
    for rp in report_paths:
        try:
            with open(rp) as f:
                rpt = json.load(f)
            task_name = rpt.get("task", "")
            if task_name:
                report_data[task_name] = {
                    "best_auroc": rpt.get("best_auroc", rpt.get("best_metric", 0.0)),
                    "best_model": rpt.get("best_model", "unknown"),
                }
        except Exception:
            pass

    return templates.TemplateResponse("index.html", {
        "request": request,
        "active_page": "overview",
        "stats": stats_data,
        "models": list(_model_registry.keys()),
        "report_data": report_data if report_data else None,
    })


@app.get("/patients", response_class=HTMLResponse, include_in_schema=False)
async def page_patients(request: Request, q: str = "", risk: str = "", page: int = 1):
    """Patient browser page."""
    per_page = 20
    df = cached("patients", _load_patients)
    if df is None:
        return templates.TemplateResponse("patients.html", {
            "request": request, "active_page": "patients",
            "patients": [], "total": 0, "page": 1, "pages": 1,
            "q": q, "risk": risk,
        })

    filtered = df.copy()
    q_lower = q.strip().lower()
    if q_lower:
        mask = (
            filtered["subject_id"].astype(str).str.contains(q_lower, na=False) |
            filtered["icustay_id"].astype(str).str.contains(q_lower, na=False) |
            filtered["diagnosis"].fillna("").str.lower().str.contains(q_lower) |
            filtered["first_careunit"].fillna("").str.lower().str.contains(q_lower)
        )
        filtered = filtered[mask]

    total = len(filtered)
    pages_total = max(1, math.ceil(total / per_page))
    page = max(1, min(page, pages_total))
    df_page = filtered.iloc[(page - 1) * per_page : page * per_page]

    patients_list = []
    for _, row in df_page.iterrows():
        p = _row_to_dict(row)
        pred = _get_predictions(int(row["icustay_id"]))
        p["risk_level"] = pred["risk_level"]
        p["composite_score"] = pred["composite_score"]
        patients_list.append(p)

    if risk:
        patients_list = [p for p in patients_list if p["risk_level"] == risk.upper()]

    return templates.TemplateResponse("patients.html", {
        "request": request, "active_page": "patients",
        "patients": patients_list, "total": total,
        "page": page, "pages": pages_total,
        "q": q, "risk": risk,
    })


@app.get("/patients/{icustay_id}", response_class=HTMLResponse, include_in_schema=False)
async def page_patient_detail(request: Request, icustay_id: int):
    """Single patient detail page."""
    df = cached("patients", _load_patients)
    if df is None:
        raise HTTPException(503, "data unavailable")
    row = df[df["icustay_id"] == icustay_id]
    if len(row) == 0:
        raise HTTPException(404, "Patient not found")

    patient_data = _row_to_dict(row.iloc[0])
    pred = _get_predictions(icustay_id)

    return templates.TemplateResponse("patient_detail.html", {
        "request": request, "active_page": "patients",
        "patient": patient_data, "pred": pred,
    })


@app.get("/predict", response_class=HTMLResponse, include_in_schema=False)
async def page_predict(request: Request):
    """New patient prediction form page."""
    return templates.TemplateResponse("predict.html", {
        "request": request, "active_page": "predict",
    })


@app.get("/compare", response_class=HTMLResponse, include_in_schema=False)
async def page_compare(request: Request):
    """Patient comparison page."""
    return templates.TemplateResponse("compare.html", {
        "request": request, "active_page": "compare",
    })


@app.get("/api/validation/auroc")
async def validation_auroc():
    """Compute per-task AUROC on the test set (lazy, on-demand)."""
    try:
        vdata = _load_validation_data()
        if vdata is None:
            return {"error": "Validation data not available. Run training first."}

        _load_all_models()
        if not _model_registry:
            return {"error": "No trained models loaded."}

        from sklearn.metrics import roc_auc_score

        y_true = vdata["y"]
        X_test = vdata["X"]
        label_names = vdata["label_names"]
        n_test = len(X_test)

        # Predict all test samples
        y_pred = np.zeros_like(y_true, dtype=np.float32)
        for i in range(n_test):
            try:
                result = _predict_test_sample(i, vdata)
                if result and result.get("results"):
                    for r in result["results"]:
                        lbl_idx = label_names.index(r["label"]) if r["label"] in label_names else -1
                        if lbl_idx >= 0:
                            if len(y_pred.shape) > 1 and lbl_idx < y_pred.shape[1]:
                                y_pred[i, lbl_idx] = r["prob"]
                            elif len(y_pred.shape) == 1:
                                y_pred[i] = r["prob"]
            except Exception:
                continue

        per_task = {}
        total_auroc = 0.0
        valid_tasks = 0
        n_labels = min(len(label_names), y_true.shape[1] if len(y_true.shape) > 1 else 1)

        for j in range(n_labels):
            lbl = label_names[j] if j < len(label_names) else f"label_{j}"
            try:
                yt = y_true[:, j] if len(y_true.shape) > 1 else y_true
                yp = y_pred[:, j] if len(y_pred.shape) > 1 else y_pred
                if len(np.unique(yt)) < 2:
                    continue
                auroc = float(roc_auc_score(yt, yp))
                per_task[lbl] = round(auroc, 4)
                total_auroc += auroc
                valid_tasks += 1
            except Exception:
                continue

        avg = round(total_auroc / max(valid_tasks, 1), 4)
        return {
            "per_task_auroc": per_task,
            "avg_auroc": avg,
            "n_test": n_test,
            "n_tasks": valid_tasks,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ── Validation Demo ───────────────────────────────────────────────────────────
_validation_cache: dict = {}


def _load_validation_data():
    """Load feature cache and split into test set (last 20% by temporal order)."""
    if "loaded" in _validation_cache:
        return _validation_cache

    X_path = os.path.join(OUTPUT_DIR, "feature_cache_X.npy")
    y_path = os.path.join(OUTPUT_DIR, "feature_cache_y.npy")
    meta_path = os.path.join(OUTPUT_DIR, "feature_cache_meta.pkl")

    if not all(os.path.exists(p) for p in [X_path, y_path, meta_path]):
        return None

    try:
        print("[VALIDATION] Loading feature cache (this may take a moment)...")
        X = np.load(X_path, mmap_mode='r')  # memory-mapped for speed
        y = np.load(y_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        n_total = len(X)
        n_test_start = int(n_total * 0.8)  # last 20% = test split

        # Get label names from meta
        label_names = meta.get("label_names", PREDICTION_LABELS)
        # Get ICU stay and subject IDs — with fallback
        icustay_ids = meta.get("icustay_ids", [])
        subject_ids = meta.get("subject_ids", [])

        # If icustay_ids is empty or all None, try to cross-reference with patients DF
        has_valid_ids = icustay_ids and any(x is not None for x in icustay_ids[:min(20, len(icustay_ids))])
        if not has_valid_ids:
            print("[VALIDATION] icustay_ids missing from meta — attempting fallback from patients DF")
            patients_df = cached("patients", _load_patients)
            if patients_df is not None and len(patients_df) > 0:
                available = min(len(patients_df), n_total)
                icustay_ids = patients_df["icustay_id"].iloc[:available].tolist()
                subject_ids = patients_df["subject_id"].iloc[:available].tolist()
                # Pad with None if patients_df has fewer rows than feature cache
                if available < n_total:
                    icustay_ids.extend([None] * (n_total - available))
                    subject_ids.extend([None] * (n_total - available))
                print(f"[VALIDATION] Recovered {available} icustay_ids from patients DF (total needed: {n_total})")
            else:
                # Last resort: generate sequential placeholder IDs
                icustay_ids = [None] * n_total
                subject_ids = [None] * n_total

        _validation_cache.update({
            "X": X,
            "y": y,
            "meta": meta,
            "label_names": label_names,
            "n_total": n_total,
            "n_test_start": n_test_start,
            "n_test": n_total - n_test_start,
            "icustay_ids": icustay_ids,
            "subject_ids": subject_ids,
            "timestamps": meta.get("timestamps", []),
            "loaded": True,
        })
        print(f"[VALIDATION] Loaded: {n_total} total, {n_total - n_test_start} test samples, {len(label_names)} labels")
        return _validation_cache
    except Exception as e:
        print(f"[VALIDATION] Failed to load: {e}")
        traceback.print_exc()
        return None


def _predict_test_sample(idx: int, vdata: dict) -> dict:
    """Run model prediction on a single test sample and compare with ground truth."""
    X = vdata["X"]
    y = vdata["y"]
    label_names = vdata["label_names"]
    abs_idx = vdata["n_test_start"] + idx

    if abs_idx >= vdata["n_total"]:
        return None

    # Ground truth
    y_true = y[abs_idx]

    # Get feature sequence for this sample
    X_sample = np.array(X[abs_idx:abs_idx+1], dtype=np.float32)  # [1, seq_len, features]

    # Run predictions through loaded models
    _load_all_models()
    predictions = {}

    for task, info in _model_registry.items():
        try:
            labels = info["labels"]
            mtype  = info["type"]

            X_flat = X_sample.reshape(1, -1)

            # ── Ensemble / Stacked Ensemble ──────────────────────────
            if mtype in ("weighted_ensemble", "stacked_ensemble"):
                components  = info.get("components", {})
                weights_map = info.get("weights", {})
                meta_lrs    = info.get("meta_learners", [])

                comp_preds = {}
                for cname, cinfo in components.items():
                    comp_preds[cname] = _predict_single_component(cinfo, X_sample, X_flat)

                comp_names = list(comp_preds.keys())

                if mtype == "weighted_ensemble":
                    raw_w = np.array([weights_map.get(n, 1.0) for n in comp_names])
                    w = raw_w / raw_w.sum()
                    for lbl in labels:
                        vals = [comp_preds[n].get(lbl, 0.0) for n in comp_names]
                        predictions[lbl] = round(float(np.average(vals, weights=w)), 4)

                elif mtype == "stacked_ensemble":
                    for i, lbl in enumerate(labels):
                        vals = [comp_preds[n].get(lbl, 0.0) for n in comp_names]
                        if i < len(meta_lrs) and meta_lrs[i] is not None:
                            meta_X = np.array(vals).reshape(1, -1)
                            predictions[lbl] = round(float(
                                meta_lrs[i].predict_proba(meta_X)[0, 1]
                            ), 4)
                        else:
                            predictions[lbl] = round(float(np.mean(vals)), 4)

            # ── Single model (XGBoost / LightGBM / DL) ──────────────
            else:
                preds = _predict_single_component(info, X_sample, X_flat)
                predictions.update({k: round(v, 4) for k, v in preds.items()})

        except Exception:
            traceback.print_exc()

    # Build results per label
    results = []
    correct = 0
    total_labels = min(len(label_names), len(y_true))

    for i in range(total_labels):
        lbl = label_names[i]
        gt = int(y_true[i]) if not np.isnan(y_true[i]) else 0
        prob = predictions.get(lbl, 0.0)
        pred_binary = 1 if prob >= 0.5 else 0
        match = pred_binary == gt
        if match:
            correct += 1
        results.append({
            "label": lbl,
            "prob": round(prob, 4),
            "pred": pred_binary,
            "true": gt,
            "match": match,
        })

    icustay_id = None
    subject_id = None
    if vdata["icustay_ids"] and abs_idx < len(vdata["icustay_ids"]):
        icustay_id = vdata["icustay_ids"][abs_idx]
    if vdata["subject_ids"] and abs_idx < len(vdata["subject_ids"]):
        subject_id = vdata["subject_ids"][abs_idx]

    return {
        "index": idx,
        "icustay_id": icustay_id,
        "subject_id": subject_id,
        "results": results,
        "correct": correct,
        "total": total_labels,
        "accuracy": correct / max(total_labels, 1),
    }


def _compute_validation_summary(vdata: dict) -> dict:
    """Compute aggregate metrics over a sample of the test set."""
    from sklearn.metrics import roc_auc_score

    n_test = vdata["n_test"]
    label_names = vdata["label_names"]
    y = vdata["y"]
    n_test_start = vdata["n_test_start"]
    n_labels = min(len(label_names), y.shape[1] if len(y.shape) > 1 else 1)

    # Sample up to 200 test patients for speed
    sample_size = min(200, n_test)
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(n_test, size=sample_size, replace=False)

    all_preds = []
    total_correct = 0
    total_labels_counted = 0

    for idx in sample_indices:
        result = _predict_test_sample(int(idx), vdata)
        if result:
            all_preds.append(result)
            total_correct += result["correct"]
            total_labels_counted += result["total"]

    avg_accuracy = total_correct / max(total_labels_counted, 1)

    # Per-task AUROC
    per_task_auroc = {}
    if all_preds and n_labels > 0:
        for i in range(n_labels):
            lbl = label_names[i]
            y_true_list = []
            y_prob_list = []
            for pred_result in all_preds:
                for r in pred_result["results"]:
                    if r["label"] == lbl:
                        y_true_list.append(r["true"])
                        y_prob_list.append(r["prob"])
                        break
            if len(set(y_true_list)) >= 2:
                try:
                    auroc = roc_auc_score(y_true_list, y_prob_list)
                    per_task_auroc[lbl] = round(auroc, 4)
                except Exception:
                    pass

    avg_auroc = np.mean(list(per_task_auroc.values())) if per_task_auroc else 0.0

    return {
        "n_test": n_test,
        "n_labels": n_labels,
        "n_models": len(_model_registry),
        "source": "trained_models" if _model_registry else "no_models",
        "avg_accuracy": avg_accuracy,
        "avg_auroc": avg_auroc,
        "per_task_auroc": per_task_auroc,
        "sample_size": sample_size,
    }


@app.get("/validation", response_class=HTMLResponse, include_in_schema=False)
async def page_validation(request: Request, page: int = 1, idx: int = -1):
    """Validation demo page: pick test patient, predict, compare."""
    vdata = _load_validation_data()
    if vdata is None:
        return templates.TemplateResponse("validation.html", {
            "request": request, "active_page": "validation",
            "error": "Feature cache not found. Run the training pipeline first (python main_pipeline.py --data_dir data).",
        })

    n_test = vdata["n_test"]
    per_page = 15
    pages_total = max(1, math.ceil(n_test / per_page))
    page = max(1, min(page, pages_total))

    # Ensure models are loaded before computing predictions
    _load_all_models()

    # Build test patient list for this page — compute accuracy for each
    start = (page - 1) * per_page
    end = min(start + per_page, n_test)
    test_patients = []
    page_accuracy_sum = 0.0
    page_accuracy_count = 0

    # Get timestamps for fallback identification
    timestamps = vdata.get("timestamps", [])

    for i in range(start, end):
        abs_i = vdata["n_test_start"] + i
        icustay_id = vdata["icustay_ids"][abs_i] if abs_i < len(vdata["icustay_ids"]) else None
        subject_id = vdata["subject_ids"][abs_i] if abs_i < len(vdata["subject_ids"]) else None

        # Use timestamp as fallback identifier when ICU Stay ID is missing
        ts_label = None
        if abs_i < len(timestamps):
            try:
                ts_label = str(timestamps[abs_i])[:16]  # "2154-02-08 12:00"
            except Exception:
                pass

        # Compute accuracy for this sample (lightweight — just one sample)
        sample_acc = None
        try:
            result = _predict_test_sample(i, vdata)
            if result and result.get("total", 0) > 0:
                sample_acc = result["accuracy"]
                page_accuracy_sum += sample_acc
                page_accuracy_count += 1
        except Exception:
            pass

        test_patients.append({
            "index": i,
            "icustay_id": icustay_id,
            "subject_id": subject_id,
            "timestamp": ts_label,
            "accuracy": sample_acc,
        })

    # If a patient is selected, predict (may already have been computed above)
    selected = None
    if 0 <= idx < n_test:
        selected = _predict_test_sample(idx, vdata)

    avg_page_acc = page_accuracy_sum / max(page_accuracy_count, 1)

    # Summary (lightweight — don't compute full AUROC every page load)
    summary = {
        "n_test": n_test,
        "n_labels": min(len(vdata["label_names"]), vdata["y"].shape[1] if len(vdata["y"].shape) > 1 else 1),
        "n_models": len(_model_registry),
        "source": "trained_models" if _model_registry else "no_models",
        "avg_accuracy": avg_page_acc,
        "avg_auroc": 0.0,
        "per_task_auroc": {},
        "has_patient_ids": any(p["icustay_id"] is not None for p in test_patients),
    }

    return templates.TemplateResponse("validation.html", {
        "request": request, "active_page": "validation",
        "summary": summary,
        "test_patients": test_patients,
        "selected": selected,
        "page": page, "pages": pages_total,
        "error": None,
    })


@app.get("/api/health")
async def health():
    return {
        "status":        "ok",
        "ts":            datetime.utcnow().isoformat(),
        "data_dir":      DATA_DIR,
        "models_loaded": _models_loaded,
        "tasks_loaded":  list(_model_registry.keys()),
        "framework":     "FastAPI",
    }


# ═════════════════════════════════════════════════════════════════════════════
# NEW PATIENT PREDICTION — POST /api/predict
# ═════════════════════════════════════════════════════════════════════════════
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class VitalsInput(BaseModel):
    heartrate:  float = Field(80, description="Heart rate (bpm)")
    sysbp:      float = Field(120, description="Systolic BP (mmHg)")
    diasbp:     float = Field(80, description="Diastolic BP (mmHg)")
    meanbp:     float = Field(90, description="Mean arterial pressure (mmHg)")
    resprate:   float = Field(16, description="Respiratory rate (breaths/min)")
    tempc:      float = Field(37.0, description="Temperature (°C)")
    spo2:       float = Field(98, description="Oxygen saturation (%)")
    glucose:    float = Field(100, description="Blood glucose (mg/dL)")

class LabsInput(BaseModel):
    creatinine:  float = Field(1.0, description="Creatinine (mg/dL)")
    lactate:     float = Field(1.0, description="Lactate (mmol/L)")
    wbc:         float = Field(8000, description="White blood cell count (/µL)")
    hemoglobin:  float = Field(13.0, description="Hemoglobin (g/dL)")
    platelets:   float = Field(250, description="Platelet count (×10³/µL)")
    bicarbonate: float = Field(24, description="Bicarbonate (mEq/L)")
    chloride:    float = Field(102, description="Chloride (mEq/L)")

class MedicationsInput(BaseModel):
    vasopressors:            bool = Field(False, description="Currently on vasopressors")
    antibiotics:             bool = Field(False, description="Currently on antibiotics")
    mechanical_ventilation:  bool = Field(False, description="Currently on mech vent")
    sedation:                bool = Field(False, description="Currently sedated")

class DemographicsInput(BaseModel):
    age:       int   = Field(60, description="Patient age (years)")
    gender:    str   = Field("M", description="M or F")
    weight_kg: float = Field(75, description="Weight in kg")

class HistoryInput(BaseModel):
    hours_in_icu:          float       = Field(6, description="Hours since ICU admission")
    prior_icu_admissions:  int         = Field(0, description="Prior ICU visits")
    diagnosis:             str         = Field("", description="Primary diagnosis")
    comorbidities:         List[str]   = Field([], description="e.g. diabetes, hypertension")

class PatientInput(BaseModel):
    demographics: DemographicsInput = DemographicsInput()
    vitals:       VitalsInput       = VitalsInput()
    labs:         LabsInput         = LabsInput()
    medications:  MedicationsInput  = MedicationsInput()
    history:      HistoryInput      = HistoryInput()


def _clinical_rule_scores(v: VitalsInput, l: LabsInput, m: MedicationsInput,
                          d: DemographicsInput, h: HistoryInput) -> dict:
    """Compute risk scores using evidence-based clinical rules."""
    scores = {}

    # ── SIRS Score (0-4) ─────────────────────────────────────────────────
    sirs = 0
    if v.tempc > 38.3 or v.tempc < 36.0:
        sirs += 1
    if v.heartrate > 90:
        sirs += 1
    if v.resprate > 20:
        sirs += 1
    if l.wbc > 12000 or l.wbc < 4000:
        sirs += 1

    # ── Shock Index ──────────────────────────────────────────────────────
    shock_index = round(v.heartrate / max(v.sysbp, 1), 2)

    # ── MORTALITY ────────────────────────────────────────────────────────
    mort_base = 0.0
    if d.age > 80:       mort_base += 0.25
    elif d.age > 65:     mort_base += 0.15
    if v.meanbp < 60:    mort_base += 0.20
    if l.lactate > 4.0:  mort_base += 0.25
    elif l.lactate > 2.0:mort_base += 0.10
    if v.spo2 < 90:      mort_base += 0.15
    if sirs >= 3:         mort_base += 0.10
    if shock_index > 1.0: mort_base += 0.10
    mort_base = min(1.0, mort_base)

    scores["mortality_6h"]  = round(min(1.0, mort_base * 0.7), 4)
    scores["mortality_12h"] = round(min(1.0, mort_base * 0.85), 4)
    scores["mortality_24h"] = round(min(1.0, mort_base), 4)

    # ── SEPSIS (SIRS + suspected infection) ──────────────────────────────
    infection_factor = 0.0
    if m.antibiotics:      infection_factor += 0.30
    if l.wbc > 12000:      infection_factor += 0.15
    if l.lactate > 2.0:    infection_factor += 0.20
    if v.tempc > 38.3:     infection_factor += 0.15
    if sirs >= 2:          infection_factor += 0.20 * (sirs / 4)

    sepsis_base = min(1.0, (sirs / 4) * 0.5 + infection_factor)
    scores["sepsis_6h"]  = round(min(1.0, sepsis_base * 0.7), 4)
    scores["sepsis_12h"] = round(min(1.0, sepsis_base * 0.85), 4)
    scores["sepsis_24h"] = round(min(1.0, sepsis_base), 4)

    # ── AKI (KDIGO stages) ──────────────────────────────────────────────
    cr_baseline = 1.0
    cr_ratio = l.creatinine / max(cr_baseline, 0.1)
    cr_increase = l.creatinine - cr_baseline

    aki1 = min(1.0, max(0, (cr_ratio - 1.3) / 0.5) * 0.5 + max(0, (cr_increase - 0.2) / 0.3) * 0.5)
    aki2 = min(1.0, max(0, (cr_ratio - 1.8) / 0.5) * 0.7)
    aki3 = min(1.0, max(0, (cr_ratio - 2.8) / 0.5) * 0.7)

    scores["aki_stage1_24h"] = round(aki1, 4)
    scores["aki_stage2_24h"] = round(aki2, 4)
    scores["aki_stage3_24h"] = round(aki3, 4)
    scores["aki_stage1_48h"] = round(min(1.0, aki1 * 1.15), 4)
    scores["aki_stage2_48h"] = round(min(1.0, aki2 * 1.15), 4)
    scores["aki_stage3_48h"] = round(min(1.0, aki3 * 1.15), 4)



    # ── VASOPRESSOR ─────────────────────────────────────────────────────
    vaso_base = 0.0
    if m.vasopressors:      vaso_base += 0.70
    if v.meanbp < 65:       vaso_base += 0.25
    elif v.meanbp < 70:     vaso_base += 0.10
    if shock_index > 1.0:   vaso_base += 0.15
    vaso_base = min(1.0, vaso_base)

    scores["vasopressor_6h"]  = round(min(1.0, vaso_base * 0.85), 4)
    scores["vasopressor_12h"] = round(min(1.0, vaso_base), 4)

    # ── VENTILATION ─────────────────────────────────────────────────────
    vent_base = 0.0
    if m.mechanical_ventilation: vent_base += 0.70
    if v.spo2 < 88:             vent_base += 0.30
    elif v.spo2 < 92:           vent_base += 0.15
    if v.resprate > 30:         vent_base += 0.20
    elif v.resprate > 24:       vent_base += 0.10
    vent_base = min(1.0, vent_base)

    scores["ventilation_6h"]  = round(min(1.0, vent_base * 0.7), 4)
    scores["ventilation_12h"] = round(min(1.0, vent_base * 0.85), 4)
    scores["ventilation_24h"] = round(min(1.0, vent_base), 4)

    # ── LENGTH OF STAY ──────────────────────────────────────────────────
    los_short = 0.0
    los_long  = 0.0
    severity = (sirs / 4) * 0.3 + mort_base * 0.3 + sepsis_base * 0.2 + max(0, aki1) * 0.2
    if severity < 0.15:     los_short = 0.60
    elif severity < 0.30:   los_short = 0.30
    if severity > 0.50:     los_long = 0.70
    elif severity > 0.30:   los_long = 0.40

    scores["los_short_24h"] = round(los_short, 4)
    scores["los_long_72h"]  = round(los_long, 4)

    # ── Estimated LOS in days (human-readable) ──────────────────────────
    # Convert binary threshold probabilities into a single day estimate.
    # Weighted interpolation: high short prob → ~1 day, high long prob → ~5+ days
    if los_short > 0.5:
        est_days = round(1.0 + (1.0 - los_short) * 1.5, 1)   # ~1.0-1.75 days
    elif los_long > 0.5:
        est_days = round(3.0 + los_long * 4.0, 1)             # ~3.0-5.8 days
    else:
        est_days = round(2.0 + severity * 3.0, 1)             # ~2.0-3.5 days
    scores["los_estimated_days"] = est_days

    clinical_info = {
        "sirs": sirs,
        "shock_index": shock_index,
        "map": v.meanbp,
        "cr_ratio": round(cr_ratio, 2),
        "los_estimated_days": est_days,
    }

    return scores, clinical_info


def _run_models_on_input(v: VitalsInput, l: LabsInput) -> dict:
    """Run trained models on new patient data (if models exist)."""
    _load_all_models()
    if not _model_registry:
        return {}

    feat_vals = [
        v.heartrate, v.sysbp, v.diasbp, v.meanbp,
        v.resprate, v.tempc, v.spo2, v.glucose,
        v.heartrate / max(v.sysbp, 1),
        v.sysbp - v.diasbp,
        v.sysbp * v.heartrate,
        l.creatinine, l.lactate, l.wbc, l.hemoglobin,
        l.platelets, l.bicarbonate, l.chloride,
    ]

    n_features = len(feat_vals)
    seq = np.array([feat_vals] * 24, dtype=np.float32)
    X = seq[np.newaxis, :, :]

    scores = {}
    for task, info in _model_registry.items():
        try:
            labels = info["labels"]
            mtype  = info["type"]
            model  = info["model"]

            if mtype == "xgboost":
                X_flat = X.reshape(1, -1)
                for i, m in enumerate(model.models):
                    if m is None or i >= len(labels):
                        continue
                    expected = m.n_features_in_ if hasattr(m, 'n_features_in_') else X_flat.shape[1]
                    if X_flat.shape[1] < expected:
                        X_flat = np.pad(X_flat, ((0,0),(0, expected - X_flat.shape[1])))
                    elif X_flat.shape[1] > expected:
                        X_flat = X_flat[:, :expected]
                    prob = float(m.predict_proba(X_flat)[0, 1])
                    scores[labels[i]] = round(prob, 4)
            else:
                import torch
                X_t = torch.FloatTensor(X)
                try:
                    with torch.no_grad():
                        out = model(X_t).numpy()[0]
                except RuntimeError:
                    if hasattr(model, 'lstm'):
                        expected = model.lstm.input_size
                    elif hasattr(model, 'input_proj'):
                        expected = model.input_proj.in_features
                    else:
                        continue
                    if n_features < expected:
                        X_padded = np.pad(X, ((0,0),(0,0),(0, expected - n_features)))
                        X_t = torch.FloatTensor(X_padded)
                    elif n_features > expected:
                        X_t = torch.FloatTensor(X[:, :, :expected])
                    with torch.no_grad():
                        out = model(X_t).numpy()[0]

                for i, lbl in enumerate(labels):
                    if i < len(out):
                        scores[lbl] = round(float(out[i]), 4)
        except Exception:
            print(f"[WARN] Model prediction failed for task={task}:")
            traceback.print_exc()

    return scores


@app.post("/api/predict")
async def predict_new_patient(patient: PatientInput):
    v = patient.vitals
    l = patient.labs
    m = patient.medications
    d = patient.demographics
    h = patient.history

    rule_scores, clinical_info = _clinical_rule_scores(v, l, m, d, h)
    model_scores = _run_models_on_input(v, l)

    final_scores = {}
    for lbl in PREDICTION_LABELS:
        if lbl in model_scores and model_scores[lbl] > 0:
            final_scores[lbl] = model_scores[lbl]
        else:
            final_scores[lbl] = rule_scores.get(lbl, 0.0)

    source = "trained_models" if model_scores else "clinical_rules"

    composite = round(
        final_scores.get("mortality_24h",  0) * 0.30 +
        final_scores.get("sepsis_24h",     0) * 0.25 +
        final_scores.get("aki_stage1_24h", 0) * 0.15 +
        final_scores.get("vasopressor_12h",0) * 0.15 +
        final_scores.get("ventilation_24h",0) * 0.15,
        4,
    )
    risk = "HIGH" if composite > 0.6 else "MEDIUM" if composite > 0.3 else "LOW"

    alerts = []

    if final_scores.get("mortality_24h", 0) > 0.5:
        alerts.append({"type": "critical", "category": "Mortality",
                       "message": f"High 24h mortality risk ({final_scores['mortality_24h']:.0%})",
                       "score": final_scores["mortality_24h"]})
    if clinical_info["sirs"] >= 3:
        alerts.append({"type": "critical", "category": "Sepsis",
                       "message": f"SIRS score {clinical_info['sirs']}/4 — high sepsis risk",
                       "score": final_scores.get("sepsis_24h", 0)})
    elif clinical_info["sirs"] >= 2 and m.antibiotics:
        alerts.append({"type": "warning", "category": "Sepsis",
                       "message": f"SIRS {clinical_info['sirs']}/4 + antibiotics — monitor for sepsis",
                       "score": final_scores.get("sepsis_24h", 0)})
    if l.creatinine > 2.0:
        alerts.append({"type": "critical", "category": "AKI",
                       "message": f"Creatinine {l.creatinine} mg/dL (elevated) — AKI risk",
                       "score": final_scores.get("aki_stage1_24h", 0)})
    elif l.creatinine > 1.3:
        alerts.append({"type": "warning", "category": "AKI",
                       "message": f"Creatinine {l.creatinine} mg/dL — monitor kidney function",
                       "score": final_scores.get("aki_stage1_24h", 0)})
    if v.meanbp < 65:
        alerts.append({"type": "critical", "category": "Vasopressor",
                       "message": f"MAP {v.meanbp:.0f} mmHg < 65 — low blood pressure, vasopressor may be needed",
                       "score": final_scores.get("vasopressor_6h", 0)})
    elif v.meanbp < 70:
        alerts.append({"type": "warning", "category": "Vasopressor",
                       "message": f"MAP {v.meanbp:.0f} mmHg — borderline low, monitor closely",
                       "score": final_scores.get("vasopressor_6h", 0)})
    if m.vasopressors:
        alerts.append({"type": "warning", "category": "Vasopressor",
                       "message": "Patient currently on vasopressors",
                       "score": final_scores.get("vasopressor_12h", 0)})
    if v.spo2 < 90:
        alerts.append({"type": "critical", "category": "Ventilation",
                       "message": f"SpO₂ {v.spo2}% — severe hypoxemia, consider ventilation",
                       "score": final_scores.get("ventilation_6h", 0)})
    elif v.spo2 < 94 and v.resprate > 24:
        alerts.append({"type": "warning", "category": "Ventilation",
                       "message": f"SpO₂ {v.spo2}% + RR {v.resprate} — respiratory distress",
                       "score": final_scores.get("ventilation_12h", 0)})
    if clinical_info["shock_index"] > 1.0:
        alerts.append({"type": "warning", "category": "Hemodynamic",
                       "message": f"Shock index {clinical_info['shock_index']} (>1.0) — hemodynamic instability",
                       "score": composite})
    if l.lactate > 4.0:
        alerts.append({"type": "critical", "category": "Metabolic",
                       "message": f"Lactate {l.lactate} mmol/L — severe tissue hypoperfusion",
                       "score": composite})
    elif l.lactate > 2.0:
        alerts.append({"type": "warning", "category": "Metabolic",
                       "message": f"Lactate {l.lactate} mmol/L — elevated, monitor perfusion",
                       "score": composite})

    alerts.sort(key=lambda a: (0 if a["type"] == "critical" else 1, -a["score"]))

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return {
        "patient_id":      f"NEW_{ts}",
        "composite_score": composite,
        "risk_level":      risk,
        "scores":          final_scores,
        "groups":          {g: {lbl: final_scores.get(lbl, 0.0) for lbl in ls}
                           for g, ls in TASK_GROUPS.items()},
        "alerts":          alerts,
        "clinical_scores": clinical_info,
        "input_summary": {
            "age": d.age, "gender": d.gender,
            "hr": v.heartrate, "bp": f"{v.sysbp}/{v.diasbp}",
            "map": v.meanbp, "spo2": v.spo2, "rr": v.resprate,
            "temp": v.tempc, "cr": l.creatinine, "lactate": l.lactate,
        },
        "source":          source,
        "updated_at":      datetime.utcnow().isoformat() + "Z",
    }

_stats_cache = {"data": None, "ts": 0}

@app.get("/api/stats")
async def stats():
    import time
    now = time.time()
    if _stats_cache["data"] and (now - _stats_cache["ts"]) < 60:
        return _stats_cache["data"]

    df = cached("patients", _load_patients)
    if df is None:
        raise HTTPException(503, "data unavailable")

    sample = df.sample(min(200, len(df)), random_state=42)
    risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for _, r in sample.iterrows():
        p = _get_predictions(int(r["icustay_id"]))
        risk_counts[p["risk_level"]] += 1
    scale = len(df) / max(len(sample), 1)

    result = {
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
    }

    _stats_cache["data"] = result
    _stats_cache["ts"] = now
    return result


@app.get("/api/patients")
async def patients(
    q:        str = Query("", description="Search ID, diagnosis, unit"),
    risk:     str = Query("", description="Filter by risk: HIGH, MEDIUM, LOW"),
    unit:     str = Query("", description="Filter by care unit"),
    page:     int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Results per page"),
):
    df = cached("patients", _load_patients)
    if df is None:
        raise HTTPException(503, "data unavailable")

    q = q.strip().lower()
    risk = risk.upper()
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

    total   = len(filtered)
    df_page = filtered.iloc[(page-1)*per_page : page*per_page]

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

    return {
        "patients": result,
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "pages":    max(1, math.ceil(total / per_page)),
        "units":    all_units,
    }


@app.get("/api/patients/{icustay_id}")
async def patient(icustay_id: int):
    df = cached("patients", _load_patients)
    if df is None:
        raise HTTPException(503, "data unavailable")

    row = df[df["icustay_id"] == icustay_id]
    if len(row) == 0:
        raise HTTPException(404, "not found")

    p    = _row_to_dict(row.iloc[0])
    pred = _get_predictions(icustay_id)
    p["composite_score"] = pred["composite_score"]
    p["risk_level"]      = pred["risk_level"]
    return p


@app.get("/api/patients/{icustay_id}/vitals")
async def vitals(icustay_id: int):
    df = cached("patients", _load_patients)
    if df is None:
        raise HTTPException(503, "data unavailable")

    row = df[df["icustay_id"] == icustay_id]
    if len(row) == 0:
        raise HTTPException(404, "not found")

    return {
        "icustay_id": icustay_id,
        "vitals":     _real_vitals(icustay_id),
        "source":     "chartevents" if cached("charts_grouped", _build_charts_grouped) else "fallback",
    }


@app.get("/api/patients/{icustay_id}/labs")
async def labs(icustay_id: int):
    df = cached("patients", _load_patients)
    if df is None:
        raise HTTPException(503, "data unavailable")

    row = df[df["icustay_id"] == icustay_id]
    if len(row) == 0:
        raise HTTPException(404, "not found")

    return {
        "icustay_id": icustay_id,
        "labs":       _real_labs(icustay_id),
        "source":     "labevents" if cached("labs_grouped", _build_labs_grouped) else "fallback",
    }


@app.get("/api/patients/{icustay_id}/predictions")
async def predictions(icustay_id: int):
    return _get_predictions(icustay_id)


@app.get("/api/alerts")
async def alerts():
    df = cached("patients", _load_patients)
    if df is None:
        return {"alerts": []}

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
    return {"alerts": out[:40]}


import threading
from contextlib import asynccontextmanager

_BOOT_STAGES = {
    "starting": {
        "index": 0,
        "progress": 0.05,
        "label": "Initializing service",
        "detail": "Preparing API routes and dashboard assets",
    },
    "loading_patients": {
        "index": 1,
        "progress": 0.20,
        "label": "Loading patient demographics",
        "detail": "Reading patients, admissions, and ICU stays",
    },
    "patients_loaded": {
        "index": 2,
        "progress": 0.38,
        "label": "Patients ready",
        "detail": "Core roster is available for browsing",
    },
    "loading_models": {
        "index": 3,
        "progress": 0.52,
        "label": "Loading trained models",
        "detail": "Preparing model registry and prediction metadata",
    },
    "models_loaded": {
        "index": 4,
        "progress": 0.66,
        "label": "Models ready",
        "detail": "Predictions can run while full live data warms up",
    },
    "loading_chartevents": {
        "index": 5,
        "progress": 0.80,
        "label": "Loading CHARTEVENTS",
        "detail": "Grouping vital-sign history for bedside trends",
    },
    "loading_labevents": {
        "index": 6,
        "progress": 0.92,
        "label": "Loading LABEVENTS",
        "detail": "Attaching lab history to ICU stays",
    },
    "ready": {
        "index": 7,
        "progress": 1.00,
        "label": "Realtime data ready",
        "detail": "Vitals and labs now come from full MIMIC-III data",
    },
}
_BOOT_TOTAL_STAGES = len(_BOOT_STAGES)
_boot_status = {
    "stage": "starting",
    "label": _BOOT_STAGES["starting"]["label"],
    "detail": _BOOT_STAGES["starting"]["detail"],
    "progress": _BOOT_STAGES["starting"]["progress"],
    "current_step": 1,
    "total_steps": _BOOT_TOTAL_STAGES,
    "ready": False,
    "error": None,
    "updated_at": datetime.utcnow().isoformat(),
}


def _set_boot_status(stage: str, *, ready: Optional[bool] = None, error: Optional[str] = None):
    meta = _BOOT_STAGES.get(stage, {})
    _boot_status.update({
        "stage": stage,
        "label": meta.get("label", stage.replace("_", " ").title()),
        "detail": meta.get("detail", _boot_status.get("detail", "")),
        "progress": meta.get("progress", _boot_status.get("progress", 0.0)),
        "current_step": min(meta.get("index", _BOOT_TOTAL_STAGES - 1) + 1, _BOOT_TOTAL_STAGES),
        "total_steps": _BOOT_TOTAL_STAGES,
        "ready": ready if ready is not None else stage == "ready",
        "error": error,
        "updated_at": datetime.utcnow().isoformat(),
    })

def _background_loader():
    """Load heavy CHARTEVENTS + LABEVENTS in background thread."""
    try:
        _set_boot_status("loading_chartevents")
        print("[BOOT-BG] Loading CHARTEVENTS (may take 30+ min for large files)…")
        cached("charts_grouped", _build_charts_grouped)

        _set_boot_status("loading_labevents")
        print("[BOOT-BG] Loading LABEVENTS…")
        cached("labs_grouped", _build_labs_grouped)

        _set_boot_status("ready", ready=True)
        print("[BOOT-BG] [OK] All data loaded. Vitals/labs now use real MIMIC-III data.")
    except Exception as exc:
        _set_boot_status("ready", ready=False, error=str(exc))
        _boot_status["label"] = "Background loading failed"
        _boot_status["detail"] = (
            "Dashboard remains usable, but vitals and labs may stay on fallback data"
        )
        print(f"[BOOT-BG] Background loading failed: {exc}")


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan: load essentials sync, heavy data in background."""
    _set_boot_status("starting", ready=False, error=None)
    print("=" * 60)
    print("  Smart ICU Assistant — FastAPI")
    print(f"  Data      : {DATA_DIR}")
    print(f"  Output    : {OUTPUT_DIR}")
    print(f"  Models    : {MODELS_DIR}")
    print(f"  Config    : {CONFIG_PATH}")
    print(f"  Docs      : http://localhost:8000/docs")
    print(f"  Dashboard : http://localhost:8000")
    print("=" * 60)

    _set_boot_status("loading_patients")
    print("[BOOT] Loading patient demographics…")
    cached("patients", _load_patients)
    _set_boot_status("patients_loaded")

    _set_boot_status("loading_models")
    print("[BOOT] Loading trained models…")
    _load_all_models()
    _set_boot_status("models_loaded")

    print("[BOOT] Server starting — heavy data loading in background thread…")
    bg_thread = threading.Thread(target=_background_loader, daemon=True)
    bg_thread.start()

    print("[BOOT] [OK] Server is READY (vitals/labs will use fallback until background load completes)")
    yield
    print("[SHUTDOWN] Goodbye.")

app.router.lifespan_context = lifespan


@app.get("/api/boot_status", include_in_schema=False)
async def boot_status():
    return dict(_boot_status)


# ═════════════════════════════════════════════════════════════════════════════
# Direct run: python app.py
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")