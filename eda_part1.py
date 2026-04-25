"""
EDA Part 1 — Demographics & Distribution Analysis
Generates publication-ready figures from MIMIC-III for the project report.

Figures produced (saved to output/eda/):
  1. Dataset overview table
  2. Age distribution (survivors vs non-survivors)
  3. Gender distribution with mortality overlay
  4. ICU Length of Stay distribution
  5. Care unit distribution
  6. Vital signs box plots
  7. Lab values box plots

Usage:
    python eda_part1.py [--sample N]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from tqdm import tqdm
import logging

from data_loader import MIMICDataLoader

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'axes.titleweight': 'bold',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

PALETTE = sns.color_palette('deep')
COLOR_SURV = '#2196F3'
COLOR_DEAD = '#F44336'
COLOR_MALE = '#42A5F5'
COLOR_FEMALE = '#EF5350'
OUTPUT_DIR = os.path.join('output', 'eda')


def ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Dataset Overview Table
# ═══════════════════════════════════════════════════════════════════════════════

def fig_dataset_overview(merged: pd.DataFrame, loader: MIMICDataLoader):
    """Styled table summarizing dataset dimensions."""
    logger.info("Figure 1: Dataset overview table")

    n_patients = merged['subject_id'].nunique()
    n_admissions = merged['hadm_id'].nunique()
    n_stays = len(merged)
    pct_male = (merged['gender'] == 'M').mean() * 100
    mortality = merged['expire_flag'].mean() * 100
    median_age = merged['age'].median()
    mean_age = merged['age'].mean()
    median_los = merged['los'].median() if 'los' in merged.columns else np.nan

    # Compute median LOS from intime/outtime if 'los' column missing
    if pd.isna(median_los):
        los_hours = (merged['outtime'] - merged['intime']).dt.total_seconds() / 3600
        median_los_h = los_hours.median()
        los_str = f"{median_los_h:.1f} hours"
    else:
        los_str = f"{median_los:.1f} days"

    chart_rows = len(loader.chartevents) if loader.chartevents is not None else 0
    lab_rows = len(loader.labevents) if loader.labevents is not None else 0

    rows = [
        ['Total Dataset Size', '43.6 GB (uncompressed)'],
        ['Unique Patients', f'{n_patients:,}'],
        ['Hospital Admissions', f'{n_admissions:,}'],
        ['ICU Stays', f'{n_stays:,}'],
        ['Median Age', f'{median_age:.1f} years'],
        ['Mean Age', f'{mean_age:.1f} years'],
        ['Male Patients', f'{pct_male:.1f}%'],
        ['In-Hospital Mortality', f'{mortality:.1f}%'],
        ['Median ICU LOS', los_str],
        ['CHARTEVENTS Rows (filtered)', f'{chart_rows:,}'],
        ['LABEVENTS Rows (in-stay)', f'{lab_rows:,}'],
        ['Engineered Features / Timestep', '81'],
        ['Prediction Labels', '19'],
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    table = ax.table(
        cellText=rows,
        colLabels=['Statistic', 'Value'],
        loc='center',
        cellLoc='left',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)

    # Style header
    for j in range(2):
        cell = table[0, j]
        cell.set_facecolor('#1565C0')
        cell.set_text_props(color='white', weight='bold')

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(2):
            cell = table[i, j]
            cell.set_facecolor('#E3F2FD' if i % 2 == 0 else 'white')

    ax.set_title('MIMIC-III Dataset Overview', fontsize=16, fontweight='bold', pad=20)
    path = os.path.join(OUTPUT_DIR, 'eda_dataset_overview.png')
    plt.savefig(path)
    plt.close()
    logger.info(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Age Distribution
# ═══════════════════════════════════════════════════════════════════════════════

def fig_age_distribution(merged: pd.DataFrame):
    """Histogram + KDE of age, stratified by survival."""
    logger.info("Figure 2: Age distribution")

    ages = merged[['age', 'expire_flag']].dropna(subset=['age']).copy()
    ages['Outcome'] = ages['expire_flag'].map({0: 'Survived', 1: 'Deceased'})

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram for each group
    for label, color in [('Survived', COLOR_SURV), ('Deceased', COLOR_DEAD)]:
        subset = ages[ages['Outcome'] == label]['age']
        ax.hist(subset, bins=40, alpha=0.55, color=color, label=label, edgecolor='white')

    # KDE overlay
    for label, color in [('Survived', COLOR_SURV), ('Deceased', COLOR_DEAD)]:
        subset = ages[ages['Outcome'] == label]['age']
        if len(subset) > 10:
            subset.plot.kde(ax=ax, color=color, linewidth=2, label=f'{label} (KDE)')

    # Reference lines
    median_age = ages['age'].median()
    mean_age = ages['age'].mean()
    ax.axvline(median_age, color='#FF9800', linestyle='--', linewidth=2,
               label=f'Median = {median_age:.1f}')
    ax.axvline(mean_age, color='#9C27B0', linestyle=':', linewidth=2,
               label=f'Mean = {mean_age:.1f}')

    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Count')
    ax.set_title('Age Distribution: Survivors vs Deceased')
    ax.legend(framealpha=0.9)
    ax.set_xlim(15, 95)

    path = os.path.join(OUTPUT_DIR, 'eda_age_distribution.png')
    plt.savefig(path)
    plt.close()
    logger.info(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Gender Distribution
# ═══════════════════════════════════════════════════════════════════════════════

def fig_gender_distribution(merged: pd.DataFrame):
    """Bar chart of gender counts with mortality rate overlay."""
    logger.info("Figure 3: Gender distribution")

    gender_stats = merged.groupby('gender').agg(
        count=('subject_id', 'count'),
        mortality=('expire_flag', 'mean')
    ).reset_index()
    gender_stats['mortality_pct'] = gender_stats['mortality'] * 100
    gender_stats['gender_label'] = gender_stats['gender'].map({'M': 'Male', 'F': 'Female'})

    fig, ax1 = plt.subplots(figsize=(8, 6))

    bars = ax1.bar(
        gender_stats['gender_label'],
        gender_stats['count'],
        color=[COLOR_MALE, COLOR_FEMALE],
        edgecolor='white',
        width=0.5,
        zorder=3,
    )

    # Add count labels on bars
    for bar, count in zip(bars, gender_stats['count']):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                 f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=13)

    ax1.set_ylabel('Number of ICU Stays', fontsize=12)
    ax1.set_title('Gender Distribution with Mortality Rate', fontsize=14, fontweight='bold')

    # Mortality overlay on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(gender_stats['gender_label'], gender_stats['mortality_pct'],
             'D-', color='#F44336', markersize=10, linewidth=2, label='Mortality Rate')
    ax2.set_ylabel('Mortality Rate (%)', color='#F44336', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#F44336')
    ax2.set_ylim(0, max(gender_stats['mortality_pct']) * 1.5)
    ax2.legend(loc='upper right')

    path = os.path.join(OUTPUT_DIR, 'eda_gender_distribution.png')
    plt.savefig(path)
    plt.close()
    logger.info(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — ICU Length of Stay
# ═══════════════════════════════════════════════════════════════════════════════

def fig_los_distribution(merged: pd.DataFrame):
    """Histogram of ICU LOS with clinical thresholds marked."""
    logger.info("Figure 4: Length of stay distribution")

    los_hours = (merged['outtime'] - merged['intime']).dt.total_seconds() / 3600
    los_hours = los_hours[los_hours > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: linear scale (capped at 500h for readability)
    los_capped = los_hours.clip(upper=500)
    ax1.hist(los_capped, bins=60, color='#26A69A', edgecolor='white', alpha=0.8)
    ax1.axvline(24, color='#FF9800', linestyle='--', linewidth=2, label='24h (Short Stay)')
    ax1.axvline(72, color='#F44336', linestyle='--', linewidth=2, label='72h (Long Stay)')
    ax1.axvline(los_hours.median(), color='#9C27B0', linestyle=':', linewidth=2,
                label=f'Median = {los_hours.median():.1f}h')
    ax1.set_xlabel('ICU Length of Stay (hours)')
    ax1.set_ylabel('Count')
    ax1.set_title('ICU LOS Distribution (Linear Scale)')
    ax1.legend(fontsize=9)

    # Right: log scale (full range)
    ax2.hist(los_hours, bins=80, color='#26A69A', edgecolor='white', alpha=0.8)
    ax2.set_yscale('log')
    ax2.axvline(24, color='#FF9800', linestyle='--', linewidth=2, label='24h')
    ax2.axvline(72, color='#F44336', linestyle='--', linewidth=2, label='72h')
    ax2.set_xlabel('ICU Length of Stay (hours)')
    ax2.set_ylabel('Count (log scale)')
    ax2.set_title('ICU LOS Distribution (Log Scale)')
    ax2.legend(fontsize=9)

    # Add stats text box
    stats_text = (
        f"N = {len(los_hours):,}\n"
        f"Mean = {los_hours.mean():.1f}h\n"
        f"Median = {los_hours.median():.1f}h\n"
        f"<24h: {(los_hours < 24).mean()*100:.1f}%\n"
        f">72h: {(los_hours > 72).mean()*100:.1f}%"
    )
    ax2.text(0.97, 0.95, stats_text, transform=ax2.transAxes, fontsize=9,
             va='top', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('ICU Length of Stay Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'eda_los_distribution.png')
    plt.savefig(path)
    plt.close()
    logger.info(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Care Unit Distribution
# ═══════════════════════════════════════════════════════════════════════════════

def fig_care_units(merged: pd.DataFrame):
    """Horizontal bar chart of ICU care units."""
    logger.info("Figure 5: Care unit distribution")

    col = None
    for c in ['first_careunit', 'curr_careunit', 'last_careunit']:
        if c in merged.columns:
            col = c
            break
    if col is None:
        logger.warning("  No care unit column found, skipping")
        return

    unit_counts = merged[col].value_counts().sort_values(ascending=True)

    # Compute mortality per unit
    unit_mort = merged.groupby(col)['expire_flag'].mean() * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette('Blues_d', len(unit_counts))
    bars = ax.barh(unit_counts.index, unit_counts.values, color=colors, edgecolor='white')

    # Add count + mortality labels
    for i, (unit, count) in enumerate(unit_counts.items()):
        mort = unit_mort.get(unit, 0)
        ax.text(count + max(unit_counts) * 0.01, i,
                f' {count:,}  (mort: {mort:.1f}%)',
                va='center', fontsize=10)

    ax.set_xlabel('Number of ICU Stays')
    ax.set_title('ICU Care Unit Distribution', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'eda_care_units.png')
    plt.savefig(path)
    plt.close()
    logger.info(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 — Vital Signs Box Plots
# ═══════════════════════════════════════════════════════════════════════════════

def fig_vitals_boxplots(loader: MIMICDataLoader, sample_n=5000):
    """Box plots of vital signs with clinical reference ranges."""
    logger.info(f"Figure 6: Vital signs distributions (sampling {sample_n} stays)")

    if loader.chartevents is None or len(loader.chartevents) == 0:
        logger.warning("  No chartevents loaded, skipping")
        return

    vital_itemids = {
        'Heart Rate': [220045, 211],
        'Systolic BP': [220050, 220179, 51, 455],
        'Diastolic BP': [220051, 220180, 8368, 8441],
        'Resp Rate': [220210, 224690, 618, 615],
        'SpO₂ (%)': [220277, 646],
        'Temp (°C)': [223761, 223762, 676, 678],
        'Glucose': [220621, 226537, 807, 811, 1529],
    }

    # Sample stays
    all_stays = loader.chartevents['icustay_id'].dropna().unique()
    if len(all_stays) > sample_n:
        sampled = np.random.choice(all_stays, sample_n, replace=False)
        charts = loader.chartevents[loader.chartevents['icustay_id'].isin(sampled)]
    else:
        charts = loader.chartevents

    # Collect distributions
    data_frames = []
    for name, ids in vital_itemids.items():
        vals = charts[charts['itemid'].isin(ids)]['valuenum'].dropna()
        if len(vals) == 0:
            continue
        # Remove extreme outliers (0.5th and 99.5th percentile)
        lo, hi = vals.quantile(0.005), vals.quantile(0.995)
        vals = vals[(vals >= lo) & (vals <= hi)]
        df = pd.DataFrame({'Vital Sign': name, 'Value': vals.values})
        data_frames.append(df)

    if not data_frames:
        logger.warning("  No vital data found")
        return

    all_vitals = pd.concat(data_frames, ignore_index=True)

    # Separate plots for different scales
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    # Clinical reference ranges (normal)
    ref_ranges = {
        'Heart Rate': (60, 100),
        'Systolic BP': (90, 140),
        'Diastolic BP': (60, 90),
        'Resp Rate': (12, 20),
        'SpO₂ (%)': (94, 100),
        'Temp (°C)': (36.0, 38.3),
        'Glucose': (70, 180),
    }

    for i, name in enumerate(vital_itemids.keys()):
        ax = axes[i]
        subset = all_vitals[all_vitals['Vital Sign'] == name]['Value']
        if len(subset) == 0:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(subset.values, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='#42A5F5', alpha=0.7),
                        medianprops=dict(color='#F44336', linewidth=2),
                        whiskerprops=dict(color='#666'),
                        flierprops=dict(marker='.', markersize=2, alpha=0.3))

        # Add reference range band
        if name in ref_ranges:
            lo, hi = ref_ranges[name]
            ax.axhspan(lo, hi, alpha=0.15, color='#4CAF50', label='Normal')

        ax.set_title(name, fontweight='bold')
        ax.set_xticklabels([f'n={len(subset):,}'], fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        # Stats annotation
        ax.text(0.02, 0.98,
                f'μ={subset.mean():.1f}\nσ={subset.std():.1f}\nmed={subset.median():.1f}',
                transform=ax.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Hide last subplot if unused
    if len(vital_itemids) < len(axes):
        for j in range(len(vital_itemids), len(axes)):
            axes[j].set_visible(False)

    plt.suptitle('Vital Signs Distributions (with Normal Ranges)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'eda_vitals_boxplots.png')
    plt.savefig(path)
    plt.close()
    logger.info(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 7 — Lab Values Box Plots
# ═══════════════════════════════════════════════════════════════════════════════

def fig_labs_boxplots(loader: MIMICDataLoader, sample_n=5000):
    """Box plots of laboratory values with clinical reference ranges."""
    logger.info(f"Figure 7: Lab values distributions (sampling {sample_n} stays)")

    if loader.labevents is None or len(loader.labevents) == 0:
        logger.warning("  No labevents loaded, skipping")
        return

    # Build lab itemid mapping
    lab_keywords = {
        'Creatinine': 'creatinine',
        'Lactate': 'lactate',
        'WBC': 'wbc',
        'Hemoglobin': 'hemoglobin',
        'Platelets': 'platelet',
        'Bicarbonate': 'bicarbonate',
        'Chloride': 'chloride',
    }

    lab_itemids = {}
    if loader.d_labitems is not None:
        for display_name, keyword in lab_keywords.items():
            mask = loader.d_labitems['label'].str.lower().str.contains(keyword, na=False)
            ids = loader.d_labitems[mask]['itemid'].unique().tolist()
            if ids:
                lab_itemids[display_name] = ids

    if not lab_itemids:
        logger.warning("  No lab mappings found")
        return

    # Sample stays
    all_stays = loader.labevents['icustay_id'].dropna().unique()
    if len(all_stays) > sample_n:
        sampled = np.random.choice(all_stays, sample_n, replace=False)
        labs = loader.labevents[loader.labevents['icustay_id'].isin(sampled)]
    else:
        labs = loader.labevents

    # Clinical reference ranges
    ref_ranges = {
        'Creatinine': (0.6, 1.2),
        'Lactate': (0.5, 2.0),
        'WBC': (4.0, 12.0),
        'Hemoglobin': (12.0, 17.5),
        'Platelets': (150, 400),
        'Bicarbonate': (22, 28),
        'Chloride': (96, 106),
    }

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    for i, (name, ids) in enumerate(lab_itemids.items()):
        ax = axes[i]
        vals = labs[labs['itemid'].isin(ids)]['valuenum'].dropna()
        if len(vals) == 0:
            ax.set_visible(False)
            continue

        # Remove extreme outliers
        lo, hi = vals.quantile(0.005), vals.quantile(0.995)
        vals = vals[(vals >= lo) & (vals <= hi)]

        bp = ax.boxplot(vals.values, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='#66BB6A', alpha=0.7),
                        medianprops=dict(color='#F44336', linewidth=2),
                        whiskerprops=dict(color='#666'),
                        flierprops=dict(marker='.', markersize=2, alpha=0.3))

        if name in ref_ranges:
            lo_r, hi_r = ref_ranges[name]
            ax.axhspan(lo_r, hi_r, alpha=0.15, color='#4CAF50', label='Normal')

        ax.set_title(name, fontweight='bold')
        ax.set_xticklabels([f'n={len(vals):,}'], fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        ax.text(0.02, 0.98,
                f'μ={vals.mean():.2f}\nσ={vals.std():.2f}\nmed={vals.median():.2f}',
                transform=ax.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    for j in range(len(lab_itemids), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Laboratory Values Distributions (with Normal Ranges)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'eda_labs_boxplots.png')
    plt.savefig(path)
    plt.close()
    logger.info(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='EDA Part 1: Demographics & Distributions')
    parser.add_argument('--sample', type=int, default=5000,
                        help='Number of ICU stays to sample for vitals/labs (default: 5000)')
    args = parser.parse_args()

    ensure_dir()

    print("=" * 60)
    print("  EDA PART 1 — Demographics & Distributions")
    print("=" * 60)

    # Load data
    print("\nLoading MIMIC-III data...")
    loader = MIMICDataLoader('data', 'config.yaml')
    merged = loader.merge_data()
    print(f"✓ Loaded {len(merged):,} ICU stays\n")

    # Generate all figures
    fig_dataset_overview(merged, loader)
    fig_age_distribution(merged)
    fig_gender_distribution(merged)
    fig_los_distribution(merged)
    fig_care_units(merged)
    fig_vitals_boxplots(loader, sample_n=args.sample)
    fig_labs_boxplots(loader, sample_n=args.sample)

    print(f"\n✅ Part 1 complete — 7 figures saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
