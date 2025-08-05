# -------------------------------------------------------------
# This script merges the clinical data with the unsupervised
# image cluster data. It then analyzes the clinical
# characteristics of each visual cluster to discover correlations.
# -------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import re

# --- Configuration ---
CLINICAL_DATA_PATH = Path("clinicaldata_april5_2021_wDx_updated.xlsx")
IMAGE_CLUSTER_PATH = Path("image_clusters.csv")
MERGED_DATA_OUTPUT = "merged_clinical_and_image_data.csv"
ANALYSIS_PLOT_OUTPUT = "clinical_feature_by_cluster.png"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ============================================================================
# 1.  Preprocessing function for clinical data (provided by user)
# ============================================================================
def preprocess_visits(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = (df_raw.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))
              .copy())
    rename_map = {
        "length_(cm)": "length_cm", "width_(cm)": "width_cm", "depth_(cm)": "depth_cm",
        "ph_both_values": "ph_combined", "ph_": "ph_alt",
        "peripheral_temperature": "peripheral_temp", "wound_temperature": "wound_temp"
    }
    df = df.rename(columns=rename_map)
    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        logging.warning(f"Duplicate column names found and handled: {dupes}")
    recode_maps = {
        'ishealed': {'Yes': 1, 'No': 0}, 'gender': {'male': 0, 'female': 1, 'other': 2},
        'tobacco_use': {'non-smoker': 0, 'previous smoker': 1, 'current smoker': 2}
    }
    for col, mapping in recode_maps.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)
    df = df.infer_objects(copy=False)
    numeric_cols = [
        'impedance', 'length_cm', 'width_cm', 'depth_cm', 'wound_temp', 'ph',
        'ph_combined', 'ph_alt', 'tewl', 'age', 'weight', 'bmi', 'study_id'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

# ============================================================================
# 2.  Load and Prepare Both Datasets
# ============================================================================

# Load Clinical Data
logging.info(f"Loading clinical data from: {CLINICAL_DATA_PATH}")
try:
    df_clinical_raw = pd.read_excel(CLINICAL_DATA_PATH)
    df_clinical = preprocess_visits(df_clinical_raw)
    
    visit_col_name = None
    possible_visit_cols = ['redcap_event_name', 'visit', 'visit_number', 'visit_id'] 
    for col in possible_visit_cols:
        if col in df_clinical.columns:
            visit_col_name = col
            logging.info(f"Found visit column in clinical data as: '{visit_col_name}'")
            break
            
    if visit_col_name:
        df_clinical['visit_standardized'] = 'V' + df_clinical[visit_col_name].astype(str).str.extract(r'(\d+)').fillna('0')
    else:
        logging.error(f"FATAL: Could not find a usable visit column in the clinical data.")
        logging.error(f"Available columns are: {df_clinical.columns.tolist()}")
        exit()

    logging.info(f"Clinical data loaded. Shape: {df_clinical.shape}")
except FileNotFoundError:
    logging.error(f"FATAL: Clinical data file not found at '{CLINICAL_DATA_PATH}'. Exiting.")
    exit()

# Load Image Cluster Data
logging.info(f"Loading image cluster data from: {IMAGE_CLUSTER_PATH}")
try:
    df_images = pd.read_csv(IMAGE_CLUSTER_PATH)
    logging.info(f"Image cluster data loaded. Shape: {df_images.shape}")
except FileNotFoundError:
    logging.error(f"FATAL: Image cluster file not found at '{IMAGE_CLUSTER_PATH}'. Exiting.")
    exit()

# --- Create a common merge key ---
def parse_filename(filename):
    match = re.match(r'(\d+)_V(\d+)_o2sat\.png', filename)
    if match:
        study_id = int(match.group(1))
        visit = f"V{match.group(2)}"
        return study_id, visit
    return None, None

df_images[['study_id', 'visit_standardized']] = df_images['filename'].apply(
    lambda x: pd.Series(parse_filename(x))
)
df_images.dropna(subset=['study_id', 'visit_standardized'], inplace=True)
df_images['study_id'] = df_images['study_id'].astype(int)

logging.info("Successfully parsed study_id and visit from image filenames.")

# ============================================================================
# 3.  Merge the Datasets
# ============================================================================
logging.info("Merging clinical and image cluster data...")

df_merged = pd.merge(
    df_clinical,
    df_images[['study_id', 'visit_standardized', 'cluster_id']],
    on=['study_id', 'visit_standardized'],
    how='left'
)

df_merged['cluster_id'] = df_merged['cluster_id'].fillna(-2)

df_merged.to_csv(MERGED_DATA_OUTPUT, index=False)
logging.info(f"Merged data saved to '{MERGED_DATA_OUTPUT}'. Shape: {df_merged.shape}")

# ============================================================================
# 4.  Analyze and Visualize Correlations
# ============================================================================
logging.info("Analyzing clinical features by visual cluster...")

features_to_analyze = ['impedance', 'wound_temp', 'ph', 'tewl', 'ishealed', 'age', 'weight']

df_analysis = df_merged[df_merged['cluster_id'] >= 0].copy()

if df_analysis.empty:
    logging.error("No matching data found after merging. Cannot perform analysis.")
else:
    df_analysis['cluster_id'] = df_analysis['cluster_id'].astype(int)
    
    cluster_summary = df_analysis.groupby('cluster_id')[features_to_analyze].mean().reset_index()
    
    print("\n--- Clinical Feature Averages per Visual Cluster ---")
    print(cluster_summary)
    print("\nNote: 'ishealed' average represents the proportion of healed cases in that cluster.")

    # --- Create a visualization ---
    feature_to_plot = 'ishealed'  # Change this to any feature you want to visualize
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))
    
    sns.barplot(
        data=cluster_summary,
        x='cluster_id',
        y=feature_to_plot,
        palette='viridis',
        hue='cluster_id', # Assign hue to the x-variable
        legend=False,     # Disable the legend as it's redundant
        ax=ax
    )
    
    ax.set_title(f'Average {feature_to_plot.replace("_", " ").title()} by Visual Cluster', fontsize=18, weight='bold')
    ax.set_xlabel('Visual Cluster ID', fontsize=14)
    ax.set_ylabel(f'Average {feature_to_plot.title()}', fontsize=14)
    ax.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(ANALYSIS_PLOT_OUTPUT, dpi=300)
    logging.info(f"Analysis plot saved to '{ANALYSIS_PLOT_OUTPUT}'")
    plt.show()

logging.info("--- Analysis Complete! ---")
