# -------------------------------------------------------------
# characteristics of all discovered visual clusters and saves
# the final summary table.
# -------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import re

# --- Configuration ---
# Ensure this path points to the merged data file
MERGED_DATA_PATH = Path("merged_clinical_and_image_data.csv")
SUMMARY_OUTPUT_CSV = "cluster_summary_with_demographics.csv"
HEATMAP_OUTPUT_PLOT = "clinical_heatmap_by_cluster.png"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ============================================================================
# 1.  Load Merged Data
# ============================================================================
logging.info(f"Loading merged clinical and image data from: {MERGED_DATA_PATH}")
try:
    df_merged = pd.read_csv(MERGED_DATA_PATH)
except FileNotFoundError:
    logging.error(f"FATAL: Merged data file not found at '{MERGED_DATA_PATH}'.")
    logging.error("Please run the previous analysis script first to generate this file.")
    exit()

# ============================================================================
# 2.  Analyze and Summarize Data
# ============================================================================
logging.info("Analyzing clinical features by visual cluster...")

features_to_analyze = ['impedance', 'wound_temp', 'ph', 'tewl', 'ishealed', 'age', 'weight']

# Filter out noise points (-1) and points without a corresponding image (-2)
df_analysis = df_merged[df_merged['cluster_id'] >= 0].copy()

if df_analysis.empty:
    logging.error("No matching data found for analysis. Cannot proceed.")
else:
    df_analysis['cluster_id'] = df_analysis['cluster_id'].astype(int)
    
    # Calculate the mean of each feature for each cluster
    cluster_summary = df_analysis.groupby('cluster_id')[features_to_analyze].mean()
    
    # Save the summary table to a new CSV file
    cluster_summary.to_csv(SUMMARY_OUTPUT_CSV)
    logging.info(f"Full cluster summary saved to '{SUMMARY_OUTPUT_CSV}'")
    
    print("\n--- Clinical & Demographic Averages per Visual Cluster ---")
    print(cluster_summary)
    print("\nNote: 'ishealed' average represents the proportion of healed cases in that cluster.")

    # ============================================================================
    # 3.  Generate Heatmap Visualization
    # ============================================================================
    logging.info(f"Generating heatmap visualization: {HEATMAP_OUTPUT_PLOT}")
    
    # Standardize the data for better color mapping (so one feature doesn't dominate)
    # This converts each feature's values to a scale of 0 to 1.
    summary_normalized = cluster_summary.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 10))
    
    sns.heatmap(
        summary_normalized,
        annot=cluster_summary,  # Annotate with the original (non-normalized) values
        fmt=".2f",              # Format annotations to two decimal places
        cmap='viridis',         # A vibrant and clear colormap
        linewidths=.5,
        ax=ax
    )
    
    ax.set_title('Clinical Profile of Visual Clusters', fontsize=20, weight='bold', pad=20)
    ax.set_xlabel('Clinical & Demographic Features', fontsize=14)
    ax.set_ylabel('Visual Cluster ID', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(HEATMAP_OUTPUT_PLOT, dpi=300)
    logging.info(f"Heatmap saved to '{HEATMAP_OUTPUT_PLOT}'")
    plt.show()

logging.info("--- Analysis Complete! ---")
