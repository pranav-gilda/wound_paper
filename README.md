# Project: Unsupervised Discovery of Wound Healing Phenotypes

**Date:** August 4, 2025
**Status:** Initial image analysis complete. Next Phase (Longitudinal Analysis) Proposed.

## 1. Overview

This project aims to discover objective, data-driven biomarkers for wound healing from O2Sat spectral images. Due to a severe class imbalance in the dataset (<10% "healed" cases), we have adopted an unsupervised learning framework. Instead of predicting a label, our goal is to discover natural patterns within the image data and then investigate the clinical meaning of these patterns.

---

## 2. Project Structure

The repository is organized into distinct modules for a clear and reproducible workflow:

```text
SMART_SENSOR_DATA/
│
├── crop_pdf/
│   ├── extract_pdfs_crop_o2sat.py  # Main script to run cropping
│   ├── worker.py                   # Helper script for isolated processing
│   └── extracted_imgs/             # OUTPUT: Cropped O2Sat images
│
├── image_analysis/
│   ├── cluster_images.py           # Script for feature extraction & clustering
│   ├── analyze_clusters.py         # Script for clinical correlation & heatmap
│   ├── image_clusters.csv          # OUTPUT: UMAP coordinates and cluster IDs
│   └── ...                         # OUTPUT: All plots and summary tables
│
├── sensor_project_images/
│   └── ...                         # INPUT: Raw PDF reports from sensor (ignored by .gitignore)
│
├── clinicaldata_april5_2021_wDx_updated.xlsx # INPUT: Raw clinical data 
├── sensor_ml_validated.ipynb       # Jupyter Notebook for Patient-Level Cross Validated Analysis
├── requirements.txt                # All Python dependencies
└── README.md                       # This file
```

---

## 3. How to Run the Analysis

Follow these steps in order to reproduce the analysis from raw data to final insights.

### Step 1: Setup

1.  Clone the repository.
2.  Set up a Python virtual environment: `python -m venv .venv` and activate it.
3.  Install all required dependencies: `pip install -r requirements.txt`

### Step 2: Place Raw Data

1.  Ensure your raw clinical data file, `clinicaldata_april5_2021_wDx_updated.xlsx`, is in the project's root directory.
2.  Ensure your raw PDF reports are placed inside the `sensor_project_images/` directory.

### Step 3: Run Image Cropping (Phase 1)

Navigate to the cropping directory and run the main extraction script. This will populate the `crop_pdf/extracted_imgs/` folder.

```bash
cd crop_pdf
python extract_pdfs_crop_o2sat.py
```

### Step 4: Run Unsupervised Clustering (Phase 2)

Navigate to the analysis directory and run the clustering script. This uses the images from the previous step to discover visual phenotypes.

```bash
cd ../image_analysis
python cluster_images.py
```

### Step 5: Run Clinical Correlation (Phase 3)

Finally, run the last analysis script to merge the visual clusters with the clinical data and generate the final heatmap and summary table.

```bash
python analyze_clusters.py
python clusters_viz.py
```

---

## 4. Methodology & Findings

### Phase 1: Automated Image Preparation

* **Script:** `crop_pdf/extract_pdfs_crop_o2sat.py`
* **Process:** A Python script automates the extraction of O2Sat scans from source PDFs. A `worker.py` script ensures each PDF is handled in an isolated process to prevent library caching issues.
* **Output:** A clean folder (`crop_pdf/extracted_imgs/`) containing one cropped PNG file per visit.

### Phase 2: Unsupervised Visual Clustering

* **Script:** `image_analysis/cluster_images.py`
* **Process:** We use a pre-trained **ResNet50** model to convert each image into a numerical feature vector. These vectors are then projected into 2D space using **UMAP** and grouped into "visual phenotypes" using **HDBSCAN**.

### Phase 3: Clinical Correlation & Key Findings

* **Script:** `image_analysis/analyze_clusters.py`
* **Process:** The discovered visual clusters are merged with the main clinical dataset to analyze the clinical profile of each group.
* **Key Findings:**
    * **The "Healing Phenotype" (Cluster 5):** A visual group with a **71.4% healing rate**, characterized by high impedance, high wound temperature, and moderately low pH.
    * **Two Distinct Non-Healing States (Clusters 7 & 11):** We found two non-healing phenotypes with opposite clinical profiles (e.g., high vs. low impedance), suggesting different biological reasons for stalled healing.
    * **Biomarker "Sweet Spots":** The data suggests that extreme impedance values (both high and low) are often associated with poor outcomes, while lower wound temperature & TEWL are better and the most successful clusters occupy a more moderate range.

---

## 5. Next Steps: Longitudinal Analysis with Siamese Networks

* **Objective:** To transition from static, visit-level analysis to a dynamic, patient-level analysis that models the **change** in wound status over time.
* **Proposed Methodology:**
    * We will build a **Siamese Network**, a neural network designed to compare two inputs.
    * It will be trained to learn what constitutes a meaningful visual **change** between two O2Sat scans from the same patient.
    * The output will be a "change vector" representing the wound's evolution. We hypothesize that this vector will be a powerful biomarker for signalling healing status.
