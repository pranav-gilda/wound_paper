# -------------------------------------------------------------
# cluster_images.py
#
# Unsupervised analysis of O2Sat images. This script uses a
# pre-trained CNN (ResNet50) to extract features from images, then uses
# UMAP and HDBSCAN to discover and visualize clusters of
# visually similar images.
# -------------------------------------------------------------
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pathlib
import numpy as np
import pandas as pd
import umap
import hdbscan
import matplotlib.pyplot as plt
import logging

# --- Configuration ---
IMG_DIR = pathlib.Path("extracted_imgs")
OUTPUT_CSV = "image_clusters.csv"
OUTPUT_PLOT = "image_clusters_visualization.png"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# --- 1. Set up the Feature Extractor (ResNet50) ---
# Use a pre-trained ResNet50 model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
# Set the model to evaluation mode (no training)
model.eval()

# Remove the final classification layer to get feature embeddings
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

# Check for GPU availability and move the model to the GPU if possible
device = "cuda" if torch.cuda.is_available() else "cpu"
feature_extractor.to(device)
logging.info(f"Using device: {device}")

# --- 2. Define Image Transformation ---
# Images must be resized and normalized in the same way the model was trained
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 3. Process Images and Extract Features ---
image_paths = []
all_features = []

logging.info(f"Starting feature extraction from images in: {IMG_DIR}")

image_files = sorted(list(IMG_DIR.glob("*.png")))
if not image_files:
    logging.error(f"No PNG images found in '{IMG_DIR}'. Please check the path.")
else:
    for img_path in image_files:
        try:
            # Open and preprocess the image
            img = Image.open(img_path).convert("RGB")
            input_tensor = preprocess(img)
            input_batch = input_tensor.unsqueeze(0).to(device) # Create a mini-batch

            # Extract features with the model
            with torch.no_grad():
                features = feature_extractor(input_batch)
            
            # Flatten the features and move to CPU
            all_features.append(features.squeeze().cpu().numpy())
            image_paths.append(img_path.name)

        except Exception as e:
            logging.warning(f"Could not process {img_path.name}. Error: {e}")

    all_features = np.array(all_features)
    logging.info(f"Successfully extracted features from {len(all_features)} images.")

    # --- 4. Dimensionality Reduction with UMAP ---
    logging.info("Reducing feature dimensions with UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(all_features)

    # --- 5. Clustering with HDBSCAN ---
    logging.info("Discovering clusters with HDBSCAN...")
    # min_cluster_size can be tuned based on your dataset size
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(embedding)
    
    # -1 is considered noise by HDBSCAN
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    logging.info(f"Found {num_clusters} clusters (plus noise points labeled -1).")

    # --- 6. Save Results to CSV ---
    results_df = pd.DataFrame({
        'filename': image_paths,
        'umap_x': embedding[:, 0],
        'umap_y': embedding[:, 1],
        'cluster_id': cluster_labels
    })
    results_df.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Saved cluster results to {OUTPUT_CSV}")

    # --- 7. Visualize the Clusters ---
    logging.info(f"Generating visualization plot: {OUTPUT_PLOT}")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create a scatter plot of the UMAP embedding, colored by cluster ID
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=cluster_labels,
        cmap='Spectral', # A good colormap for clusters
        s=50,
        alpha=0.7
    )

    ax.set_title('UMAP Projection of O2Sat Images with HDBSCAN Clusters', fontsize=16)
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    
    # Create a legend with unique cluster labels
    unique_labels = np.unique(cluster_labels)
    legend_elements = scatter.legend_elements(num=unique_labels.size)
    ax.legend(handles=legend_elements[0], labels=[f"Cluster {l}" if l != -1 else "Noise" for l in unique_labels], title="Clusters")

    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    plt.show()

    logging.info("--- Process Complete! ---")