# MSDS565-Images-TeamKAI

## Overview
This project implements an end-to-end CRISP-DM computer vision pipeline using a custom web-scraped image dataset. It covers data collection, manual cleaning and exploratory data analysis (EDA), transfer learning across pretrained Keras applications of varying sizes, model interpretability (Saliency, SmoothGrad, and Grad-CAM), and deployment via a live Streamlit web application.

---


## Repository Structure

The following represents the required repository structure. Project files will be added as they are completed.

```text
MSDS565-Images-TeamKAI/
├── README.md
├── environment.yml
├── data/
│   └── P2_manifest.csv
├── models/
│   ├── P2_N3_<application>.keras
│   └── [label_mapping_file]
├── notebooks/
│   ├── ddgs_scraper.py
│   ├── image_explainers.py
│   ├── P2_Images-1_Scrape_TeamKAI.ipynb
│   ├── P2_Images-2_Preprocessing_TeamKAI.ipynb
│   ├── P2_Images-3_TransferLearning_TeamKAI.ipynb
│   └── P2_Images-4_xAI_TeamKAI.ipynb
└── app/
    ├── image_explainers.py
    └── P2_Images-5_App_TeamKAI.py
```

Scraped images, cleaned images, and preprocessed image arrays are stored locally outside the GitHub repository. Only the dataset manifest is committed to the `data/` folder.
  ```

---

## Chosen Classes & Rationale
We selected **[Number]** visually distinguishable, non-benchmark classes (minimum 200 usable images per class after cleaning):

1. **`[Class 1 Name]`** — [Brief justification: why chosen and what makes it distinct]
2. **`[Class 2 Name]`** — [Brief justification: why chosen and what makes it distinct]
3. **`[Class 3 Name]`** — [Brief justification: why chosen and what makes it distinct]
*(Add or remove classes as needed, 3–10 total)*

* **Expected Confusion Pair:** Before modeling, we predicted that **`[Class A]`** and **`[Class B]`** would be the most frequently confused pair because [explain visual overlap].

---

## Dataset Construction & Search Terms
Images were scraped locally (outside this repository) using `ddgs_scraper.py` via the Bing image index. Multiple queries were used per class to ensure visual diversity and avoid narrow search slices.

| Class Label | Search Queries Used | Raw Scraped | Kept (Post-Cleaning) |
| :--- | :--- | :---: | :---: |
| `[Class 1]` | `"query 1"`, `"query 2"`, `"query 3"` | `[Count]` | `[Count >= 200]` |
| `[Class 2]` | `"query 1"`, `"query 2"`, `"query 3"` | `[Count]` | `[Count >= 200]` |
| `[Class 3]` | `"query 1"`, `"query 2"`, `"query 3"` | `[Count]` | `[Count >= 200]` |

* **Data Privacy & Copyright Note:** No raw, cleaned, or preprocessed image files are stored in this repository. `data/P2_manifest.csv` serves as the complete record of image URLs, queries, labels, and survival status (`kept` column).

---

## Pipeline Walkthrough (Notebooks & Artifacts)
Each notebook is self-contained, executed top-to-bottom, and produces artifacts consumed by subsequent stages:

1. **`notebooks/P2_Images-1_Scrape_<TeamName>.ipynb` (Dataset Construction)**
   * **Action:** Executes multi-query web scraping into local directories, validates image readability, and flags any class at risk of falling below 200 usable images.
   * **Produces:** `data/P2_manifest.csv` containing `filename`, `label`, `source_url`, `query`, and `readable`.
2. **`notebooks/P2_Images-2_Preprocessing_<TeamName>.ipynb` (Cleaning, Preprocessing & EDA)**
   * **Action:** Documents manual removal of duplicates, wrong subjects, watermarks, and collages. Performs EDA (dimension distributions, per-class mean images, class support), converts images to 3-channel RGB, applies **[Stretch / Pad / Center-Crop]** resizing to **`[Width x Height]`**, and executes a stratified 80/20 train/test split (`random_state=42`).
   * **Produces:** Local preprocessed train/test datasets, the saved label index mapping, and updates `data/P2_manifest.csv` with a `kept` boolean column.
3. **`notebooks/P2_Images-3_TransferLearning_<TeamName>.ipynb` (Transfer Learning)**
   * **Action:** Handles class imbalance via **[Oversampling / Class Weights]** and applies **[list augmentations]** to the training set only. Extracts embeddings using three frozen pretrained Keras applications spanning small, medium, and large parameter counts (**`[Small Model]`**, **`[Medium Model]`**, **`[Large Model]`**) and trains identical linear probes (`Dense` softmax layer) on each.
   * **Produces:** Trained linear probes saved to `models/P2_N3_<application>.keras` alongside the label index mapping.
   * **Large File Storage:** `[State whether your .keras files are committed directly, tracked via Git LFS, or hosted externally]`.
4. **`notebooks/P2_Images-4_xAI_<TeamName>.ipynb` (Interpretability & Shortcut Audit)**
   * **Action:** Rebuilds joined models (`Input -> Pretrained App -> Linear Probe`) and generates raw Saliency, SmoothGrad (`smooth=32`), and Grad-CAM maps using `image_explainers.py` on correct predictions, misclassifications, and cross-model comparisons. Audits for background/border shortcuts via occlusion ablation.
   * **Produces:** Diagnostic visualizations and shortcut ablation metrics.

---

## Setup & Running the Streamlit App

### 1. Environment Setup
Recreate the exact Conda environment used to build and run this pipeline:
```bash
conda env create -f environment.yml
conda activate <your-env-name>
```

### 2. Launch the App
From the **root directory** of the repository, run:
```bash
streamlit run app/P2_Images-5_App_<TeamName>.py
```

### 3. App Features
* **Image Upload:** Accepts user-uploaded images and applies the exact **`[Resize Strategy]`** (`[Width x Height]`) used in Notebook 2.
* **Side-by-Side Model Comparison:** Evaluates the image across all three models simultaneously (one row per model size).
* **Probabilities & xAI Maps:** Displays each model's sorted class probability bar chart alongside overlaid **Raw Saliency**, **SmoothGrad**, and **Grad-CAM** explanations.
