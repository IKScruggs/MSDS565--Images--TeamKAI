
# MSDS565-Images-TeamKAI

**Course:** MSDS 565 — Project 2: Image Classification  
**Team:** TeamKAI — KeiTonyia, Ashley, India

## Overview

TeamKAI developed an end-to-end computer vision pipeline to classify five cloud types using a custom web-scraped image dataset.

Following the CRISP-DM framework, the project includes image collection, cleaning, exploratory data analysis (EDA), preprocessing, transfer learning, model evaluation, explainable AI (XAI), and deployment through a Streamlit web application.

The project compares four pretrained Keras applications and uses Saliency, SmoothGrad, and Grad-CAM to investigate model predictions and potential classification shortcuts.

---

## Repository Structure

The following represents the intended final repository structure. Generated artifacts are added as they become available.

```text
MSDS565-Images-TeamKAI/
├── README.md
├── environment.yml
├── data/
│   └── P2_manifest.csv
├── models/
│   ├── P2_N3_mobilenetv2.keras
│   ├── P2_N3_effnetv2s.keras
│   ├── P2_N3_resnet50.keras
│   ├── P2_N3_effnetv2m.keras
│   └── P2_label_map.json
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

Raw images, cleaned images, and preprocessed image arrays are stored locally outside the GitHub repository.

---

## Chosen Classes and Rationale

We selected five cloud categories that exhibit recognizable visual differences while presenting realistic image-classification challenges.

1. **Cumulus:** Puffy clouds with relatively distinct shapes and boundaries.
2. **Stratus:** Broad, flat cloud layers that often produce overcast skies.
3. **Cirrus:** Thin, wispy clouds with distinctive feather-like formations.
4. **Cumulonimbus:** Large, vertically developed clouds commonly associated with thunderstorms.
5. **Altocumulus:** Smaller, patchy cloud formations that frequently appear in groups.

These categories allow the models to learn differences in cloud shape, texture, density, and structure.

However, variations in lighting, camera angle, scale, and background conditions can make some cloud types difficult to distinguish.

**Expected Confusion Pair:**

Before modeling, we predicted that **Cumulus and Altocumulus** would be the most frequently confused classes because both can exhibit rounded, puffy formations. Their differences in altitude and scale may not be obvious from individual photographs.

Notebook 3 did not confirm this prediction as the most frequent confusion pair for any of the four models.

---

## Dataset Construction and Search Terms

Images were collected using the instructor-provided `ddgs_scraper.py` module with the Bing image-search backend.

Three search queries were used per class to improve visual diversity and reduce dependence on a single search phrase.

The scraper was configured with:

- Minimum image resolution: 224 × 224 pixels
- Acceptable aspect ratio: 0.5–2.0
- Maximum requested images per class: 400
- Per-image timeout: 10 seconds
- Moderate safe search
- US-English search region

### Image Collection Summary

| Class | Search Queries | Downloaded and Readable |
|---|---|---:|
| Cumulus | `cumulus clouds`; `fair weather cumulus sky`; `cumulus humilis clouds` | 400 |
| Stratus | `stratus clouds overcast`; `stratus cloud layer sky`; `low stratus fog clouds sky` | 400 |
| Cirrus | `cirrus clouds`; `wispy cirrus clouds sky`; `cirrus fibratus clouds` | 400 |
| Cumulonimbus | `cumulonimbus clouds`; `towering cumulonimbus thunderstorm`; `cumulonimbus incus cloud` | 400 |
| Altocumulus | `altocumulus clouds`; `altocumulus cloud patches sky`; `altocumulus stratiformis clouds` | 400 |
| **Total** | **15 search queries** | **2,000** |

Notebook 1 records 2,000 downloaded and readable images across the five classes.

### Dataset Manifest

The file `data/P2_manifest.csv` documents the collected images, including:

- Image filenames
- Class labels
- Source URLs
- Search queries
- Image readability
- Cleaning status (`kept`)

Notebook 2 updates the manifest to reflect preprocessing and dataset preparation.

**Final cleaning counts:** Post-cleaning removal totals should be confirmed against Notebook 2's accounting results.

**Data Storage:** Raw images, cleaned images, and preprocessed arrays are maintained locally and are not committed to GitHub.

---

## Pipeline Walkthrough: Notebooks and Artifacts

The project consists of four sequential notebooks that document the image-classification workflow.

### Notebook 1 — Dataset Construction

**File:** `notebooks/P2_Images-1_Scrape_TeamKAI.ipynb`

**Purpose:**

Collect a custom image dataset representing the five selected cloud categories.

**Methodology:**

- Uses the instructor-provided scraping module.
- Executes multiple image-search queries per category.
- Validates image readability.
- Records source information and labels.
- Exports the dataset manifest.

**Primary Output:**

`data/P2_manifest.csv`

---

### Notebook 2 — Cleaning, Preprocessing, and EDA

**File:** `notebooks/P2_Images-2_Preprocessing_TeamKAI.ipynb`

**Purpose:**

Prepare the collected images for consistent model training and evaluation.

**Methodology:**

- Reviews images for duplicates and unsuitable content.
- Examines class distributions and image characteristics.
- Converts images to three-channel RGB.
- Resizes images to 224 × 224 pixels.
- Preserves aspect ratios using letterbox resizing.
- Centers images on a gray canvas with padding value 128.
- Performs a stratified 80/20 training/testing split.
- Uses `random_state=42` for reproducibility.

**Primary Outputs:**

Locally stored preprocessed datasets:

- `x_train.npy`
- `y_train.npy`
- `x_test.npy`
- `y_test.npy`
- `label_map.json`

The dataset manifest is also updated to document image cleaning and retention.

---

### Notebook 3 — Transfer Learning and Model Evaluation

**File:** `notebooks/P2_Images-3_TransferLearning_TeamKAI.ipynb`

**Purpose:**

Compare pretrained convolutional neural network architectures of different sizes using transfer learning.

**Models Evaluated:**

| Model | Size Category |
|---|---|
| MobileNetV2 | Small |
| EfficientNetV2S | Medium |
| ResNet50 | Medium |
| EfficientNetV2M | Large |

**Methodology:**

- Uses pretrained Keras applications with frozen feature-extraction layers.
- Extracts image embeddings.
- Trains a separate single-layer softmax linear probe for each model.
- Uses training-only augmentation, including horizontal flipping, rotation, zoom, brightness, and contrast adjustments.
- Includes an oversampling procedure for class imbalance.
- Evaluates each model using held-out test images.

The training split contained 320 images per class, so the oversampling procedure did not alter the already balanced training data.

### Model Performance

Notebook 3 reports results from 400 test images, with 80 images per class.

| Model | Test Accuracy | Macro-F1 |
|---|---:|---:|
| MobileNetV2 | 77.2% | 0.772 |
| EfficientNetV2S | 76.7% | 0.766 |
| ResNet50 | **81.0%** | **0.811** |
| EfficientNetV2M | 73.0% | 0.728 |

**Key Finding:**

ResNet50 achieved the highest test accuracy and macro-F1 score among the four evaluated models.

The largest architecture, EfficientNetV2M, did not produce the strongest results, demonstrating that a larger pretrained model does not necessarily improve performance on this dataset.

**Primary Outputs:**

```text
models/P2_N3_mobilenetv2.keras
models/P2_N3_effnetv2s.keras
models/P2_N3_resnet50.keras
models/P2_N3_effnetv2m.keras
models/P2_label_map.json
```

**Model Storage:** Confirm that the four trained probe files and label mapping are committed to `models/`. If any model requires Git LFS or external storage, document its location here.

---

### Notebook 4 — Explainable AI and Shortcut Audit

**File:** `notebooks/P2_Images-4_xAI_TeamKAI.ipynb`

**Purpose:**

Investigate how the trained models make classification decisions and whether predictions rely on relevant cloud features or potentially misleading image characteristics.

**Explainability Methods:**

1. **Raw Saliency:** Highlights image regions where pixel changes influence predictions.
2. **SmoothGrad:** Averages saliency maps from multiple noisy inputs to produce more stable explanations.
3. **Grad-CAM:** Uses convolutional feature activations to highlight regions contributing to a predicted class.

SmoothGrad uses 32 samples in the notebook's configured analysis.

**Additional Analyses:**

- Correctly classified examples
- Misclassified examples
- Cross-model comparisons
- Background and border shortcut investigations
- Occlusion-based ablation tests
- Potential influence of watermarks and banners

**Primary Outputs:**

Diagnostic visualizations, explanation maps, and shortcut-analysis results.

**Verification Note:**

The uploaded Notebook 4 reports 321 test images, while Notebook 3 reports 400. The saved datasets should be reconciled before interpreting their results as directly comparable.

Some Notebook 4 interpretation sections also require completion.

---

## Setup and Running the Streamlit App

### 1. Environment Setup

This project uses a Conda environment documented in `environment.yml`.

Open Anaconda Prompt and navigate to the repository's root directory.

Create the environment:

```bash
conda env create -f environment.yml
```

Open `environment.yml` to identify the environment name specified by the `name:` field.

Activate the environment:

```bash
conda activate YOUR_ENVIRONMENT_NAME
```

Replace `YOUR_ENVIRONMENT_NAME` with the actual environment name.

### 2. Required Model Files

Before launching the application, verify that the following files are available:

```text
models/P2_N3_mobilenetv2.keras
models/P2_N3_effnetv2s.keras
models/P2_N3_resnet50.keras
models/P2_N3_effnetv2m.keras
models/P2_label_map.json
app/image_explainers.py
```

The application reconstructs the pretrained Keras feature extractors and joins them with the saved linear probes.

### 3. Launch the Streamlit Application

From the repository's root directory, run:

```bash
streamlit run app/P2_Images-5_App_TeamKAI.py
```

The application opens in a web browser.

### 4. Application Features

**Image Upload**

Users can upload cloud photographs in supported image formats.

Uploaded images are converted to RGB and resized to 224 × 224 pixels using the same aspect-ratio-preserving letterbox method applied in Notebook 2.

**Model Comparison**

The application evaluates uploaded images using each available pretrained model and its trained classification probe.

It displays:

- Predicted cloud class
- Class probability distributions
- Model comparison results
- Raw Saliency overlays
- SmoothGrad overlays
- Grad-CAM overlays

**Prediction Diagnostics**

The application also provides:

- Model agreement and disagreement notifications
- Low-confidence warnings
- Prediction uncertainty summaries
- Information about limitations of the classifier

The low-confidence warning uses a heuristic threshold rather than a calibrated probability guarantee.

### 5. Application Limitations

The classifier is trained to recognize only five cloud categories.

Images outside these categories may still receive a confident prediction because the models must select among the available classes.

The dataset also includes web-scraped photographs with potentially distracting backgrounds, banners, or watermarks.

Explainability methods help investigate these limitations but do not guarantee that the models rely exclusively on cloud features.

Live application functionality should be verified using the saved model files and label mapping.

---

## Project Summary

TeamKAI developed a custom cloud image-classification pipeline integrating web scraping, preprocessing, transfer learning, model evaluation, explainable AI, and Streamlit deployment.

Four pretrained Keras applications were evaluated on five cloud categories.

ResNet50 achieved the highest recorded test accuracy of **81.0%**, outperforming both smaller and larger alternatives in this experiment.

The project demonstrates how transfer learning and explainability methods can be combined to investigate model performance, interpret predictions, and identify potential limitations in a custom image dataset.

---

## Team

**TeamKAI**

- KeiTonia
- Ashley
- India

**Course:** MSDS 565 — Predictive Modeling and Analytics
